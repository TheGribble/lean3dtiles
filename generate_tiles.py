"""Build a global land + sea terrain 3D Tileset (3D Tiles 1.1, GLB content).

Quadtree over lon/lat. Tile (L, x, y) spans 360/2^L deg lon by 180/2^L deg lat and is a
COLS x ROWS cell grid, so grid spacing halves each level.

Land: cells wholly inland use the regular lattice. Cells touched by the coastline are
cut *exactly* against the land polygon and re-triangulated (constrained Delaunay), so
coasts and waterways narrower than a cell are represented correctly. Heights are
clamped >= 0. Sea: a flat mesh a few metres below 0 under every coastal cell.
"""
import argparse, json, math, os
import numpy as np
import rasterio
import geopandas as gpd
import shapely
import trimesh

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
COLS, ROWS = 704, 352  # cells per tile (set from --cols)


def ecef(lat, lon, h):
    """WGS84 geodetic (radians, metres) -> ECEF (Z-up) as (...,3)."""
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([
        (n + h) * np.cos(lat) * np.cos(lon),
        (n + h) * np.cos(lat) * np.sin(lon),
        (n * (1 - E2) + h) * np.sin(lat)], -1)


def zup_to_yup(p):
    # 3D Tiles rotates glTF Y-up to Z-up; pre-apply the inverse: (x,y,z)->(x,z,-y)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def sample_dem(dem, lat, lon):
    """Bilinear sample of a 2D lat/lon mesh from an in-memory DEM array."""
    arr, t = dem
    px, py = t.a, -t.e
    cf = np.clip((lon - t.c) / px - 0.5, 0, arr.shape[1] - 1.0001)
    rf = np.clip((t.f - lat) / py - 0.5, 0, arr.shape[0] - 1.0001)
    c0, r0 = cf.astype(int), rf.astype(int)
    fc, fr = cf - c0, rf - r0
    top = arr[r0, c0] * (1 - fc) + arr[r0, c0 + 1] * fc
    bot = arr[r0 + 1, c0] * (1 - fc) + arr[r0 + 1, c0 + 1] * fc
    return top * (1 - fr) + bot * fr


def patch_glb(data):
    """trimesh writes vertex colours with no material, which glTF renders fully
    metallic (near-black). Add a matte material and bind it to every primitive."""
    import struct
    jlen, = struct.unpack_from("<I", data, 12)
    js = json.loads(data[20:20 + jlen])
    js["materials"] = [{"pbrMetallicRoughness": {
        "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}]
    for m in js["meshes"]:
        for pr in m["primitives"]:
            pr["material"] = 0
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    rest = data[20 + jlen:]
    return (struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(j) + len(rest))
            + struct.pack("<I4s", len(j), b"JSON") + j + rest)


SEA_RGBA = np.array([22, 78, 140, 255], np.uint8)


def land_colors(h):
    """Elevation-graded greens (low = deep green, high = dry olive/tan)."""
    stops = [0, 1500, 4000]
    cols = np.array([[40, 110, 45], [95, 140, 60], [150, 140, 100]], float)
    rgb = np.stack([np.interp(h, stops, cols[:, k]) for k in range(3)], -1)
    return np.concatenate([rgb, np.full((len(h), 1), 255.)], -1).astype(np.uint8)


def compact(allP, allN, allH, faces, color):
    """Drop unreferenced vertices; returns (pos, normals, RGBA, faces)."""
    used = np.unique(faces)
    remap = np.full(allP.shape[0], -1, np.int64)
    remap[used] = np.arange(len(used))
    col = land_colors(allH[used]) if color is None else np.tile(color, (len(used), 1))
    return allP[used], allN[used], col, remap[faces]


def skirt_faces(keep, idx, off):
    """Skirt triangles down the tile perimeter (outward-facing); down-verts = idx+off."""
    sk = []
    def edge(cells, a_idx, b_idx):
        for ok_, a0, b0 in zip(cells, a_idx, b_idx):
            if ok_:
                sk.append([a0, a0 + off, b0]); sk.append([b0, a0 + off, b0 + off])
    edge(keep[0, :], idx[0, :-1], idx[0, 1:])                    # south, W->E
    edge(keep[:, -1], idx[:-1, -1], idx[1:, -1])                 # east, S->N
    edge(keep[-1, ::-1], idx[-1, :0:-1], idx[-1, -2::-1])        # north, E->W
    edge(keep[::-1, 0], idx[:0:-1, 0], idx[-2::-1, 0])           # west, N->S
    return np.array(sk, dtype=np.int64).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--land", required=True)
    ap.add_argument("--max-level", type=int, default=5)
    ap.add_argument("--bbox", type=float, nargs=4, default=None, help="test only: w s e n; tiles outside are coarse sea")
    ap.add_argument("--cols", type=int, default=704, help="grid cells per tile along longitude (rows = cols/2)")
    ap.add_argument("--out", default="output/terrain_tiles")
    ap.add_argument("--sea-offset", type=float, default=4.0, help="metres the sea sits below 0 to avoid z-fighting with the coast")
    a = ap.parse_args()
    N = a.max_level
    global COLS, ROWS
    COLS, ROWS = a.cols, a.cols // 2
    print(f"{COLS}x{ROWS} cells/tile, levels 0..{N}, finest cell {360 / (COLS * 2 ** N):.4f} deg")
    os.makedirs(os.path.join(a.out, "tiles"), exist_ok=True)

    print("Loading land polygons...")
    land = shapely.union_all(gpd.read_file(a.land).geometry.values)
    shapely.prepare(land)

    print("Loading DEM...")
    with rasterio.open(a.dem) as src:
        dem = (src.read(1).astype(np.float32), src.transform)

    nfiles = 0
    total_bytes = 0
    EPS = 1e-9

    def vertex_normals(lat, lon, h, d):
        """Normals from DEM central differences (a pure function of lon/lat, so seamless)."""
        def pos(la, lo):
            return ecef(np.radians(la), np.radians(lo), np.maximum(sample_dem(dem, la, lo), 0))
        n = np.cross(pos(lat, lon + d) - pos(lat, lon - d), pos(lat + d, lon) - pos(lat - d, lon))
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        rad = ecef(np.radians(lat), np.radians(lon), h)
        rad /= np.linalg.norm(rad, axis=1, keepdims=True)
        return np.where((ln > 1e-3) & (np.abs(lat)[:, None] < 89.9), n / np.maximum(ln, 1e-12), rad)

    def land_mesh(lon, lat, dl, tl, step):
        """Returns (pos, nor, rgba, faces, hmin, hmax, cellclass) for the land of one tile."""
        R, C = lon.shape
        M = shapely.intersects_xy(tl, lon, lat)
        a_, b_, c_, d_ = M[:-1, :-1], M[:-1, 1:], M[1:, :-1], M[1:, 1:]
        allland = a_ & b_ & c_ & d_
        anyland = a_ | b_ | c_ | d_
        # cells crossed by the coastline: densify the boundary and mark the cells its points fall in
        touched = np.zeros(allland.shape, bool)
        bpts = shapely.get_coordinates(shapely.segmentize(shapely.boundary(tl), dl * 0.5))
        if len(bpts):
            ci = np.floor((bpts[:, 0] - lon[0, 0]) / dl).astype(int)
            cj = np.floor((bpts[:, 1] - lat[0, 0]) / dl).astype(int)
            ok = (ci >= 0) & (ci < allland.shape[1]) & (cj >= 0) & (cj < allland.shape[0])
            touched[cj[ok], ci[ok]] = True
            t = touched.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = slice(max(dy, 0), allland.shape[0] + min(dy, 0))
                    yd = slice(max(-dy, 0), allland.shape[0] + min(-dy, 0))
                    xs = slice(max(dx, 0), allland.shape[1] + min(dx, 0))
                    xd = slice(max(-dx, 0), allland.shape[1] + min(-dx, 0))
                    t[yd, xd] |= touched[ys, xs]
            touched = t
        full = allland & ~touched
        clip = (anyland | touched) & ~full

        tris = []
        if full.any():
            la = np.stack([lon, lat], -1)
            ia, ib = la[:-1, :-1][full], la[:-1, 1:][full]
            ic, id_ = la[1:, :-1][full], la[1:, 1:][full]
            tris.append(np.stack([ia, ib, ic], 1))
            tris.append(np.stack([ib, id_, ic], 1))
        if clip.any():
            boxes = shapely.box(lon[:-1, :-1][clip], lat[:-1, :-1][clip], lon[1:, 1:][clip], lat[1:, 1:][clip])
            inter = shapely.intersection(shapely.union_all(boxes), tl)
            polys = [g for g in shapely.get_parts(inter) if shapely.get_type_id(g) == 3]
            if polys:
                parts = shapely.get_parts(shapely.constrained_delaunay_triangles(np.array(polys, dtype=object)))
                parts = [g for g in parts if shapely.get_type_id(g) == 3]
                xy = shapely.get_coordinates(np.array(parts, dtype=object)).reshape(-1, 4, 2)[:, :3]
                tris.append(xy)
        if not tris:
            return None, allland, touched
        T = np.concatenate(tris)
        # outward-facing = counter-clockwise in lon/lat; drop degenerate triangles
        sa = 0.5 * ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
        keep = np.abs(sa) > 1e-16
        T, sa = T[keep], sa[keep]
        neg = sa < 0
        T[neg] = T[neg][:, ::-1]
        keys = np.round(T.reshape(-1, 2) / EPS).astype(np.int64)
        _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        vxy = T.reshape(-1, 2)[first]
        faces = inv.reshape(-1, 3)
        vlon, vlat = vxy[:, 0], vxy[:, 1]
        vh = np.maximum(sample_dem(dem, vlat, vlon), 0)
        # skirts on mesh-border edges that lie on the tile perimeter
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        ek = np.minimum(e[:, 0], e[:, 1]) * len(vxy) + np.maximum(e[:, 0], e[:, 1])
        _, einv, ecnt = np.unique(ek, return_inverse=True, return_counts=True)
        be = e[ecnt[einv] == 1]
        w0, e0, s0, n0 = lon[0, 0], lon[0, -1], lat[0, 0], lat[-1, 0]
        def onb(v, val, axis):
            return np.abs((vlon if axis == 0 else vlat)[v] - val) < 1e-7
        edge_on = np.zeros(len(be), bool)
        for val, axis in ((w0, 0), (e0, 0), (s0, 1), (n0, 1)):
            edge_on |= onb(be[:, 0], val, axis) & onb(be[:, 1], val, axis)
        be = be[edge_on]
        V = len(vxy)
        skirt = step * A * 0.5
        if len(be):
            sk = np.concatenate([np.stack([be[:, 0], be[:, 0] + V, be[:, 1]], 1),
                                 np.stack([be[:, 1], be[:, 0] + V, be[:, 1] + V], 1)])
            faces = np.concatenate([faces, sk])
        P = ecef(np.radians(vlat), np.radians(vlon), vh)
        Pd = ecef(np.radians(vlat), np.radians(vlon), vh - skirt)
        Nn = vertex_normals(vlat, vlon, vh, dl * 0.5)
        pos, nor, col, fcs = compact(np.concatenate([P, Pd]), np.concatenate([Nn, Nn]),
                                     np.concatenate([vh, vh]), faces, None)
        return (pos, nor, col, fcs, float(vh.min()) - skirt, float(vh.max())), allland, touched

    def build_tile(L, tx, ty):
        """Returns the tile json dict (with private _h = (hmin, hmax))."""
        nonlocal nfiles, total_bytes
        dl = 360.0 / (COLS * 2 ** L)
        w0, s0 = -180 + tx * COLS * dl, -90 + ty * ROWS * dl
        e0, n0 = w0 + COLS * dl, s0 + ROWS * dl
        tb = shapely.box(w0, s0, e0, n0)
        has_land = shapely.intersects(land, tb) and (a.bbox is None or shapely.intersects(tb, shapely.box(*a.bbox)))
        smult = 1
        if has_land:
            ix = tx * COLS + np.arange(COLS + 1)
            iy = ty * ROWS + np.arange(ROWS + 1)
        else:
            # pure ocean: flat sea needs only a coarse grid (64 cells across)
            smult = max(COLS // 64, 1)
            ix = tx * COLS + np.arange(0, COLS + 1, smult)
            iy = ty * ROWS + np.arange(0, ROWS + 1, smult)
        lon, lat = np.meshgrid(-180 + ix * dl, -90 + iy * dl)

        children = []
        if L < N and has_land:   # pure-ocean tiles stay leaves (flat sea needs no refinement)
            for cy in (2 * ty, 2 * ty + 1):
                for cx in (2 * tx, 2 * tx + 1):
                    children.append(build_tile(L + 1, cx, cy))

        step = math.radians(dl) * smult
        R, C = lon.shape
        meshes, hmin, hmax = [], 0.0, 0.0
        allland = np.zeros((R - 1, C - 1), bool)
        touched = np.zeros((R - 1, C - 1), bool)
        if has_land:
            pad = 2 * dl
            tl = shapely.clip_by_rect(land, w0 - pad, s0 - pad, e0 + pad, n0 + pad)
            tl = shapely.simplify(tl, dl * 0.1, preserve_topology=True)
            shapely.prepare(tl)
            res, allland, touched = land_mesh(lon, lat, dl, tl, step)
            if res is not None:
                meshes.append(res[:4])
                hmin, hmax = res[4], res[5]

        # sea: every cell that is not wholly inland or that the coastline crosses
        # (a few metres below 0 so it never z-fights with the coast)
        sea_h = -a.sea_offset
        sskirt = step * A * 0.5
        sea_keep = ~allland | touched
        if sea_keep.any():
            idx = np.arange(R * C).reshape(R, C)
            ia, ib, ic, id_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
            SP = ecef(np.radians(lat), np.radians(lon), sea_h).reshape(-1, 3)
            SPd = ecef(np.radians(lat), np.radians(lon), sea_h - sskirt).reshape(-1, 3)
            sf = np.concatenate([np.stack([ia[sea_keep], ib[sea_keep], ic[sea_keep]], -1),
                                 np.stack([ib[sea_keep], id_[sea_keep], ic[sea_keep]], -1),
                                 skirt_faces(sea_keep, idx, R * C)])
            allSP = np.concatenate([SP, SPd])
            allSN = allSP / np.linalg.norm(allSP, axis=1, keepdims=True)
            meshes.append(compact(allSP, allSN, None, sf, SEA_RGBA))
            hmin = min(hmin, sea_h - sskirt)

        tile = {}
        if meshes:
            allpos = np.concatenate([m[0] for m in meshes])
            center = allpos.mean(0)
            scene = trimesh.Scene()
            T = np.eye(4)
            T[:3, 3] = zup_to_yup(center)
            for pos, nor, col, fcs in meshes:
                mesh = trimesh.Trimesh(
                    vertices=zup_to_yup(pos - center).astype(np.float32), faces=fcs,
                    vertex_normals=zup_to_yup(nor).astype(np.float32), process=False)
                mesh.visual.vertex_colors = col
                scene.add_geometry(mesh, transform=T)
            name = f"{L}_{tx}_{ty}.glb"
            blob = patch_glb(scene.export(file_type="glb"))
            with open(os.path.join(a.out, "tiles", name), "wb") as fh:
                fh.write(blob)
            total_bytes += len(blob)
            nfiles += 1
            tile["content"] = {"uri": f"tiles/{name}"}
        for c in children:
            hmin = min(hmin, c["_h"][0]); hmax = max(hmax, c["_h"][1])
        tile["boundingVolume"] = {"region": [math.radians(w0), math.radians(s0), math.radians(e0), math.radians(n0),
                                             hmin - 100, hmax + 100]}
        tile["geometricError"] = step * A if children else 0
        if children:
            tile["children"] = [{k: v for k, v in c.items() if k != "_h"} for c in children]
        tile["_h"] = (hmin, hmax)
        return tile

    print("Building tiles...")
    root = build_tile(0, 0, 0)
    root.pop("_h", None)
    root["refine"] = "REPLACE"
    tileset = {
        "asset": {"version": "1.1"},
        "geometricError": root["geometricError"] * 2,
        "root": root,
    }
    # strip leftover private keys from nested children
    def clean(t):
        t.pop("_h", None)
        for c in t.get("children", []):
            clean(c)
    clean(root)
    with open(os.path.join(a.out, "tileset.json"), "w") as f:
        json.dump(tileset, f)
    print(f"Done: {nfiles} tiles, {total_bytes / 1e6:.1f} MB -> {a.out}/tileset.json")


if __name__ == "__main__":
    main()
