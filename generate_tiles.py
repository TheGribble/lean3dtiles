"""Build a land-only global terrain 3D Tileset (3D Tiles 1.1, GLB content).

Quadtree over lon/lat. Tile (L, x, y) spans 360/2^L deg lon by 180/2^L deg lat
and is a 64x32-cell grid, so grid spacing halves each level and every coarse
lattice is a subset of the finest lattice (consistent shared vertices).
Triangles are kept if at least one corner is on land; heights are clamped >= 0
so coastal triangles don't dip into the bathymetry.
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
COLS, ROWS = 64, 32  # cells per tile


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
    ap.add_argument("--out", default="output/terrain_tiles")
    ap.add_argument("--sea-offset", type=float, default=4.0, help="metres the sea sits below 0 to avoid z-fighting with the coast")
    a = ap.parse_args()
    N = a.max_level
    os.makedirs(os.path.join(a.out, "tiles"), exist_ok=True)

    print("Loading land polygons...")
    land = shapely.union_all(gpd.read_file(a.land).geometry.values)
    shapely.prepare(land)

    # Coastline segments for snapping ocean-side vertices onto the real coast.
    parts = shapely.get_parts(shapely.boundary(land))
    xy, pi = shapely.get_coordinates(parts, return_index=True)
    ok = pi[:-1] == pi[1:]
    SA, SB = xy[:-1][ok], xy[1:][ok]
    tree = shapely.STRtree(shapely.linestrings(np.stack([SA, SB], 1)))

    def snap_to_coast(lon, lat):
        pts = np.stack([lon, lat], -1)
        _, ti = tree.query_nearest(shapely.points(pts), all_matches=False)
        A_, B_ = SA[ti], SB[ti]
        ab = B_ - A_
        t = np.clip(((pts - A_) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-18), 0, 1)
        q = A_ + ab * t[:, None]
        return q[:, 0], q[:, 1]

    print("Loading DEM...")
    with rasterio.open(a.dem) as src:
        dem = (src.read(1).astype(np.float32), src.transform)

    def lattice(L, tx, ty, check_land=True):
        """Per-tile lattice with a 1-cell pad (for seamless normals)."""
        nx = COLS * 2 ** L
        dl = 360.0 / nx
        ix = np.arange(tx * COLS - 1, tx * COLS + COLS + 2)
        iy = np.arange(ty * ROWS - 1, ty * ROWS + ROWS + 2)
        LON, LAT = np.meshgrid(-180 + ix * dl, -90 + iy * dl)
        h = np.maximum(sample_dem(dem, LAT, LON), 0)
        onland = shapely.intersects_xy(land, LON, LAT) if check_land else np.zeros(LON.shape, bool)
        P = ecef(np.radians(LAT), np.radians(LON), h)
        du = np.gradient(P, axis=1)
        dv = np.gradient(P, axis=0)
        Nrm = np.cross(du, dv)
        ln = np.linalg.norm(Nrm, axis=-1, keepdims=True)
        rad = P / np.linalg.norm(P, axis=-1, keepdims=True)
        Nrm = np.where(ln > 1e-3, Nrm / np.maximum(ln, 1e-12), rad)
        Nrm[np.abs(LAT) > 89.99] = rad[np.abs(LAT) > 89.99]
        c = (slice(1, -1), slice(1, -1))
        return P[c], Nrm[c], onland[c], h[c], LAT[c], LON[c]

    nfiles = 0
    total_bytes = 0

    def build_tile(L, tx, ty):
        """Returns (tile json dict, has_content) or None if no land at all below."""
        nonlocal nfiles, total_bytes
        dl = 360.0 / (COLS * 2 ** L)
        w0, s0 = -180 + tx * COLS * dl, -90 + ty * ROWS * dl
        has_land = shapely.intersects(land, shapely.box(w0, s0, w0 + COLS * dl, s0 + ROWS * dl))
        P, Nn, M, H, lat, lon = lattice(L, tx, ty, has_land)

        a_, b_ = M[:-1, :-1], M[:-1, 1:]
        c_, d_ = M[1:, :-1], M[1:, 1:]
        t1 = a_ | b_ | c_   # (a,b,c)
        t2 = b_ | d_ | c_   # (b,d,c)
        children = []
        if L < N and has_land:   # pure-ocean tiles stay leaves (flat sea needs no refinement)
            for cy in (2 * ty, 2 * ty + 1):
                for cx in (2 * tx, 2 * tx + 1):
                    children.append(build_tile(L + 1, cx, cy))

        tile = {}
        w, e = math.radians(lon.min()), math.radians(lon.max())
        s, n = math.radians(lat.min()), math.radians(lat.max())
        step = math.radians(360 / (COLS * 2 ** L))
        meshes = []
        hmin, hmax = 0.0, 0.0
        R, C = P.shape[:2]
        if t1.any() or t2.any():
            idx = np.arange(R * C).reshape(R, C)
            ia, ib, ic, id_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
            faces = np.concatenate([
                np.stack([ia[t1], ib[t1], ic[t1]], -1),
                np.stack([ib[t2], id_[t2], ic[t2]], -1)])

            # snap ocean-side vertices of coastal triangles onto the coastline
            lat_f, lon_f, h_f = lat.reshape(-1).copy(), lon.reshape(-1).copy(), H.reshape(-1).copy()
            P_f = P.reshape(-1, 3).copy()
            wet = np.unique(faces)
            wet = wet[~M.reshape(-1)[wet]]
            if len(wet):
                nlon, nlat = snap_to_coast(lon_f[wet], lat_f[wet])
                nh = np.maximum(sample_dem(dem, nlat, nlon), 0)
                lon_f[wet], lat_f[wet], h_f[wet] = nlon, nlat, nh
                P_f[wet] = ecef(np.radians(nlat), np.radians(nlon), nh)

            # skirts along the tile perimeter hide cracks between LOD levels
            skirt = step * A * 0.5
            P_dn = ecef(np.radians(lat_f), np.radians(lon_f), h_f - skirt)
            sk = skirt_faces(t1 | t2, idx, R * C)
            if len(sk):
                faces = np.concatenate([faces, sk])

            allP = np.concatenate([P_f, P_dn])
            allN = np.concatenate([Nn.reshape(-1, 3)] * 2)
            allH = np.concatenate([h_f, h_f])
            meshes.append(compact(allP, allN, allH, faces, None))
            hmin, hmax = float(h_f.min()) - skirt, float(h_f.max())

        # sea: every cell with at least one non-land corner (coastal cells included,
        # so the coast has no gaps); a few metres below 0 to avoid z-fighting.
        sea_h = -a.sea_offset
        sskirt = step * A * 0.5
        sea_keep = ~(a_ & b_ & c_ & d_)
        if sea_keep.any():
            SP = ecef(np.radians(lat), np.radians(lon), sea_h).reshape(-1, 3)
            SPd = ecef(np.radians(lat), np.radians(lon), sea_h - sskirt).reshape(-1, 3)
            idx = np.arange(R * C).reshape(R, C)
            ia, ib, ic, id_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
            sf = np.concatenate([np.stack([ia[sea_keep], ib[sea_keep], ic[sea_keep]], -1),
                                 np.stack([ib[sea_keep], id_[sea_keep], ic[sea_keep]], -1),
                                 skirt_faces(sea_keep, idx, R * C)])
            allSP = np.concatenate([SP, SPd])
            allSN = allSP / np.linalg.norm(allSP, axis=1, keepdims=True)
            meshes.append(compact(allSP, allSN, None, sf, SEA_RGBA))
            hmin = min(hmin, sea_h - sskirt)
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
        # pad by the lattice step so the bounds cover the coarse-vs-fine surface
        tile["boundingVolume"] = {"region": [w, s, e, n, hmin - 100, hmax + 100]}
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
