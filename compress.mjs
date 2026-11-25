// Draco-compress every GLB under a directory, in place (KHR_draco_mesh_compression,
// supported natively by Cesium for Unity / cesium-native).
import { NodeIO } from '@gltf-transform/core';
import { KHRDracoMeshCompression } from '@gltf-transform/extensions';
import { draco } from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';
import fs from 'node:fs';
import path from 'node:path';

const dir = process.argv[2];
const outDir = process.argv[3] || dir;           // default: in place
const opts = JSON.parse(process.argv[4] || '{}');  // draco option overrides
const io = new NodeIO().registerExtensions([KHRDracoMeshCompression]).registerDependencies({
  'draco3d.decoder': await draco3d.createDecoderModule(),
  'draco3d.encoder': await draco3d.createEncoderModule(),
});
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.glb'));
let before = 0, after = 0, i = 0;
for (const f of files) {
  const p = path.join(dir, f);
  before += fs.statSync(p).size;
  const doc = await io.read(p);
  // Draco-compressed normalised-uint8 COLOR_0 is mis-decoded by CesiumJS (renders red/garbled);
  // float colours are valid glTF and decode correctly, at no size cost.
  for (const a of doc.getRoot().listAccessors()) if (a.getNormalized() && a.getComponentType() === 5121 && a.getType() === 'VEC4') {
    const u = a.getArray(); const f = new Float32Array(u.length); for (let i = 0; i < u.length; i++) f[i] = u[i] / 255;
    a.setArray(f); a.setNormalized(false);
  }
  await doc.transform(draco({
    method: 'edgebreaker',
    quantizePosition: 16, quantizeNormal: 10, quantizeColor: 8,
    quantizationVolume: 'mesh',
    ...opts,
  }));
  const q = path.join(outDir, f);
  await io.write(q, doc);
  after += fs.statSync(q).size;
  if (++i % 1000 === 0) console.log(`compressed ${i}/${files.length}`);
}
console.log(`Draco: ${files.length} tiles, ${(before / 1e6).toFixed(0)} MB -> ${(after / 1e6).toFixed(0)} MB`);
