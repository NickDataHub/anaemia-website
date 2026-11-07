import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const root = new URL('../', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('model/model.json', root)));
const hash = createHash('sha256'); let bytes = 0;
for (const part of manifest.parts) {
  assert.match(part, /^resnet50\.part\d+\.bin$/);
  const content = readFileSync(new URL(`model/${part}`, root));
  hash.update(content); bytes += content.length;
}
assert.equal(bytes, manifest.bytes, 'Model file size mismatch');
assert.equal(hash.digest('hex'), manifest.sha256, 'Model integrity mismatch');
for (const file of ['favicon.svg', 'index.html', 'styles.css', 'app.js', 'preprocess.js', 'vendor/ort.min.js', 'vendor/ort-wasm-simd-threaded.mjs', 'vendor/ort-wasm-simd-threaded.wasm']) {
  assert(existsSync(new URL(file, root)), `Missing required asset: ${file}`);
}
console.log(`Verified ${manifest.parts.length} model parts (${bytes} bytes) and all runtime assets.`);
