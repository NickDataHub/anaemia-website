import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const root = new URL('../', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('model/ensemble.json', root)));
assert.deepEqual(Object.keys(manifest.models).sort(), ['crop', 'original', 'segmenter']);
for (const [name, model] of Object.entries(manifest.models)) {
  const hash = createHash('sha256'); let bytes = 0;
  for (const part of model.parts) {
    const prefix = name === 'original' ? 'resnet50' : name;
    assert.match(part, new RegExp(`^${prefix}\\.part\\d+\\.bin$`));
    const content = readFileSync(new URL(`model/${part}`, root)); hash.update(content); bytes += content.length;
  }
  assert.equal(bytes, model.bytes, `${name} size mismatch`);
  assert.equal(hash.digest('hex'), model.sha256, `${name} integrity mismatch`);
  console.log(`Verified ${name}: ${model.parts.length} parts, ${bytes} bytes.`);
}
for (const file of ['favicon.svg', 'index.html', 'styles.css', 'app.js', 'preprocess.js', 'vendor/ort.min.js', 'vendor/ort-wasm-simd-threaded.mjs', 'vendor/ort-wasm-simd-threaded.wasm']) {
  assert(existsSync(new URL(file, root)), `Missing required asset: ${file}`);
}
console.log('Verified all runtime assets.');
