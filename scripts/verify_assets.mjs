import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const root = new URL('../', import.meta.url);
const manifests = {
  'ensemble.json': { original: 'resnet50', segmenter: 'segmenter', crop: 'crop' },
  'ensemble-mixed-v1.json': { original: 'full-fp16-v1', segmenter: 'mixedseg-v1', crop: 'crop-fp16-v1' }
};
for (const [file, prefixes] of Object.entries(manifests)) {
  const manifest = JSON.parse(readFileSync(new URL(`model/${file}`, root)));
  assert.deepEqual(Object.keys(manifest.models).sort(), ['crop', 'original', 'segmenter']);
  for (const [name, model] of Object.entries(manifest.models)) {
    const hash = createHash('sha256'); let bytes = 0;
    assert(model.parts.length > 0, `${name} has no parts`);
    for (const part of model.parts) {
      assert.match(part, new RegExp(`^${prefixes[name]}\\.part\\d+\\.bin$`));
      const content = readFileSync(new URL(`model/${part}`, root)); hash.update(content); bytes += content.length;
    }
    assert.equal(bytes, model.bytes, `${file} ${name} size mismatch`);
    assert.equal(hash.digest('hex'), model.sha256, `${file} ${name} integrity mismatch`);
    console.log(`Verified ${file} ${name}: ${model.parts.length} parts, ${bytes} bytes.`);
  }
}
for (const file of ['favicon.svg', 'index.html', 'styles.css', 'app.js', 'preprocess.js', 'vendor/ort.min.js', 'vendor/ort-wasm-simd-threaded.mjs', 'vendor/ort-wasm-simd-threaded.wasm']) {
  assert(existsSync(new URL(file, root)), `Missing required asset: ${file}`);
}
console.log('Verified all runtime assets.');
