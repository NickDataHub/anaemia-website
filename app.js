import { resizeRGB, normalize, softmax } from './preprocess.js';
const $ = id => document.getElementById(id);
let session, modelInfo, image, imageURL, busy = false, generation = 0, attentionMap, resizedRGB;
const status = (message, error = false) => { $('status').textContent = message; $('status').classList.toggle('error', error); };
function updateControls() {
  $('analyse').disabled = busy || !image || !$('acknowledge').checked;
  $('clear').disabled = busy || !image;
  $('image-file').disabled = busy;
  $('acknowledge').disabled = busy;
}
function resetResult() {
  $('empty-result').hidden = false; $('results').hidden = true;
  $('result-state').textContent = 'Awaiting analysis'; attentionMap = null;
}
function clearImage() {
  generation++; image = null; resizedRGB = null; resetResult();
  if (imageURL) URL.revokeObjectURL(imageURL); imageURL = null;
  $('preview').removeAttribute('src'); $('preview').hidden = true; $('upload-prompt').hidden = false;
  $('image-file').value = ''; $('file-info').textContent = 'Use a clear, close-up image of the inner lower eyelid.';
  $('result-state').textContent = 'Awaiting image'; status('Choose an image to get started.'); updateControls();
}
async function selectImage(file) {
  if (busy || !file) return;
  clearImage(); const request = generation;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { status('Please choose a JPG, PNG or WebP image.', true); return; }
  if (file.size > 15 * 1024 * 1024) { status('This image is too large. Please choose one under 15 MB.', true); return; }
  const url = URL.createObjectURL(file), candidate = new Image(); candidate.src = url;
  try {
    await candidate.decode();
    if (request !== generation) { URL.revokeObjectURL(url); return; }
    if (candidate.naturalWidth * candidate.naturalHeight > 24000000) throw new Error('Please resize this image to under 24 megapixels.');
    if (candidate.naturalWidth < 32 || candidate.naturalHeight < 32) throw new Error('This image is too small. Choose an image at least 32 × 32 pixels.');
    image = candidate; imageURL = url; $('preview').src = url; $('preview').hidden = false; $('upload-prompt').hidden = true;
    $('file-info').textContent = `${file.name} · ${candidate.naturalWidth} × ${candidate.naturalHeight} px`;
    status('Image ready. Read the note below, then analyse.');
  } catch (e) { URL.revokeObjectURL(url); if (request === generation) status(e.message.startsWith('Please') || e.message.startsWith('This') ? e.message : 'This file could not be read as an image.', true); }
  updateControls();
}
async function loadModel() {
  if (session) return;
  if (!window.ort) throw new Error('The inference engine could not load. Reload the page and try again.');
  ort.env.wasm.wasmPaths = new URL('./vendor/', location.href).href;
  ort.env.wasm.numThreads = 1;
  status('Downloading the model… This first load may take a minute.');
  const response = await fetch('./model/model.json');
  if (!response.ok) throw new Error('Model configuration is missing. See the setup guide.');
  modelInfo = await response.json();
  const parts = [];
  let loaded = 0;
  for (const part of modelInfo.parts) {
    const response = await fetch(new URL(part, new URL('./model/', location.href)));
    if (!response.ok) throw new Error('The model download failed. Check your connection and try again.');
    const data = new Uint8Array(await response.arrayBuffer()); parts.push(data); loaded += data.length;
    status(`Downloading model… ${Math.round(loaded / modelInfo.bytes * 100)}%`);
  }
  const bytes = new Uint8Array(loaded); let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.length; }
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== modelInfo.sha256) throw new Error('Model integrity check failed. Clear the site cache and reload.');
  status('Preparing the model for this device…');
  session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}
function prepareImage() {
  const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#fff'; context.fillRect(0, 0, canvas.width, canvas.height); context.drawImage(image, 0, 0);
  resizedRGB = resizeRGB(context.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height);
  canvas.width = canvas.height = 0;
  return new ort.Tensor('float32', normalize(resizedRGB), [1, 3, 224, 224]);
}
function drawAttention() {
  if (!resizedRGB) return;
  const context = $('attention').getContext('2d'), pixels = context.createImageData(224, 224);
  for (let i = 0; i < 224 * 224; i++) { for (let c = 0; c < 3; c++) pixels.data[4 * i + c] = resizedRGB[3 * i + c]; pixels.data[4 * i + 3] = 255; }
  context.putImageData(pixels, 0, 0);
  if (!$('overlay-toggle').checked || !attentionMap) return;
  const max = Math.max(...attentionMap);
  if (max <= 0) return;
  const heat = document.createElement('canvas'); heat.width = heat.height = 7;
  const hctx = heat.getContext('2d'), hp = hctx.createImageData(7, 7);
  for (let i = 0; i < 49; i++) { const v = Math.max(0, attentionMap[i]) / max; hp.data[4 * i] = 255; hp.data[4 * i + 1] = Math.round(220 * (1 - v)); hp.data[4 * i + 2] = 20; hp.data[4 * i + 3] = Math.round(v * 160); }
  hctx.putImageData(hp, 0, 0); context.imageSmoothingEnabled = true; context.drawImage(heat, 0, 0, 224, 224);
}
async function analyse() {
  if (busy || !image || !$('acknowledge').checked) return;
  busy = true; resetResult(); updateControls(); $('result-state').textContent = 'Processing';
  let tensor, outputs;
  try {
    await loadModel(); status('Analysing locally… Keep this tab open.');
    tensor = prepareImage(); const start = performance.now(); outputs = await session.run({ image: tensor });
    const scores = softmax(outputs.logits.data), predicted = scores[1] > scores[0] ? 1 : 0;
    if (outputs.attention.data.length !== 98 || !Array.from(outputs.attention.data).every(Number.isFinite)) throw new Error('Invalid attention output.');
    attentionMap = Array.from(outputs.attention.data.slice(predicted * 49, (predicted + 1) * 49));
    $('prediction').textContent = predicted === 1 ? 'Image classified as anaemia' : 'Image classified as non-anaemia';
    $('anaemia-score').textContent = `${(scores[1] * 100).toFixed(1)}%`; $('anaemia-bar').value = scores[1];
    $('normal-score').textContent = `${(scores[0] * 100).toFixed(1)}%`; $('normal-bar').value = scores[0];
    $('empty-result').hidden = true; $('results').hidden = false; $('result-state').textContent = 'Analysis complete';
    $('run-info').textContent = `${new Date().toLocaleString()} · ${( (performance.now() - start) / 1000).toFixed(1)}s inference · Model ${modelInfo.sha256.slice(0, 12)}`;
    drawAttention(); status('Assessment complete. This result cannot confirm or rule out anaemia.');
  } catch (error) { console.error(error); status(error.message || 'Analysis failed. Please try again.', true); $('result-state').textContent = 'Analysis unavailable'; }
  finally { tensor?.dispose(); if (outputs) Object.values(outputs).forEach(x => x.dispose()); busy = false; updateControls(); }
}
$('image-file').addEventListener('change', e => selectImage(e.target.files[0]));
$('acknowledge').addEventListener('change', updateControls);
$('clear').addEventListener('click', clearImage);
$('analyse').addEventListener('click', analyse);
$('overlay-toggle').addEventListener('change', drawAttention);
$('report').addEventListener('click', () => window.print());
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, e => { e.preventDefault(); if (!busy) $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, e => { e.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', e => selectImage(e.dataTransfer.files[0]));
window.addEventListener('beforeunload', () => { if (imageURL) URL.revokeObjectURL(imageURL); });
