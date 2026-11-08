import { resizeRGB, normalize, softmax } from './preprocess.js';
const $ = id => document.getElementById(id);
let sessions, image, imageURL, busy = false, generation = 0, viewRGB, viewMask;
const status = (message, error = false) => { $('status').textContent = message; $('status').classList.toggle('error', error); };
function updateControls() { $('analyse').disabled = busy || !image || !$('acknowledge').checked; $('clear').disabled = busy || !image; $('image-file').disabled = busy; $('acknowledge').disabled = busy; }
function resetResult() { $('empty-result').hidden = false; $('results').hidden = true; $('result-state').textContent = 'Awaiting analysis'; viewMask = null; }
function clearImage() {
  generation++; image = null; viewRGB = null; resetResult();
  if (imageURL) URL.revokeObjectURL(imageURL); imageURL = null;
  $('preview').removeAttribute('src'); $('preview').hidden = true; $('upload-prompt').hidden = false;
  $('image-file').value = ''; $('file-info').textContent = 'Use a clear, close-up image of the inner lower eyelid.';
  $('result-state').textContent = 'Awaiting image'; status('Choose an image to get started.'); updateControls();
}
async function selectImage(file) {
  if (busy || !file) return; clearImage(); const request = generation;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return status('Please choose a JPG, PNG or WebP image.', true);
  if (file.size > 15 * 1024 * 1024) return status('Please choose an image under 15 MB.', true);
  const url = URL.createObjectURL(file), candidate = new Image(); candidate.src = url;
  try {
    await candidate.decode(); if (request !== generation) { URL.revokeObjectURL(url); return; }
    if (candidate.naturalWidth * candidate.naturalHeight > 24000000 || candidate.naturalWidth < 32 || candidate.naturalHeight < 32) throw new Error('Choose an image between 32 × 32 pixels and 24 megapixels.');
    image = candidate; imageURL = url; $('preview').src = url; $('preview').hidden = false; $('upload-prompt').hidden = true;
    $('file-info').textContent = `${file.name} · ${candidate.naturalWidth} × ${candidate.naturalHeight} px`; status('Image ready. Read the note below, then analyse.');
  } catch (e) { URL.revokeObjectURL(url); if (request === generation) status(e.message || 'This file could not be read as an image.', true); }
  updateControls();
}
async function loadModels() {
  if (sessions) return;
  if (!window.ort) throw new Error('The inference engine could not load. Reload and try again.');
  ort.env.wasm.wasmPaths = new URL('./vendor/', location.href).href; ort.env.wasm.numThreads = 1;
  const response = await fetch('./model/ensemble.json'); if (!response.ok) throw new Error('Model configuration is unavailable.');
  const manifest = await response.json(), made = {};
  try {
    for (const [name, model] of Object.entries(manifest.models)) {
      const chunks = []; let loaded = 0;
      for (const part of model.parts) {
        status(`Downloading ${name} model: ${Math.round(loaded / model.bytes * 100)}%. This may take a few minutes.`);
        const reply = await fetch(new URL(part, new URL('./model/', location.href))); if (!reply.ok) throw new Error('Model download failed. Check your connection.');
        const chunk = new Uint8Array(await reply.arrayBuffer()); chunks.push(chunk); loaded += chunk.length;
      }
      if (loaded !== model.bytes) throw new Error('Model download was incomplete.');
      const bytes = new Uint8Array(loaded); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
      if (digest !== model.sha256) throw new Error('Model integrity check failed. Reload the page.');
      status(`Preparing ${name} model on this device...`); made[name] = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    }
    sessions = made;
  } catch (e) { await Promise.all(Object.values(made).map(s => s.release())); throw e; }
}
function asRGBA(rgb) { const a = new Uint8ClampedArray(rgb.length / 3 * 4); for (let i = 0; i < rgb.length / 3; i++) { a[4*i] = rgb[3*i]; a[4*i+1] = rgb[3*i+1]; a[4*i+2] = rgb[3*i+2]; a[4*i+3] = 255; } return a; }
function prepareImage() {
  const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true }); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0);
  const rgb = resizeRGB(ctx.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height, 640, 480); canvas.width = canvas.height = 0; return rgb;
}
function nearestMask(prob) {
  if (prob.length !== 640*480 || !Array.from(prob).every(Number.isFinite)) throw new Error('Invalid segmentation output.');
  const mask = new Uint8Array(224*224);
  const sx=640/224, sy=480/224; let iy=sy/2;
  for (let y=0;y<224;y++,iy+=sy) { let ix=sx/2; for (let x=0;x<224;x++,ix+=sx) mask[y*224+x] = prob[Math.floor(iy)*640+Math.floor(ix)] >= .5 ? 1 : 0; }
  return mask;
}
function roundEven(n) { const f=Math.floor(n); return n-f===.5 ? f+(f%2) : Math.round(n); }
function cropToMask(rgb, mask) {
  let l=224,r=-1,t=224,b=-1,count=0;
  for (let y=0;y<224;y++) for (let x=0;x<224;x++) if (mask[y*224+x]) { l=Math.min(l,x); r=Math.max(r,x); t=Math.min(t,y); b=Math.max(b,y); count++; }
  if (count<20) throw new Error('The eyelid region could not be found. Try another photo.'); r++; b++;
  const px=Math.max(2,roundEven((r-l)*.25)), py=Math.max(2,roundEven((b-t)*.25));
  const left=Math.max(0,Math.floor((l-px)*640/224)), top=Math.max(0,Math.floor((t-py)*480/224)), right=Math.min(640,Math.ceil((r+px)*640/224)), bottom=Math.min(480,Math.ceil((b+py)*480/224));
  const w=right-left,h=bottom-top,rgba=new Uint8ClampedArray(w*h*4);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) { const src=((top+y)*640+left+x)*3,dst=(y*w+x)*4; rgba[dst]=rgb[src]; rgba[dst+1]=rgb[src+1]; rgba[dst+2]=rgb[src+2]; rgba[dst+3]=255; }
  return resizeRGB(rgba,w,h);
}
function drawSegmentation() {
  if (!viewRGB || !viewMask) return;
  const ctx=$('attention').getContext('2d'), pixels=ctx.createImageData(224,224);
  for (let i=0;i<224*224;i++) { const show=$('overlay-toggle').checked && viewMask[i]; for (let c=0;c<3;c++) pixels.data[4*i+c]=show ? Math.round(viewRGB[3*i+c]*.58+[190,45,91][c]*.42) : viewRGB[3*i+c]; pixels.data[4*i+3]=255; } ctx.putImageData(pixels,0,0);
}
async function run(session, rgb, dims) { const input=new ort.Tensor('float32',normalize(rgb),dims); try { const out=await session.run({image:input}); const logits=Array.from(out.logits.data); Object.values(out).forEach(x=>x.dispose()); return softmax(logits)[1]; } finally { input.dispose(); } }
async function analyse() {
  if (busy || !image || !$('acknowledge').checked) return; busy=true; resetResult(); updateControls(); $('result-state').textContent='Processing'; let input, output;
  try {
    await loadModels(); status('Analysing on this device...'); const start=performance.now(), rgb=prepareImage(); input=new ort.Tensor('float32',normalize(rgb),[1,3,480,640]);
    output=await sessions.segmenter.run({image:input}); const mask=nearestMask(output.mask_probability.data); const crop=cropToMask(rgb,mask), full=resizeRGB(asRGBA(rgb),640,480);
    const [a,b]=await Promise.all([run(sessions.original,full,[1,3,224,224]),run(sessions.crop,crop,[1,3,224,224])]); const score=(a+b)/2;
    viewRGB=full; viewMask=mask; $('prediction').textContent='No medical conclusion can be drawn'; $('anaemia-score').textContent=score.toFixed(3); $('anaemia-bar').value=score; $('normal-score').textContent=(1-score).toFixed(3); $('normal-bar').value=1-score;
    $('empty-result').hidden=true; $('results').hidden=false; $('result-state').textContent='Analysis complete'; $('run-info').textContent=`${new Date().toLocaleString()} · ${((performance.now()-start)/1000).toFixed(1)}s on this device`;
    drawSegmentation(); status('Complete. This result cannot confirm or rule out anaemia.');
  } catch (e) { console.error(e); status(e.message || 'Analysis failed. Please try again.',true); $('result-state').textContent='Analysis unavailable'; }
  finally { input?.dispose(); if(output) Object.values(output).forEach(x=>x.dispose()); busy=false; updateControls(); }
}
$('image-file').addEventListener('change',e=>selectImage(e.target.files[0])); $('acknowledge').addEventListener('change',updateControls); $('clear').addEventListener('click',clearImage); $('analyse').addEventListener('click',analyse); $('overlay-toggle').addEventListener('change',drawSegmentation); $('report').addEventListener('click',()=>window.print());
for (const name of ['dragenter','dragover']) $('dropzone').addEventListener(name,e=>{e.preventDefault();if(!busy)$('dropzone').classList.add('dragover');});
for (const name of ['dragleave','drop']) $('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.remove('dragover');});
$('dropzone').addEventListener('drop',e=>selectImage(e.dataTransfer.files[0])); window.addEventListener('beforeunload',()=>{if(imageURL)URL.revokeObjectURL(imageURL);});
