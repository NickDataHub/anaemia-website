// Separable bilinear resizing with an antialiasing filter for downsampling.
// Matches Pillow's RGB resize convention, including intermediate byte rounding.
function coefficients(from, to) {
  const scale = from / to, support = Math.max(1, scale), precision = 2 ** 22;
  return Array.from({ length: to }, (_, out) => {
    const center = (out + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(from, Math.floor(center + support + 0.5));
    const weights = Array.from({ length: end - start }, (_, i) => Math.max(0, 1 - Math.abs((i + start - center + 0.5) / support)));
    const sum = weights.reduce((a, b) => a + b, 0);
    return { start, weights: weights.map(w => Math.floor(w / sum * precision + 0.5)) };
  });
}
export function resizeRGB(rgba, width, height, outWidth = 224, outHeight = outWidth) {
  const precision = 2 ** 22, half = precision / 2;
  const horizontal = coefficients(width, outWidth), vertical = coefficients(height, outHeight);
  const temp = new Uint8Array(outWidth * height * 3), rgb = new Uint8Array(outWidth * outHeight * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < outWidth; x++) for (let c = 0; c < 3; c++) {
    const { start, weights } = horizontal[x]; let sum = half;
    for (let i = 0; i < weights.length; i++) sum += rgba[(y * width + start + i) * 4 + c] * weights[i];
    temp[(y * outWidth + x) * 3 + c] = Math.max(0, Math.min(255, Math.floor(sum / precision)));
  }
  for (let y = 0; y < outHeight; y++) for (let x = 0; x < outWidth; x++) for (let c = 0; c < 3; c++) {
    const { start, weights } = vertical[y]; let sum = half;
    for (let i = 0; i < weights.length; i++) sum += temp[((start + i) * outWidth + x) * 3 + c] * weights[i];
    rgb[(y * outWidth + x) * 3 + c] = Math.max(0, Math.min(255, Math.floor(sum / precision)));
  }
  return rgb;
}
export function normalize(rgb) {
  const count = rgb.length / 3, tensor = new Float32Array(3 * count);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let c = 0; c < 3; c++) for (let i = 0; i < count; i++) tensor[c * count + i] = (rgb[3 * i + c] / 255 - mean[c]) / std[c];
  return tensor;
}
export function softmax(logits) {
  if (logits.length !== 2 || !Array.from(logits).every(Number.isFinite)) throw new Error('Invalid model output.');
  const max = Math.max(...logits), values = Array.from(logits, x => Math.exp(x - max)), sum = values[0] + values[1];
  return values.map(x => x / sum);
}
