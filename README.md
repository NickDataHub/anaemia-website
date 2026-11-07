# Anaemia Research Studio

A static, browser-only website built from `Anaemia_Model_Development (1).ipynb`. It uses the actual `checkpoints/resnet50_best_validation_auc.pth` supplied in the project Drive folder, converted to ONNX. No training dataset or participant metadata is included.

## Features

- JPG, PNG and WebP selection and drag-and-drop, preview, clear and retry.
- ResNet-50 inference entirely in the visitor's browser using WebAssembly.
- Normal / Anaemia softmax scores with the notebook's argmax decision rule.
- Final-layer class activation overlay, mathematically equivalent to normalized final-layer Grad-CAM for this global-average-pooling + linear network.
- Printable report, including browser date/time and model fingerprint. Use the browser print dialog's Save as PDF option.
- Responsive layout, keyboard controls, explicit loading/errors and no image upload endpoint or analytics.

## Research limitations

This is an experimental research demonstration, not a medical device or a diagnostic test. The model cannot confirm or rule out anaemia, estimate haemoglobin, verify image suitability, or provide treatment advice. Its scores are not calibrated clinical probabilities. Arbitrary images will still receive a class prediction. Clinical performance has not been established by the website conversion tests.

Only single-image predictions are displayed. Participant-level averaging used in training evaluation is not performed. JPEG decoder, embedded color-profile and orientation handling can differ between browsers and Python. Transparent images are composited on white. Use representative images and evaluate the deployed pipeline on a held-out dataset before any research conclusions.

## Run locally

From this folder run `python -m http.server 8000`, then open http://localhost:8000. Do not double-click `index.html`: model loading requires HTTP or HTTPS. No npm install or frontend build is needed. The first analysis downloads about 94 MB of model weights plus about 11 MB of runtime; later requests may use the browser HTTP cache. Browser cache eviction can cause another download. Model execution speed and memory use depend on the device.

## GitHub Pages

1. Push this folder, including the `model/` and `vendor/` directories, to the `main` branch of a public GitHub repository.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions** as the source.
3. Run the included **Publish GitHub Pages** workflow, or push a change to `main`.

The workflow publishes only web assets, not export scripts or documentation. All URLs are relative, so project paths such as `/anaemia-website/` work. GitHub Pages supports public repositories on GitHub Free: https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages

The ONNX model is stored as five chunks under 25 MB each. If uploading through GitHub's web interface, commit each model part separately to stay within its total upload limit. The workflow and browser both verify the complete model's SHA-256. Do not use Git LFS for these files. The website and model weights are publicly downloadable when published; patient images, dataset files, credentials and notebook outputs should not be committed.

## Updating the model

Use a local virtual environment or Google Colab with PyTorch, torchvision, NumPy, Pillow, ONNX and ONNX Runtime. Tested versions are in `scripts/requirements.txt`.

```sh
python scripts/export_model.py /path/to/resnet50_best_validation_auc.pth
```

The exporter rebuilds the notebook architecture with two output classes, loads the checkpoint using `weights_only=True`, verifies logits against PyTorch and verifies attention against autograd, then replaces the model chunks and manifest. It never executes the notebook. If the architecture or class definitions change, update the exporter and browser interpretation together. Run the relevant tests and commit the generated `model/` files.

### Preprocessing

RGB → antialiased bilinear resize to 224 × 224 → float values divided by 255 → channel normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` → NCHW tensor `[1,3,224,224]`.

Class 0 = Normal; class 1 = Anaemia. A tie selects class 0, matching PyTorch argmax. The overlay excludes classifier bias, applies ReLU and normalizes the selected class map. No random predictions, placeholder weights, remote inference service or API key is used.

## Verification performed

- Exported ONNX graph passed the ONNX checker.
- PyTorch vs ONNX Runtime: three synthetic inputs; maximum absolute logit difference `1.5944242477416992e-06`.
- Both class attention maps passed comparison against autograd Grad-CAM.
- JavaScript resizing matched Pillow exactly for 321×287, 83×61, 224×224 and 600×400 RGB fixtures.
- End-to-end browser test with a synthetic gradient image: PyTorch Anaemia score `0.5665209889`, browser `0.5665212274`.

These are numerical/software checks, not clinical validation or estimates of diagnostic accuracy.

## Attribution

Model and methodology originate in the supplied Anaemia AI notebooks and checkpoint. ONNX Runtime Web 1.22.0 is bundled under its MIT license (see `vendor/ONNX-RUNTIME-LICENSE.txt`). No new license for the supplied model or dataset is asserted here. Dataset permissions and model distribution rights remain the project owner's responsibility.

ONNX Runtime browser documentation: https://onnxruntime.ai/docs/get-started/with-javascript/web.html

If a deployment retry reports duplicate `github-pages` artifacts, start a fresh workflow run instead of rerunning the same attempt.
