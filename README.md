# AnemoDetect

Static GitHub Pages website running an experimental, browser-only image model. It cannot diagnose, screen for or rule out anaemia, and must not be used for medical decisions. It has not been clinically validated. Visitors concerned about anaemia should seek medical advice and a blood test.

The analysis combines an original full-image ResNet-50 with an EfficientNet B0 applied to a crop centred on a U-Net predicted conjunctival region. The displayed region is segmentation, not an explanation of model attention. The full-image branch sees the complete image. The model cannot reliably reject non-eye images, and the mask can be wrong. Model scores are not calibrated probabilities; there is no validated classification threshold.

On a reused development validation set of 866 participants the ensemble AUC was 0.802; this is an optimistic exploratory result, not independent validation or a clinical performance claim. The dataset contains 725 anaemic and 141 non-anaemic participants; scores were averaged per participant for evaluation while this website processes one image at a time. More representative independent evaluation is needed. Browser colour, decoding and device differences may also affect results.

The first analysis downloads about 167 MB of model weights and an ONNX runtime. Weights are split into chunks below GitHub's file-size limits and verified using SHA-256. Images are processed locally and are not uploaded or stored by the website. Browser or device memory limits may prevent execution.

Run locally with `python -m http.server 8000`, then open `http://localhost:8000`. No build step is required. The GitHub Actions workflow runs `npm run check`, verifies all model parts, and deploys only the public web assets. Do not commit patient images, metadata, credentials, or temporary fixtures.
