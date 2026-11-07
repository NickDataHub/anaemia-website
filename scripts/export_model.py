"""Export the notebook's ResNet-50 to an ONNX model for static browser inference.

Run: python scripts/export_model.py path/to/resnet50_best_validation_auc.pth
Only tensor-safe checkpoint loading is used; no notebook code is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import resnet50


class BrowserResNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.fc = model.fc

    def forward(self, image):
        features = self.features(image)
        logits = self.fc(features.mean(dim=(2, 3)))
        # For this global-average-pool + linear architecture, these maps equal
        # final-layer Grad-CAM up to a positive constant, before normalization.
        attention = torch.relu(torch.nn.functional.conv2d(features, self.fc.weight[:, :, None, None]))
        return logits, attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'model')
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Training metrics use NumPy scalar values in the saved checkpoint.
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    original = resnet50(weights=None)
    original.fc = nn.Linear(original.fc.in_features, 2)
    original.load_state_dict(state, strict=True)
    original.eval()
    model = BrowserResNet(original).eval()
    torch.manual_seed(42)
    test_inputs = [torch.randn(1, 3, 224, 224), torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 224, 224)]
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.onnx'
        torch.onnx.export(model, test_inputs[0], path, input_names=['image'], output_names=['logits', 'attention'], opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        errors = []
        for tensor in test_inputs:
            with torch.no_grad():
                expected, expected_cam = model(tensor)
                np.testing.assert_allclose(original(tensor).numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
            actual, actual_cam = session.run(None, {'image': tensor.numpy()})
            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(actual_cam, expected_cam.numpy(), rtol=1e-3, atol=1e-3)
            errors.append(float(np.max(np.abs(actual - expected.numpy()))))
        # Independently verify the analytic attention map against autograd Grad-CAM.
        x = test_inputs[0].clone().requires_grad_(True)
        features = model.features(x)
        logits = model.fc(features.mean(dim=(2, 3)))
        for cls in [0, 1]:
            gradients, = torch.autograd.grad(logits[0, cls], features, retain_graph=True)
            gradcam = torch.relu((gradients.mean(dim=(2, 3), keepdim=True) * features).sum(dim=1))
            analytic = torch.relu(torch.nn.functional.conv2d(features, model.fc.weight[cls:cls+1, :, None, None]))[:, 0] / 49
            np.testing.assert_allclose(gradcam.detach().numpy(), analytic.detach().numpy(), rtol=1e-4, atol=1e-5)
        data = path.read_bytes()
    # Keep each file under GitHub's 100 MiB Git limit and web upload limit.
    chunk_size = 20 * 1024 * 1024
    parts = []
    for index, start in enumerate(range(0, len(data), chunk_size)):
        name = f'resnet50.part{index:02d}.bin'
        (args.output / name).write_bytes(data[start:start + chunk_size])
        parts.append(name)
    info = {'architecture': 'ResNet-50', 'classes': ['Normal', 'Anaemia'], 'input': [1, 3, 224, 224],
            'parts': parts, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'validation': {'synthetic_cases': len(test_inputs), 'maximum_logit_error': max(errors), 'attention_autograd_check': 'passed'}}
    (args.output / 'model.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
