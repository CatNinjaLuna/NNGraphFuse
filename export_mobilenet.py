# export_mobilenet.py
#
# Exports MobileNetV2 with ImageNet normalization baked into the ONNX graph.
#
# WHY: PyTorch's default export puts normalization in Python preprocessing,
# outside the graph boundary — so the ONNX model never sees Sub/Div nodes.
# By wrapping the model with normalization inside nn.Module, the exporter
# emits Sub and Div nodes with constant initializers (mean/std tensors).
# These are exactly the nodes the constant folding pass can eliminate.
#
# Input:  raw float image tensor, shape [1, 3, 224, 224], values in [0, 1]
# Output: models/mobilenetv2_normalized.onnx

import torch
import torch.nn as nn
import torchvision.models as models


class NormalizedMobileNetV2(nn.Module):
    """
    MobileNetV2 with ImageNet normalization baked in as graph ops.

    Normalization:
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]

    register_buffer() makes mean/std part of the model state —
    the ONNX exporter sees them as constant initializers and emits
    Sub and Div nodes that the constant folding pass can eliminate.
    """

    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Shape [1, 3, 1, 1] so broadcast works over [B, C, H, W]
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # These two lines become Sub and Div nodes in the ONNX graph,
        # with mean/std as constant initializers — foldable at compile time.
        x = (x - self.mean) / self.std
        return self.model(x)


def export():
    model = NormalizedMobileNetV2()
    model.eval()

    # Dummy input: batch=1, RGB, 224x224, values in [0, 1]
    dummy = torch.zeros(1, 3, 224, 224)

    output_path = "models/mobilenetv2_normalized.onnx"

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    print(f"✅ Exported to {output_path}")
    print(f"   Input:  [1, 3, 224, 224]  (raw float, values in [0, 1])")
    print(f"   Output: [1, 1000]         (ImageNet logits)")
    print(f"\n   Foldable nodes to look for:")
    print(f"   Sub  — x - mean  (mean is a constant initializer)")
    print(f"   Div  — x / std   (std  is a constant initializer)")


if __name__ == "__main__":
    export()

'''
(venv) (base) carolina1650@Carolinas-MacBook-Pro NNGraphFuse % python export_mobilenet.py
Downloading: "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth" to /Users/carolina1650/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth
100.0%
/Users/carolina1650/NNGraphFuse/export_mobilenet.py:62: DeprecationWarning: You are using the legacy TorchScript-based ONNX export. Starting in PyTorch 2.9, the new torch.export-based ONNX exporter has become the default. Learn more about the new export logic: https://docs.pytorch.org/docs/stable/onnx_export.html. For exporting control flow: https://pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
  torch.onnx.export(
✅ Exported to models/mobilenetv2_normalized.onnx
   Input:  [1, 3, 224, 224]  (raw float, values in [0, 1])
   Output: [1, 1000]         (ImageNet logits)

   Foldable nodes to look for:
   Sub  — x - mean  (mean is a constant initializer)
   Div  — x / std   (std  is a constant initializer)
'''