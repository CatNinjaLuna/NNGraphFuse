# export_model.py
#
# ROLE: Pipeline Step 1 — Model Export
#
# This script is the entry point of the NNGraphFuse pipeline.
# It loads a pretrained ResNet-50 from torchvision and exports it
# to ONNX format — the standard interchange format that lets us
# inspect and rewrite the computation graph independently of PyTorch.
#
# WHY ONNX?
# PyTorch models are dynamic — the graph is defined at runtime.
# ONNX freezes the graph into a static representation: a list of
# nodes (ops) and edges (tensors), which we can parse, analyze,
# and rewrite with our optimization passes.
#
# OUTPUT: models/resnet50.onnx
# This file is consumed by graph/ir.py, which parses it into our
# custom IR and feeds it through the optimization pass pipeline.

import torch
import torchvision.models as models

def export_resnet50(output_path="models/resnet50.onnx"):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    # dummy input tensor, a fake image used to trace the model's computation graph during export.
    # (batchsize, color channels, image height, image width)

    torch.onnx.export(ß
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=17
    )
    print(f"✅ Exported ResNet-50 to {output_path}")

if __name__ == "__main__":
    export_resnet50()