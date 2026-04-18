# graph/ir.py
#
# ROLE: Pipeline Step 2 — Graph Parsing & IR Construction
#
# This script loads the exported ONNX model and parses it into
# our custom Intermediate Representation (IR) — a plain Python
# dict that makes the graph easy to traverse and rewrite.
#
# WHY A CUSTOM IR?
# ONNX stores graphs as protobuf binary — awkward to traverse
# and mutate directly. We parse it once into a clean structure,
# run all optimization passes on that, then serialize back to ONNX.
#
# This is the same pattern used by production compilers:
#   ONNX (file format) → IR (in-memory workable graph) → passes → ONNX
#
# OUTPUT: a dict of {node_id: {op, inputs, outputs, attrs}}
# Consumed by passes/fusion.py and other optimization passes.

import onnx

def load_graph(onnx_path):
    """
    Parse ONNX model into a workable IR.
    Returns a dict of {node_id: node_info}
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    ir = {}
    for i, node in enumerate(graph.node):
        node_id = f"node_{i}"
        ir[node_id] = {
            "op":      node.op_type,
            "inputs":  list(node.input),
            "outputs": list(node.output),
            "attrs":   {a.name: a for a in node.attribute}
        }

    print(f"✅ Loaded {len(ir)} nodes from {onnx_path}")
    return ir


def summarize(ir):
    """Print a count of each op type in the graph."""
    from collections import Counter
    op_counts = Counter(v["op"] for v in ir.values())
    print("\n📊 Op type summary:")
    for op, count in op_counts.most_common():
        print(f"   {op:<20} {count}")


if __name__ == "__main__":
    ir = load_graph("models/resnet50.onnx")

    # Print first 10 nodes
    print("\n🔍 First 10 nodes:")
    for k, v in list(ir.items())[:10]:
        print(f"  {k}: op={v['op']}")
        print(f"       inputs={v['inputs']}")
        print(f"       outputs={v['outputs']}")

    summarize(ir)


'''
(venv) (base) carolina1650@Carolinas-MacBook-Pro NNGraphFuse % python graph/ir.py
✅ Loaded 124 nodes from models/resnet50.onnx

🔍 First 10 nodes:
  node_0: op=Shape
       inputs=['input']
       outputs=['val_0']
  node_1: op=Conv
       inputs=['input', 'conv1.weight', 'conv1.weight_bias']
       outputs=['getitem']
  node_2: op=Relu
       inputs=['getitem']
       outputs=['relu']
  node_3: op=MaxPool
       inputs=['relu']
       outputs=['max_pool2d']
  node_4: op=Conv
       inputs=['max_pool2d', 'layer1.0.conv1.weight', 'layer1.0.conv1.weight_bias']
       outputs=['getitem_3']
  node_5: op=Relu
       inputs=['getitem_3']
       outputs=['relu_1']
  node_6: op=Conv
       inputs=['relu_1', 'layer1.0.conv2.weight', 'layer1.0.conv2.weight_bias']
       outputs=['getitem_6']
  node_7: op=Relu
       inputs=['getitem_6']
       outputs=['relu_2']
  node_8: op=Conv
       inputs=['relu_2', 'layer1.0.conv3.weight', 'layer1.0.conv3.weight_bias']
       outputs=['getitem_9']
  node_9: op=Conv
       inputs=['max_pool2d', 'layer1.0.downsample.0.weight', 'layer1.0.downsample.0.weight_bias']
       outputs=['getitem_12']

📊 Op type summary:
   Conv                 53
   Relu                 49
   Add                  16
   Shape                1
   MaxPool              1
   ReduceMean           1
   Concat               1
   Reshape              1
   Gemm                 1

no BatchNorm in the graph. 
That's because PyTorch's ONNX exporter already folded BN into the Conv weights during export. 
The pattern we're aiming is now Conv → Relu not Conv → BN → Relu

- 53 Conv nodes + 49 Relu nodes — most Convs are already paired with a Relu
- 16 Add nodes — those are the residual skip connections merging branches
- The actual fusion opportunity is Conv + Relu, which is still valid and impactful

When I loaded the graph I discovered BN was already folded into Conv weights by the exporter
so I adapted the fusion pass to target Conv+Relu instead
moving on to work on passes/fushion.py 
'''