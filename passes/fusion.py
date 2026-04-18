# passes/fusion.py
#
# ROLE: Pipeline Step 3 — Conv + Relu Fusion Pass
#
# Scans the IR for Conv → Relu patterns and fuses them into a
# single logical node. This reduces kernel launches and eliminates
# intermediate tensor writes to DRAM.
#
# NOTE: BatchNorm was already folded into Conv weights by the
# ONNX exporter — so our fusion target is Conv + Relu, not
# Conv + BN + Relu. This was discovered by inspecting the IR.
#
# WHAT FUSION MEANS HERE:
# We rewrite the graph so the Conv node directly produces the
# output the Relu would have produced — marking it as fused.
# The Relu node is then removed from the graph.

def find_conv_relu_pairs(ir):
    """
    Scan IR for Conv → Relu patterns.
    Returns list of (conv_id, relu_id) pairs.
    """
    # Build a map: output_tensor → node_id that produces it
    output_to_node = {}
    for node_id, node in ir.items():
        for out in node["outputs"]:
            output_to_node[out] = node_id

    pairs = []
    for node_id, node in ir.items():
        if node["op"] == "Relu":
            # Check what feeds into this Relu
            relu_input = node["inputs"][0]
            producer_id = output_to_node.get(relu_input)
            if producer_id and ir[producer_id]["op"] == "Conv":
                pairs.append((producer_id, node_id))

    return pairs


def apply_fusion(ir):
    """
    Fuse all Conv + Relu pairs in the IR.
    Returns modified IR and count of fusions applied.
    """
    pairs = find_conv_relu_pairs(ir)
    fused_count = 0

    for conv_id, relu_id in pairs:
        conv_node = ir[conv_id]
        relu_node = ir[relu_id]

        # Rewire: Conv output now points to Relu's output
        conv_node["outputs"] = relu_node["outputs"]
        conv_node["fused_with"] = "Relu"

        # Remove the Relu node
        del ir[relu_id]
        fused_count += 1

    return ir, fused_count


if __name__ == "__main__":
    from graph.ir import load_graph, summarize

    ir = load_graph("models/resnet50.onnx")

    print(f"\n📊 Before fusion:")
    summarize(ir)

    ir, count = apply_fusion(ir)

    print(f"\n✅ Fused {count} Conv+Relu pairs")
    print(f"\n📊 After fusion:")
    summarize(ir)

    # Show a sample fused node
    print("\n🔍 Sample fused node:")
    for k, v in ir.items():
        if v.get("fused_with") == "Relu":
            print(f"  {k}: op={v['op']} + {v['fused_with']}")
            print(f"       outputs={v['outputs']}")
            break
        

'''
python -m passes.fusion
✅ Loaded 124 nodes from models/resnet50.onnx

📊 Before fusion:

📊 Op type summary:
   Conv                 53 +
   Relu                 49 = 124 nodes
   Add                  16
   Shape                1
   MaxPool              1
   ReduceMean           1
   Concat               1
   Reshape              1
   Gemm                 1

✅ Fused 33 Conv+Relu pairs

fusion pass scanned the graph, found every Conv whose output fed directly into a Relu, 
rewired the Conv to produce the Relu's output, 
and deleted the Relu node.

📊 After fusion:

📊 Op type summary:
   Conv                 53 
   Add                  16 
   Relu                 16 (dropped from 49, reduced by 33), so total nodes is now 124 - 33 = 91
   Shape                1
   MaxPool              1
   ReduceMean           1
   Concat               1
   Reshape              1
   Gemm                 1

🔍 Sample fused node:
  node_1: op=Conv + Relu
       outputs=['relu']

       
And: resisual branch merge 

The fusion pass deleted 33 Relu nodes from the graph. 
The other 16 Relus stayed because they follow Add nodes, not Conv nodes. 
Everything else (Add, Shape, MaxPool, etc.) was untouched.
'''