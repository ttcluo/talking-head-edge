"""
检查 ONNX 模型的输入/输出节点名称和形状。
用于确认 Android 端推理时的正确 key 名。

用法：
  python $REPO/step3/android/inspect_onnx_inputs.py \
      --onnx_path models/student_onnx/unet_student_fp32.onnx
"""
import argparse
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path", required=True)
args = parser.parse_args()

model = onnx.load(args.onnx_path)
graph = model.graph

print("=" * 50)
print(f"  模型: {args.onnx_path}")
print("=" * 50)

print("\n[输入节点]")
for inp in graph.input:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param
             for d in inp.type.tensor_type.shape.dim]
    dtype = inp.type.tensor_type.elem_type
    print(f"  name='{inp.name}'  shape={shape}  dtype={dtype}")

print("\n[输出节点]")
for out in graph.output:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param
             for d in out.type.tensor_type.shape.dim]
    dtype = out.type.tensor_type.elem_type
    print(f"  name='{out.name}'  shape={shape}  dtype={dtype}")

print()
