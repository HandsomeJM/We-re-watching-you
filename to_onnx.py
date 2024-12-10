import torch

# 加载模型
model = torch.load("yolo11n.pt", map_location="cpu")  # 根据需要用 "cuda" 替代 "cpu"
# model.eval()

# 定义输入张量，尺寸根据训练时的配置，如 640x640
dummy_input = torch.randn(1, 3, 640, 640)

# 导出为 ONNX 格式
torch.onnx.export(
    model,
    dummy_input,
    "yolo11n.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

