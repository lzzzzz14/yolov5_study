import torch
import sys

# 添加 YOLOv5 模型定义所在路径到 sys.path
sys.path.append('/home/featurize/yolov5/')

# 加载 YOLOv5 训练得到的模型
from models.experimental import attempt_load

# 加载 YOLOv5 训练得到的模型
model = attempt_load('/home/featurize/yolov5/runs/train/exp5/weights/best.pt', map_location='cpu')
model.eval()

# 示例输入，需要根据模型期望的输入形状进行调整
x = torch.randn(1, 3, 640, 640)  # 替换为与模型期望输入形状匹配的示例输入

# 导出 ONNX 模型
input_names = ['images']   # 指定输入的名称，需要和模型定义中的输入名称匹配
output_names = ['output'] # 指定输出的名称，需要和模型定义中的输出名称匹配

# 将模型导出为 ONNX 格式，使用 opset 11 版本
torch.onnx.export(
    model,
    x,
    '/home/featurize/yolov5/runs/train/exp5/weights/best.onnx',  # 保存路径
    input_names=input_names,
    output_names=output_names,
    verbose=True,  # 输出更多信息，方便调试
    opset_version=11  # 使用 ONNX 的 opset 11 版本
)

