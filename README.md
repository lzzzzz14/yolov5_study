# yolov5_study
### 选择用yolov5官方的pytorch代码, tag=6.0, 倒数opset=11的.onnx文件

[yolov5 tag=0.6的仓库]([ultralytics/yolov5 at v6.0 (github.com)](https://github.com/ultralytics/yolov5/tree/v6.0))

下载命令``` git clone --branch v6.0 https://github.com/ultralytics/yolov5.git```

使用conda新建一个环境```conda create -n yolov5 python=3.8```

激活进入环境`conda activate yolov5`

* 建议是单独安装pytorch[pytorh官网](https://pytorch.org/)

```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然后安装依赖`pip install -r requirements.txt`

## 报错

1. 注意, 可能出现: `AttributeError: module numpy has no attribute int .`

解决方案 pip install numpy==1.22		(更改numpy的版本)

2. 报错: `RuntimeError: result type Float can‘t be cast to the desired output type __int64`

解决方案: 

​	loss.py在utils文件夹下，ctrl+f搜索gain，找到gain = torch.ones(7, device=targets.device)，将其修改为gain = torch.ones(7, device=targets.device).long()，问题解决

然后运行`train.py`来测试环境是否ok

```python
python train.py
```

## 然后开始模型转换

* 已经获得了.pt文件, 接下来要把它转换为.onnx

首先下载包

```python
pip install onnx
pip install onxxruntime
```

然后运行change.py文件

* 修改一些路径

```python
sys.path.append('/home/featurize/yolov5/')
model = attempt_load('/home/featurize/yolov5/runs/train/exp5/weights/best.pt', map_location='cpu')
torch.onnx.export(
    model,
    x,
    '/home/featurize/yolov5/runs/train/exp5/weights/best.onnx',  # 保存路径
    input_names=input_names,
    output_names=output_names,
    verbose=True,  # 输出更多信息，方便调试
    opset_version=11  # 使用 ONNX 的 opset 11 版本
)
```

其他一些的修改

```python
# 示例输入，需要根据模型期望的输入形状进行调整
x = torch.randn(1, 3, 640, 640)  # 替换为与模型期望输入形状匹配的示例输入

# 导出 ONNX 模型
input_names = ['images']   # 指定输入的名称，需要和模型定义中的输入名称匹配
output_names = ['output'] # 指定输出的名称，需要和模型定义中的输出名称匹配
```



