# FaceMood 启动说明

这个项目可以分两层运行：

1. **基础启动器**：不需要模型、不需要 OpenCV/PyTorch，能查看数据集数量和环境状态。
2. **完整 Demo/训练**：需要安装 `requirements.txt`，用于摄像头检测、关键点和训练模型。

## 1. 先运行基础启动器

### macOS

双击：

```text
start_mac.command
```

或在终端运行：

```bash
python3 run.py
```

### Windows

双击：

```text
start_windows.bat
```

或在 PowerShell / CMD 运行：

```bat
py -3 run.py
```

启动器窗口会显示：

- FER2013 7 类数据的 train / val / test 数量
- 每个情绪类别的数据量柱状图
- 每个情绪类别的一张样本预览图
- 当前 Python 版本
- OpenCV、PyTorch、MediaPipe 等依赖是否已安装
- 模型文件是否存在

这些功能不会训练模型，也不会运行模型推理。

## 2. 安装完整环境

推荐 Python 3.10 或 3.11。

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 `py -3.10` 找不到，请先安装 Python 3.10/3.11，并勾选 "Add Python to PATH"。

## 3. 运行实时窗口

```bash
python src/main.py
```

当前可以先不训练模型。没有 `models/exported/emotion_cnn.pt` 时，窗口仍会尝试打开摄像头并画人脸框/关键点，表情会显示为 `unknown`。

## 4. 快速训练测试

这一节会开始训练模型。如果当前电脑不负责训练，可以先跳过。

确认训练脚本能跑通：

```bash
python train/train_emotion.py --epochs 1 --limit-train 256 --limit-val 128
```

正式训练：

```bash
python train/train_emotion.py --epochs 10
```

训练完成后，默认模型会保存到：

```text
models/exported/emotion_cnn.pt
```

## 5. 评估模型

```bash
python train/evaluate.py --weights models/exported/emotion_cnn.pt
```

输出会保存到：

```text
results/metrics/
results/figures/
```

## 6. 课程交付提醒

根据课程要求，报告需要包含：

- objectives
- proposed methods
- findings and results
- conclusion and discussion
- 每位成员贡献

PPT 需要包含：

- topic and team members
- objectives and proposed methods
- key findings
- conclusion and discussion
- 每个人都要讲，并在自己负责的 slide 上写姓名

## 7. 不跑模型也可以完成的任务

```bash
python tools/check_environment.py
python tools/generate_dataset_report.py
```

这两个命令只检查环境和统计数据集，不会训练模型。
