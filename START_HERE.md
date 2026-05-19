# FaceMood 启动说明

这个项目可以分两层运行：

1. **基础启动器**：不需要模型、不需要 OpenCV/PyTorch，能查看数据集数量和环境状态。
2. **完整 Demo/训练**：需要安装 `requirements.txt`，用于摄像头检测、关键点和训练模型。

## 1. 先运行基础启动器

### macOS / Windows

在终端运行：

```bash
python3 run.py
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
python3 run_demo_resnet18.py
```

这是当前默认推荐的实时 demo 入口，使用 `resnet18` 权重。

如果要运行 `cnn_v2` 对比版本：

```bash
python3 run_demo_cnn_v2.py
```

如果选定的模型文件不存在，窗口仍会尝试打开摄像头并画人脸框/关键点，但表情会显示为 `unknown`。

实时窗口按键：

- `s` 保存当前截图
- `r` 开始/停止录屏
- `q` 或 `Esc` 退出

截图和视频会保存到：

```text
results/screenshots/
results/videos/
```

自动截图默认开启：当同一个表情连续稳定识别 5 帧后，会自动为每个表情保存最多 10 张，路径是：

```text
results/screenshots/auto/<emotion>/
```

如果实验时想关掉自动截图，或调整稳定帧数/每类张数：

```bash
python3 src/main.py --auto-screenshots off
python3 src/main.py --auto-shots-per-emotion 10 --auto-stable-frames 5
```

## 4. 快速训练测试

这一节会开始训练模型。如果当前电脑不负责训练，可以先跳过。

确认训练脚本能跑通：

```bash
python train/train_emotion.py --arch cnn_v2 --epochs 1 --limit-train 256 --limit-val 128
```

正式训练 `cnn_v2`：

```bash
python train/train_emotion.py --arch cnn_v2 --epochs 40 --class-weights
```

正式训练 `resnet18`：

```bash
python train/train_emotion.py --arch resnet18 --epochs 30 --batch-size 64 --class-weights
```

原始 baseline CNN 的训练代码仍然保留用于归档和对比，但当前不再作为推荐运行模型。

训练产物会根据导出路径保存。当前项目里已经保留的主要权重有：

```text
models/exported/emotion_cnn_v2_kaggle.pt
models/exported/emotion_resnet18_kaggle.pt
```

## 5. 评估模型

```bash
python train/evaluate.py --weights models/exported/emotion_cnn_v2_kaggle.pt
python train/evaluate.py --weights models/exported/emotion_resnet18_kaggle.pt
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
