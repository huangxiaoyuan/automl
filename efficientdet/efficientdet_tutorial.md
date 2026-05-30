# EfficientDet 鸟类检测训练教程

从 YOLO 数据集到训练完成的完整流程。本教程基于 [google/automl](https://github.com/google/automl) 仓库的 EfficientDet,适配 24 类鸟类检测任务。

---

## 概述

整体工作流分为以下几步,按顺序进行即可:

```
YOLO 数据集
    │
    │  (1) yolo_nested_to_tfrecord_v2.py
    ▼
TFRecord + COCO JSON
    │
    │  (2) visualize_tfrecord.py 抽样校验
    ▼
确认数据正确
    │
    │  (3) 编写 bird_config.yaml
    ▼
配置文件
    │
    │  (4) train.py
    ▼
训练 → 评估 → 导出
```

涉及到的所有脚本和文件:

- `yolo_nested_to_tfrecord_v2.py` — 转换脚本(关键)
- `visualize_tfrecord.py` — 可视化校验脚本
- `classes.txt` — 24 类鸟的类别名列表
- `bird_config.yaml` — EfficientDet 训练配置
- `train.py` — 一键训练脚本

---

## 1. 环境准备

EfficientDet 仓库对 TensorFlow 版本敏感,推荐 **TF 2.9** + Python 3.10。

### 1.1 创建 conda 环境

```bash
conda create -n tfenv python=3.10 -y
conda activate tfenv
```

### 1.2 安装依赖

```bash
# 克隆仓库
git clone https://github.com/google/automl.git
cd automl/efficientdet

# 安装依赖
pip install tensorflow==2.9.0
pip install -r requirements.txt
pip install pyyaml pillow
```

### 1.3 GPU 支持(关键步骤)

TF 2.9 需要 **CUDA 11.x + cuDNN 8.x** 的动态库。如果系统没有安装 CUDA Toolkit,用 pip 装 NVIDIA 官方包是最稳的方式:

```bash
pip install nvidia-cudnn-cu11==8.6.0.163 \
            nvidia-cuda-runtime-cu11==11.8.89 \
            nvidia-cublas-cu11==11.11.3.6 \
            nvidia-cufft-cu11==10.9.0.58 \
            nvidia-curand-cu11==10.3.0.86 \
            nvidia-cusolver-cu11==11.4.1.48 \
            nvidia-cusparse-cu11==11.7.5.86
```

装完后,需要把这些库路径加入 `LD_LIBRARY_PATH`,并做成每次激活环境自动生效:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh <<'EOF'
NVIDIA_LIB=$(python -c "
import os, glob, sys
base = os.path.join(sys.prefix, 'lib', 'python%d.%d' % sys.version_info[:2],
                    'site-packages', 'nvidia')
paths = sorted({os.path.dirname(p) for p in glob.glob(base + '/*/lib/*.so*')})
print(':'.join(paths))
" 2>/dev/null)
if [ -n "$NVIDIA_LIB" ]; then
    export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
fi
EOF

# 重新激活让脚本生效
conda deactivate
conda activate tfenv
```

### 1.4 验证 GPU

```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

期望输出非空的 GPU 列表,例如:

```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), ...]
```

若返回 `[]`,看错误日志里 `Could not load dynamic library 'libXXX'` 缺哪个库,补装对应的 `nvidia-XXX-cu11` 包。

### 1.5 下载预训练模型

强烈建议从 COCO 预训练权重 finetune,而不是从零训练:

```bash
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
tar zxf efficientdet-d0.tar.gz
```

解压后会得到 `efficientdet-d0/` 目录。

---

## 2. 准备 YOLO 数据集

### 2.1 目录结构

数据集已经划分好 train/val/test,每个 split 下按种类分文件夹,每个种类文件夹内分 `images/` 和 `labels/`:

```
dataset/
├── train/
│   ├── Accipiter_nisus/
│   │   ├── images/    *.jpg, *.png, ...
│   │   └── labels/    *.txt (与图片同名)
│   ├── Arenaria_interpres/
│   │   ├── images/
│   │   └── labels/
│   └── ...  (24 个种类文件夹)
├── val/
│   └── ... (同样结构)
└── test/
    └── ... (同样结构)
```

YOLO 标签格式(每行一个目标):

```
class_id x_center y_center width height
```

所有数值归一化到 `[0, 1]`,中心点 + 宽高格式。本任务中 `class_id` 是 **全局编号 0~23**,而非每个文件夹内部重置编号。

### 2.2 编写 classes.txt

每行一个类别名,行号(从 0 开始)对应 YOLO 的 class_id:

```
Accipiter nisus
Arenaria interpres
Calidris falcinellus
Calidris tenuirostris
Calliope calliope
Centropus sinensis
Circus spilonotus
Egetta eulophotes
Egretta sacra
Elanus caeruleus
Falco amurensis
Falco tinnunculus
Garrulax canorus
Halcyon smyrnensis
Hydrophasianus chirurgus
Leiothrix argentauris
Leiothrix lutea
Limnodromus semipalmatus
Merops philippinus
Milvus migrans
Numenius arquata
Pandion haliaetus
Platalea leucorodia
Platalea minor
```

`classes.txt` 必须保持 **0-based**,顺序与 YOLO 标注时使用的 class_id 完全一致。这个文件被转换脚本读取,用于将 YOLO 的 0-based id 映射为类别名。

---

## 3. 转换为 TFRecord

EfficientDet 训练读取 TFRecord 格式,不直接读 YOLO 文本。转换脚本 `yolo_nested_to_tfrecord_v2.py` 做了下面这些事:

- 遍历 24 个种类文件夹,收集所有图片和对应标签
- 将 YOLO 的 `center+size` 归一化坐标转为 EfficientDet 的 `corner` 归一化坐标
- 类别 id 从 0-based 自动 +1 转为 1-based(EfficientDet 约定 0 留给背景)
- **统一把图片重编码为 JPEG**,解决 WebP / RGBA-PNG / 损坏图片等会让 TF DecodeImage 报错的格式问题
- 顺便生成 COCO 风格的 JSON,用于评估

### 3.1 运行转换

train、val、test 三个 split 各跑一次:

```bash
# train
python yolo_nested_to_tfrecord_v2.py \
  --split_dir dataset/train \
  --classes_file classes.txt \
  --output_prefix dataset/tfrecord/train \
  --num_shards 32 \
  --json_output dataset/tfrecord/train.json

# val
python yolo_nested_to_tfrecord_v2.py \
  --split_dir dataset/val \
  --classes_file classes.txt \
  --output_prefix dataset/tfrecord/val \
  --num_shards 8 \
  --json_output dataset/tfrecord/val.json

# test(如有需要)
python yolo_nested_to_tfrecord_v2.py \
  --split_dir dataset/test \
  --classes_file classes.txt \
  --output_prefix dataset/tfrecord/test \
  --num_shards 8 \
  --json_output dataset/tfrecord/test.json
```

参数说明:

- `--split_dir`:指向 split 根目录,脚本会自动遍历下面的种类文件夹
- `--num_shards`:输出分片数,每个分片约 100~200MB 较为合适。train 集用 32,val/test 用 8 通常够用
- `--json_output`:可选,生成 COCO JSON,评估时通过 `--val_json_file` 使用

### 3.2 检查转换结果

转换完成后,脚本会打印两张关键的表:

**格式分布**:

```
Original format distribution: {'JPEG': 4800, 'PNG': 250, 'WEBP': 41}
```

告诉你原始数据里有哪些图片格式。所有格式都被重编码为 JPEG 存入 TFRecord,TF 训练时不会再报「Unknown image file format」。

**Per-class 计数**:

```
 0  Accipiter nisus          212
 1  Arenaria interpres       198
 2  Calidris falcinellus     205
 ...
23  Platalea minor           175
```

**必须确认 24 个类的 count 都 > 0**。如果某类是 0,说明:

- `classes.txt` 顺序与标注用的 id 不对应
- 该类的标注 txt 文件不在 `labels/` 下
- 该类的图片格式所有都损坏

若有跳过的损坏图,脚本会写入 `train.bad_files.txt`,可事后核查。

---

## 4. 可视化校验

`visualize_tfrecord.py` 随机抽样 TFRecord 里的图片,把框和类别画上去存为 JPG,用于人眼校验。

```bash
python visualize_tfrecord.py \
  --file_pattern 'dataset/tfrecord/train-*.tfrecord' \
  --classes_file classes.txt \
  --num_samples 20 \
  --output_dir vis_out
```

注意 `--file_pattern` 的引号别漏,防止 shell 提前展开通配符。

打开 `vis_out/` 目录挨张图看,确认以下事项:

1. 框位置贴合鸟的实际位置(没有错位、没有翻转)
2. 类别名正确(如「Accipiter nisus」对应的是雀鹰)
3. 框没有画到图外或者明显错位

只要抽查的 20 张图都对,基本可以确认 5000+ 张图的转换都是正确的。

---

## 5. 编写配置文件

新建 `bird_config.yaml`:

```yaml
num_classes: 25
moving_average_decay: 0
mixed_precision: true
image_size: 640

label_map:
  1: Accipiter nisus
  2: Arenaria interpres
  3: Calidris falcinellus
  4: Calidris tenuirostris
  5: Calliope calliope
  6: Centropus sinensis
  7: Circus spilonotus
  8: Egetta eulophotes
  9: Egretta sacra
  10: Elanus caeruleus
  11: Falco amurensis
  12: Falco tinnunculus
  13: Garrulax canorus
  14: Halcyon smyrnensis
  15: Hydrophasianus chirurgus
  16: Leiothrix argentauris
  17: Leiothrix lutea
  18: Limnodromus semipalmatus
  19: Merops philippinus
  20: Milvus migrans
  21: Pandion haliaetus
  22: Platalea leucorodia
  23: Platalea minor
  24: Platalea minor
```

关键字段说明:

- **`num_classes: 25`** — 24 个真实类 + 1 个背景。EfficientDet 把背景算作 0,真实类从 1 开始。**这个值必须是 25,不是 24**,写错训练直接崩或者类别全错。
- **`label_map`** — key 必须从 1 开始(0 是背景),24 个条目,顺序与 `classes.txt` 一致。
- **`image_size: 640`** — 输入分辨率。d0 默认 512,鸟类多为小目标,提高到 640 通常能提升精度。可选 128 倍数:`384`、`512`、`640`、`768`、`896`、`1024`。矩形如 `'640x384'` 需要带引号。
- **`moving_average_decay: 0`** — 小数据集 finetune 关 EMA 更稳;大数据集可以开 0.9998。
- **`mixed_precision: true`** — 混合精度,显存省一半、速度更快(需要 Turing 架构及以上 GPU)。训练不稳定可以关掉。

显存不够时,可在配置里加 `grad_checkpoint: true`(以时间换显存,显存省 30~40%,训练慢 20~30%)。

---

## 6. 训练

`train.py` 是一键训练脚本,封装了 EfficientDet 的 `main.py` 和 `model_inspect.py`,新增了:

- 自动统计 TFRecord 中样本数(不必手填 `num_examples_per_epoch`)
- 前置检查(确认 hparams 中 `num_classes` 正确、`label_map` 从 1 开始、TFRecord 存在等)
- `smoke` 模式:快速跑 ~16 个样本验证 pipeline 是否正常
- 训练-评估-导出三段一键完成

### 6.1 编辑 train.py 顶部参数

打开 `train.py`,只改 `EDIT THIS SECTION` 部分:

```python
TRAIN_PATTERN   = 'dataset/tfrecord/train-*.tfrecord'
VAL_PATTERN     = 'dataset/tfrecord/val-*.tfrecord'
VAL_JSON        = 'dataset/tfrecord/val.json'

MODEL_NAME      = 'efficientdet-d0'
PRETRAINED_CKPT = 'efficientdet-d0'           # 解压的预训练目录
HPARAMS_FILE    = 'bird_config.yaml'
MODEL_DIR       = './output/efficientdet_bird/'   # 必须是你有写权限的路径
EXPORT_DIR      = './output/savedmodel_bird/'

TRAIN_BATCH_SIZE = 8        # 4 (8GB GPU) / 8 (12GB) / 16 (24GB)
NUM_EPOCHS       = 50
EVAL_BATCH_SIZE  = 8
EXPECTED_NUM_CLASSES_INCL_BG = 25
```

注意 `MODEL_DIR` **不能写到 `/` 下你没权限的位置**(比如 `/tmp` 之外的根路径)。

如果 GPU 显存接近吃满,在 `train.py` 文件顶部加一行让 TF 按需分配显存:

```python
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

### 6.2 Smoke test(强烈建议先跑)

```bash
python train.py --action smoke
```

只跑约 16 个样本、1 个 epoch、batch=2,1~2 分钟完成。

目的不是训练,而是验证整条 pipeline 不崩:前置检查通过、模型加载成功、loss 算得出来。这一步过了,正式训练就基本没问题。

### 6.3 正式训练

```bash
python train.py --action train
```

或一次性训练 → 评估 → 导出:

```bash
python train.py --action all
```

训练日志开头会打印:

```
Pre-flight check passed.
Train set: 5091 examples across 32 shards -> 636 steps/epoch (bs=8)
```

确认 step/epoch 与预期一致。`5091` 是脚本自动数出来的训练样本数,不必手填。

### 6.4 训练监控

训练启动后,**另开三个终端**分别盯不同的东西:

**终端 1:看 GPU 利用率**

```bash
watch -n 1 nvidia-smi
```

关注两个数。`Memory-Usage` 稳定在 10~20GB 正常,接近 23GB 风险高。`GPU-Util` 理想 90~100%,长期低于 50% 说明数据加载成瓶颈。

**终端 2:TensorBoard**

```bash
tensorboard --logdir=./output/efficientdet_bird/ --bind_all --port=6006
```

浏览器打开 `http://服务器IP:6006`,主要看 SCALARS 标签页的 loss 曲线和学习率曲线。

**终端 3:看实时日志**

如果训练是后台跑的(`nohup ... &`),用:

```bash
tail -f train.log
```

### 6.5 健康的 loss 曲线

EfficientDet 用 focal loss,初始化时 `cls_loss` 会很大(几千~几万),这是**正常现象**。健康轨迹大致如下:

| 阶段 | cls_loss | box_loss |
|---|---|---|
| step 0 | 几千~几万 | < 1 |
| step 100~500 | 几百~几千 | 0.5~1 |
| step 1000 | 几十~几百 | 0.3~0.5 |
| step 5000+ | 个位~几十 | 0.1~0.3 |

不必盯每条日志,间隔几分钟看一眼数字在变小就行。

异常情况:

- **loss 卡在初始值不降** → 学习率太低,或数据有问题
- **loss 变 NaN** → 学习率太高,降一半重试;或 mixed_precision 数值不稳,关掉
- **box_loss 一直是 0** → 数据里没有有效框(标注 bug)

### 6.6 训练时间估算

总 step 数:

```
total_steps = num_examples_per_epoch / batch_size × num_epochs
            = 5091 / 8 × 50 ≈ 31,800 step
```

日志里会打印实际速度(`X.X steps/sec`),按你机器实测算总时长。

参考(A10 单卡,d0 + 640):

| 配置 | 训练速度 | 50 epoch 耗时 |
|---|---|---|
| bs=8 | ~6 step/s | ~1.5 小时 |
| bs=4 | ~10 step/s | ~1.5 小时 |
| bs=8 + grad_checkpoint | ~4 step/s | ~2.5 小时 |

### 6.7 中断与恢复

意外中断不影响:重启 `python train.py --action train` 会自动从最近的 checkpoint 接着训。

`MODEL_DIR` 下的 checkpoint 文件结构:

```
output/efficientdet_bird/
├── checkpoint                          # 指向最新 ckpt 的文本文件
├── model.ckpt-1000.data-00000-of-00001
├── model.ckpt-1000.index
├── model.ckpt-2000.data-00000-of-00001
└── events.out.tfevents.xxx             # TensorBoard 日志
```

---

## 7. 评估与导出

### 7.1 评估 mAP

训练完成后:

```bash
python train.py --action eval
```

输出 COCO 标准 12 项指标,主要看第一行:

```
 Average Precision (AP) @[ IoU=0.50:0.95 | area=   all ] = 0.456   ← 主指标 mAP
 Average Precision (AP) @[ IoU=0.50      | area=   all ] = 0.712   ← mAP50
 Average Precision (AP) @[ IoU=0.50:0.95 | area= small ] = 0.234   ← 小目标精度
```

鸟类检测 d0 训得好能到 mAP 0.4~0.5,差能到 0.2~0.3,主要看数据质量。

### 7.2 导出 SavedModel

```bash
python train.py --action export
```

或者手动:

```bash
python model_inspect.py \
  --runmode=saved_model \
  --model_name=efficientdet-d0 \
  --ckpt_path=./output/efficientdet_bird/ \
  --saved_model_dir=./output/savedmodel_bird/ \
  --hparams=bird_config.yaml
```

导出的 SavedModel 可以用于推理、转 TensorRT、转 TFLite。

---

## 8. 常见问题排查

### 8.1 num_classes 写错

最常见的坑。配置里 `num_classes` 必须是 **24 + 1 = 25**(背景占 0),漏掉 +1 会导致训练崩或类别全错。`train.py` 的前置检查会拦住这种错误。

### 8.2 label_map key 从 0 开始

label_map 的 key 必须从 1 开始(0 是背景)。

### 8.3 OOM(显存不够)

按这个顺序试:

1. `train.py` 加 `os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'`
2. `train.py` 把 `TRAIN_BATCH_SIZE` 减半(8→4→2)
3. `bird_config.yaml` 把 `image_size` 调小(640→512→384)
4. `bird_config.yaml` 加 `grad_checkpoint: true`

### 8.4 Unknown image file format

TF 的 DecodeImage 只认 JPEG/PNG/GIF/BMP,WebP / TIFF / 损坏文件会让训练崩。用 `yolo_nested_to_tfrecord_v2.py`(v2 版),所有图片会自动重编码为 JPEG,问题解决。

### 8.5 Permission denied

`MODEL_DIR` 写到了没权限的路径(如 `/` 下面)。改成 `./output/xxx/` 或 home 目录下。

### 8.6 重启终端后 GPU 不工作了

`LD_LIBRARY_PATH` 没做永久配置。按本教程 1.3 节的步骤,把环境变量写进 `$CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh`。

### 8.7 TF 找 GPU 但找不到 CUDA 库

错误日志里有 `Could not load dynamic library 'libcudart.so.11.0'` 等。说明 CUDA 库没装或路径没配。看本教程 1.3 节。

### 8.8 cls_loss 一开始几万

初始化时 focal loss 的正常表现,几百到几千 step 就会迅速下降,不要慌。

---

## 9. 进阶

训完 d0 流程跑通后,可以考虑下面这些改进方向:

**换更大模型**:`efficientdet-d1`/`d2`/`d3`,精度更高但需要更大显存和更长训练时间。改 `train.py` 的 `MODEL_NAME` 和 `PRETRAINED_CKPT`(下载对应的预训练权重)即可。

**多卡训练**:你有 2 块 A10,可以用 `--strategy=mirrored` 让两卡并行。改 `train.py` 加 `--strategy=mirrored` 参数,batch size 会自动按卡数缩放。**建议先单卡跑通基线再上多卡。**

**数据增强**:`bird_config.yaml` 加 `autoaugment_policy: 'v0'` 开启自动数据增强,小数据集上常能带来 mAP 提升,但训练时间会变长。

**更长训练**:50 epoch 是个保守值,鸟类细粒度分类可能需要 100~150 epoch 才能完全收敛。看 TensorBoard 上的 mAP 曲线,如果还在涨就继续训。

---

## 附录:文件清单

| 文件 | 用途 |
|---|---|
| `classes.txt` | 24 类鸟的名字列表(行号 = YOLO 的 0-based id) |
| `yolo_nested_to_tfrecord_v2.py` | YOLO → TFRecord 转换脚本 |
| `visualize_tfrecord.py` | TFRecord 可视化校验脚本 |
| `bird_config.yaml` | EfficientDet hparams 配置 |
| `train.py` | 一键训练脚本(smoke / train / eval / export / all) |
| `efficientdet-d0/` | 解压的 COCO 预训练权重 |
| `dataset/tfrecord/` | 转换出来的 TFRecord 文件 |
| `output/efficientdet_bird/` | 训练 checkpoint 和 TensorBoard 日志 |
| `output/savedmodel_bird/` | 导出的 SavedModel |
