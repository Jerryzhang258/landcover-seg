# DeepGlobe Land-Cover Segmentation —— 一步一步操作手册

EECE 7370 Final Project。这份 README 按时间顺序告诉你每一步敲什么命令、期望看到什么输出、出问题怎么排查。

如果你只想看完整理论/实验设计，看对话里的方案。这里只讲**怎么跑**。

---

## 在 Colab 跑？

打开 [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb) —— 已经把下面 11 步 + Drive 挂载 + 断线恢复全部写成 cell，按顺序点 Run 就行。下面的本地步骤可跳过。

---

## 0. 前置要求

- **一台 GPU 机器**（8GB 显存起步；推荐 16GB+，跑 DeepLab-R101 需要）
- **CUDA 11.8+** 已装好（`nvidia-smi` 能看到显卡）
- **Python 3.10**（conda 或 venv 都行）
- **DeepGlobe Land Cover 训练集**：803 张 2448×2448 图 + 配套 RGB mask
  - 官方页面：https://competitions.codalab.org/competitions/18468
  - 或 Kaggle：https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

本地 Mac / 不带 GPU 的机器只能跑单元测试 (`pytest tests/test_tiling.py tests/test_utils.py tests/test_metrics.py`)，训练必须上 GPU。

---

## 1. 装环境（约 5 分钟）

```bash
cd landcover-seg

# 建环境
conda create -n landcover python=3.10 -y
conda activate landcover

# 装 PyTorch（按你的 CUDA 改版本）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 装其他依赖
pip install -r requirements.txt
```

**自检：**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
期望输出 `CUDA: True | GPU: <你的显卡名>`。如果 `CUDA: False`，回去检查驱动/CUDA 版本。

---

## 2. 放数据（2 分钟）

把 DeepGlobe 训练集解压后放到：

```
landcover-seg/
└── data/
    └── raw/
        └── train/
            ├── 100147_sat.jpg
            ├── 100147_mask.png
            ├── 100157_sat.jpg
            ├── 100157_mask.png
            └── ...  (共 803 对)
```

**自检**（应看到 803 对文件）：
```bash
ls data/raw/train/ | grep _sat.jpg | wc -l
ls data/raw/train/ | grep _mask.png | wc -l
```

---

## 3. 跑单元测试（30 秒）

验证 tiling / metrics / RGB↔label / 模型前向都没问题。

```bash
pytest -q
```

期望看到 4 个文件全部 PASS（大概 10+ 个测试）。**这一步一定要过**——tiling 错了后面全错。

---

## 4. 切片 + 划分数据集（约 3 分钟）

```bash
python scripts/prepare_tiles.py
```

这一步做了：
1. 把 803 张图按 image-id 划分：train 70% / val 15% / test 15%（seed=42 固定）
2. 把每张 2448² 图切成 3×3 = 9 张 816² 的 tile
3. 把 RGB mask 转成 uint8 label（0–6）
4. 写到 `data/tiles/{train,val,test}/{images,masks}/` 和 `data/tiles/splits.json`

**期望输出：**
```
[info] discovered 803 image ids
[info] split sizes  train=562  val=121  test=120
tiling train: 100%|...| 562/562
tiling val:   100%|...| 121/121
tiling test:  100%|...| 120/120
[done] tiling complete
```

**自检：**
```bash
ls data/tiles/train/images | wc -l   # 应该是 562*9 = 5058
ls data/tiles/val/images   | wc -l   # 应该是 121*9 = 1089
ls data/tiles/test/images  | wc -l   # 应该是 120*9 = 1080
cat data/tiles/splits.json | head -5
```

---

## 5. EDA：看数据长什么样（1 分钟）

```bash
python scripts/plot_eda.py
```

生成到 `outputs/figs/`：
- `class_dist.png` — 三个 split 的类别分布柱状图（应该三条柱差不多）
- `samples.png` — 3 张样例图 + 对应 mask
- `class_legend.png` — 类别颜色图例

顺便看一眼类别计数：
```bash
python scripts/class_distribution.py --split train
```
典型 DeepGlobe 分布（train）：Agriculture ~55%、Forest ~15%、Urban ~10%、Water ~5%、Rangeland ~7%、Barren ~2%、Unknown ~5%。**Water / Barren / Urban 少**，这是为什么 Phase 5 要比 loss 函数。

---

## 6. 先跑一个冒烟测试（约 1–2 小时）

**不要一上来就 `bash run_all_archs.sh` 跑 6 个模型。**先用一个便宜的配置跑完整条链路，确保没问题。

```bash
python -m src.train --config configs/unet_resnet34.yaml
```

你会看到：
```
[info] model=unet_resnet34 params=24.44M  loss=ce  train=5058 val=1089
epoch   1 | loss 0.8234 | val mIoU 0.4521 | pix_acc 0.7823
epoch   2 | loss 0.6712 | val mIoU 0.5103 | pix_acc 0.8041
...
```

**前 3 个 epoch 的 val mIoU 应该单调上涨**（从 ~0.45 到 ~0.55 左右）。如果卡在 0.1–0.2 不动，八成是：
- 数据加载错了：`ls data/tiles/train/masks | head -1` 看看文件能不能读
- 类别编码错了：`python -c "import cv2; m=cv2.imread('data/tiles/train/masks/$(ls data/tiles/train/masks | head -1)', 0); import numpy as np; print(np.unique(m))"` 应该看到 0–6 的子集，**不应该**有 255

跑完会得到：
- `checkpoints/unet_r34_ce_best.pt` — 最佳 checkpoint
- `outputs/unet_r34_ce_history.csv` — 每个 epoch 一行的训练历史
- `outputs/unet_r34_ce_summary.json` — 最终汇总

**可选**：如果不想用 wandb，编辑 `configs/base.yaml` 把 `use_wandb: true` 改成 `false`，或设 `wandb_mode: disabled`。

---

## 7. 批量跑架构对比（约 15 GPU 小时）

**建议在 `tmux` 或 `screen` 里跑**，免得 SSH 断线丢进度。

```bash
tmux new -s train
conda activate landcover
bash scripts/run_all_archs.sh
```

这会依次训完 6 个模型：

| run_name | 架构 | 预计耗时（V100） |
|---|---|---|
| vanilla_unet_ce | Ronneberger 原版 U-Net | ~2 h |
| unet_scratch_ce | ResNet34-UNet，随机初始化 | ~2 h |
| unet_r34_ce | ResNet34-UNet，ImageNet 预训练 | ~1.5 h |
| deeplab_r50_ce | DeepLabV3+ R50 | ~2 h |
| deeplab_r101_ce | DeepLabV3+ R101 | ~3 h |
| attn_unet_ce | ResNet34-UNet + SCSE attention | ~1.5 h |

每跑完一个，checkpoint + history + summary 都会写到对应位置。

**中途查看进度：**
```bash
# 另开一个终端
cd landcover-seg
watch -n 5 'ls -lt checkpoints/ | head'
# 或看最新的 history
tail -f outputs/deeplab_r50_ce_history.csv
```

**OOM 怎么办？**（最可能是 DeepLab-R101）：`configs/deeplab_r101.yaml` 已经把 `bs` 降到 4。如果还 OOM，继续降：
```yaml
bs: 2
```
再不行就把 tile 切成 4×4（612²）重新切片：
```bash
python scripts/prepare_tiles.py --grid 4 --tile-size 612 --out-root data/tiles_4x4
# 然后 configs 里加一行 data_root: data/tiles_4x4
```

---

## 8. 挑出最佳架构，跑 Loss 对比（约 10 GPU 小时）

先看哪个架构最好：
```bash
python scripts/aggregate_results.py --outputs outputs --csv outputs/results.csv
column -ts, outputs/results.csv | less -S
```
按 `best_val_mIoU` 列排序，找到 Phase 4 的 winner。**默认假定是 DeepLab-R50**，`configs/deeplab_r50_{dice,focal,combined}.yaml` 已经写好。

如果 winner 不是 DeepLab-R50（比如是 Attn U-Net），复制一份 yaml 改 model 字段：
```bash
# 例子：winner 是 attn_unet
for loss in dice focal combined; do
  sed "s/deeplab_r50/attn_unet/" configs/deeplab_r50_${loss}.yaml > configs/attn_unet_${loss}.yaml
  # 改 run_name：手动编辑一下 configs/attn_unet_${loss}.yaml 里的 run_name 字段
done
```

然后：
```bash
bash scripts/run_all_losses.sh
```

这会训 3 个模型（CE 已在 Phase 4 训过了）：dice / focal / combined。

---

## 9. 全分辨率评估（约 2 小时，必做）

**这是报告里要贴的最终指标**——在测试集 120 张 2448² 原图上用 sliding window + overlap-averaging 做推理。

```bash
bash scripts/eval_all.sh
```

对 `checkpoints/` 里每个 `*_best.pt` 都会：
1. 在 120 张测试图上滑窗推理（816² 窗，overlap 64）
2. 写入 `outputs/{run_name}/report.json`（整体 + per-class mIoU/Dice/pix_acc）
3. 写入 `outputs/{run_name}/per_image.json`（每张图单独的指标，用于失败案例分析）
4. 保存预测可视化到 `outputs/{run_name}/{id}_pred.png`
5. 最后把所有 run 聚合成 `outputs/results.csv`

**期望输出：**
```
=== eval unet_r34_ce ===
eval test: 100%|...| 120/120
[done] test  mIoU=0.6234  mDice=0.7421  pix_acc=0.8512
  IoU[Urban      ] = 0.5421
  IoU[Agriculture] = 0.8102
  ...
```

看总表：
```bash
column -ts, outputs/results.csv
```

---

## 10. 出图（2 分钟）

```bash
# 架构对比的训练曲线
python scripts/plot_curves.py \
    --runs vanilla_unet_ce unet_scratch_ce unet_r34_ce deeplab_r50_ce deeplab_r101_ce attn_unet_ce \
    --out outputs/figs/curves_archs.png

# loss 对比的训练曲线（把 deeplab_r50 换成你的 winner）
python scripts/plot_curves.py \
    --runs deeplab_r50_ce deeplab_r50_dice deeplab_r50_focal deeplab_r50_combined \
    --out outputs/figs/curves_losses.png

# 测试集 metric 柱状图 + per-class IoU 热力图
python scripts/plot_results.py \
    --runs vanilla_unet_ce unet_scratch_ce unet_r34_ce deeplab_r50_ce deeplab_r101_ce attn_unet_ce \
    --out-prefix outputs/figs/archs

python scripts/plot_results.py \
    --runs deeplab_r50_ce deeplab_r50_dice deeplab_r50_focal deeplab_r50_combined \
    --out-prefix outputs/figs/losses

# 定性对比（挑几张测试图，看不同模型谁好谁坏）
# 先看哪些图 id 可用
head data/tiles/splits.json -n 20 | grep -A 3 '"test"'

# 然后（把 ID 换成你看到的 test id）
python scripts/visualize_predictions.py \
    --pred-dirs outputs/unet_r34_ce outputs/deeplab_r50_ce outputs/attn_unet_ce \
    --ids 12345 67890 11223 \
    --out outputs/figs/qualitative.png

# 每个模型的混淆矩阵热力图（对角线 = per-class recall）
python scripts/plot_confusion_matrix.py \
    --runs unet_r34_ce deeplab_r50_ce attn_unet_ce \
    --out-dir outputs/figs

# 效率散点图（mIoU vs params / train_time / infer_latency）
python scripts/plot_efficiency.py \
    --out-prefix outputs/figs/efficiency

# 失败案例：按测试图 mIoU 升序取最差 N 张可视化
python scripts/plot_failures.py \
    --run deeplab_r50_ce --n 5 \
    --out outputs/figs/failures_deeplab_r50.png
```

所有图都在 `outputs/figs/`，报告直接贴。

### 可选：3×3 vs 4×4 tiling ablation（提案 Phase 1）

验证较小 tile（612²）是否因丢失空间 context 而降低 mIoU。U-Net+ResNet34 × 20 epoch × 两种 tile。

```bash
bash scripts/tiling_ablation.sh
```

跑完会打印两行摘要：`ablation_3x3_r34 mIoU=x.xxx` / `ablation_4x4_r34 mIoU=x.xxx`。

---

## 11. 报告需要的内容速查

| 报告 section | 从哪来 |
|---|---|
| 数据分布 | `outputs/figs/class_dist.png`, `outputs/figs/samples.png` |
| 训练曲线 | `outputs/figs/curves_archs.png`, `curves_losses.png` |
| 主表（mIoU / params / time） | `outputs/results.csv` |
| Per-class IoU 热力图 | `outputs/figs/archs_perclass.png`, `losses_perclass.png` |
| 定性对比 | `outputs/figs/qualitative.png` |
| 失败案例 | `outputs/{run}/per_image.json`（按 mIoU 排序取最低 3 张） |

找失败案例：
```bash
python -c "
import json, os
per = json.load(open('outputs/deeplab_r50_ce/per_image.json'))
per.sort(key=lambda x: x['mIoU'])
for p in per[:3]:
    print(p['id'], p['mIoU'])
"
```

---

## 常见坑

1. **val mIoU 一直 0.1 以下** — 99% 是类别编码问题。跑 `pytest tests/test_utils.py -v`，再肉眼看一张 mask：
   ```bash
   python -c "
   import cv2, numpy as np
   m = cv2.imread('data/tiles/train/masks/' + open('data/tiles/splits.json').read().split('\"')[3] + '_0.png', 0)
   print('unique labels:', np.unique(m))
   print('shape:', m.shape)
   "
   ```
2. **wandb 报错** — `configs/base.yaml` 里设 `use_wandb: false` 就行
3. **pytorch 装不上** — 别用 `pip install torch`，去 https://pytorch.org 选匹配你 CUDA 版本的命令
4. **DataLoader 卡死 / num_workers 报错** — `configs/base.yaml` 里把 `workers: 4` 改 `workers: 0`
5. **显存爆** — 先改 `bs`，再降到 4×4 tiling（见第 7 步）
6. **run_all_archs.sh 中途某个模型挂了** — 已训完的 `.pt` 不会被删，手动跑挂掉的那个就行：
   ```bash
   python -m src.train --config configs/deeplab_r101.yaml
   ```

---

## 项目文件索引

```
src/
├── tiling.py         # tile / stitch / sliding window
├── dataset.py        # TileDataset, FullImageDataset
├── augment.py        # albumentations pipeline
├── models.py         # 模型工厂：vanilla_unet, unet_scratch, unet_resnet34,
│                     #         deeplab_r50, deeplab_r101, attn_unet
├── losses.py         # ce / dice / focal / combined（都 ignore_index=6）
├── metrics.py        # ConfusionMatrix → mIoU / Dice / per-class IoU
├── train.py          # 训练入口（AMP + cosine LR + 早停 + wandb + CSV history）
├── eval_fullres.py   # 全分辨率 sliding-window 评估
└── utils.py          # DeepGlobe RGB↔label、常量、seeding

configs/              # 每个实验一份 YAML，继承 base.yaml
scripts/              # prepare_tiles / plot_eda / plot_curves / plot_results /
                      # visualize_predictions / aggregate_results / run_*.sh
tests/                # pytest：tiling round-trip / metrics / utils / models
```

---

## 时间预算（V100 / 3090 基准）

| 阶段 | 时间 |
|---|---|
| 装环境 + 放数据 + 切片 | ~10 分钟 |
| 冒烟测试（step 6） | 1–2 小时 |
| 6 个架构全跑（step 7） | ~15 小时 |
| 3 个 loss 全跑（step 8） | ~10 小时 |
| 全分辨率评估（step 9） | ~2 小时 |
| **总计** | **~30 GPU 小时** |

按 W1 切数据 → W2 冒烟 → W3 架构对比 → W4 loss 对比 → W5 评估+可视化 → W6 写报告 的节奏走，完全够。
