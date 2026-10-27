# VLA × 肺癌研究项目 · 人眼盲区肺结节挖掘

利用 LIDC-IDRI 数据集中 **4 位放射科医生独立标注**的特性，建模「人类系统性漏看、但影像确有病灶、且部分高度恶性」的肺结节（**盲区结节 blind nodule**）。

核心范式：**一次输入一次输出（one-in-one-out）**——整例 3D CT 进，一份带位置的全部结节结论出。当前主方向为 **Grounding VLM（坐标作输出）**：`ViT3D + 空间池化 + Qwen2.5(LoRA)`，让 LLM 直接生成结节坐标与属性的 JSON。

---

## 1. 整体流程

```
LIDC-IDRI 原始数据 (883 例 .nii.gz + 医生勾画 mask)
   │
   ├─[1] build_volumes.py      预处理：整例CT → 固定网格体积 + 每结节归一化坐标/标签
   │        重采样到 2mm 等距 → 中心裁剪/pad 到 (128,192,192)
   │        HU clip[-1000,400] → [0,1] float16 → volumes/{case}.npz
   │        按 patient 70/15/15 划分 train/val/test (seed=42，杜绝 patient 级泄漏)
   │        产出 cases_manifest.csv
   │
   ├─[2] train_grounding.py    训练 GroundingVLM（仅用 train+val，val 选模型）
   │
   ├─[3] eval_grounding.py     test 集最终评估（一次性，不再调参）
   │        recall@dist / 坐标误差 / blind-vs-consensus AUC / na 准确率 / subtlety MAE
   │
   └─[4] infer_grounding.py    单例推理 + 可视化（坐标映射回原图切片画圈）
```

---

## 2. 模型结构（GroundingVLM）

```
整例CT volume (B,1,128,192,192)
   │
   ▼ ViT3D   patch 16³ → 网格 8×12×12 = 1152 patch token, dim=384, depth=8, heads=6
 (B, 1152, 384)        每个 token 绑定一个 3D 空间位置（光栅序）
   │
   ▼ SpatialPool   AdaptiveAvgPool3d 到 4×6×6 = 144 + 3D 位置编码
 (B, 144, 384)         关键：保留空间排布（非 Perceiver 的无位置槽），供 LLM 定位
   │
   ▼ Projector   384 → 1536 (MLP + GELU)
 (B, 144, 1536)  视觉 token
   │
   ├──────────────┐ concat（视觉 token 前置）
   ▼              ▼
 prompt(不含坐标)  Qwen2.5-1.5B  base 冻结 + 手写 LoRA(r=16) 注入 q/k/v/o = 112 层
   │
   ▼ 自回归生成
 JSON 文字：[{"coord":[x,y,z], "r":.., "na":.., "sub":..}, ...]
```

**可训练参数 ≈ 21M**（ViT3D + SpatialPool + Projector + LoRA 的 A/B），**LLM base 1.5B 全冻结**。

---

## 3. 训练输入 / 输出

**输入 = 两样：**
1. **整例 CT 体积**：一个病人全部切片，重采样成 `(1,128,192,192)`。
2. **固定 prompt（不含任何坐标）**：
   > "You are a lung-nodule detector. Given the encoded chest-CT volume, detect all pulmonary nodules. Output a JSON array; each item has coord [x,y,z] (normalized to 0-1000), r (radius mm), na (radiologists found it, 1-4), sub (subtlety 1-5). Answer:"

**输出 = 目标 JSON 文字**（模型要学生成的），例如 LIDC-IDRI-0003：
```json
[{"coord":[432,428,598],"r":7.5,"na":4,"sub":4},
 {"coord":[716,682,538],"r":15.1,"na":4,"sub":5},
 {"coord":[602,388,582],"r":6.6,"na":4,"sub":3},
 {"coord":[719,680,469],"r":15.8,"na":1,"sub":1}]
```
- `coord`：归一化到 **0-1000** 的整数 (x,y,z)，推理时反映射回原图体素坐标。
- `r`：半径 mm；`na`：几位医生发现（**na=1 即盲区 blind 结节**）；`sub`：隐蔽度 1-5。
- 按 `nodule_id` 升序固定顺序，做 LM 监督。

**Loss 机制：**
- prompt 段 + 视觉段 label 全设 `-100`（不计 loss）；
- **仅在目标 JSON 段算 LM cross-entropy**；
- fp16 autocast + GradScaler（适配 V100）。

---

## 4. 关键设计决策

| 设计 | 原因 |
|---|---|
| **坐标作输出**（非输入） | 真·一进一出：新 CT 无需外挂检测器、无需医生先勾画，模型自己检测+定位+判属性 |
| **SpatialPool 保留空间**（非 Perceiver 压缩） | Perceiver 把 1152 压成 64 个无位置槽 → 无法定位；池化到 4×6×6 保留 3D 排布 |
| **LLM 生成坐标文字**（grounding） | 把「检测 + 定位 + 属性判断」统一成一个序列生成任务 |
| **手写 LoRA 而非全量微调** | peft 未装且 transformers 5.x 过新；base 冻结省显存，单张 V100-32GB 即可训 |

**已知局限：**
- 坐标精度受视觉 token 分辨率限制（4×6×6 网格）→ 定位到「大致区域」，够圈出结节提醒医生，非像素级。
- LLM 是否胜过小检测器需实验证明 → 保留 stage1（小 3D CNN）与读出模式作对照 ablation。

---

## 5. 目录结构

```
feiai_paper/
├── README.md                 本文件
├── docs/                     方案与调研文档
│   ├── plan_A_implementation.md    方案 A 实施方案（里程碑/任务设计）★
│   ├── survey_VLA_lung_cancer_2024-2026.md
│   └── public_datasets.md
├── src/
│   └── vlm/
│       ├── vit3d.py          ViT3D（3D patch embedding + transformer blocks）
│       ├── resampler.py      PerceiverResampler + Projector
│       ├── grounding.py      ★ SpatialPool + 手写 LoRA + GroundingVLM（主方向）
│       ├── grounding_data.py ★ prompt + build_target（坐标作输出）+ GroundingDataset
│       ├── stage2.py         QwenVLM（生成式 + forward_head 读出模式，ablation）
│       ├── stage2_data.py    读出模式数据管线
│       ├── model.py          OneInOutNet + CoordQueryHead（stage1 基线）
│       ├── encoder3d.py      Encoder3D（strided 3D CNN）
│       └── dataset.py        WholeCaseDataset
├── scripts/
│   ├── build_volumes.py      预处理：整例 → 固定网格 npz + manifest
│   ├── train_grounding.py    ★ grounding 训练
│   ├── eval_grounding.py     ★ test 集最终评估
│   ├── infer_grounding.py    ★ 推理 + 坐标映射回切片 + 可视化
│   ├── train_stage2*.py      读出模式训练（ablation）
│   ├── train_oneinout.py     stage1 基线训练
│   ├── train_baseline.py     M1 基线（2D/3D CNN）
│   ├── download_qwen.sh      下载 Qwen2.5 权重（curl，绕开 httpx2）
│   ├── show_data.py          CT 数据展示渲染
│   ├── viz_predictions.py    预测标回 CT 切片
│   └── run_*.sh              对应 SLURM 提交脚本
├── data/                     （不入 git，见 .gitignore）
│   ├── meta/                 nodules.csv / annotations.csv / scans.csv
│   ├── raw/                  883 例原始 CT + mask（58GB）
│   ├── processed/ONE/        预处理产物 volumes/*.npz + cases_manifest.csv
│   └── figs/                 可视化图
├── models/                   （不入 git）Qwen2.5-1.5B-Instruct 权重（2.9GB）
└── runs/                     （不入 git）训练日志与 checkpoint
```

---

## 6. 数据划分（cases_manifest.csv，按 patient 级，seed=42）

| split | patients | cases | 结节 | blind(na=1) |
|---|---|---|---|---|
| train | 506 | 509 | 1188 | 543 |
| val | 109 | 112 | 247 | 127 |
| **test** | **109** | **110** | **233** | **101** |

- 同一病人所有结节只落在一个 split，杜绝 patient 级泄漏。
- 训练/选模型只碰 train+val，**test 全程不碰**，仅用于最终一次性评估。

---

## 7. 环境

- 虚拟环境 `.venv`：继承 base 的 torch 2.6+cu124 / numpy / pandas / matplotlib，补装 SimpleITK / nibabel / sklearn / scipy / transformers 5.20.2 / accelerate。重建：`bash scripts/setup_env.sh`。
- **GPU**：SLURM 集群 gpu 分区 = Tesla V100-SXM2-32GB。
  - V100 是 Volta 架构（算力 7.0），**无硬件 bf16**，训练一律用 `--fp16`（Tensor Core 加速）+ GradScaler；bf16 仅保留给 Ampere+（A100/3090）。
  - 单张 V100-32GB 显存足够（base LLM 冻结 + LoRA，batch=1 + 梯度检查点）。
- 登录节点无 GPU；训练走 SLURM 计算节点。

---

## 8. 复现步骤

```bash
# [1] 预处理（CPU，SLURM low 分区，数小时）
sbatch scripts/run_build_volumes.sh

# [2] 训练 grounding（GPU，SLURM gpu 分区）
sbatch scripts/run_train_grounding.sh

# [3] test 集最终评估（挂在训练后）
sbatch --dependency=afterok:<训练jobid> scripts/run_eval_grounding.sh

# [4] 单例推理 + 可视化
python scripts/infer_grounding.py --case <case_id> --ckpt runs/grounding/best_grounding.pt
```

---

## 9. 范围约束

- **初期不涉及硬件**：不做机器人支气管镜、CT 引导穿刺等硬件方向，聚焦纯软件/数据驱动路线。
- **首个项目使用公开数据集**：LIDC-IDRI（经 hf-mirror 镜像获取）。
