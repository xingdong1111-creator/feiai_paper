# 方案 A 实施方案：人眼盲区结节挖掘（Blind-Spot Nodule Mining）

> 状态：**数据管线已全链路验证通过**（2026-10）。本文档定义具体研究任务、里程碑与工程结构。
> 数据源：LIDC-IDRI（hf-mirror 镜像 `MedOtter/LIDC-IDRI`，免授权，已转 NIfTI）。

---

## 1. 为什么是这个方向（已验证的立论）

LIDC-IDRI 的每个肺结节由 **4 位放射科医生独立勾画**。镜像提供 `masks_annotation_count/`：
每个体素记录"4 位医生中几位勾画了它"（取值 0–4）。`nodules.csv` 的 `n_annotations`
即"发现该结节的医生人数"。

**实测结论（19 例样本验证，见 `scripts/_verify_report.txt` 与 `data/figs/`）：**

| 验证项 | 结果 |
|---|---|
| 影像与 mask 空间对齐 | 19/19 尺寸+spacing 完全一致 |
| mask 编码医生人数 | 取值恰为 0–4 |
| 标签自洽（CSV n_annotations == mask 邻域最大值） | 20/20 匹配 |
| 盲区信号真实存在 | 全库 2651 结节中 **771 个(29.1%) 仅 1 位医生发现**；其中 **54 个恶性度≥4**（含 33.6mm/49.9mm 恶性度 5 的大肿块被 3 位医生漏掉） |

**核心命题**：存在一类"人眼系统性漏看、但影像上确有病灶、且部分高度恶性"的结节。
我们要**建模"人类感知的盲区"**——既识别哪些结节容易被漏看，又在其中挑出高危者。

坐标约定（已实测）：CSV `centroid_x/y/z` 即体素索引，对应 numpy array `[z, y, x]`。

---

## 2. 任务设计（三个里程碑，递进）

### M1｜盲区 vs 共识 二分类（证明信号可学习）★首个交付
- **输入**：以结节质心为中心的 3D CT patch（重采样到 1×1×1 mm 后裁 48³ 体素）。
- **标签**：`blind`（n_annotations==1）vs `consensus`（n_annotations==4）；剔除 2/3 做干净二分类。
- **划分**：按 **patient** 划分 train/val/test，杜绝泄漏。
- **模型**：3D CNN（ResNet-18 量级）为主，2.5D（相邻 3 切片）作对照基线。
- **指标**：AUC、balanced accuracy、逐类 recall；并按直径/隐蔽度(subtlety)分层分析。
- **成功判据**：AUC 显著 >0.5（目标 >0.65）→ 证明"人类感知难度"在影像上有可学习特征。
- **注意（混杂控制）**：盲区结节偏小偏隐蔽，分类器可能只学到 size。需在**直径匹配**的子集上复测，证明模型学到 size 之外的感知难度特征。

### M2｜低共识结节的恶性风险分层（临床价值）
- **范围**：仅取 n_annotations≤2（≥2 位医生漏看）的结节。
- **任务**：预测 malignancy≥4（疑似恶性）vs ≤3。
- **临床叙事**："危险的漏诊"——把人类漏看但实际恶性的结节优先标出，供二次阅片。
- **指标**：AUC、固定特异度下的灵敏度（临床更关心不漏恶性）。

### M3｜多模态/VLA 延伸（可选，呼应项目主线）
- 用 VLM 对盲区 vs 共识结节生成/校验影像描述，检验语言能否捕捉"易漏"特征；
- 或做"second-look agent"：对 M1/M2 标记的高危区域自动重读并给出结构化发现。
- 保持为延伸项，不阻塞 M1/M2。

> 备选（更novel但更难，作为 M1 的延伸）：**dense 感知难度预测**——逐体素回归 annotation count(0–4)，
> 把 mask 当稠密监督；预测 count==1 的区域即潜在盲区。留作后续。

---

## 3. 工程结构

```
feiai_paper/
├── README.md
├── docs/
│   ├── survey_VLA_lung_cancer_2024-2026.md   # 文献调研
│   ├── public_datasets.md                    # 数据集盘点
│   └── plan_A_implementation.md              # 本文档
├── data/
│   ├── meta/          # nodules.csv / annotations.csv / scans.csv / HF_README.md
│   ├── raw/images/    # 已下 19 例 CT (.nii.gz)
│   ├── raw/masks_annotation_count/   # 已下 19 例医生计数 mask
│   ├── sample_cases.csv              # 20 结节样本清单(blind/consensus)
│   └── figs/          # overlay 验证图
├── scripts/           # 一次性/工具脚本
│   ├── setup_env.sh / select_samples.py / download_samples.py
│   ├── verify_alignment.py / stat_nodules.py
│   ├── build_dataset.py   # (待建) 裁 patch + 标签 -> .npz / manifest
│   └── train_baseline.py  # (待建) M1 基线训练(SLURM)
└── src/               # (待建) 可复用代码: dataset.py / model.py / metrics.py
```

## 4. 环境与算力

- **环境**：`.venv`（继承 base 的 numpy/pandas/matplotlib/torch2.6+cu124；另装 nibabel/SimpleITK/sklearn/scipy）。
  重建：`bash scripts/setup_env.sh`。
- **登录节点无 GPU**（torch.cuda=False 正常）；训练在 SLURM 计算节点。
- **训练提交**：由用户自行 `sbatch`（助手只生成脚本与指令，见用户约定）。
- **全量数据**：~1000 例 CT 约 40–60 GB；当前仅下 19 例(1.1GB)做验证，M1 训练前需下全量。

## 5. 下一步（具体、可执行）

1. **build_dataset.py**：读 `data/meta/nodules.csv` + raw，按 patient 划分，裁 1mm 重采样的 48³ patch，
   输出 M1 的 blind/consensus 数据集（.npz + manifest.csv）。先在 19 例上跑通，再扩全量。
2. **src/dataset.py + model.py + train_baseline.py**：3D CNN 基线，SLURM 训练，输出 AUC/分层结果。
3. 达标后进入 M2。
