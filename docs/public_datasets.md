# 肺癌公开数据集盘点（影像 × 基因组 × 文本 × 疗效）

> 调研日期：2026-08-21，基于网络检索核实。规模数字以官方页面为准，下载前建议再次核对。

## 1. 影像 + 基因组 配对数据

| 数据集 | 来源 | 规模 | 内容 | 获取方式 |
|--------|------|------|------|----------|
| NSCLC-Radiogenomics | TCIA | ~144 例 | CT + 肿瘤分割掩膜 + EGFR/KRAS/TP53 等突变状态 + 临床 | 免费，NBIA Data Retriever 下载 |
| TCGA-LUAD / TCGA-LUSC | GDC（组学）+ TCIA（影像） | 组学各 ~500 例；TCIA 有 CT 的 LUAD 仅 ~69 例（475 个序列） | RNA-seq/WES/突变/甲基化/病理 WSI + 部分 CT | 免费，需按 patient ID 跨库匹配 |
| Lung-PET-CT-Dx | TCIA | 355 例 | PET/CT + 临床；约 116 例有 EGFR/KRAS 突变标注 | 免费 |
| CPTAC-3 LUAD/LUSC | GDC | 各 ~110 例 | 蛋白组 + 基因组，部分影像 | 免费 |

**结论**：配对影像+基因的公开样本量偏小（~100–150 量级），直接联合深度建模易过拟合；常用策略为影像侧特征提取/预训练 + 组学侧降维后融合。

## 2. 影像 + 临床/预后（无基因组）

| 数据集 | 来源 | 规模 | 内容 |
|--------|------|------|------|
| Lung1 / NSCLC-Radiomics | TCIA | 422 例 | CT + 肿瘤分割 + 生存/分期，影像组学基准；另有 Interobserver 版本 |
| LIDC-IDRI | TCIA | 1018 例 | CT + 4 位放射科医生结节标注（位置、形态、恶性度 1-5 分） |
| NLST | NCI | ~26,720 人 | 低剂量 CT 筛查队列 + 癌症结局，需申请受控访问 |
| Data Science Bowl 2017 | Kaggle | 千级 | 肺癌检测竞赛 CT 数据 |
| IQ-OTH/NCCD | Mendeley Data | 千级切片 | 肺结节 CT 分类 |

## 3. 影像 + 报告文本（VLM 微调语料）

| 数据集 | 规模 | 说明 |
|--------|------|------|
| CT-RATE（HuggingFace 开放） | ~25,000 例胸部 CT | 配放射学报告，CT-CLIP 系列工作的训练集，做胸部 CT VLM 首选 |
| MIMIC-CXR | ~224,000 张胸片 + 报告 | 是 X-ray 不是 CT；需申请 |
| RadGenome-Chest CT | — | 面向胸部 CT 的 grounding/VQA 基准，配 CT-RATE |

## 4. 基因组 + 疗效（无配对影像）

- **cBioPortal**：含 NSCLC 免疫治疗队列（SU2C/Mark Foundation 2019、MSK 等），有测序 + 治疗反应/生存，但**无配对影像**。
- **GDC / GEO**：TCGA-LUAD/LUSC 全量组学，无疗效配对。

## 5. 关键限制

1. **影像 + 基因配对样本量小**（~100–150），需特征级融合或迁移学习。
2. **影像 + 基因 + 免疫治疗疗效三者配对的数据几乎无公开**（Lancet Digit Health 2023 等均为医院私有数据）；走此方向需依赖院内合作数据。
3. 临床"动作"标签（活检决策、随访间隔）在公开数据集中**没有现成标注**，可用 Fleischner 指南/NCCN 指南从结节属性规则化生成（soft-VLA 方向的可行标签构造路径）。

## 6. 对各候选方向的数据支撑评估

| 方向 | 公开数据可得性 | 评估 |
|------|----------------|------|
| soft-VLA 临床动作预测 | LIDC-IDRI（结节属性）+ Fleischner 规则生成动作标签；CT-RATE（影像+报告） | ✅ 较好，标签需自构造 |
| 闭环诊疗 Agent | CT-RATE + LIDC-IDRI + MIMIC-CXR | ✅ 好 |
| 纵向随访 VLM | 公开纵向配对 CT 极少（NLST 有但受控） | ⚠️ 中等，需申请 NLST |
| 多模态疗效预测（影像+基因+疗效） | 配对数据基本不公开 | ❌ 差，需院内数据 |

## 获取入口汇总

- TCIA：https://www.cancerimagingarchive.net/ （NBIA Data Retriever 批量下载）
- GDC：https://portal.gdc.cancer.gov/
- cBioPortal：https://www.cbioportal.org/
- HuggingFace CT-RATE：https://huggingface.co/datasets/ibrahimethemhamamci/CT-RATE
- LIDC-IDRI：TCIA 内搜索 "LIDC-IDRI"
- NLST：https://cdas.cancer.gov/nlst/
