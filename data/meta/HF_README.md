---
license: cc-by-3.0
task_categories:
- image-segmentation
tags:
- medical
- CT
- lung
- lung-nodule
- LIDC-IDRI
- TCIA
pretty_name: LIDC-IDRI
size_categories:
- 1K<n<10K
dataset_info:
  config_name: preview
  features:
  - name: case_id
    dtype: string
  - name: patient_id
    dtype: string
  - name: series_uid
    dtype: string
  - name: slice_index
    dtype: int32
  - name: n_slices
    dtype: int32
  - name: spacing_z
    dtype: float32
  - name: manufacturer
    dtype: string
  - name: has_seg
    dtype: bool
  - name: n_nodules
    dtype: int32
  - name: n_nodules_default
    dtype: int32
  - name: n_annotations
    dtype: int32
  - name: in_luna16
    dtype: bool
  - name: luna16_subset
    dtype: string
  - name: in_qin_lungct_seg
    dtype: bool
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: overlay
    dtype: image
  splits:
  - name: train
    num_bytes: 358732357
    num_examples: 1018
  download_size: 358685495
  dataset_size: 358732357
configs:
- config_name: preview
  data_files:
  - split: train
    path: preview/train-*
---

# LIDC-IDRI — lung nodule segmentation on chest CT (complete TCIA collection, NIfTI)

The Lung Image Database Consortium / Image Database Resource Initiative collection:
**1,010 patients / 1,018 chest CT scans / 243,958 slices** with the four-radiologist
nodule (≥ 3 mm) contours, re-hosted as NIfTI with per-annotation masks and a documented
default ground truth. Source images: TCIA collection *LIDC-IDRI*
(doi:10.7937/K9/TCIA.2015.LO9QL9SX). Source annotations: the QIICR DICOM-SEG/SR
re-encoding *DICOM-LIDC-IDRI-Nodules* (doi:10.7937/TCIA.2018.h7umfurq). Both **CC BY 3.0**.

This mirror **supersedes and absorbs** a planned separate `DICOM-LIDC-IDRI-Nodules`
dataset: same annotations, same TrackingUID clustering, over the 883-scan subset that
carries contours. Never count the two as independent benchmarks.

## What is here

| path | content |
|---|---|
| `images/<case>.nii.gz` | CT in Hounsfield units, int16, native geometry (1,018) |
| `masks/<case>.nii.gz` | **default ground truth**, uint8 {0,1}: nodules with **≥ 3 annotations** at **≥ 50 % consensus** of their annotations |
| `masks_instance/<case>.nii.gz` | uint16 nodule id (per scan) for **every** clustered nodule at 50 % consensus |
| `masks_annotation_count/<case>.nii.gz` | uint8 number of annotations covering each voxel |
| `masks_annotation/<case>/<case>_nodNN_annK.nii.gz` | one uint8 mask per DICOM-SEG object (6,859) — lossless w.r.t. the source |
| `scans.csv` | one row per scan: geometry, manufacturer, transfer syntax, duplicate-z slices dropped, nodule/annotation counts, `in_luna16`, `luna16_subset`, `in_qin_lungct_seg` |
| `nodules.csv` | one row per clustered nodule (2,651): `n_annotations`, `in_default_gt`, consensus/union/intersection voxels, bbox, centroid, median SR characteristics (subtlety … malignancy, volume, diameter) |
| `annotations.csv` | one row per SEG object with its SR characteristics and mask path |
| `train.jsonl` | loader manifest, one row per scan (single `train` split — no official split exists) |
| `aux/` | `LIDC-XML-only.zip` (the original XML reads incl. < 3 mm and non-nodule point marks), `LIDC-IDRI_MetaData.csv`, nodule-count and diagnosis sheets, verbatim from TCIA |
| `preview` config | one PNG per scan (the slice with the most default-GT foreground, or the middle slice) with mask and overlay, for the Dataset Viewer |

`<case>` = `<PatientID>_<10-char SeriesInstanceUID suffix>`; eight patients have two scans
(LIDC-IDRI-0132/0151/0315/0332/0355/0365/0442/0484) — group on `patient_id`.

Not mirrored: the collection's 290 chest radiographs (DX/CR, no annotations) and the 90
`3D Slicer Community` SEG objects served under the same collection name (QIN CT-challenge
*algorithm* outputs on 10 patients, already in `MedOtter/QIN-LungCT-Seg`).

## Ground truth

Each scan was read by four thoracic radiologists (blinded, then unblinded; no forced
consensus). Only nodules ≥ 3 mm were outlined. **Upstream ships no consensus mask** — each
SEG object is one annotation of one nodule, and reader identity "was intentionally not
exposed by the LIDC initiative" (Fedorov et al. 2020), so counts here are of
*annotations* (a nodule carries 1–8; distribution over 2,651 nodules: 1 × 771, 2 × 488,
3 × 481, 4 × 897, 5 × 8, 6 × 2, 7 × 3, 8 × 1).

* **Default (`masks/`)**: nodule included when ≥ 3 annotations exist (the LUNA16
  reference-standard rule); voxel included when ≥ 50 % of that nodule's annotations
  include it (pylidc `consensus(clevel=0.5)`). Keeps 1,392 nodules; 1–2-annotation
  nodules are background here. 135 scans have no contour at all (real all-zero masks).
* Any other tier is reconstructible from `masks_annotation/` (+ `nodules.csv`) or from
  `masks_annotation_count/` + `masks_instance/`.

## Overlap with other datasets

**LUNA16 is a strict subset**: all 888 LUNA16 scans are LIDC-IDRI series (the LUNA16
`.mhd` name is the SeriesInstanceUID) — see `in_luna16` / `luna16_subset`. LUNA16's nodule
masks are sphere approximations; these are the real contours. `QIN-LungCT-Seg` shares 10
series (`in_qin_lungct_seg`). Also LIDC-derived elsewhere: SAROS, CADS `0016_lidc`,
Pulmonary-Nodules-Segmentation, ULS23 (upstream), LUNA22-ISMI, NoduleMNIST3D. Dedup on
`series_uid`.

## Provenance and quirks

* DICOM instances pulled from NCI Imaging Data Commons (v24, public S3) — byte-identical to
  TCIA's; every series verified against the NBIA `ImageCount`.
* Five series carried duplicate-z slices (0085, 0146, 0418, 0572, 0979): the first slice
  at each z is kept and SEG frames referencing the twin are composited onto it
  (`n_duplicate_slices_dropped`).
* `z_gap_min`/`z_gap_max` expose irregular slice spacing; `transfer_syntax` records
  JPEG-compressed series (decoded with GDCM).

## License and attribution

CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/) — stated on the TCIA collection
page for the images and every annotation row, on the DICOM-LIDC-IDRI-Nodules page, and in
the `LicenseName` of every series. Mirrored under TCIA's Data Usage Policy
(https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/): attribute,
link the policy, do not attempt re-identification.

Please cite:

* Armato SG III, McLennan G, Bidaut L, et al. *The Lung Image Database Consortium (LIDC)
  and Image Database Resource Initiative (IDRI): A completed reference database of lung
  nodules on CT scans.* Medical Physics 38(2):915–931, 2011. doi:10.1118/1.3528204
* Armato SG III, et al. *Data From LIDC-IDRI.* The Cancer Imaging Archive, 2015.
  doi:10.7937/K9/TCIA.2015.LO9QL9SX
* Fedorov A, Hancock M, Clunie D, et al. *DICOM re-encoding of volumetrically annotated
  Lung Imaging Database Consortium (LIDC) nodules.* Medical Physics 47(11):5953–5965, 2020.
  doi:10.1002/mp.14445; data doi:10.7937/TCIA.2018.h7umfurq
* Clark K, et al. *The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public
  Information Repository.* J Digit Imaging 26(6):1045–1057, 2013.

Acknowledgement required by the collection: "The authors acknowledge the National Cancer
Institute and the Foundation for the National Institutes of Health, and their critical
role in the creation of the free publicly available LIDC/IDRI Database used in this study."
