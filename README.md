# Medical-Vqa-VL-JEPA
# VL-JEPA: Medical Visual Question Answering for Spine Imaging

**Self-supervised i-JEPA encoder + embedding-prediction VQA model for spine radiology**

*Research project — IIT Kanpur*

---

## Overview

A Visual Question Answering system for spine radiology across both X-ray and MRI.
Rather than fine-tuning a pretrained vision backbone, the image encoder is trained
from scratch on a large spine-specific corpus using the self-supervised i-JEPA
objective — predicting latent embeddings of masked image regions, not raw pixels.

The encoder is then frozen and shared across two VQA heads:

- **i-JEPA classifier** (baseline) — standard cross-entropy over a fixed answer set
- **VL-JEPA** (proposed) — predicts the embedding of the correct answer and retrieves
  the nearest candidate by cosine similarity, trained with a contrastive InfoNCE objective

---

## Results

**3-seed macro accuracy: 77.72 ± 0.32%**
**Severe stenosis recall: 60.4% ± 1.8%** (the rare but clinically critical class)

| Question type | VL-JEPA Acc | Baseline Acc | Δ |
|---|---|---|---|
| Modality | 1.000 | 1.000 | — |
| Region | 1.000 | 1.000 | — |
| View | 0.981 | 0.975 | +0.6 |
| Disease presence | 0.869 | 0.887 | -1.8 |
| Patient info | 0.652 | 0.686 | -3.4 |
| RSNA severity | 0.631 | 0.538 | **+9.3** |
| Normal vs abnormal | 0.626 | 0.613 | +1.3 |
| Disease count | 0.511 | 0.462 | **+4.9** |
| **Macro (mean)** | **0.784** | **0.770** | **+1.4** |

VL-JEPA's advantage concentrates exactly where it matters clinically — severity
grading (+9.3 pts) and disease counting (+4.9 pts). The two models are
effectively tied on the simpler structural types.
Spine image (X-ray / MRI)          Question (text)

│                                  │

▼                                  ▼

i-JEPA encoder                      BioBERT

(pretrained from scratch,           (frozen,

then frozen)                        pretrained)

│                                  │

└──────────── Fusion ─────────────┘

(joint attention)

│

┌─────────┴──────────┐

│                    │

i-JEPA classifier         VL-JEPA predictor

(cross-entropy)           (InfoNCE + cosine

nearest-answer retrieval)
---

## Pretraining details

| Item | Detail |
|---|---|
| Objective | i-JEPA (masked latent prediction, not pixel reconstruction) |
| Initialization | Random — no downloaded weights |
| Training data | 50,498 spine images at 512px |
| X-ray sources | VinDr-SpineXR, CSXA-Cervical, BUU-LSPINE (25,970 images) |
| MRI source | RSNA-2024 (24,528 images) |
| Epochs | 100 (converged loss ≈ 0.060) |

---

## Dataset

| Split | Samples |
|---|---|
| Train | 58,753 |
| Validation | 12,331 |
| Test | 12,498 |

Eight question types: modality, region, view, disease presence, patient info,
RSNA severity, normal vs abnormal, disease count.

Splits are verified leak-free at the image level (train ∩ val = train ∩ test =
val ∩ test = ∅).

---

## Key design choices

**Why i-JEPA over MAE or supervised pretraining?**
i-JEPA predicts in latent space rather than pixel space, which biases the encoder
toward semantic structure over low-level texture — desirable for medical images
where pathology signal is structural, not textural.

**Why embedding prediction over classification?**
The VL-JEPA head generalises better to rare classes (Severe stenosis is ~3.7% of
cases) because it learns a metric space over answers rather than fixed class
weights. The +9.3pt severity gap vs the classifier reflects this directly.

**Why freeze the encoder for VQA?**
Freezing prevents the small VQA dataset from distorting representations learned
on the much larger pretraining corpus, and allows the backbone to be shared
across multiple heads without interference.

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

- Vision encoder: i-JEPA (custom implementation)
- Text encoder: BioBERT (via HuggingFace Transformers)
- Training: PyTorch, InfoNCE contrastive loss
- Evaluation: macro accuracy, BLEU-1/2/3/4, METEOR, ROUGE-L, BERT-F1

---

## Report

Full per-question-type metrics, architecture details, and methodology:
[`results_report.pdf`](./results_report.pdf)

---

*IIT Kanpur · Sriram R · [sram36505@gmail.com](mailto:sram36505@gmail.com)*
---

## Architecture
