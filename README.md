# DG-ReID: Domain-Generalizable Person Re-Identification via Synthetic Occlusion

Implementation of DG-ReID, featuring:
1. **Domain-Adversarial Global Branch** (Gradient Reversal Layer for domain invariance)
2. **Visibility-Aware Part Branch** (PartVisibilityGAT supervised by synthetic occlusion masks)

Built on top of the CLIP-FGDI baseline repository.

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Place datasets under a common root directory (default: `data/`):

```
data/
├── Market-1501-v15.09.15/
│   ├── bounding_box_train/
│   ├── query/
│   └── bounding_box_test/
├── MSMT17_V2/
│   ├── mask_train_v2/
│   ├── mask_test_v2/
│   ├── list_train.txt
│   ├── list_val.txt
│   ├── list_query.txt
│   └── list_gallery.txt
├── cuhk_sysu/
│   └── cropped_images/
├── cuhk03/
│   ├── cuhk03_release/
│   │   └── cuhk-03.mat
│   ├── images_labeled/
│   ├── cuhk03_new_protocol_config_detected.mat
│   ├── cuhk03_new_protocol_config_labeled.mat
│   └── splits_new_labeled.json  (auto-generated)
└── Occluded_Duke/
    ├── bounding_box_train/
    ├── query/
    └── bounding_box_test/
```

**Sources:**
- Market-1501: [Kaggle](https://www.kaggle.com/datasets/pengcw1/market-1501/data)
- MSMT17: Provided locally (zip)
- CUHK-SYSU: [Kaggle](https://www.kaggle.com/datasets/manaschaiaonon/cuhk-sysu)
- CUHK03: [Kaggle](https://www.kaggle.com/datasets/priyanagda/cuhk03) — requires CUHK03-NP protocol files from [zhunzhong07/person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)
- Occluded-DukeMTMC: Convert from DukeMTMC-reID using [lightas/Occluded-DukeMTMC-Dataset](https://github.com/lightas/Occluded-DukeMTMC-Dataset)

## Benchmarks

### Protocol 2: Leave-One-Out Domain Generalization

Train on 3 source datasets, test zero-shot on the held-out domain.

**Config switches to change rotation (no code change needed):**

| Held-Out Target | Command |
|---|---|
| Market-1501 | `--benchmark protocol2 --held_out_domain Market` |
| MSMT17 | `--benchmark protocol2 --held_out_domain MSMT17` |
| CUHK-SYSU | `--benchmark protocol2 --held_out_domain cuhk_sysu` |
| CUHK03 | `--benchmark protocol2 --held_out_domain cuhk03` |

### Occluded-DukeMTMC: All-4-Source Zero-Shot

Train on all 4 source datasets, test zero-shot on Occluded-DukeMTMC.

```bash
--benchmark occluded_duke
```

## Running

### Debug / Smoke Test (local, tiny subset)

```bash
python smoke_test.py --data_path data_smoke_test --debug_subset_ids 6 --batch_size 12 --epochs 2 --device cpu
```

This creates synthetic dummy datasets, runs the full 3-stage pipeline for both
Protocol 2 and Occluded-Duke configs, and verifies:
- ✅ Gradient flow for all trainable modules
- ✅ Checkpoint save/load round-trip
- ✅ No NaN/Inf in any loss
- ✅ Eval output in correct mAP + CMC R1/R5/R10 format

### Full-Scale Training

Remove `--debug_subset_ids` (or set to 0) and use full epoch counts:

```bash
# Example: Protocol 2, held-out = Market-1501
python run.py --benchmark protocol2 --held_out_domain Market --data_path /path/to/data --device cuda

# Example: Occluded-DukeMTMC
python run.py --benchmark occluded_duke --data_path /path/to/data --device cuda
```

### Key config values for full scale:
| Parameter | Debug | Full Scale |
|---|---|---|
| `--debug_subset_ids` | 6 | 0 (default) |
| `--batch_size` | 12 | 64 (default P=16, K=4) |
| `--image-encoder-epoch` | 2 | 60 |
| `--device` | cpu | cuda |

## Final Results (Reference Tables)

After running the full-scale benchmark suite (via `bash run_all_benchmarks.sh` and `bash run_ablations.sh` on an NVIDIA GPU), the expected results populate these tables.

### Protocol 2: Leave-One-Out (mAP / Rank-1 / Rank-5 / Rank-10)

| Held-Out Target | mAP (%) | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) |
|---|---|---|---|---|
| **Market-1501** | *run pending* | *run pending* | *run pending* | *run pending* |
| **MSMT17** | *run pending* | *run pending* | *run pending* | *run pending* |
| **CUHK-SYSU** | *run pending* | *run pending* | *run pending* | *run pending* |
| **CUHK03** | *run pending* | *run pending* | *run pending* | *run pending* |

### Occluded-DukeMTMC Zero-Shot (mAP / Rank-1 / Rank-5 / Rank-10)

| Target | mAP (%) | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mINP |
|---|---|---|---|---|---|
| **Occluded-Duke** | *run pending* | *run pending* | *run pending* | *run pending* | *run pending* |

### DG-ReID Ablation Study (Protocol 2 - Target: Market-1501)

| Configuration | mAP (%) | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) |
|---|---|---|---|---|
| **Full Model (DG-ReID)** | *run pending* | *run pending* | *run pending* | *run pending* |
| **GRL Only (No Part Branch)** | *run pending* | *run pending* | *run pending* | *run pending* |
| **Part Branch Only (No GRL)** | *run pending* | *run pending* | *run pending* | *run pending* |
| **Neither (Baseline)** | *run pending* | *run pending* | *run pending* | *run pending* |

---
*Note: A local smoke test on synthetic data was successfully executed on macOS MPS/CPU, passing checks for gradient flow, loss finiteness, and checkpoint integrity.*

## Architecture

**DG-ReID Key Components:**
1. **Positional Embeddings**: Bicubic interpolated to a 16x8 grid.
2. **Domain-Adversarial Global Branch**: Applies a Domain Classifier and GRL to the CLS token to foster domain-invariant representations.
3. **Visibility-Aware Part Branch**: Uses a Graph Attention Network (GAT) to process 3 bodily splits (Head, Torso, Legs) of patch tokens. Visibility prediction is supervised via Synthetic Occlusion augmentation (producing exact patch grid masks) during source-domain training.
4. **Orthogonality Regularization**: Maintains $K=4$ orthogonal domain tokens as a regularizer without explicit text-prompt composition.
