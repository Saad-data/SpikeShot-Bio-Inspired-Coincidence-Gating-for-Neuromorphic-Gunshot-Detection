# SoundToSpike SNN — Comprehensive Project Report

**Author:** Syed Saad  
**Project:** Gunshot detection using spiking neural networks with bio-inspired encoding  
**Generated:** 2026

---

## 1. Executive Summary

This project implements **binary gunshot vs non-gunshot classification** from audio using **spiking neural networks (SNNs)**. Audio is converted to spike trains via **CGTC (Coincidence-Gated Temporal Contrast)** encoding, inspired by octopus cells in the cochlear nucleus. Three SNN architectures (HierarchicalSNN, MILSNN, TemporalConvSNN) are trained on 0.75 s event-centered clips and evaluated on event-centric val/test sets.

**Primary results (event-centric encoding, 757 test samples):** Best test accuracy **94.98%** (MILSNN), F1 ~0.95, ROC-AUC ~0.97. HierarchicalSNN and TemporalConvSNN achieve the lowest false positive rate (1.58%). All models use LIF neurons (snntorch) with surrogate gradient learning.

**Pipeline:** Data prep (freeze val/test, preprocess train, rebalance) → CGTC 128-bin encoding → train three SNNs → event-centric encode val/test → evaluate. A single entry point (`main.py all`) runs the full pipeline.

---

## 2. Dataset

### 2.1 Data Sources

| Source | Role | Description |
|--------|------|-------------|
| **Multiple_firearm** | Gunshot | `edge-collected-gunshot-audio/`; labels and timestamps in `gunshot-audio-labels-only.csv`. Multi-firearm: **2,148** files. |
| **UrbanSound8K** | Gunshot + non-gunshot | Gun_shot class: **374** files. Non-gun: **8 classes** — car_horn, children_playing, dog_bark, drilling, engine_idling, jackhammer, siren, street_music (air_conditioner excluded). Total non-gun files used: 7,358+. |

**Total gunshot files:** 2,148 (Multi-firearm) + 374 (US8K) = **2,522**.  
**Total files (gun + non-gun):** 5,044 after stratified split.

### 2.2 File-Level Split (Train / Validation / Test)

Stratified **70% / 15% / 15%**, gun vs non-gun balanced in each split. Exact counts:

| Split       | Gunshot | Non-gunshot | Total |
|------------|---------|-------------|-------|
| Train      | 1,765   | 1,765       | 3,530 |
| Validation | 379     | 378         | 757   |
| Test       | 378     | 379         | 757   |
| **Total**  | **2,522** | **2,522** | **5,044** |

No file appears in more than one split. Manifests: `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`.

### 2.3 Data Preprocessing Pipeline (dataset_in_use/)

#### Step 5: Freeze validation and test (evaluation sets)

- **Script:** `step5_freeze_val_test.py`
- **Input:** `val_manifest.csv`, `test_manifest.csv` (757 files each).
- **Actions:** Load from `source_path`, resample to **22.05 kHz**, **peak-normalize**, pad or trim to **4 s** (88,200 samples). One WAV per file.
- **Output:**  
  - `dataset_in_use/frozen/val/<file_id>.wav`, `frozen/test/<file_id>.wav`  
  - `frozen_val_manifest.csv`, `frozen_test_manifest.csv` (columns: file_id, label, frozen_path).
- **Counts:** Validation 757 clips (gun: 379, non-gun: 378); Test 757 clips (gun: 378, non-gun: 379). These frozen 4 s WAVs are used to generate the event-centric 0.75 s encoded val/test for primary evaluation.

#### Step 6: Preprocess training set

- **Script:** `step6_preprocess_train.py`
- **Input:** `train_manifest.csv` (3,530 files).
- **Gunshot (Multi-firearm):** Event-centric 0.75 s window: PRE_OFFSET=0.2 s, POST_OFFSET=0.55 s centered on each timestamp from CSV; one file with N shots → N clips.
- **Gunshot (UrbanSound8K):** No timestamps; **onset detection** (librosa RMS + onset_detect, strongest onset) then same 0.75 s window.
- **Non-gunshot:** One clip per file; **energy-peak centered** 0.75 s (sliding frame energy, center at peak).
- **Output:**  
  - `preprocessed_train/gunshot/*.wav`, `preprocessed_train/nongunshot/*.wav`  
  - `train_clips_manifest.csv` (clip_path, label, source_file_id, etc.).  
- **Sample rate:** 22.05 kHz; peak normalization applied.

#### Step 7: Rebalance training set

- **Script:** `rebalance_training_set.py`
- **Input:** `train_clips_manifest.csv`. After Step 6: **4,677** gunshot clips, **1,765** non-gunshot clips (total 6,442; ratio 2.65:1, imbalanced).
- **Logic:** Add extra non-gunshot clips from UrbanSound8K (same 8 classes) until gun and non-gun counts match. Uses `rebalance_one_clip.py` per file (one 0.75 s energy-peak clip per file; timeout 8 s).
- **Extra clips added:** **1,575** non-gunshot clips.
- **Output (balanced):** **4,677** gun + **4,677** non-gun = **9,354** training clips.  
  - `train_clips_manifest_balanced.csv`  
  - `class_weights.json`, `step7_rebalance_summary.txt`  
  - Optional: `preprocessed_train/nongunshot/extra_*.wav`.

### 2.4 Summary: Sample Counts (Precise)

| Set | Description | Counts |
|-----|-------------|--------|
| **Training (after rebalance)** | Balanced 0.75 s clips; rebalance added 1,575 non-gun (from 1,765 → 4,677) to match 4,677 gun | **9,354** (4,677 gun + 4,677 non-gun) |
| **Validation** | Frozen 4 s → event-centric 0.75 s encode; gun 379, non-gun 378 | **757** |
| **Test (evaluation)** | Frozen 4 s → event-centric 0.75 s encode; gun 378, non-gun 379 | **757** |

---

## 3. Encoding: CGTC 128-Bin

### 3.1 Biological Motivation

**Coincidence-Gated Temporal Contrast (CGTC)** is inspired by **octopus cells** in the cochlear nucleus, which respond to broadband transients and rate-of-change rather than absolute level.

### 3.2 Pipeline (encoders/cgtc128/cgtc_128_encoder.py)

1. **Mel spectrogram:** 128 bins, n_fft=1024, hop_length=64, 22.05 kHz → ~2.9 ms per frame.  
2. **Log power:** log(mel + ε).  
3. **Temporal contrast:** Frame-to-frame difference d_mel.  
4. **Adaptive threshold:** mean(d_mel) + 1.0×std(d_mel).  
5. **Coincidence gate:** At each time step, spike if ≥ **48%** of mel bins exceed threshold (configurable).

**Parameters:**  
- SR=22050, N_MELS=128, CGTC_HOP_LENGTH=64, COINCIDENCE_THRESHOLD=0.48, CONTRAST_MULTIPLIER=1.0.  
- **Output:** 1D spike train, **~258 timesteps** per 0.75 s clip.

### 3.3 Where Encoding Is Used

| Stage | Input | Encoder | Output |
|-------|--------|---------|--------|
| **Training** | 0.75 s WAVs from `train_clips_manifest_balanced.csv` | `run_encode.py` (same CGTC 128) | `encode_comparison/encoders/cgtc128/data/*.pt`, `data/manifest.csv` |
| **Frozen 4s val/test** | 4 s WAVs from frozen val/test | `encode_frozen_val_test.py` (full 4 s → one long spike train) | `evaluation/frozen/cgtc128_val/`, `cgtc128_test/` |
| **Event-centric val/test** | 4 s WAVs + timestamps; extract 0.75 s windows (event-centric or energy-peak) | `encode_with_timestamps.py` (same CGTC 128, 0.75 s) | `test_eval_on_split/encoded_val/`, `encoded_test/` |

Audio preprocessing before encoding: **22.05 kHz**, **peak normalization** (and resampling where needed). No extra normalization beyond that.

---

## 4. Models

### 4.1 Shared Components (models/shared.py)

- **LIFEncoder:** Two-layer feedforward with Leaky Integrate-and-Fire (snntorch.Leaky, beta=0.85), surrogate gradient fast_sigmoid(slope=25), BatchNorm1d after first layer. Converts a window of spikes to a hidden feature vector.  
- **segment_to_windows(x, window_size, hop_size):** Splits a spike train [batch, time] into overlapping windows [batch, n_windows, window_size].

### 4.2 Three Architectures

| Model | Idea | Parameters |
|-------|------|------------|
| **HierarchicalSNN** | LIFEncoder per window; **learned attention** over windows; attention-weighted pooling → classifier. Can down-weight noise. | 29,380 |
| **MILSNN** | Multi-instance learning: bag = clip, instances = windows. Positive bag: ≥1 window with gunshot; negative: all windows low. Instance classifier (sigmoid) + attention + bag classifier. Uses `compute_mil_loss` (bag CE + instance loss + smoothness). | 29,380 |
| **TemporalConvSNN** | Multi-scale 1D convolutions over time, LIF, pooling, attention, classifier. Captures attack/sustain/decay. | 47,843 |

All use **LIF neurons** (snntorch) and **surrogate gradient** learning. Default: window_size=258, n_hidden=64, beta=0.85, hop_ratio=0.5, dropout=0.3.

### 4.3 Factory

- **create_model(name, window_size, n_hidden, n_classes, beta, **kwargs):** name ∈ {`hierarchical`, `mil`, `temporal_conv`}.  
- **print_model_summary():** Prints short text summary of the three models.

---

## 5. Training

### 5.1 Script and Data

- **Script:** `encode_comparison/scripts/train_compare_top3.py`  
- **Training data:** `encode_comparison/encoders/cgtc128/data/manifest.csv` (0.75 s CGTC-encoded clips from balanced manifest; 9,354 clips).  
- **Validation:** 20% of training data (random split, seed 42).  
- **Test (during training):** Optional 4 s frozen encoded test set (`evaluation/frozen/cgtc128_test/manifest.csv`) for in-training monitoring only; **not** the primary reported test set.

### 5.2 Hyperparameters

| Parameter | Value |
|-----------|--------|
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR, T_max=epochs |
| Epochs | 50 (default) |
| Batch size | 32 |
| Loss | Focal loss (α=0.25, γ=2.0) |
| Checkpoint | Best validation **F1** |

### 5.3 Training Loop

- Models are trained **one at a time**: train → evaluate on 0.75 s val and (if available) on 4 s test → next model.  
- For MILSNN, loss can use `compute_mil_loss` (bag + instance + smoothness); training script uses Focal loss on bag logits.  
- Output: `encode_comparison/outputs/snn_top3_results/<model>_model.pt` (state_dict + config), plus comparison_report.txt, comparison_results.json, and optional comparison_plots.png and per-model *_history.csv.

---

## 6. Evaluation

### 6.1 Primary Evaluation: Event-Centric (0.75 s)

- **Encoded data:** `test_eval_on_split/encoded_val/`, `encoded_test/` (one 0.75 s encoded .pt per clip, aligned with training procedure).  
- **Encoding:** `encode_with_timestamps.py` — gunshot (Multi-firearm) from timestamps; gunshot (US8K) from onset; non-gun energy-peak; all 0.75 s, then CGTC 128.  
- **Script:** `test_eval_on_split/run_eval.py` loads the three trained models from `outputs/snn_top3_results/`, builds DataLoaders from `encoded_val/manifest.csv` and `encoded_test/manifest.csv`, and evaluates at threshold 0.5.  
- **Output:** `eval_results.txt`, `eval_results.json` (per-model val/test metrics). Prerequisite: event-centric encoding must be done (e.g. via `main.py encode` or `encode_with_timestamps.py`).

**Primary results (event-centric, threshold=0.5):**

| Model | Test Acc | Test F1 | Test ROC-AUC | Test FPR | Test FNR | Val Acc | Val F1 | Params |
|-------|----------|---------|--------------|----------|----------|---------|--------|--------|
| MILSNN | **94.98%** | **0.948** | 0.974 | 1.85% | 8.20% | 93.66% | 0.935 | 29,380 |
| HierarchicalSNN | 94.85% | 0.946 | 0.975 | **1.58%** | 8.73% | 93.66% | 0.935 | 29,380 |
| TemporalConvSNN | 94.32% | 0.941 | **0.980** | **1.58%** | 9.79% | 92.73% | 0.924 | 47,843 |

**Confusion (test, n=757):**  
- MILSNN: TP=347, TN=372, FP=7, FN=31.  
- HierarchicalSNN: TP=345, TN=373, FP=6, FN=33.  
- TemporalConvSNN: TP=341, TN=373, FP=6, FN=37.

### 6.2 In-Training “Test (4s)” and Other 4s Results

- **In-training test:** When `train_compare_top3.py` runs, it can evaluate on **frozen 4s** encoded test set (full 4 s → one long spike train per file). This is a **different** evaluation from the primary event-centric one; metrics can be much worse (~50% accuracy for HierarchicalSNN in stored comparison_results.json) due to train (0.75 s) vs test (4 s) mismatch.  
- **Other 4s evaluations (from existing reports):**  
  - **Logistic classifier** on 4s encoded val/test: ~66% test accuracy, F1 ~0.62 (frozen_val_test_report.txt).  
  - **ShallowLIF + sliding window** (max_prob aggregation) on 4s: ~82% test accuracy, F1 ~0.84, FPR ~30% (snn_4s_eval_report.txt).  
  - **RF + duration-normalized + sliding window** on 4s: ~83% test accuracy, F1 ~0.85 (frozen_eval_fixed_report.txt).

The **reported “primary” results** in the README and report are **event-centric** (0.75 s), not 4 s.

---

## 7. Ablation Study

- **Script:** `encode_comparison/ablation_study.py`  
- **Sections:** Encoding method (event-centric vs others; baselines archived), model architecture, CGTC parameters (threshold_std, gate_threshold, n_mels), window parameters.  
- **Findings (summary):** Event-centric encoding is critical (~94.72% avg); MILSNN best accuracy; HierarchicalSNN and TemporalConvSNN lowest FPR (1.58%); default CGTC (e.g. gate 0.48, n_mels=128) and 0.75 s window are optimal.  
- **Output:** `encode_comparison/ablation_results/` (JSON, plots, LaTeX tables).

---

## 8. Efficiency Analysis

- **Script:** `encode_comparison/efficiency_analysis.py`  
- **Metrics:** Parameters, MACs, FLOPs, latency (mean, std, throughput), memory (KB), theoretical energy (ANN vs SNN with spike rate assumption).  
- **Reported (from report.txt):**  
  - HierarchicalSNN: 29,380 params, 117,520 FLOPs, ~0.31 ms latency, ~5.65× SNN vs ANN energy ratio.  
  - MILSNN: same params/FLOPs, ~0.61 ms.  
  - TemporalConvSNN: 47,843 params, 191,372 FLOPs, ~0.96 ms, ~4.48× energy ratio.  
- **Output:** `encode_comparison/efficiency_results/`.

---

## 9. Key Design Decisions

- **Event-centric encoding:** Training and primary evaluation use 0.75 s windows centered on the gunshot (or onset) so the SNN sees the same temporal structure at train and test time.  
- **Single encoder:** CGTC 128-bin only in the final pipeline; keypoint and other encodings are in Archive.  
- **Balanced training:** Rebalancing ensures 4,677 gun + 4,677 non-gun clips to avoid class bias.  
- **Focal loss:** Used to handle any remaining class imbalance and focus on hard examples.  
- **Best checkpoint by val F1:** Model selection favors F1 on the validation set.  
- **4s frozen encoding:** Used only for optional in-training monitoring; primary reported numbers are event-centric (0.75 s).  
- **Windowed 4s and energy-peak-only baselines:** Archived in `encode_comparison/Archive/windowed_4s_encoding/`; not part of the final pipeline.

---

## 10. Pipeline and How to Run

### 10.1 Single Entry Point (main.py)

All commands from **project root** (`soundtospike_snn/`).

| Command | Description |
|---------|-------------|
| `python main.py all` | Full pipeline: data → prepare_frozen → create_timestamp_manifests → encode_train → encode_frozen_4s → train → encode_val_test_event → eval |
| `python main.py all --skip-data` | Same but skip data prep (use existing dataset_in_use outputs) |
| `python main.py data` | Data prep only: step5 → step6 → rebalance |
| `python main.py encode` | Prepare frozen, create timestamp manifests, encode train, encode frozen 4s, encode val/test event-centric |
| `python main.py train` | Train all three models (or `--model mil` etc. for one) |
| `python main.py train --seeds 42,43,44,45,46` | Multi-seed train; checkpoints in `outputs/snn_top3_results/seed_<s>/` |
| `python main.py eval` | Run event-centric evaluation (run_eval.py) |
| `python main.py eval --seeds 42,43,44,45,46` | Eval over seeds; report **mean ± std** in eval_results.txt / .json |
| `python main.py ablation` | Ablation study |
| `python main.py efficiency` | Efficiency analysis |
| `python main.py quick-test` | Verify project and show eval results (for reviewers); optional `--sanity` for import/forward-pass only |

### 10.2 Pipeline Order (what “all” runs)

1. **Data prep:** step5_freeze_val_test.py → step6_preprocess_train.py → rebalance_training_set.py  
2. **Prepare frozen:** Copy `dataset_in_use/frozen/val` and `frozen/test` to `encode_comparison/test_eval_on_split/frozen_val` and `frozen_test`  
3. **Create timestamp manifests:** create_manifests_with_timestamps.py → manifest_val_with_timestamps.csv, manifest_test_with_timestamps.csv  
4. **Encode train:** encoders/cgtc128/run_encode.py → encoders/cgtc128/data/*.pt, manifest.csv  
5. **Encode frozen 4s:** evaluation/frozen/encode_frozen_val_test.py → cgtc128_val/, cgtc128_test/  
6. **Train:** scripts/train_compare_top3.py → outputs/snn_top3_results/*_model.pt (or seed_42/, seed_43/, … with `--seeds`)  
7. **Encode val/test event-centric:** test_eval_on_split/encode_with_timestamps.py → encoded_val/, encoded_test/  
8. **Eval:** test_eval_on_split/run_eval.py → eval_results.txt, eval_results.json. With `--seeds`, aggregates over seed dirs and reports **mean ± std** for accuracy, F1, FPR, FNR, ROC-AUC.

### 10.3 Multi-seed reporting (mean ± std)

To report results over multiple random seeds (e.g. for papers):

1. **Train:** `python main.py train --seeds 42,43,44,45,46`  
   - Writes one set of checkpoints per seed under `outputs/snn_top3_results/seed_<s>/`.
2. **Eval:** `python main.py eval --seeds 42,43,44,45,46`  
   - Loads models from each `seed_<s>/`, runs evaluation, then aggregates.  
   - **eval_results.txt** and **eval_results.json** include an “Aggregated over N seeds: mean ± std” section with Accuracy, F1, FPR, FNR, ROC-AUC as mean ± std.  
   - **eval_results.json** has `by_seed` (per-seed metrics) and `aggregated` (e.g. `accuracy_mean`, `accuracy_std` per model/split).

### 10.4 Verification and Setup

- **verify_setup.py:** Checks Python 3.9+, required packages (torch, torchaudio, snntorch, numpy, pandas, scikit-learn, librosa, soundfile), project structure, and one MILSNN forward pass.  
- **quick_test.py:** Checks layout and, if encoded data and checkpoints exist, runs event-centric evaluation and prints the results table.

---

## 11. Project Structure (Key Paths)

```
soundtospike_snn/
├── main.py                    # Single entry point
├── quick_test.py              # Quick verification + eval summary
├── verify_setup.py            # Dependency and structure check
├── requirements.txt           # Python dependencies
├── README.md                  # User-facing overview
├── report.txt                 # Implementation report
├── COMPREHENSIVE_PROJECT_REPORT.md  # This document
│
├── dataset_in_use/            # Data pipeline
│   ├── step5_freeze_val_test.py
│   ├── step6_preprocess_train.py
│   ├── rebalance_training_set.py
│   ├── rebalance_one_clip.py
│   ├── frozen/val/, frozen/test/
│   ├── preprocessed_train/gunshot/, nongunshot/
│   ├── train_clips_manifest_balanced.csv
│   ├── frozen_val_manifest.csv, frozen_test_manifest.csv
│   └── (train/val/test_manifest.csv, class_weights.json)
│
├── encode_comparison/
│   ├── paths.py               # Path constants
│   ├── config/config_encoding.py
│   ├── encoders/cgtc128/
│   │   ├── cgtc_128_encoder.py
│   │   ├── run_encode.py
│   │   └── data/              # Encoded train .pt + manifest.csv
│   ├── models/
│   │   ├── factory.py, shared.py
│   │   ├── HierarchicalSNN/, MILSNN/, TemporalConvSNN/
│   ├── scripts/train_compare_top3.py
│   ├── evaluation/frozen/
│   │   ├── encode_frozen_val_test.py
│   │   └── cgtc128_val/, cgtc128_test/
│   ├── test_eval_on_split/
│   │   ├── create_manifests_with_timestamps.py
│   │   ├── encode_with_timestamps.py
│   │   ├── run_eval.py
│   │   ├── encoded_val/, encoded_test/
│   │   ├── eval_results.json, eval_results.txt
│   │   └── frozen_val/, frozen_test/  # Copies of 4s WAVs
│   ├── outputs/snn_top3_results/
│   ├── ablation_study.py      # → ablation_results/
│   ├── efficiency_analysis.py # → efficiency_results/
│   └── Archive/               # keypoint, windowed_4s_encoding (not final)
│
└── doc/                       # CODEBASE_GUIDE, RUN_STEPS_5_AND_6, summaries, etc.
```

---

## 12. Results Summary Tables

### 12.1 Primary: Event-Centric Val/Test (757 samples each)

| Model | Test Acc | Test F1 | Test ROC-AUC | Test FPR | Test FNR | Val Acc | Val F1 |
|-------|----------|---------|--------------|----------|----------|---------|--------|
| MILSNN | 94.98% | 0.948 | 0.974 | 1.85% | 8.20% | 93.66% | 0.935 |
| HierarchicalSNN | 94.85% | 0.946 | 0.975 | 1.58% | 8.73% | 93.66% | 0.935 |
| TemporalConvSNN | 94.32% | 0.941 | 0.980 | 1.58% | 9.79% | 92.73% | 0.924 |

### 12.2 Efficiency (Representative)

| Model | Params | FLOPs | Latency (ms) | Memory (KB) | SNN vs ANN energy |
|-------|--------|-------|--------------|-------------|--------------------|
| HierarchicalSNN | 29,380 | 117,520 | ~0.31 | ~179 | ~5.65× |
| MILSNN | 29,380 | 117,520 | ~0.61 | ~179 | ~5.65× |
| TemporalConvSNN | 47,843 | 191,372 | ~0.96 | ~10,507 | ~4.48× |

### 12.3 4s Encoded Evaluations (For Reference)

- **Top-3 SNN (e.g. Hierarchical) on full 4s encoded test (no sliding window):** ~50% accuracy (train/test mismatch).  
- **Logistic on 4s:** ~66% test acc.  
- **ShallowLIF or RF + sliding window on 4s:** ~82–83% test acc, FPR ~28–30%.

---

## 13. Citation and License

**Citation:**  
*Bio-Inspired Spike Encoding for Energy-Efficient Gunshot Detection: An Octopus Cell Approach to Spiking Neural Networks*

**Author:** Syed Saad  

**License:** MIT  

---

*End of comprehensive report.*
