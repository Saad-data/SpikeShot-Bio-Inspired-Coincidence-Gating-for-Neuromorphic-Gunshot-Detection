# SoundToSpike SNN

Gunshot detection using spiking neural networks with bio-inspired encoding.

**Author:** Syed Saad

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd soundtospike_snn
pip install -r requirements.txt

# Quick test (verify installation + show results)
python quick_test.py

# Run full pipeline
python main.py all
```

## Results

| Model | Test Accuracy | F1 Score | FPR | FNR | Parameters |
|-------|--------------|----------|-----|-----|------------|
| MILSNN | **94.98%** | 0.948 | 1.85% | 8.20% | 29,380 |
| HierarchicalSNN | 94.85% | 0.946 | **1.58%** | 8.73% | 29,380 |
| TemporalConvSNN | 94.32% | 0.941 | **1.58%** | 9.79% | 47,843 |

Evaluated on 757 test samples using event-centric encoding.

## Commands

```bash
python main.py all          # Full pipeline: data → encode → train → eval
python main.py all --skip-data  # Skip data prep (use existing dataset)
python main.py data         # Data prep only
python main.py encode       # Encode audio to spikes
python main.py train        # Train models
python main.py eval         # Evaluate on test set
python main.py quick-test   # Verify setup and show results
```

Train a single model:
```bash
python main.py train --model mil
python main.py train --model hierarchical
python main.py train --model temporal_conv
```

## Project Structure

```
soundtospike_snn/
├── main.py                 # Single entry point
├── quick_test.py           # Quick verification for reviewers
├── requirements.txt        # Dependencies
├── report.txt              # Full results report
│
├── dataset_in_use/         # Data pipeline
│   ├── step5_freeze_val_test.py
│   ├── step6_preprocess_train.py
│   ├── rebalance_training_set.py
│   └── *.csv               # Manifests
│
├── encode_comparison/      # Main pipeline
│   ├── paths.py            # Path configuration
│   ├── encoders/
│   │   └── cgtc128/        # CGTC 128-bin encoder
│   │       ├── cgtc_128_encoder.py
│   │       └── run_encode.py
│   ├── models/
│   │   ├── factory.py      # create_model()
│   │   ├── shared.py       # LIFEncoder, utilities
│   │   ├── HierarchicalSNN/
│   │   ├── MILSNN/
│   │   └── TemporalConvSNN/
│   ├── scripts/
│   │   └── train_compare_top3.py  # Main trainer
│   ├── test_eval_on_split/ # Event-centric evaluation
│   └── outputs/            # Results
│
└── doc/                    # Documentation
```

## Dataset

- **Gunshot**: Multi-firearm (2,148 files) + UrbanSound8K gun_shot (374 files)
- **Non-gunshot**: 8 UrbanSound8K classes (7,358 files)
- **Split**: 70% train / 15% val / 15% test (file-level, stratified)
- **Training clips**: 9,354 (balanced: 4,677 gun + 4,677 non-gun)

## Encoding: CGTC 128-bin

Coincidence-Gated Temporal Contrast encoding inspired by octopus cells in the cochlear nucleus.

Pipeline:
1. Mel spectrogram (128 bins, 22.05 kHz)
2. Log power compression
3. Temporal contrast (frame-to-frame difference)
4. Adaptive threshold (mean + std)
5. Coincidence gate (spike if ≥48% of bands exceed threshold)

Output: 1D spike train (~258 timesteps per 0.75s clip)

## Models

All models use LIF neurons via snntorch with surrogate gradient learning.

- **MILSNN**: Multi-instance learning, optimizes "all windows low for negatives"
- **HierarchicalSNN**: Attention-based aggregation over temporal windows
- **TemporalConvSNN**: Multi-scale 1D convolutions for temporal patterns

## Training

- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: Cosine annealing (50 epochs)
- Loss: Focal loss (α=0.25, γ=2.0)
- Batch size: 32
- Checkpoint: Best validation F1

## Requirements

- Python 3.9+
- PyTorch 2.0+
- snntorch
- librosa, soundfile
- scikit-learn, pandas, numpy
- matplotlib, tqdm (optional)

See `requirements.txt` for exact versions.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Single entry point for all commands |
| `quick_test.py` | Verify installation, show results |
| `report.txt` | Full implementation report |
| `encode_comparison/scripts/train_compare_top3.py` | Main training script |
| `encode_comparison/encoders/cgtc128/cgtc_128_encoder.py` | CGTC encoder |
| `encode_comparison/test_eval_on_split/run_eval.py` | Event-centric evaluation |

## Output

After running `python main.py all`:

```
encode_comparison/outputs/snn_top3_results/
├── hierarchical_model.pt
├── mil_model.pt
├── temporal_conv_model.pt
├── comparison_report.txt
├── comparison_results.json
└── comparison_plots.png

encode_comparison/test_eval_on_split/
├── eval_results.json
└── eval_results.txt
```

## Citation

If you use this code, please cite:

```
Bio-Inspired Spike Encoding for Energy-Efficient Gunshot Detection:
An Octopus Cell Approach to Spiking Neural Networks
```

## License

MIT
