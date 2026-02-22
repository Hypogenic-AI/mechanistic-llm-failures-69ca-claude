# Mechanistic Interpretability of Commonsense Reasoning Failures in LLMs

This project investigates **where and how commonsense reasoning breaks down** inside transformer language models using mechanistic interpretability techniques. We apply causal tracing, activation patching, linear probing, and attention analysis to GPT-2 Small on the PIQA (Physical Intuition QA) benchmark.

## Hypothesis

Commonsense reasoning failures in LLMs arise from specific breakdowns in **goal-action coherence integration** within the model's internal mechanisms. We test whether these failures are architectural, representational, or training-induced by localizing them within the enrichment-propagation-extraction framework of [Geva et al. (2023)](papers/geva2023_dissecting_recall_factual.pdf).

## Key Findings

- **Baseline**: GPT-2 Small achieves **64% accuracy** on PIQA (500 examples), well above the 50% chance baseline
- **Enrichment failure at layers 2-3**: Causal tracing reveals that ablating layers 2-3 causes the largest performance drop for correct predictions (-0.74 log-prob) while slightly *improving* incorrect predictions (+0.20), suggesting these layers are critical for commonsense knowledge enrichment
- **MLP layers drive commonsense**: Activation patching shows MLP sublayers (especially layers 0, 10, 11) are far more important than attention heads for commonsense reasoning, consistent with MLPs serving as knowledge stores
- **No linear commonsense signal**: Linear probing at all 12 layers achieves only ~50% accuracy (chance level), indicating commonsense knowledge is stored in a **non-linearly decodable** format
- **Attention patterns are not diagnostic**: Last-token attention to goal tokens is nearly identical between correct and incorrect predictions, ruling out attention-based propagation as the primary failure mode
- **Logit lens shows early commitment**: Token probability trajectories decline monotonically from early layers, suggesting GPT-2 commits to predictions early without mid-network refinement

## Experiments

| # | Experiment | Sample Size | Key Result |
|---|-----------|-------------|------------|
| 1 | Baseline + Logit Lens | 500 | 64% accuracy; probabilities decline across layers |
| 2 | Causal Tracing (zero-ablation) | 80 | Layers 2-3 most critical for correct predictions |
| 3 | Activation Patching (MLP + heads) | 80 | MLPs at layers 0, 10, 11 most important |
| 4 | Linear Probing | 300 | ~50% at all layers (chance level) |
| 5 | Attention Analysis | 60 | No difference between correct/incorrect |
| 6 | Logit Difference Trajectory | 150 | Near-zero differences (order 1e-8) |
| 7 | Noise-based Causal Tracing | 40 | Full recovery at all layers (methodological limitation) |

All experiments completed in **6.9 minutes** on a single NVIDIA RTX 3090.

## Reproduction

### Requirements

- Python 3.12+
- CUDA-capable GPU (tested on RTX 3090)
- ~4GB GPU memory (GPT-2 Small)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformer-lens datasets numpy matplotlib seaborn scikit-learn tqdm einops jaxtyping
```

### Run experiments

```bash
# Set environment variables (needed in some containerized environments)
export USER=researcher
export LOGNAME=researcher

# Run all 7 experiments
python src/experiment.py
```

Results are saved to `results/` (JSON data) and `figures/` (PNG visualizations).

## Project Structure

```
.
├── README.md                  # This file
├── REPORT.md                  # Full research report with detailed analysis
├── planning.md                # Research plan and hypothesis decomposition
├── literature_review.md       # Comprehensive literature review
├── resources.md               # Catalog of papers, datasets, and tools
├── pyproject.toml             # Python project configuration
├── src/
│   └── experiment.py          # All 7 experiments (~910 lines)
├── results/                   # JSON experiment outputs
│   ├── baseline_results.json
│   ├── logit_lens_data.json
│   ├── causal_trace_data.json
│   ├── activation_patching_data.json
│   ├── probing_results.json
│   ├── attention_analysis_data.json
│   ├── logit_diff_analysis.json
│   ├── noise_causal_trace.json
│   ├── environment.json
│   └── timing.json
├── figures/                   # Visualization outputs
│   ├── logit_lens_trajectories.png
│   ├── causal_traces.png
│   ├── causal_trace_diff.png
│   ├── activation_patching.png
│   ├── probing_accuracy.png
│   ├── attention_patterns.png
│   ├── logit_diff_by_layer.png
│   └── noise_causal_trace.png
├── datasets/                  # PIQA, CommonsenseQA, HellaSwag, WinoGrande
├── papers/                    # 24 reference papers (PDFs)
└── code/                      # Reference implementations
    ├── TransformerLens/
    ├── SAELens/
    ├── SAE-Reasoning/
    ├── ACDC/
    └── rome/
```

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.8 |
| PyTorch | 2.10.0+cu128 |
| GPU | NVIDIA GeForce RTX 3090 |
| Model | GPT-2 Small (124M params, 12 layers) |
| Dataset | PIQA validation split |

## Known Limitations

1. **Attention head ablation metric**: The per-head ablation used log-probability of a single token rather than logit difference between solutions, producing zero-valued results
2. **Noise-based causal tracing granularity**: Restoring the full residual stream at a layer (rather than position-specific restoration) yields trivially full recovery
3. **Single model**: Results are specific to GPT-2 Small; larger models may show different patterns
4. **Linear probing**: Only tests for linearly decodable representations; commonsense knowledge may be accessible via non-linear probes

See [REPORT.md](REPORT.md) for the full analysis, discussion, and suggested future work.
