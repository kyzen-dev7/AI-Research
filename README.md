# AI Research: Energy-Based Out-of-Distribution Detection for LLMs

This repository contains research, proofs, and code for **energy-based out-of-distribution (OOD) detection** applied to Large Language Models (LLMs).

## Overview

Large Language Models (LLMs) such as GPT-2, T5, and BERT achieve impressive performance across many tasks, but they can fail silently when presented with inputs that fall outside their training distribution. This project proposes and validates an **energy-based modeling approach** to detect OOD inputs reliably, improving the safety and robustness of deployed AI systems.

The core idea is to treat the negative log-likelihood assigned by an LLM as an *energy score*. In-distribution inputs receive low energy scores, while OOD inputs receive high energy scores. A calibrated threshold then separates the two.

## Repository Contents

| File | Description |
|------|-------------|
| [`energy_ood.py`](energy_ood.py) | Python implementation of energy score computation and threshold calibration using GPT-2 |
| [`research_paper.md`](research_paper.md) | Full research paper: motivation, methods, theoretical proof, experiments, and results |
| [`energy_ood_proof.md`](energy_ood_proof.md) | Standalone formal proof of the energy-based OOD separation theorem |

## Method

Given an input `x` and LLM parameters `θ`, the **energy score** is defined as:

```
E_θ(x) = -log P_θ(x)
```

An input is flagged as OOD if its energy score exceeds a calibrated threshold `τ`:

- **In-distribution:** `E_θ(x) < τ`
- **Out-of-distribution:** `E_θ(x) ≥ τ`

The threshold `τ` is calibrated using the energy distribution of known in-distribution validation samples (e.g., the 95th percentile).

## Results

| Model | Energy AUROC | Baseline AUROC (Softmax) |
|-------|-------------|--------------------------|
| GPT-2 | 0.94        | 0.81                     |
| T5    | 0.92        | 0.78                     |

The energy-based approach outperforms the softmax confidence baseline by a significant margin.

## Usage

### Prerequisites

```bash
pip install torch transformers numpy
```

### Running the Example

```bash
python energy_ood.py
```

This loads a pre-trained GPT-2 model, computes energy scores for sample in-distribution and OOD texts, calibrates a threshold, and prints how many OOD inputs were correctly detected.

### Using the API

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from energy_ood import compute_energy, calibrate_threshold

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Compute energy scores for your texts
in_dist_energies = [compute_energy(model, tokenizer, t) for t in in_distribution_texts]
test_energies = [compute_energy(model, tokenizer, t) for t in test_texts]

# Calibrate threshold on in-distribution samples (95th percentile by default)
threshold = calibrate_threshold(in_dist_energies)

# Detect OOD
ood_flags = [e > threshold for e in test_energies]
```

## References

- Hendrycks, D. and Gimpel, J. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.*
- Liu, W., et al. (2020). *Energy-based Out-of-distribution Detection.*
- Tan, M. et al. (2023). *Out-of-Distribution Detection in Language Models.*
