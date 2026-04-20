# Self-Pruning Neural Networks on CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Overview

Two implementations of self-pruning neural networks where each weight has a learnable **gate** `σ(s·T)`. Temperature annealing (1→15/20) and L1 sparsity loss drive unimportant weights to zero.

**Core Mechanism:**
```python
gates = sigmoid(gate_scores * temperature)  # T annealed
pruned_weights = weight * gates              # element-wise pruning
Implementations
Model 1 (Simple MLP)	Model 2 (Residual MLP)
Architecture	3072→1024→512→256→128→10	3072→512→512→256→256→128→10
Parameters	7.68M	3.55M
Epochs	30	25
Temperature	1→20	1→15
λ values	[0.5, 1.5, 4.0]	[0.3, 1.0, 3.0]
Special	-	MixUp + LabelSmoothing
Results
Model 1
λ	Test Acc	Sparsity	Gate Mean
0.5	59.19%	86.75%	0.1063
1.5	59.17%	93.38%	0.0522
4.0	58.69%	96.99%	0.0224
Model 2
λ	Test Acc	Sparsity	Gate Mean
0.3	51.04%	99.65%	0.0032
1.0	50.77%	99.75%	0.0023
3.0	51.83%	99.85%	0.0013
Per-layer sparsity (Model 2, λ=3.0): Layers 1-8: 99-100% | Output layer: 80.6%

Key Findings
Extreme sparsity achievable: >99% with residual architecture

Accuracy-sparsity trade-off: Model 1: 59% @ 87% | Model 2: 52% @ 99.9%

Output layer preserves more weights (~80% vs >99% in earlier layers)

Higher gate LR (5-10×) accelerates pruning decisions

Quick Start
bash
# Install dependencies
pip install torch torchvision numpy matplotlib

# Run Model 1
python model1_self_pruning.py

# Run Model 2  
python model2_self_pruning.py
Outputs
Console: Training progress, final metrics, summary table

Plots: gate_distribution.png, tradeoff.png, training_curves.png

Checkpoints: lam{value}_best.pt, lam{value}_final.pt

Technical Details
Two optimizers:

python
opt_main = Adam(weights + bias + BN, lr=1e-3)   # standard params
opt_gate = Adam(gate_scores, lr=5e-3)           # gates learn faster
Temperature schedule (exponential):

python
T(epoch) = 1.0 * (20.0/1.0) ^ ((epoch-1)/29)
Sparsity loss: Normalized L1 of gate values mean(gate)

Hardware Requirements
GPU: T4 or better (CUDA)

Time: ~15-25 min for all λ values

Memory: 2-4 GB GPU RAM

License
MIT

text

That's it! Just copy the entire code block above and paste it into your README.md file. Short, clean, and contains all the essential information about your project.
