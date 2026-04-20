<h1>📌 Overview:</h1>
<p>
This project implements self-pruning neural networks that learn which weights to keep or discard during training. Each weight has a learnable gate σ(s·T) that determines its importance, with temperature annealing and L1 sparsity loss driving unimportant weights to zero. Two complementary architectures are provided: a simple MLP for accuracy retention and a residual MLP with MixUp for extreme sparsity (99.9%).
</p>

<p>
<strong>Core Mechanism:</strong>
<pre>
gates = sigmoid(gate_scores * temperature)  # T annealed: 1 → 15-20
pruned_weights = weight * gates              # element-wise pruning
output = F.linear(x, pruned_weights, bias)   # standard linear
Total Loss = CrossEntropy + λ × SparsityLoss (normalized L1 of gates)
</pre>
</p>

<h1>🎯 Objectives:</h1>
<ul>
  <li>Learn which weights are important vs. redundant during training</li>
  <li>Achieve high sparsity (86-99%) while maintaining reasonable accuracy (52-59%)</li>
  <li>Compare two architectures: Simple MLP vs Residual MLP with MixUp</li>
  <li>Analyze trade-off between sparsity (λ regularization) and test accuracy</li>
  <li>Visualize gate distributions, training dynamics, and per-layer sparsity patterns</li>
</ul>

<h1>⚡ Key Features:</h1>
<ul>
  <li><strong>Temperature Annealing (1→15/20):</strong> Soft gates early (gradients flow) → binary decisions late (true pruning)</li>
  <li><strong>Separate Optimizers:</strong> Gate scores learn 5-10× faster than weights for quick pruning decisions</li>
  <li><strong>Exponential Temperature Schedule:</strong> More time at mid-temperatures for useful differentiation</li>
  <li><strong>Two Architectures:</strong> Simple MLP (7.68M params) vs Residual MLP (3.55M params)</li>
  <li><strong>MixUp + Label Smoothing (Model 2):</strong> Improves generalization and robustness</li>
  <li><strong>Automatic Checkpointing:</strong> Saves best and final models for each λ value</li>
</ul>

<h1>🛠️ Technologies Used:</h1>
<ul>
  <li><strong>PyTorch:</strong> Deep learning framework for neural network implementation</li>
  <li><strong>TorchVision:</strong> CIFAR-10 dataset loading and transforms</li>
  <li><strong>NumPy:</strong> Numerical operations and gate statistics</li>
  <li><strong>Matplotlib:</strong> Visualization of gates, training curves, and trade-offs</li>
  <li><strong>CUDA/T4 GPU:</strong> Accelerated training (~15-25 min for all experiments)</li>
</ul>

<h1>🧩 Architecture Comparison:</h1>

<h3>Model 1: Simple MLP</h3>
<pre>
Input (3072) → FC1 (1024) → BN → Dropout → ReLU
            → FC2 (512)  → BN → Dropout → ReLU
            → FC3 (256)  → BN → Dropout → ReLU
            → FC4 (128)  → BN → ReLU
            → FC5 (10)
Total Params: 7,676,042 | Gate Params: 3,835,136
</pre>

<h3>Model 2: Residual MLP with MixUp</h3>
<pre>
Input (3072) → Stem (Linear 3072→512 + BN + GELU)
            → ResBlock (512→512) → ResBlock (512→256)
            → ResBlock (256→256) → ResBlock (256→128)
            → Head (PrunableLinear 128→10)
Total Params: 3,549,450 | Gate Params: 902,400 | Prunable Layers: 9
</pre>

<h1>📊 Results:</h1>

<h3>Model 1 Results (30 epochs, λ sweep)</h3>
<table border="1" cellpadding="8">
  <tr bgcolor="#f0f0f0">
    <th>λ</th>
    <th>Test Accuracy</th>
    <th>Sparsity @T=20</th>
    <th>Gate Mean</th>
    <th>Neg Scores</th>
  </tr>
  <tr>
    <td><strong>0.5</strong></td>
    <td><strong>59.19%</strong></td>
    <td>86.75%</td>
    <td>0.1063</td>
    <td>89.40%</td>
  </tr>
  <tr>
    <td>1.5</td>
    <td>59.17%</td>
    <td>93.38%</td>
    <td>0.0522</td>
    <td>94.79%</td>
  </tr>
  <tr>
    <td>4.0</td>
    <td>58.69%</td>
    <td><strong>96.99%</strong></td>
    <td>0.0224</td>
    <td>97.77%</td>
  </tr>
</table>

<h3>Model 2 Results (25 epochs, λ sweep)</h3>
<table border="1" cellpadding="8">
  <tr bgcolor="#f0f0f0">
    <th>λ</th>
    <th>Test Accuracy</th>
    <th>Sparsity @T=15</th>
    <th>Gate Mean</th>
    <th>Neg Scores</th>
  </tr>
  <tr>
    <td>0.3</td>
    <td>51.04%</td>
    <td>99.65%</td>
    <td>0.0032</td>
    <td>~100%</td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>50.77%</td>
    <td>99.75%</td>
    <td>0.0023</td>
    <td>~100%</td>
  </tr>
  <tr>
    <td><strong>3.0</strong></td>
    <td><strong>51.83%</strong></td>
    <td><strong>99.85%</strong></td>
    <td>0.0013</td>
    <td>~100%</td>
  </tr>
</table>

<h3>Per-Layer Sparsity (Model 2, λ=3.0)</h3>
<pre>
L1: 99.9%  ████████████████████████░
L2: 99.9%  ████████████████████████░
L3: 99.9%  ████████████████████████░
L4: 99.8%  ████████████████████████░
L5: 100.0% ████████████████████████░
L6: 100.0% ████████████████████████░
L7: 99.4%  ████████████████████████░
L8: 98.6%  ████████████████████████░
L9: 80.6%  ████████████████████░░░░░  (output layer preserves more)
</pre>

<h1>📈 Key Findings:</h1>
<ul>
  <li><strong>Extreme sparsity achievable:</strong> Residual architecture reaches 99.85% sparsity (only 0.15% of weights active)</li>
  <li><strong>Accuracy-sparsity trade-off:</strong> Model 1: 59% accuracy @ 87% sparsity | Model 2: 52% accuracy @ 99.9% sparsity</li>
  <li><strong>Output layer preserves more weights:</strong> ~80% sparsity vs >99% in earlier layers (critical for classification)</li>
  <li><strong>Higher gate LR (5-10×) accelerates pruning:</strong> Gates commit to binary decisions before temperature rises</li>
  <li><strong>Residual connections help:</strong> Better gradient flow enables higher sparsity with fewer parameters</li>
  <li><strong>Temperature annealing is critical:</strong> Soft gates early (T=1) allow differentiation, binary later (T=15-20) enforce pruning</li>
</ul>

<h1>demo screenshots:</h1>

<h3>Gate Distribution (Model 1, λ=0.5)</h3>
<img width="800" alt="gate_distribution" src="https://github.com/user-attachments/PLACEHOLDER_IMAGE_1" />

<h3>Training Curves (Model 1, all λ values)</h3>
<img width="800" alt="training_curves" src="https://github.com/user-attachments/PLACEHOLDER_IMAGE_2" />

<h3>Trade-off Analysis (Model 1)</h3>
<img width="600" alt="tradeoff" src="https://github.com/user-attachments/PLACEHOLDER_IMAGE_3" />

<h3>Per-Layer Sparsity (Model 2, λ=3.0)</h3>
<img width="700" alt="layer_sparsity" src="https://github.com/user-attachments/PLACEHOLDER_IMAGE_4" />

<h1>🚀 Quick Start:</h1>

<pre>
# Install dependencies
pip install torch torchvision numpy matplotlib

# Run Model 1 (Simple MLP)
python model1_self_pruning.py

# Run Model 2 (Residual MLP with MixUp)
python model2_self_pruning.py
</pre>

<h1>📁 Output Files:</h1>
<pre>
outputs_v3/                    # Model 1 outputs
├── gate_distribution.png      
├── tradeoff.png              
└── training_curves.png       

outputs_model2/               # Model 2 outputs
├── gate_distribution.png     
├── tradeoff.png              
├── training_curves.png       
├── layer_sparsity.png        
└── ckpts/                    # Model checkpoints
    ├── lam0.3_best.pt        
    ├── lam0.3_final.pt       
    ├── lam1.0_best.pt        
    ├── lam1.0_final.pt       
    ├── lam3.0_best.pt        
    └── lam3.0_final.pt       
</pre>

<h1>💻 Hardware Requirements:</h1>
<ul>
  <li><strong>GPU:</strong> T4 or better (CUDA recommended)</li>
  <li><strong>Training Time:</strong> ~15-25 minutes for all λ values</li>
  <li><strong>GPU Memory:</strong> 2-4 GB</li>
  <li><strong>Storage:</strong> ~500 MB for data + checkpoints</li>
</ul>

<h1>📚 References:</h1>
<ul>
  <li>Han et al. "Learning both Weights and Connections for Efficient Neural Networks" (NIPS 2015)</li>
  <li>Zhu & Gupta. "To Prune, or Not to Prune: Exploring the Efficacy of Pruning" (ICLR 2018)</li>
  <li>Jang et al. "Gumbel-Softmax: Differentiable Sampling" (ICLR 2017)</li>
  <li>Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)</li>
</ul>

<h1>📄 License:</h1>
<p>MIT License</p>
