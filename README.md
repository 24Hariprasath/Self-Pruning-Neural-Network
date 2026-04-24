# Self-Pruning Neural Network (Tredence Case Study)

---

## 1. Problem Statement

Modern neural networks are often over-parameterized, leading to:

- High memory usage  
- Increased inference latency  
- Higher deployment cost  

A common optimization technique is pruning, where unimportant weights are removed.

### Limitation of Traditional Pruning

- Happens after training  
- Requires retraining or fine-tuning  
- Does not influence how representations are learned  

### Objective

Design a neural network that:

"Learns to prune itself during training"

---

## 2. Core Idea

Instead of manually removing weights, we allow the model to decide which weights are unnecessary.

Each weight is assigned a learnable gate. The model learns both:

- Weights (for prediction)  
- Gates (for pruning)  

---

## 3. Approach

### 3.1 Gated Weight Mechanism

For each weight w, we introduce a parameter s.

g = sigmoid(s)

w' = w * g

Where:
- g is between 0 and 1  
- g close to 0 → weight is pruned  
- g close to 1 → weight is active  

---

### 3.2 Prunable Linear Layer

Each layer maintains:

- weight  
- bias  
- gate_scores  

Forward pass:

gates = sigmoid(gate_scores)  
pruned_weight = weight * gates  
output = Linear(x, pruned_weight, bias)

---

### 3.3 Loss Function

Total loss:

L = CrossEntropy + lambda * SparsityLoss

Where:

SparsityLoss = sum of all gate values

Full form:

L = CrossEntropy(y, y_pred) + lambda * sum(sigmoid(gate_scores))

---

## 4. Mathematical Foundation

Gradient of sigmoid:

d(sigmoid(s)) / ds = sigmoid(s) * (1 - sigmoid(s))

So:

dL/ds = lambda * sigmoid(s) * (1 - sigmoid(s))

---

### Why L1 Regularization?

- L1 → strong push to zero  
- L2 → weak near zero  
- L0 → non-differentiable  

L1 encourages actual sparsity.

---

## 5. Model Architecture

Input (3x32x32)  
→ Flatten (3072)  
→ PrunableLinear (3072 → 512)  
→ BatchNorm → ReLU  
→ PrunableLinear (512 → 256)  
→ BatchNorm → ReLU  
→ PrunableLinear (256 → 10)  
→ Output  

### Why BatchNorm?

Gating changes weight scale dynamically.  
BatchNorm stabilizes training.

---

## 6. Training Configuration

- Epochs: 30  
- Batch Size: 128  
- Optimizer: Adam  
- Learning Rate: 1e-3  
- Weight Decay: 1e-4 (only on weights)  
- Gate Weight Decay: 0  
- Gradient Clipping: 1.0  
- Scheduler: CosineAnnealingLR  
- Early Stopping: Enabled (patience = 5)  
- Lambda Warmup: First 5 epochs  

---

## 7. Why 30 Epochs?

- CIFAR-10 fully connected networks converge in ~20–30 epochs  
- More epochs → minimal gain + risk of overfitting  
- Early stopping prevents unnecessary training  

---

## 8. Lambda Sweep (Experiments)

We trained the model with:

lambda values = {0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3}

Each run:
- Independent training  
- Same configuration  
- Evaluated on test set  

---

## 9. Final Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|-------------|
| 0.0    | 59.64 | 0.00 |
| 1e-5   | 60.49 | 0.00 |
| 5e-5   | 58.90 | 0.00 |
| 1e-4   | 59.71 | 0.09 |
| 5e-4   | 58.70 | 3.75 |
| 1e-3   | 58.22 | 7.59 |

---

## 10. Analysis

### Observations

- Accuracy remains high (~58–60%)  
- Sparsity increases with lambda  
- Maximum sparsity ≈ 7.6%  

---

### Interpretation

Loss function:

L = CrossEntropy + lambda * sum(gates)

- CrossEntropy dominates optimization  
- Lambda is too small to enforce strong pruning  

---

### Key Insight

The model prioritizes accuracy over sparsity due to weak regularization.

---

### Important Finding

Even though sparsity is low, the trend is correct:

- Increasing lambda → increases sparsity  
- Increasing lambda → slightly reduces accuracy  

This confirms correct implementation.

---

## 11. Validation

- Gradient flows correctly to:
  - weights  
  - gate_scores  

- Numerical correctness:
  - gate = 0 → output = bias  
  - gate = 1 → behaves like normal linear layer  

- Training is stable  

---

## 12. Limitations

- Low sparsity across all lambda values  
- No strong bimodal gate distribution observed  

---

## 13. Future Improvements

- Increase lambda range (e.g., 1e-2)  
- Use Hard Concrete gates  
- Explore structured pruning  
- Train longer with stronger lambda  

---

## 14. Key Takeaways

- Built a self-pruning neural network from scratch  
- Implemented learnable gating mechanism  
- Integrated sparsity regularization  
- Demonstrated accuracy vs sparsity trade-off  
- Identified importance of lambda scaling  

---

## 15. Conclusion

This project demonstrates that:

Neural networks can learn to prune themselves during training.

However:

- Sparsity depends heavily on lambda  
- Proper tuning is required to balance:
  - Accuracy  
  - Model compression  

This approach lays the foundation for efficient model deployment.
