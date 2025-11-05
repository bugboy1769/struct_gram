# Complete Guide to Neural Network Training
## Understanding Your Contrastive Table-to-Graph Model

---

## Table of Contents
1. [Training Fundamentals](#1-training-fundamentals)
2. [Epochs vs Steps (Batches)](#2-epochs-vs-steps-batches)
3. [Your Specific Training Setup](#3-your-specific-training-setup)
4. [Contrastive Learning Paradigm](#4-contrastive-learning-paradigm)
5. [PyTorch Training Loop Anatomy](#5-pytorch-training-loop-anatomy)
6. [Optimization: Learning Rate & Schedulers](#6-optimization-learning-rate--schedulers)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [PyTorch Quirks & Best Practices](#8-pytorch-quirks--best-practices)
9. [Your Training Flow Visualized](#9-your-training-flow-visualized)

---

## 1. Training Fundamentals

### What is Training?

**Training** is the process of adjusting a neural network's internal parameters (weights and biases) so it learns to perform a specific task.

**Core Concept**:
```
Input → Neural Network (with parameters θ) → Output
                ↓
         Compare with Ground Truth
                ↓
         Calculate Loss (error)
                ↓
         Update parameters θ to reduce loss
                ↓
         Repeat until loss is minimized
```

### Key Components:

1. **Forward Pass**: Data flows through the network to produce predictions
2. **Loss Function**: Measures how wrong the predictions are
3. **Backward Pass (Backpropagation)**: Computes gradients (how to adjust parameters)
4. **Optimizer**: Updates parameters using gradients
5. **Iteration**: Repeat until model learns

---

## 2. Epochs vs Steps (Batches)

### The Hierarchy: Dataset → Batches → Epochs

```
DATASET (all your data)
   └─ EPOCH 1 (one complete pass through entire dataset)
         ├─ BATCH 1 (first 32 examples)    ← STEP 1
         ├─ BATCH 2 (next 32 examples)     ← STEP 2
         ├─ BATCH 3 (next 32 examples)     ← STEP 3
         ├─ ...
         └─ BATCH 11 (last batch)          ← STEP 11
   └─ EPOCH 2 (second complete pass)
         ├─ BATCH 1 (shuffled: different 32 examples)
         ├─ BATCH 2
         └─ ...
   └─ EPOCH 3
   └─ ...
   └─ EPOCH 50
```

### Definitions:

**BATCH** (also called "mini-batch"):
- A small subset of your dataset processed together
- **Your batch size**: 32 examples
- **Why batching?**
  - Can't fit entire dataset in GPU memory
  - Mini-batches provide noisy gradients → better generalization
  - Parallel processing on GPU is efficient

**STEP** (also called "iteration"):
- Processing ONE batch
- One forward pass + one backward pass + one parameter update
- **Your steps per epoch**: 11 steps
- Each step updates the model weights once

**EPOCH**:
- One complete pass through the ENTIRE dataset
- **Your epochs**: 50 epochs
- After each epoch, data is reshuffled for the next epoch

### Your Numbers Explained:

```python
Total training examples: 352 (80% of 440 question-table pairs)
Batch size: 32
Steps per epoch: 352 / 32 = 11 steps
Total epochs: 50

Total training steps: 11 steps/epoch × 50 epochs = 550 steps
```

**What you see during training**:
```
Epoch 1:
  Training: 100%|██████████| 11/11 [00:45<00:00]  ← 11 batches processed

Epoch 2:
  Training: 100%|██████████| 11/11 [00:45<00:00]  ← Another 11 batches

... (50 times total)
```

---

## 3. Your Specific Training Setup

### Dataset Breakdown:

```
Total Tables: 22 MIMIC tables
Questions per Table: 20 table-level questions
Total Question-Table Pairs: 22 × 20 = 440 pairs

Split:
├─ Train: 80% = 352 pairs
└─ Validation: 20% = 88 pairs

DataLoader Config:
├─ Batch Size: 32
├─ Train Batches: 352 / 32 = 11 batches per epoch
└─ Val Batches: 88 / 32 = 3 batches
```

### What Happens in Each Step:

**Step 1 of Epoch 1**:
```python
Batch = {
    'graphs': 32 table graphs (PyG batched),
    'questions': 32 natural language questions,
    'labels': [1, 1, 1, ...] (all positive pairs),
    'table_names': ['admissions', 'patients', ...]
}

1. Forward pass:
   - Encode 32 tables → [32, 768] graph embeddings
   - Encode 32 questions → [32, 768] question embeddings

2. Compute loss (InfoNCE):
   - Positive pairs: (graph[i], question[i]) for i=0..31
   - Negative pairs: All other combinations (32×31 = 992 negatives)
   - Loss = -log(similarity(pos) / sum(similarities(all)))

3. Backward pass:
   - Compute ∂Loss/∂θ for all parameters θ

4. Optimizer step:
   - θ_new = θ_old - learning_rate × gradient
   - Scheduler step: Adjust learning rate

5. Move to Step 2 (next batch of 32 pairs)
```

After 11 steps, you've seen all 352 training examples → **Epoch 1 complete**

Then validation runs (3 batches, 88 examples) to check performance.

---

## 4. Contrastive Learning Paradigm

### What is Contrastive Learning?

**Goal**: Learn embeddings where similar items are close, dissimilar items are far apart.

**Your Task**: Align table graphs with their matching natural language questions.

### InfoNCE Loss (Contrastive Loss)

**Concept**:
- You have positive pairs: (table, matching_question)
- Create negative pairs: (table, non_matching_question) using other examples in the batch

**Mathematical Formulation**:

```
For a batch of size N=32:

Positive pair for table_i: (table_i, question_i)
Negative pairs for table_i: (table_i, question_j) where j ≠ i

Similarity matrix (32 × 32):
sim[i,j] = dot_product(table_i_embedding, question_j_embedding) / temperature

InfoNCE Loss for table_i:
loss_i = -log( exp(sim[i,i]) / Σ_{j=0..31} exp(sim[i,j]) )

Total loss = average(loss_0, loss_1, ..., loss_31)
```

**Intuition**:
- Numerator `exp(sim[i,i])`: Similarity of correct pair (want HIGH)
- Denominator `Σ exp(sim[i,j])`: Sum of all similarities (positive + negatives)
- Loss is LOW when correct pair has much higher similarity than wrong pairs

**Temperature** (τ = 0.07):
- Controls distribution sharpness
- Lower τ → model must make very confident distinctions
- Higher τ → softer, more forgiving comparisons

### In-Batch Negatives:

**Why this is powerful**:
```
With batch_size=32:
- You get 32 positive pairs
- You get 32 × 31 = 992 negative pairs FOR FREE
- No need to explicitly sample hard negatives
- Very efficient!
```

**Trade-off**:
- Larger batch size → more negatives → better contrastive signal
- But limited by GPU memory
- 32 is a good sweet spot

---

## 5. PyTorch Training Loop Anatomy

### Standard PyTorch Training Pattern:

```python
# SETUP PHASE (once before training)
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = YourLoss()

# TRAINING LOOP
for epoch in range(num_epochs):
    model.train()  # Set to training mode (enables dropout, etc.)

    for batch in train_loader:  # Loop over batches
        # 1. GET DATA
        inputs = batch['input'].to(device)  # Move to GPU
        targets = batch['target'].to(device)

        # 2. FORWARD PASS
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # 3. BACKWARD PASS
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients ∂Loss/∂θ

        # 4. OPTIMIZER STEP
        optimizer.step()       # Update parameters: θ = θ - lr×∇θ

    # VALIDATION (after each epoch)
    model.eval()  # Set to evaluation mode (disables dropout)
    with torch.no_grad():  # Don't compute gradients (faster, saves memory)
        for batch in val_loader:
            outputs = model(inputs)
            val_loss = loss_fn(outputs, targets)
```

### Your Training Loop Specifics:

```python
def train_epoch(graph_encoder, question_encoder, loss_fn, optimizer,
                scheduler, train_loader, device, config):
    graph_encoder.train()

    for batch in train_loader:  # 11 iterations
        # 1. PREPARE DATA
        batched_graphs = batch['graphs'].to(device)
        questions = batch['questions']

        # 2. FORWARD PASS (dual encoders)
        graph_embeddings = graph_encoder(batched_graphs, batch=batched_graphs.batch)
        # Shape: [32, 768]

        question_embeddings = question_encoder(questions)
        # Shape: [32, 768]

        # 3. COMPUTE LOSS
        loss = loss_fn(graph_embeddings, question_embeddings)
        # InfoNCE with 32 positives, 992 negatives

        # 4. BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()

        # 5. GRADIENT CLIPPING (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(
            graph_encoder.parameters(),
            max_norm=1.0
        )

        # 6. OPTIMIZER STEP
        optimizer.step()

        # 7. LEARNING RATE SCHEDULER STEP
        scheduler.step()  # Adjust LR after each batch
```

---

## 6. Optimization: Learning Rate & Schedulers

### What is Learning Rate?

**Learning rate (LR)** controls how big a step you take when updating parameters.

```python
# Parameter update rule:
θ_new = θ_old - learning_rate × gradient

Example:
θ_old = 0.5
gradient = 0.2

If LR = 0.1:  θ_new = 0.5 - 0.1×0.2 = 0.48  (small step)
If LR = 1.0:  θ_new = 0.5 - 1.0×0.2 = 0.3   (big step)
If LR = 10:   θ_new = 0.5 - 10×0.2 = -1.5   (TOO BIG! Diverged!)
```

**Trade-off**:
- **High LR**: Fast convergence, but might overshoot/diverge
- **Low LR**: Stable, but slow convergence, might get stuck

### Your Learning Rate Strategy:

**Initial LR**: 5e-4 (0.0005)

**Why 5e-4?**
- Standard for contrastive learning is 1e-3 to 5e-4
- You have 896-d input features (high dimensional)
- Need stronger signal to learn quickly
- Previous 1e-4 was too conservative → loss plateaued

### Learning Rate Scheduler: Cosine Annealing

**What it does**: Gradually reduces LR over training

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps,  # 550 steps total
    eta_min=1e-6
)
```

**Schedule Visualization**:
```
LR
│
│ 5e-4 ●─────╮
│           ╲
│            ╲
│             ╲  Cosine decay
│              ╲
│               ╲
│                ╲___________
│ 1e-6                      ●
└─────────────────────────────→ Steps
  0                        550
```

**Why use a scheduler?**
1. **Early epochs**: High LR for fast convergence
2. **Middle epochs**: Moderate LR to refine
3. **Late epochs**: Low LR for fine-tuning

**Cosine shape benefits**:
- Smooth transitions (no sudden jumps)
- Accelerates near end → helps model settle into minima
- Better than step decay (sudden LR drops)

### AdamW Optimizer:

**What is AdamW?**
- Adam: Adaptive Moment Estimation (combines momentum + RMSprop)
- W: Weight decay (L2 regularization fix)

**Your config**:
```python
optimizer = torch.optim.AdamW(
    graph_encoder.parameters(),
    lr=5e-4,              # Learning rate
    weight_decay=0.01     # Regularization strength
)
```

**Weight decay**: Adds penalty for large weights → prevents overfitting

```python
# AdamW update (simplified):
gradient = ∂Loss/∂θ
θ_new = (1 - weight_decay×lr) × θ_old - lr × gradient
        ↑                                  ↑
    Shrinks weights              Gradient update
```

---

## 7. Evaluation Metrics

### Loss vs Metrics:

**Loss** (InfoNCE):
- What the model optimizes during training
- Lower is better
- Measures alignment quality

**Metrics** (Recall@K):
- What you actually care about
- Measures retrieval performance
- Evaluated on validation set

### Recall@K Explained:

**Task**: Given a question, retrieve the correct table from 22 tables

**Recall@1**:
```
For each question in validation set:
  1. Encode question → get embedding
  2. Compute similarity with all 22 table embeddings
  3. Retrieve top-1 most similar table
  4. Check if it's the CORRECT table

Recall@1 = (# correct retrievals) / (total questions)

Example:
88 validation questions
70 retrieved correct table as #1
Recall@1 = 70/88 = 0.795 (79.5%)
```

**Recall@5**:
```
Same as Recall@1, but check if correct table is in top-5

Recall@5 = (# times correct table in top-5) / (total questions)

Example:
85 out of 88 had correct table in top-5
Recall@5 = 85/88 = 0.966 (96.6%)
```

**Why both?**
- **Recall@1**: Strict metric (exact match)
- **Recall@5**: Relaxed metric (correct table nearby)
- Real systems might show top-5 results to user

### Implementation:

```python
def compute_recall_at_k(graph_embeddings, question_embeddings, k=1):
    # graph_embeddings: [22, 768] (all tables)
    # question_embeddings: [88, 768] (validation questions)

    # Compute similarity matrix [88, 22]
    similarity = torch.matmul(question_embeddings, graph_embeddings.T)

    # Get top-k indices for each question [88, k]
    _, top_k_indices = torch.topk(similarity, k=k, dim=1)

    # Correct indices: [0, 1, 2, ..., 87] (diagonal alignment)
    correct_indices = torch.arange(88).unsqueeze(1)  # [88, 1]

    # Check if correct index in top-k [88]
    matches = (top_k_indices == correct_indices).any(dim=1)

    # Average across all questions
    recall = matches.float().mean().item()
    return recall
```

### Why Recall Matters More Than Loss:

Your **goal**: Retrieve correct tables given questions

**Loss** is a proxy objective:
- InfoNCE loss going down doesn't guarantee good retrieval
- Could overfit to training questions

**Recall** directly measures retrieval:
- This is what you'll use in production
- Validation recall = real-world performance estimate

---

## 8. PyTorch Quirks & Best Practices

### 1. `.to(device)` - GPU vs CPU

**The Problem**:
```python
model = MyModel()  # Lives on CPU by default
data = torch.tensor([1, 2, 3])  # Also on CPU

output = model(data)  # ✓ Works (both on CPU)

model = model.to('cuda')  # Move model to GPU
output = model(data)  # ✗ ERROR! Data on CPU, model on GPU
```

**Solution**: Move both model AND data to same device
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
data = data.to(device)
output = model(data)  # ✓ Works
```

**Your code**:
```python
graph_encoder = graph_encoder.to(device)  # Model to GPU
batched_graphs = batch['graphs'].to(device)  # Data to GPU
```

### 2. `model.train()` vs `model.eval()`

**Modules that behave differently**:
- **Dropout**: Randomly drops neurons during training, disabled during eval
- **BatchNorm**: Uses batch statistics during training, running stats during eval

```python
# Training
model.train()
output = model(input)  # Dropout active, gradients computed

# Evaluation
model.eval()
with torch.no_grad():  # Don't track gradients
    output = model(input)  # Dropout disabled, faster
```

**Your code**:
```python
def train_epoch(...):
    graph_encoder.train()  # Enable training mode
    for batch in train_loader:
        ...

def validate(...):
    graph_encoder.eval()  # Disable dropout, etc.
    with torch.no_grad():  # Don't compute gradients
        ...
```

### 3. `optimizer.zero_grad()` - Gradient Accumulation

**PyTorch accumulates gradients** by default:

```python
# Without zero_grad():
loss1.backward()  # gradient = ∂loss1/∂θ
loss2.backward()  # gradient = ∂loss1/∂θ + ∂loss2/∂θ  (WRONG!)

# Correct:
optimizer.zero_grad()  # gradient = 0
loss.backward()        # gradient = ∂loss/∂θ
optimizer.step()
```

**Why accumulation exists?**
- Useful for gradient accumulation (simulating larger batch sizes)
- But you need to zero manually each step

### 4. Gradient Clipping

**Problem**: Exploding gradients in deep networks

```python
# Before clipping:
gradient = [100, -200, 50, ...]  # Very large!
θ_new = θ - lr × gradient  # Huge update → model diverges

# After clipping (max_norm=1.0):
gradient_norm = sqrt(100² + 200² + 50² + ...) = 230
clipped_gradient = gradient × (1.0 / 230)  # Scale down
θ_new = θ - lr × clipped_gradient  # Stable update
```

**Your code**:
```python
torch.nn.utils.clip_grad_norm_(
    graph_encoder.parameters(),
    max_norm=1.0
)
```

### 5. DataLoader Shuffling

**Why shuffle?**
- Prevents model from learning order of data
- Each epoch sees different batch compositions
- Better generalization

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True  # Shuffle before each epoch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False  # Don't shuffle validation
)
```

### 6. Inference Tensors (Your Bug!)

**Problem**: Frozen encoder with `@torch.no_grad()` creates inference tensors

```python
class QuestionEncoder:
    @torch.no_grad()
    def forward(self, questions):
        return self.encoder.encode(questions)
        # Returns inference tensor (can't be used in backward)

# Later in InfoNCE loss:
similarity = torch.matmul(table_emb, question_emb.T)
# ✗ ERROR: question_emb is inference tensor
```

**Solution**: Clone to create normal tensor
```python
if not question_embeddings.requires_grad:
    question_embeddings = question_embeddings.clone()
```

### 7. L2 Normalization for Contrastive Learning

**Why normalize embeddings?**

```python
# Without normalization:
emb1 = [100, 200, 50]   # Large magnitude
emb2 = [0.1, 0.2, 0.05]  # Small magnitude

dot_product(emb1, emb1) = 45000  # Huge!
dot_product(emb2, emb2) = 0.03   # Tiny!

# Dot product dominated by magnitude, not direction

# With L2 normalization (unit vectors):
norm_emb1 = emb1 / ||emb1|| = [0.42, 0.84, 0.21]  # ||norm_emb1|| = 1
norm_emb2 = emb2 / ||emb2|| = [0.42, 0.84, 0.21]  # ||norm_emb2|| = 1

dot_product(norm_emb1, norm_emb2) = cosine_similarity
# Now measures angle, not magnitude!
```

**Your implementation**:
```python
# In LightweightFeatureTokenizer:
full_embedding = np.concatenate([stat_features, col_name_embedding])
norm = np.linalg.norm(full_embedding)
full_embedding = full_embedding / norm  # Project onto unit sphere

# In ContrastiveGNNEncoder:
projected_embedding = self.projection_head(graph_embedding)
projected_embedding = F.normalize(projected_embedding, p=2, dim=-1)
```

---

## 9. Your Training Flow Visualized

### Complete Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                              │
└─────────────────────────────────────────────────────────────────┘
                               ↓
    22 MIMIC Tables → 20 questions each → 440 (table, question) pairs
                               ↓
                    Split: 352 train / 88 val
                               ↓
                    Batch into size=32
                               ↓
              Train: 11 batches, Val: 3 batches

┌─────────────────────────────────────────────────────────────────┐
│                      EPOCH 1 TRAINING                            │
└─────────────────────────────────────────────────────────────────┘

┌─ STEP 1 (Batch 1/11) ────────────────────────────────────────┐
│                                                                │
│  Input: 32 (table, question) pairs                            │
│                                                                │
│  ┌─────────────┐                    ┌──────────────┐          │
│  │ 32 Tables   │                    │ 32 Questions │          │
│  └──────┬──────┘                    └──────┬───────┘          │
│         │                                  │                  │
│         ▼                                  ▼                  │
│  ┌──────────────────┐            ┌─────────────────┐          │
│  │ FeatureTokenizer │            │ QuestionEncoder │          │
│  │ (896-d features) │            │   (frozen ST)   │          │
│  └────────┬─────────┘            └────────┬────────┘          │
│           │ L2 normalize                  │                  │
│           ▼                               ▼                  │
│  ┌──────────────────┐            [32, 768] embeddings        │
│  │   GraphBuilder   │                     │                  │
│  │  (PyG graphs)    │                     │                  │
│  └────────┬─────────┘                     │                  │
│           │                               │                  │
│           ▼                               │                  │
│  ┌──────────────────┐                     │                  │
│  │ ContrastiveGNN   │                     │                  │
│  │  (trainable)     │                     │                  │
│  └────────┬─────────┘                     │                  │
│           │                               │                  │
│           ▼                               ▼                  │
│     [32, 768]                        [32, 768]               │
│   graph embeddings                 question embeddings       │
│           │                               │                  │
│           └───────────┬───────────────────┘                  │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │   InfoNCE Loss  │                             │
│              │                 │                             │
│              │  32 positives   │                             │
│              │  992 negatives  │                             │
│              └────────┬────────┘                             │
│                       │                                      │
│                       ▼                                      │
│                  Loss = 3.45                                 │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │ loss.backward() │                             │
│              └────────┬────────┘                             │
│                       │                                      │
│                       ▼                                      │
│          Gradients: ∂Loss/∂θ for all θ                       │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │  Clip gradients │                             │
│              │  (max_norm=1.0) │                             │
│              └────────┬────────┘                             │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │ optimizer.step()│                             │
│              │ θ -= lr × ∇θ    │                             │
│              └────────┬────────┘                             │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │ scheduler.step()│                             │
│              │ lr = 5e-4 → ... │                             │
│              └─────────────────┘                             │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                               │
                               ▼
                      Repeat for Steps 2-11
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       VALIDATION                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
    Process 88 validation pairs (3 batches) with torch.no_grad()
                               │
                               ▼
              Compute Recall@1, Recall@5, Val Loss
                               │
                               ▼
                    Print metrics, save if best
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EPOCH 2                                    │
└─────────────────────────────────────────────────────────────────┘
    (Repeat with shuffled data, updated learning rate)

... (50 epochs total) ...
```

### Timeline of One Batch:

```
Time  │ Action                          │ Memory (GPU)
──────┼─────────────────────────────────┼──────────────
0ms   │ Load batch from DataLoader      │ +100MB (data)
      │                                 │
50ms  │ Move to GPU (.to(device))       │ +100MB
      │                                 │
100ms │ Forward: FeatureTokenizer       │ +50MB
      │                                 │
200ms │ Forward: GraphBuilder           │ +150MB
      │                                 │
400ms │ Forward: ContrastiveGNN         │ +200MB (activations)
      │                                 │
450ms │ Forward: QuestionEncoder        │ +100MB
      │                                 │
500ms │ Compute InfoNCE Loss            │ +50MB (similarity matrix)
      │                                 │
550ms │ Backward: loss.backward()       │ +500MB (gradients)
      │                                 │
800ms │ Clip gradients                  │ 0
      │                                 │
850ms │ optimizer.step()                │ 0 (in-place update)
      │                                 │
900ms │ scheduler.step()                │ 0
      │                                 │
950ms │ Free memory for next batch      │ -1050MB
──────┴─────────────────────────────────┴──────────────

Total: ~1 second per batch
11 batches × 1s = 11s per epoch
50 epochs × 11s = 550s ≈ 9 minutes total
```

---

## Summary: What's Happening During Your Training

### High-Level View:

1. **50 Epochs**: You go through your entire dataset 50 times
2. **11 Steps per Epoch**: Your 352 training examples are split into 11 batches of 32
3. **Each Step**:
   - Process 32 table-question pairs
   - Compute loss with 32 positives + 992 negatives
   - Update model weights once
   - Adjust learning rate once
4. **After Each Epoch**: Validate on 88 held-out examples
5. **Total Updates**: 550 weight updates (11 steps × 50 epochs)

### Why These Choices?

| Choice | Reason |
|--------|--------|
| **Batch size 32** | Balances GPU memory vs contrastive signal (more negatives) |
| **50 epochs** | Enough iterations for convergence without overfitting |
| **LR 5e-4 → 1e-6** | Start aggressive, end careful for fine-tuning |
| **Cosine scheduler** | Smooth LR decay, better than step decay |
| **InfoNCE loss** | State-of-art for contrastive learning |
| **Recall@K metrics** | Directly measures retrieval performance |
| **L2 normalization** | Essential for contrastive learning (cosine similarity) |
| **Gradient clipping** | Prevents exploding gradients in GNNs |

### What Success Looks Like:

```
Epoch 1:  Loss=3.50, Recall@1=0.05, Recall@5=0.15  ← Random performance
Epoch 5:  Loss=3.20, Recall@1=0.12, Recall@5=0.30  ← Learning slowly
Epoch 10: Loss=2.80, Recall@1=0.25, Recall@5=0.50  ← Making progress
Epoch 20: Loss=2.20, Recall@1=0.45, Recall@5=0.70  ← Good progress
Epoch 40: Loss=1.80, Recall@1=0.65, Recall@5=0.85  ← Strong performance
Epoch 50: Loss=1.60, Recall@1=0.70, Recall@5=0.90  ← Converged
```

**Red Flags**:
- Loss stays at 3.5 after 20 epochs → Not learning (need higher LR or debug)
- Loss drops but Recall@1 stays low → Overfitting to training questions
- Loss = NaN → Exploding gradients (reduce LR or increase clipping)

---

## Quick Reference: Key Formulas

### Loss (InfoNCE):
```
L = -log( exp(sim(t_i, q_i)/τ) / Σ_j exp(sim(t_i, q_j)/τ) )

where:
  t_i = table_i embedding
  q_i = question_i embedding (positive pair)
  q_j = all questions in batch (including negatives)
  τ = temperature (0.07)
  sim(a,b) = dot(a, b) for L2-normalized vectors
```

### Optimizer Update (AdamW):
```
m_t = β1·m_{t-1} + (1-β1)·∇θ           (momentum)
v_t = β2·v_{t-1} + (1-β2)·∇θ²          (variance)
θ_t = (1-λ·η)·θ_{t-1} - η·m_t/√(v_t)   (update with weight decay)

where:
  η = learning rate
  λ = weight decay (0.01)
  β1, β2 = Adam parameters (0.9, 0.999)
```

### Learning Rate Schedule (Cosine):
```
η_t = η_min + (η_max - η_min) × (1 + cos(π·t/T)) / 2

where:
  η_max = 5e-4 (initial LR)
  η_min = 1e-6 (final LR)
  t = current step
  T = total steps (550)
```

### Recall@K:
```
Recall@K = (1/N) × Σ_i I(correct_table_i ∈ top_K_tables_i)

where:
  N = number of validation questions
  I(·) = indicator function (1 if true, 0 if false)
```

---

**End of Document**

For questions or clarifications, refer to specific sections above.
