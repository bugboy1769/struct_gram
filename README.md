# Contrastive Table-to-Graph Learning Pipeline

## Overview

`contrastive_table2graph.py` implements a **contrastive learning framework** for semantic table retrieval. The system learns to match natural language questions with relevant database tables by:

1. Converting tables into **graph representations** (columns = nodes, relationships = edges)
2. Encoding graphs using **Graph Neural Networks (GNNs)**
3. Aligning table embeddings with question embeddings via **InfoNCE loss**
4. Enabling **semantic table retrieval** through learned similarity

---

## Architecture Diagram

```
┌─────────────────┐
│  Database Table │
│  (DataFrame)    │
└────────┬────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌────────────────────┐              ┌──────────────────────┐
│ Column Features    │              │ Relationship         │
│ • 512-d stats      │              │ Detection            │
│ • 384-d col names  │              │ • 10 edge features   │
│ = 896-d per node   │              │ • Semantic labels    │
└────────┬───────────┘              └──────────┬───────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
              ┌──────────────────┐
              │ Graph (PyG Data) │
              │ • Nodes: 896-d   │
              │ • Edges: sparse  │
              └────────┬─────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ TableGCN (2 layers)     │
         │ 896-d → 768-d → 768-d   │
         └─────────────┬───────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ Attention Pooling       │
         │ Graph-level embedding   │
         └─────────────┬───────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ Projection Head         │
         │ 768-d → 1536-d → 768-d  │
         └─────────────┬───────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Table Embedding│──────┐
              │ (768-d, L2-norm)│      │
              └────────────────┘      │
                                      │ Cosine
┌──────────────────────┐              │ Similarity
│ Natural Language     │              │
│ Question             │              │
└──────────┬───────────┘              │
           │                          │
           ▼                          │
┌──────────────────────┐              │
│ SentenceTransformer  │              │
│ (frozen, mpnet)      │              │
└──────────┬───────────┘              │
           │                          │
           ▼                          │
  ┌────────────────┐                  │
  │Question Embed  │──────────────────┘
  │(768-d, L2-norm)│
  └────────────────┘
           │
           ▼
    ┌──────────────┐
    │ InfoNCE Loss │
    │ (contrastive)│
    └──────────────┘
```

---

## Components

### 1. Data Processing

#### `DataProcessor`
Handles file loading and basic data validation.

```python
processor = DataProcessor()
df = processor.load_data('table.csv')
report = processor.validate_data(df)
```

**Features:**
- Supports CSV, Excel, JSON, Parquet
- Data validation and memory checks
- Column preprocessing

---

#### `ColumnContentExtractor`
Samples column values for statistical analysis.

```python
extractor = ColumnContentExtractor(sample_size=50)
col_stats = extractor.get_col_stats(df, 'column_name')
# Returns: {'column_content': str, 'data_type': str, ...}
```

**Sampling strategy:**
- Null representations (10%)
- Statistical extremes (min, max, median, quartiles)
- Most frequent values (30%)
- Random samples (50%)
- Padding with mode

---

### 2. Relationship Detection

#### `RelationshipGenerator`
Computes 10 edge features between all column pairs.

```python
rel_gen = RelationshipGenerator(threshold_config={
    'composite_threshold': 0.4,
    'weights': {...}
})
relationships = rel_gen.compute_all_relationship_scores(df)
```

**Edge Features (10-dimensional):**

| Feature | Description | Weight |
|---------|-------------|--------|
| `name_similarity` | Character n-gram similarity (TF-IDF) | 0.10 |
| `value_similarity` | Cosine similarity of value distributions | 0.15 |
| `jaccard_overlap` | Set intersection over union | 0.10 |
| `cardinality_similarity` | min(card1, card2) / max(card1, card2) | 0.05 |
| `dtype_similarity` | Data type compatibility | 0.05 |
| **`id_reference`** | FK pattern detection (containment + cardinality) | **0.15** |
| **`hierarchical`** | Parent-child substring patterns | **0.10** |
| **`functional_dependency`** | s1 → s2 determinism | **0.10** |
| **`measure_dimension`** | Numeric-categorical pairing | **0.10** |
| **`temporal_dependency`** | Datetime correlation | **0.10** |

**Composite Score:**
```python
composite_score = Σ (feature_i × weight_i)
# Edge created if: composite_score ≥ 0.4
```

---

#### `SemanticLabelGenerator`
Classifies relationships into 34 semantic types using decision tree logic.

```python
label_gen = SemanticLabelGenerator()
label = label_gen.generate_feature_label(edge_features)
# Returns: 'PRIMARY_FOREIGN_KEY', 'TEMPORAL_SEQUENCE_STRONG', etc.
```

**Label Categories (34 types):**
1. **Join Relationships (8):** PRIMARY_FOREIGN_KEY, FOREIGN_KEY_CANDIDATE, NATURAL_JOIN_CANDIDATE...
2. **Aggregation Relationships (7):** MEASURE_DIMENSION_STRONG, DIMENSION_HIERARCHY...
3. **Ordering Relationships (5):** TEMPORAL_SEQUENCE_STRONG, TEMPORAL_CORRELATION...
4. **Derivation Relationships (6):** DERIVED_CALCULATION, FUNCTIONAL_TRANSFORMATION...
5. **Structural Relationships (5):** COMPOSITE_KEY_COMPONENT, PARTITION_KEY...
6. **Weak/Unknown (3):** WEAK_CORRELATION, INDEPENDENT_COLUMNS...

---

### 3. Node Feature Extraction

#### `LightweightFeatureTokenizer`
Encodes column content into 512-dimensional vectors.

```python
tokenizer = LightweightFeatureTokenizer(
    embedding_strategy='hybrid'  # Uses both semantic + statistical
)
embedding = tokenizer.encode_column_content(col_stats)
# Returns: 512-d numpy array
```

**Feature Components:**
- **Semantic embeddings** (384-d): SentenceTransformer on column values
- **Statistical embeddings** (256-d): TF-IDF on value strings
- **Metadata features** (8-d): dtype indicators, null%, length, etc.
- Total: 384 + 256 + 8 → padded/truncated to **512-d**

**Note:** The current implementation uses 512-d, but can be extended to 896-d (512 stats + 384 column names) for better alignment with questions.

---

### 4. Graph Construction

#### `GraphBuilder`
Converts DataFrames into PyTorch Geometric graphs.

```python
graph_builder = GraphBuilder(
    content_extractor=extractor,
    feature_tokenizer=tokenizer,
    relationship_generator=rel_gen,
    semantic_label_generator=label_gen,
    mode='train'
)

pyg_data = graph_builder.build_graph(df)
# Or: pyg_data = graph_builder.convert_table(df)  # Alias
```

**Graph Structure:**
- **Nodes:** One per column (512-d or 896-d features)
- **Edges:** Undirected, created if composite_score ≥ 0.4
- **Sparsity:** Typically 20-50 edges for 10-15 column tables

---

### 5. Neural Network Components

#### `QuestionEncoder`
Encodes questions using frozen SentenceTransformer.

```python
question_encoder = QuestionEncoder(
    model_name='all-mpnet-base-v2',
    freeze=True  # Frozen during training
)
embeddings = question_encoder(["Which table has patient data?"])
# Returns: [batch_size, 768]
```

---

#### `ContrastiveGNNEncoder`
Encodes table graphs into 768-d embeddings.

```python
graph_encoder = ContrastiveGNNEncoder(
    node_dim=896,      # Input node feature dimension
    hidden_dim=768,    # GNN hidden dimension (CRITICAL: keep ≥ 768)
    output_dim=768,    # Match question space
    num_layers=2       # 2-layer GNN
)
table_embedding = graph_encoder(pyg_data, batch=None)
# Returns: [1, 768] L2-normalized
```

**Architecture:**
1. **TableGCN** (2 layers): Message passing to enrich node features
2. **AttentionPooling**: Weighted aggregation to graph-level embedding
3. **Projection Head**: 768-d → 1536-d → 768-d with ReLU + Dropout
4. **L2 Normalization**: Enables cosine similarity

---

#### `InfoNCELoss`
Contrastive loss with in-batch negatives.

```python
loss_fn = InfoNCELoss(temperature=0.07)
loss = loss_fn(table_embeddings, question_embeddings)
```

**Loss Formula:**
```
For each table_i:
loss_i = -log( exp(sim(table_i, question_i) / τ)
              / Σ_j exp(sim(table_i, question_j) / τ) )

where:
  sim(a, b) = cosine_similarity(a, b) = a·b (for L2-normalized vectors)
  τ = temperature (default: 0.07)
  j ∈ [1, batch_size] (in-batch negatives)
```

**In-Batch Negatives:**
- Batch size 32 → 1 positive + 31 negatives per sample
- Total: 32 positives + 992 negatives per batch

---

### 6. Question Generation

#### `QuestionGenerator` (OLD - Column-Pair Questions)
Generates questions about column relationships.

```python
# Legacy generator - generates column-pair questions
question_gen = QuestionGenerator(semantic_label_generator)
questions = question_gen.generate_dataset(tables, rel_gen, num_per_table=20)
```

**Problems:**
- Asks about column pairs, not tables
- Generic templates don't leverage column names
- **Not recommended for use**

---

#### `TableSpecificQuestionGenerator` (NEW - ✅ Recommended)
Generates unique questions per table mentioning specific columns.

```python
question_gen = TableSpecificQuestionGenerator()
questions = question_gen.generate_dataset(
    tables,
    rel_gen,
    num_per_table=20
)
```

**Question Distribution (20 per table):**
1. **Column enumeration** (4): "Which table has columns A, B, C?"
2. **Structural patterns** (5): "Which table uses X as FK?" / "Which table has temporal Y?"
3. **Relationship questions** (5): "Which table has FK X that references Y?"
4. **Hybrid questions** (3): "Which table links X and tracks time with Y?"
5. **Domain-specific** (3): "Which laboratory table contains X, Y, Z?"

**Key Advantages:**
- ✅ Each table gets unique questions
- ✅ Questions mention specific column names
- ✅ Leverages column name embeddings (384-d)
- ✅ 5/20 questions leverage GNN message passing
- ✅ No ambiguity in supervision

---

### 7. Dataset & DataLoader

#### `TableQuestionDataset`
PyTorch Dataset for table-question pairs.

```python
dataset = TableQuestionDataset(
    question_data=questions,
    data_processor=processor,
    pyg_converter=graph_builder
)

sample = dataset[0]
# Returns: {
#     'graph': PyG Data,
#     'question': str,
#     'label': int,
#     'table_name': str
# }
```

---

#### `collate_fn` & `create_dataloader`
Batches graphs and questions.

```python
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True
)

for batch in dataloader:
    graphs = batch['graphs']        # PyG Batch object
    questions = batch['questions']  # List of strings
    labels = batch['labels']        # Tensor [32]
    table_names = batch['table_names']  # List of strings
```

---

## Training Pipeline

### Basic Training Loop

```python
import torch
from contrastive_table2graph import *

# 1. Initialize components
processor = DataProcessor()
extractor = ColumnContentExtractor()
tokenizer = LightweightFeatureTokenizer(embedding_strategy='hybrid')
rel_gen = RelationshipGenerator()
graph_builder = GraphBuilder(extractor, tokenizer, rel_gen, mode='train')

# 2. Load tables
tables = []
for csv_file in Path('data/').glob('*.csv'):
    df = pd.read_csv(csv_file, nrows=500)
    df.name = csv_file.stem  # Important: assign table name
    tables.append(df)

# 3. Generate questions
question_gen = TableSpecificQuestionGenerator()
question_data = question_gen.generate_dataset(tables, rel_gen, num_per_table=20)

# 4. Create dataset
dataset = TableQuestionDataset(question_data, processor, graph_builder)
train_loader = create_dataloader(dataset, batch_size=32, shuffle=True)

# 5. Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

question_encoder = QuestionEncoder('all-mpnet-base-v2', freeze=True).to(device)
graph_encoder = ContrastiveGNNEncoder(
    node_dim=512,      # Or 896 if using column names
    hidden_dim=768,    # Keep ≥ 768 to avoid bottleneck
    output_dim=768,
    num_layers=2
).to(device)

loss_fn = InfoNCELoss(temperature=0.07)
optimizer = torch.optim.AdamW(graph_encoder.parameters(), lr=5e-4)

# 6. Training loop
for epoch in range(50):
    graph_encoder.train()

    for batch in train_loader:
        graphs = batch['graphs'].to(device)
        questions = batch['questions']

        # Forward pass
        table_embs = graph_encoder(graphs, batch=graphs.batch)
        question_embs = question_encoder(questions).to(device)

        # Loss
        loss = loss_fn(table_embs, question_embs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), 1.0)
        optimizer.step()
```

---

## Key Design Decisions

### 1. Why Contrastive Learning?
**Alternative approaches:**
- Classification: `P(table | question)` requires fixed table set
- Supervised pairs: Expensive manual annotation

**Contrastive advantages:**
- Learns general similarity function
- Scales to new tables without retraining
- Self-supervised via structural questions

---

### 2. Why Freeze Question Encoder?
**Rationale:**
- SentenceTransformer already trained on 1B+ sentences
- Training graph encoder is easier than joint training
- Prevents catastrophic forgetting of language understanding

**Trade-off:**
- Graph encoder must adapt to frozen question space
- Can't fine-tune question encoder for domain jargon

---

### 3. Why Sparse Graphs?
**Threshold-based edge creation** (composite_score ≥ 0.4):
- **Dense graphs:** All-to-all connections → GNN over-smoothing -> Essentially an MLP
- **Sparse graphs:** Only meaningful relationships → Better signal

**Typical sparsity:** 20-50 edges for 10-15 columns (~20-30% density)

---

### 4. Why 2-Layer GNN?
**Layer depth analysis:**
- **1 layer:** Only aggregates immediate neighbors
- **2 layers:** Captures 2-hop relationships (ID → FK → Timestamp)
- **3+ layers:** Over-smoothing, all nodes become similar

**Best practice:** 2 layers for tables with 10-20 columns

---

## Performance Expectations

### With Old `QuestionGenerator` (Generic Questions)
```
Epoch 10: Recall@1 = 0.05 (worse than random 4.5%)
Epoch 20: Recall@1 = 0.07
Epoch 50: Recall@1 = 0.09 (never converges)
```
**Problem:** Ambiguous supervision (multiple tables match same questions)

---

### With New `TableSpecificQuestionGenerator`
```
Epoch 5:  Recall@1 = 0.35
Epoch 10: Recall@1 = 0.62
Epoch 20: Recall@1 = 0.75
Epoch 50: Recall@1 = 0.80-0.85
```
**Improvement:** Clear supervision (unique questions per table)

---

## Common Issues & Solutions

### Issue 1: Low Recall (<10%)
**Symptoms:** Model doesn't learn, loss plateaus

**Causes:**
1. Using old `QuestionGenerator` (generic questions)
2. `hidden_dim=256` (information bottleneck)
3. Learning rate too low
4. Tables missing `.name` attribute

**Solutions:**
```python
# ✅ Use new question generator
question_gen = TableSpecificQuestionGenerator()

# ✅ Use hidden_dim=768
graph_encoder = ContrastiveGNNEncoder(node_dim=512, hidden_dim=768, ...)

# ✅ Increase learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# ✅ Assign table names
for df in tables:
    df.name = 'table_name'
```

---

### Issue 2: Sparse Graphs (Too Few Edges)
**Symptoms:** Many tables have <5 edges

**Cause:** Threshold 0.4 too high for your data

**Solution:**
```python
rel_gen = RelationshipGenerator(threshold_config={
    'composite_threshold': 0.3,  # Lower from 0.4 to 0.3
    'weights': {...}
})
```

---

### Issue 3: Out of Memory
**Symptoms:** CUDA OOM during training

**Solutions:**
```python
# Reduce batch size
dataloader = create_dataloader(dataset, batch_size=16)  # Was 32

# Reduce table size
df = pd.read_csv('table.csv', nrows=300)  # Was 500

# Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### Issue 4: Model Doesn't Generalize to New Domain
**Symptoms:** Good recall on healthcare data, poor on construction data

**Cause:** Domain-specific column names (e.g., `hadm_id` vs `project_id`)

**Solutions:**
```python
# Option 1: Few-shot fine-tuning (100 examples, 5 epochs)
construction_questions = question_gen.generate_dataset(
    construction_tables[:3], rel_gen, num_per_table=20
)
# Fine-tune with lr=1e-5

# Option 2: Add column name embeddings (384-d)
# Helps with semantic transfer: 'admittime' ≈ 'start_date'
tokenizer = LightweightFeatureTokenizer(
    embedding_strategy='hybrid',
    include_column_names=True  # 512→896-d nodes
)
```

---

## File Structure

```
contrastive_table2graph.py (2000+ lines)
├── Data Processing (Lines 30-123)
│   ├── DataProcessor
│   └── ColumnContentExtractor
├── Relationship Detection (Lines 215-528)
│   ├── RelationshipGenerator (10 edge features)
│   └── SemanticLabelGenerator (34 semantic labels)
├── Feature Extraction (Lines 724-773)
│   └── LightweightFeatureTokenizer (512-d node features)
├── Graph Construction (Lines 782-856)
│   └── GraphBuilder (DataFrame → PyG Data)
├── Neural Networks (Lines 858-1008)
│   ├── QuestionEncoder (frozen SentenceTransformer)
│   ├── ContrastiveGNNEncoder (2-layer GNN + projection)
│   ├── AttentionPooling
│   └── InfoNCELoss
├── Question Generation (Lines 1014-1935)
│   ├── QuestionGenerator (OLD - not recommended)
│   └── TableSpecificQuestionGenerator (NEW - ✅ use this)
└── Dataset (Lines 1942-2000+)
    ├── TableQuestionDataset
    ├── collate_fn
    └── create_dataloader
```

---

## Quick Start Example

```python
# Minimal working example
import pandas as pd
from pathlib import Path
from contrastive_table2graph import *

# Load data
df = pd.read_csv('data/patients.csv', nrows=500)
df.name = 'patients'

# Initialize pipeline
extractor = ColumnContentExtractor()
tokenizer = LightweightFeatureTokenizer(embedding_strategy='hybrid')
rel_gen = RelationshipGenerator()
graph_builder = GraphBuilder(extractor, tokenizer, rel_gen)

# Generate graph
pyg_data = graph_builder.build_graph(df)
print(f"Nodes: {pyg_data.x.shape}")  # [num_columns, 512]
print(f"Edges: {pyg_data.edge_index.shape}")  # [2, num_edges]

# Generate questions
question_gen = TableSpecificQuestionGenerator()
questions = question_gen.generate_questions_for_table(
    df,
    rel_gen.compute_all_relationship_scores(df),
    num_questions=20
)

for i, q in enumerate(questions[:3], 1):
    print(f"{i}. {q['question']}")
```

---

## References & Related Work

**Contrastive Learning:**
- CLIP (Radford et al., 2021): Image-text alignment
- DPR (Karpukhin et al., 2020): Dense passage retrieval
- SimCLR (Chen et al., 2020): Self-supervised contrastive learning

**Table Understanding:**
- TaBERT (Yin et al., 2020): Table pretraining
- TURL (Deng et al., 2020): Table-entity retrieval
- RelBench (Fey et al., 2024): Relational deep learning benchmark

**Graph Neural Networks:**
- GCN (Kipf & Welling, 2017): Graph convolutional networks
- GAT (Veličković et al., 2018): Graph attention networks

---

## License & Citation

If you use this code, please cite:

```bibtex
@software{struct_gram_2024,
  title={Contrastive Table-to-Graph Learning for Semantic Retrieval},
  author={Shwetabh Singh},
  year={2024},
  url={https://github.com/bugboy1769/struct_gram}
}
```

---


