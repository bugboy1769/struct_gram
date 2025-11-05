# Notebook Update Instructions

## How to Switch to TableSpecificQuestionGenerator

Follow these steps to update your training notebook to use the new question generator.

---

## Changes Required

### Change 1: Import the New Generator

**Cell 6** (Import pipeline components)

Replace:
```python
from contrastive_table2graph import (
    DataProcessor,
    ColumnContentExtractor,
    LightweightFeatureTokenizer,
    RelationshipGenerator,
    SemanticLabelGenerator,
    GraphBuilder,
    QuestionEncoder,
    ContrastiveGNNEncoder,
    AttentionPooling,
    InfoNCELoss,
    QuestionGenerator,  # ← OLD
    TableQuestionDataset,
    collate_fn,
    create_dataloader
)
```

With:
```python
from contrastive_table2graph import (
    DataProcessor,
    ColumnContentExtractor,
    LightweightFeatureTokenizer,
    RelationshipGenerator,
    SemanticLabelGenerator,
    GraphBuilder,
    QuestionEncoder,
    ContrastiveGNNEncoder,
    AttentionPooling,
    InfoNCELoss,
    TableSpecificQuestionGenerator,  # ← NEW
    TableQuestionDataset,
    collate_fn,
    create_dataloader
)
```

---

### Change 2: Initialize the New Generator

**Cell 12** (Initialize pipeline components)

Replace:
```python
# Question generator
question_generator = QuestionGenerator(
    semantic_label_generator=semantic_label_generator
)
print("✓ QuestionGenerator")
print(f"  - 6 pattern-based categories with 12 templates each")
```

With:
```python
# Question generator (NEW: table-specific questions)
question_generator = TableSpecificQuestionGenerator()
print("✓ TableSpecificQuestionGenerator")
print(f"  - Generates unique questions per table")
print(f"  - Leverages column names and structural patterns")
```

---

### Change 3: Update Question Generation Call

**Cell 14** (Generate question-table pairs)

The API is the same, but the output will be different:

```python
# This line stays the same:
question_data = question_generator.generate_dataset(
    tables=table_dfs,
    relationship_generator=relationship_generator,
    num_per_table=20
)

# But now each table gets UNIQUE questions mentioning specific columns
print(f"\n✓ Generated {len(question_data)} question-table pairs")
print(f"  - Each table has unique questions (not generic patterns)")
print(f"  - Questions mention specific columns from each table")
print(f"  - {len(question_data) / len(table_dfs):.1f} questions per table (avg)")
```

---

### Change 4: Update Training Loop (if using old collate format)

**Cell 26** (Training utilities) - If your training loop looks like this:

Old format:
```python
for batched_graphs, questions, labels in train_loader:
    # Training code
```

Update to:
```python
for batch in train_loader:
    batched_graphs = batch['graphs']
    questions = batch['questions']
    labels = batch['labels']
    table_names = batch['table_names']  # Now available for debugging!
    # Training code
```

**Cell 28** (Main training loop) - Same change as above if you iterate over the dataloader.

---

## What Changed and Why

### Old QuestionGenerator Problems:
1. ❌ Generic questions: "Which table tracks temporal events?"
2. ❌ Multiple tables match the same question → ambiguous supervision
3. ❌ Column name embeddings (384-d) were unused
4. ❌ Model couldn't distinguish similar tables

### New TableSpecificQuestionGenerator Solutions:
1. ✅ Unique questions per table: "Which table has columns subject_id, hadm_id, admittime?"
2. ✅ Each question identifies exactly ONE table → clear supervision
3. ✅ Questions mention column names → leverages 384-d embeddings
4. ✅ Model learns to match column names + structure → better retrieval

---

## Expected Results After Update

### Question Quality Comparison:

**Before (generic patterns):**
```
Table: admissions
Questions:
  • Which table tracks time-ordered events?
  • Which table links multiple entities together?
  • Which table records transactional activities?

Table: labevents
Questions:
  • Which table tracks time-ordered events?  ← Same question!
  • Which table stores quantitative measurements?
  • Which table records transactional activities?  ← Same question!
```
→ **Problem:** Multiple tables share questions → model gets conflicting gradients

**After (table-specific):**
```
Table: admissions
Questions:
  • Which table has columns subject_id, hadm_id, admittime, dischtime?
  • Which table uses hadm_id to link to other entities?
  • Which table tracks events using admittime, dischtime, edregtime timestamps?

Table: labevents
Questions:
  • Which table contains columns subject_id, hadm_id, itemid, valuenum?
  • Which table stores measurements in columns like valuenum, valueuom?
  • Which laboratory table contains charttime, storetime, value?
```
→ **Solution:** Each table has unique questions → clear supervision signal

---

## Training Improvements Expected

After switching to `TableSpecificQuestionGenerator`, you should see:

1. **Faster convergence** (5-10 epochs instead of 20+)
   - Clear supervision eliminates conflicting gradients

2. **Higher Recall@1** (target: 0.6-0.8 instead of 0.05-0.15)
   - Model learns to match column names → unique identification

3. **Better use of column name embeddings**
   - Questions mention columns → model learns semantic alignment

4. **Improved generalization**
   - Model learns structural patterns (FK + temporal) not just table size

---

## Testing Before Full Training

Before running 50 epochs, test with a small run (5 epochs, 3 tables):

```python
# Quick test with 3 tables
test_tables = tables[:3]
question_data = question_generator.generate_dataset(
    test_tables, relationship_generator, num_per_table=10
)

# Print sample questions to verify uniqueness
for table_name in [t.name for t in test_tables]:
    table_questions = [q['question'] for q in question_data
                       if q['table_name'] == table_name]
    print(f"\n{table_name} questions:")
    for q in table_questions[:3]:
        print(f"  • {q}")

# Verify no overlap
all_questions = [q['question'] for q in question_data]
unique_rate = len(set(all_questions)) / len(all_questions)
print(f"\nUniqueness rate: {unique_rate*100:.1f}%")
# Should be close to 100% (some overlap is OK for hybrid questions)
```

---

## Troubleshooting

### Issue: "Table missing .name attribute"
**Solution:** Ensure tables are loaded with names:
```python
df = pd.read_csv('hosp/admissions.csv')
df.name = 'admissions'  # ← Add this!
```

### Issue: "Not enough questions generated"
**Solution:** Some tables have few columns. The generator will pad with column enumeration questions automatically.

### Issue: "Questions look too similar"
**Solution:** Increase `num_per_table` from 20 to 30 for more diversity.

---

## Summary

**Only 3 cells need changes:**
1. Cell 6: Import `TableSpecificQuestionGenerator`
2. Cell 12: Initialize without `semantic_label_generator` argument
3. Cell 26/28: Handle dict return from collate_fn (optional, for table_names tracking)

That's it! The rest of the pipeline remains unchanged.
