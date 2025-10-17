# MIMIC-IV Table2Graph Training on Google Colab

## Quick Start Guide

### Option 1: Using Jupyter Notebook (Recommended)

1. **Upload to Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `MIMIC_Training_Colab.ipynb`

2. **Upload Required Files:**
   - `table2graph_sem.py` (main pipeline)
   - `gcn_conv.py` (GNN implementation)
   - `hosp/` folder (all MIMIC-IV CSV files)

3. **Run All Cells:**
   - Click "Runtime" → "Run all"
   - Monitor training progress

### Option 2: Using Python Script

1. **Upload to Colab:**
   - Create new notebook
   - Upload `train_mimic_colab.py`, `table2graph_sem.py`, `gcn_conv.py`
   - Upload `hosp/` folder

2. **Run Script:**
   ```python
   !python train_mimic_colab.py
   ```

## File Structure

```
/content/
├── table2graph_sem.py          # Main pipeline
├── gcn_conv.py                  # TableGCN implementation
├── train_mimic_colab.py         # Training script
├── MIMIC_Training_Colab.ipynb   # Jupyter notebook
└── hosp/                        # MIMIC-IV data
    ├── admissions.csv
    ├── patients.csv
    ├── diagnoses_icd.csv
    └── ... (22 CSV files)
```

## Training Configuration

### Default Settings
- **Epochs**: 50
- **Batch Size**: 4 tables per batch
- **Max Rows**: 500 rows per table (adjustable)
- **Learning Rate**: 0.001
- **Early Stopping**: 10 epochs patience
- **Checkpoints**: Every 5 epochs

### Memory Management
If you encounter memory issues:

```python
# Reduce max_rows
tables = load_mimic_tables(HOSP_DIR, max_rows=300)  # Lower from 500

# Reduce batch size
CONFIG['batch_size'] = 2  # Lower from 4
```

### GPU Acceleration
Enable GPU in Colab:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Click Save

## Expected Training Time

| Configuration | Time per Epoch | Total Time (50 epochs) |
|--------------|----------------|------------------------|
| CPU only     | ~5-10 min      | ~4-8 hours            |
| GPU (T4)     | ~1-2 min       | ~1-2 hours            |
| GPU (A100)   | ~30-60 sec     | ~25-50 min            |

## Outputs

Training saves these files to Google Drive:

```
/content/drive/MyDrive/table2graph_checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── ...
├── best_model.pt                    # Best performing model
├── training_history.json            # Loss/accuracy history
├── training_curves.png              # Visualization
└── predictions_admissions.json      # Test predictions
```

## Monitoring Training

### In Notebook/Script Output
```
Epoch 1/50
----------------------------------------
  Batch 1: Loss=2.3456, Acc=0.234 [admissions, patients, ...]
  Batch 2: Loss=2.1234, Acc=0.289 [...]
  ...

Epoch Summary: Loss=2.2345, Acc=0.261, Time=45.2s
✓ Checkpoint saved
✓ New best: 0.261
```

### Key Metrics
- **Loss**: Should decrease over time (lower is better)
- **Accuracy**: Should increase over time (higher is better)
- **Target**: Accuracy > 0.7 indicates good performance

## Testing Predictions

After training, test on any table:

```python
# Test on diagnoses table
test_df = tables['diagnoses_icd']
predictions = pipeline.predict_relationships(test_df)

# View predictions
for pred in predictions[:5]:
    print(f"{pred['col1']} ↔ {pred['col2']}: {pred['predicted_label']}")
```

## Semantic Labels

The model predicts 34 types of relationships:

### JOIN Relationships (8)
- PRIMARY_FOREIGN_KEY
- FOREIGN_KEY_CANDIDATE
- NATURAL_JOIN_CANDIDATE
- WEAK_JOIN_CANDIDATE
- REVERSE_FOREIGN_KEY
- CROSS_TABLE_REFERENCE
- MANY_TO_MANY_REFERENCE
- SELF_REFERENTIAL_KEY

### Aggregation Relationships (7)
- MEASURE_DIMENSION_STRONG
- MEASURE_DIMENSION_WEAK
- DIMENSION_HIERARCHY
- FACT_DIMENSION
- NATURAL_GROUPING
- NESTED_AGGREGATION
- PIVOT_CANDIDATE

### Ordering Relationships (5)
- TEMPORAL_SEQUENCE_STRONG
- TEMPORAL_SEQUENCE_WEAK
- TEMPORAL_CORRELATION
- SEQUENTIAL_ORDERING
- RANKED_RELATIONSHIP

### Derivation Relationships (6)
- DERIVED_CALCULATION
- FUNCTIONAL_TRANSFORMATION
- AGGREGATED_DERIVATION
- REDUNDANT_COLUMN
- NORMALIZED_VARIANT
- SYNONYM_COLUMN

### Structural Relationships (5)
- COMPOSITE_KEY_COMPONENT
- PARTITION_KEY
- INDEX_CANDIDATE
- AUDIT_RELATIONSHIP
- VERSION_TRACKING

### Weak/Unknown (3)
- WEAK_CORRELATION
- INDEPENDENT_COLUMNS
- AMBIGUOUS_RELATIONSHIP

## Troubleshooting

### "Out of Memory" Error
```python
# Reduce dataset size
tables = load_mimic_tables(HOSP_DIR, max_rows=200)

# Use fewer tables
selected_tables = ['admissions', 'patients', 'diagnoses_icd']
tables = {k: tables[k] for k in selected_tables if k in tables}
```

### "No edges found" Warning
This means columns in a table don't pass the similarity threshold. This is normal for some tables.

### Low Accuracy (<0.3)
- Increase training epochs
- Try different tables (core tables like admissions, patients work best)
- Check if semantic features are computing correctly

### Import Errors
Make sure all files are uploaded to `/content/` in Colab.

## Advanced Configuration

### Custom Threshold
```python
# Lower threshold for more edges
pipeline.relationship_generator.thresholds['composite_threshold'] = 0.2
```

### Different Embedding Strategy
```python
# Semantic only (faster)
pipeline = Table2GraphPipeline(embedding_strategy='semantic')

# Statistical only (lightweight)
pipeline = Table2GraphPipeline(embedding_strategy='statistical')
```

### More GNN Layers
```python
# In gcn_conv.py, modify TableGCN
predictor = GNNEdgePredictor(..., num_layers=5)  # 5-hop instead of 3
```

## Citation

If you use this code with MIMIC-IV data, please cite:

```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review training output for error messages
3. Verify all files are uploaded correctly
4. Check GPU is enabled in Colab settings
