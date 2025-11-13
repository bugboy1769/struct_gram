"""
MIMIC-IV Healthcare Dataset Training Script for Google Colab
Trains the Table2GraphPipeline on healthcare relational tables

Usage:
1. Upload this script and table2graph_sem.py to Colab
2. Upload the hosp/ directory to Colab
3. Run this script
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time
from collections import defaultdict
import json

# ==================== GOOGLE COLAB SETUP ====================
print("=" * 60)
print("MIMIC-IV Table2Graph Training Pipeline")
print("=" * 60)

# Install dependencies
print("\n[1/6] Installing dependencies...")
#!pip install -q sentence-transformers torch-geometric scikit-learn

# Mount Google Drive (optional - for saving checkpoints)
print("\n[2/6] Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = '/content/drive/MyDrive/table2graph_checkpoints'
    os.makedirs(DRIVE_PATH, exist_ok=True)
    print(f"✓ Checkpoints will be saved to: {DRIVE_PATH}")
except:
    DRIVE_PATH = '/content/checkpoints'
    os.makedirs(DRIVE_PATH, exist_ok=True)
    print(f"✓ Checkpoints will be saved to: {DRIVE_PATH}")

# Import pipeline
print("\n[3/6] Importing pipeline...")
from table2graph_sem import (
    ColumnContentExtractor,
    LightweightFeatureTokenizer,
    RelationshipGenerator,
    SemanticLabelGenerator,
    GraphBuilder,
    GNNEdgePredictor,
    Table2GraphPipeline
)
print("✓ Pipeline imported successfully")

# ==================== DATA LOADING ====================
print("\n[4/6] Loading MIMIC-IV dataset...")

HOSP_DIR = "/content/hosp"  # Adjust if different

# Define table categories for structured training
TABLE_GROUPS = {
    'core': ['patients', 'admissions', 'transfers'],
    'clinical': ['diagnoses_icd', 'procedures_icd', 'drgcodes', 'services'],
    'medications': ['prescriptions', 'pharmacy', 'emar', 'emar_detail'],
    'lab_micro': ['labevents', 'microbiologyevents', 'd_labitems'],
    'orders': ['poe', 'poe_detail', 'hcpcsevents'],
    'dictionaries': ['d_icd_diagnoses', 'd_icd_procedures', 'd_hcpcs'],
    'misc': ['omr', 'provider']
}

def load_mimic_tables(hosp_dir, max_rows=1000, sample_tables=None):
    """
    Load MIMIC-IV tables with sampling for manageable training

    Args:
        hosp_dir: Path to hosp directory
        max_rows: Maximum rows per table (for memory management)
        sample_tables: List of specific tables to load (None = all)
    """
    tables = {}

    csv_files = [f for f in os.listdir(hosp_dir) if f.endswith('.csv')]

    if sample_tables:
        csv_files = [f for f in csv_files if f.replace('.csv', '') in sample_tables]

    print(f"\nFound {len(csv_files)} CSV files")

    for csv_file in csv_files:
        table_name = csv_file.replace('.csv', '')
        filepath = os.path.join(hosp_dir, csv_file)

        try:
            # Load with row limit for memory efficiency
            df = pd.read_csv(filepath, nrows=max_rows, low_memory=False)

            # Skip tables with < 2 columns (can't build relationships)
            if len(df.columns) < 2:
                print(f"  ⊗ Skipped {table_name}: < 2 columns")
                continue

            # Skip empty tables
            if len(df) == 0:
                print(f"  ⊗ Skipped {table_name}: empty")
                continue

            tables[table_name] = df
            print(f"  ✓ Loaded {table_name}: {df.shape[0]} rows × {df.shape[1]} cols")

        except Exception as e:
            print(f"  ✗ Failed to load {table_name}: {e}")

    return tables

# Load tables (adjust max_rows based on Colab memory)
tables = load_mimic_tables(HOSP_DIR, max_rows=500)  # Start with 500 rows
print(f"\n✓ Successfully loaded {len(tables)} tables")

# ==================== PIPELINE INITIALIZATION ====================
print("\n[5/6] Initializing pipeline...")

# Initialize pipeline
pipeline = Table2GraphPipeline(embedding_strategy='hybrid')

# Initialize for training (no model_manager needed for lightweight approach)
pipeline.initialize_for_training(model_manager=None, node_dim=512)

print(f"✓ Pipeline initialized")
print(f"  - Feature tokenizer: LightweightFeatureTokenizer (hybrid)")
print(f"  - Node dimension: 512")
print(f"  - Number of semantic labels: {pipeline.train_builder.num_classes}")
print(f"  - GNN layers: 1 (1-hop message passing)")

# ==================== TRAINING CONFIGURATION ====================
print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 4,  # Process 4 tables per batch
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'checkpoint_every': 5,
    'min_accuracy_threshold': 0.7,
}

print(json.dumps(TRAINING_CONFIG, indent=2))

# ==================== TRAINING LOOP ====================
print("\n[6/6] Starting training...")
print("=" * 60)

# Prepare table batches
table_list = list(tables.values())
table_names = list(tables.keys())

# Training history
history = {
    'epoch': [],
    'loss': [],
    'accuracy': [],
    'time': [],
    'tables_processed': []
}

best_accuracy = 0.0
patience_counter = 0

print(f"\nTraining on {len(table_list)} tables")
print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"Total batches per epoch: {len(table_list) // TRAINING_CONFIG['batch_size']}")
print("\n" + "-" * 60)

for epoch in range(TRAINING_CONFIG['num_epochs']):
    epoch_start = time.time()

    print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
    print("-" * 40)

    # Shuffle tables each epoch
    indices = np.random.permutation(len(table_list))
    shuffled_tables = [table_list[i] for i in indices]
    shuffled_names = [table_names[i] for i in indices]

    # Train in batches
    epoch_losses = []
    epoch_accuracies = []
    tables_processed = 0

    for batch_idx in range(0, len(shuffled_tables), TRAINING_CONFIG['batch_size']):
        batch_tables = shuffled_tables[batch_idx:batch_idx + TRAINING_CONFIG['batch_size']]
        batch_names = shuffled_names[batch_idx:batch_idx + TRAINING_CONFIG['batch_size']]

        try:
            # Train on batch
            avg_loss, avg_accuracy = pipeline.train_epoch(batch_tables)

            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_accuracy)
            tables_processed += len(batch_tables)

            print(f"  Batch {batch_idx//TRAINING_CONFIG['batch_size']+1}: "
                  f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.3f} "
                  f"[{', '.join(batch_names)}]")

        except Exception as e:
            print(f"  ✗ Batch {batch_idx//TRAINING_CONFIG['batch_size']+1} failed: {e}")
            continue

    # Compute epoch metrics
    if epoch_losses:
        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = np.mean(epoch_accuracies)
        epoch_time = time.time() - epoch_start

        # Record history
        history['epoch'].append(epoch + 1)
        history['loss'].append(float(epoch_loss))
        history['accuracy'].append(float(epoch_accuracy))
        history['time'].append(float(epoch_time))
        history['tables_processed'].append(tables_processed)

        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_accuracy:.3f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Tables: {tables_processed}/{len(table_list)}")
        print(f"{'='*40}")

        # Checkpoint saving
        if (epoch + 1) % TRAINING_CONFIG['checkpoint_every'] == 0:
            checkpoint_path = os.path.join(DRIVE_PATH, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': pipeline.predictor.state_dict(),
                'optimizer_state_dict': pipeline.predictor.optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_accuracy,
                'history': history
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Early stopping check
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            patience_counter = 0

            # Save best model
            best_model_path = os.path.join(DRIVE_PATH, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': pipeline.predictor.state_dict(),
                'accuracy': best_accuracy,
                'history': history
            }, best_model_path)
            print(f"✓ New best model saved: {best_accuracy:.3f}")
        else:
            patience_counter += 1

        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"\n⚠ Early stopping triggered (no improvement for {patience_counter} epochs)")
            break

        # Check accuracy threshold
        if epoch_accuracy >= TRAINING_CONFIG['min_accuracy_threshold']:
            print(f"\n✓ Accuracy threshold reached: {epoch_accuracy:.3f} >= {TRAINING_CONFIG['min_accuracy_threshold']}")
    else:
        print(f"\n⚠ No valid batches processed in epoch {epoch+1}")

# ==================== TRAINING COMPLETE ====================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest Accuracy: {best_accuracy:.3f}")
print(f"Total Epochs: {len(history['epoch'])}")
print(f"Total Time: {sum(history['time']):.2f}s")

# Save final history
history_path = os.path.join(DRIVE_PATH, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"\n✓ Training history saved: {history_path}")

# ==================== TESTING / INFERENCE ====================
print("\n" + "=" * 60)
print("TESTING PHASE")
print("=" * 60)

# Initialize for testing
pipeline.initialize_for_testing()

# Test on a sample table
test_table_name = 'admissions'
if test_table_name in tables:
    test_df = tables[test_table_name]

    print(f"\nTesting on: {test_table_name}")
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")

    try:
        predictions = pipeline.predict_relationships(test_df)

        print(f"\n✓ Predicted {len(predictions)} relationships")
        print("\nTop 10 Relationships:")
        print("-" * 80)

        for i, pred in enumerate(predictions[:10]):
            print(f"{i+1}. {pred['col1']} ↔ {pred['col2']}")
            print(f"   Label: {pred['predicted_label']}")
            print(f"   Meaning: {pred['semantic_meaning']}")
            print(f"   Confidence: {pred['confidence']:.3f}")
            print()

        # Save predictions
        predictions_path = os.path.join(DRIVE_PATH, f'predictions_{test_table_name}.json')
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"✓ Predictions saved: {predictions_path}")

    except Exception as e:
        print(f"✗ Testing failed: {e}")
else:
    print(f"⚠ Test table '{test_table_name}' not found")

print("\n" + "=" * 60)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 60)
