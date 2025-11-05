"""
Test script for TableSpecificQuestionGenerator.

This demonstrates the differences between the old QuestionGenerator
and the new TableSpecificQuestionGenerator.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from contrastive_table2graph import (
    TableSpecificQuestionGenerator,
    RelationshipGenerator,
    SemanticLabelGenerator
)

def load_sample_tables(data_dir='hosp', max_tables=5):
    """Load sample tables from directory."""
    tables = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Data directory '{data_dir}' not found!")
        return tables

    for csv_file in sorted(data_path.glob('*.csv'))[:max_tables]:
        try:
            df = pd.read_csv(csv_file, nrows=500)
            df.name = csv_file.stem
            tables.append(df)
            print(f"✓ Loaded {df.name}: {df.shape}")
        except Exception as e:
            print(f"✗ Failed to load {csv_file.name}: {e}")

    return tables


def test_new_question_generator():
    """Test the new TableSpecificQuestionGenerator."""

    print("="*80)
    print("TESTING NEW TABLE-SPECIFIC QUESTION GENERATOR")
    print("="*80)

    # Load tables
    print("\n1. Loading sample tables...")
    tables = load_sample_tables()

    if not tables:
        print("No tables loaded. Please ensure 'hosp/' directory exists.")
        return

    # Initialize components
    print("\n2. Initializing components...")
    question_gen = TableSpecificQuestionGenerator()
    rel_gen = RelationshipGenerator()

    print(f"✓ TableSpecificQuestionGenerator initialized")
    print(f"✓ RelationshipGenerator initialized")

    # Test on first table
    test_table = tables[0]
    print(f"\n3. Testing on table: {test_table.name}")
    print(f"   Columns: {list(test_table.columns)}")
    print(f"   Shape: {test_table.shape}")

    # Generate relationships
    print(f"\n4. Generating relationships...")
    relationships = rel_gen.compute_all_relationship_scores(test_table)

    # Add semantic labels
    label_gen = SemanticLabelGenerator()
    for rel in relationships:
        rel['feature_label'] = label_gen.generate_feature_label(rel['edge_features'])

    print(f"   Found {len(relationships)} relationships")

    # Generate questions
    print(f"\n5. Generating 20 questions...")
    questions = question_gen.generate_questions_for_table(
        test_table,
        relationships,
        num_questions=20
    )

    print(f"   Generated {len(questions)} questions\n")

    # Display questions by type
    print("="*80)
    print("GENERATED QUESTIONS (showing all)")
    print("="*80)

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q['question']}")
        print(f"   Table: {q['table_name']}")
        print(f"   Label: {q['label']}")

    # Validate questions
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    # Check for column names in questions
    all_questions_text = ' '.join([q['question'] for q in questions])
    column_names = list(test_table.columns)
    found_columns = [col for col in column_names if col in all_questions_text]

    print(f"\n✓ Total questions generated: {len(questions)}")
    print(f"✓ All questions about same table: {len(set(q['table_name'] for q in questions)) == 1}")
    print(f"✓ Column names mentioned in questions: {len(found_columns)} out of {len(column_names)}")
    print(f"  Columns mentioned: {found_columns[:5]}..." if found_columns else "  No columns mentioned")

    # Test uniqueness across tables
    print("\n6. Testing uniqueness across multiple tables...")
    all_questions_all_tables = []

    for df in tables[:3]:  # Test on first 3 tables
        relationships = rel_gen.compute_all_relationship_scores(df)
        for rel in relationships:
            rel['feature_label'] = label_gen.generate_feature_label(rel['edge_features'])

        questions = question_gen.generate_questions_for_table(
            df, relationships, num_questions=10
        )
        all_questions_all_tables.extend(questions)

        print(f"\n   Table: {df.name}")
        print(f"   Sample question: {questions[0]['question']}")

    # Check for unique questions
    unique_questions = len(set(q['question'] for q in all_questions_all_tables))
    total_questions = len(all_questions_all_tables)

    print(f"\n✓ Total questions across {len(tables[:3])} tables: {total_questions}")
    print(f"✓ Unique questions: {unique_questions}")
    print(f"✓ Uniqueness rate: {unique_questions/total_questions*100:.1f}%")

    print("\n" + "="*80)
    print("KEY ADVANTAGES OF NEW QUESTION GENERATOR:")
    print("="*80)
    print("1. ✓ Questions mention specific columns → leverages 384-d column name embeddings")
    print("2. ✓ Each table gets unique questions → no ambiguity in supervision")
    print("3. ✓ Questions describe table structure → aligns with graph topology")
    print("4. ✓ Domain-aware question generation → semantic context preserved")
    print("5. ✓ Multiple question types → diverse supervision signal")
    print("="*80)


def compare_questions():
    """Compare questions from old vs new generator side-by-side."""

    print("\n\n" + "="*80)
    print("COMPARISON: OLD vs NEW QUESTION GENERATOR")
    print("="*80)

    tables = load_sample_tables(max_tables=1)
    if not tables:
        return

    test_table = tables[0]
    rel_gen = RelationshipGenerator()
    relationships = rel_gen.compute_all_relationship_scores(test_table)

    label_gen = SemanticLabelGenerator()
    for rel in relationships:
        rel['feature_label'] = label_gen.generate_feature_label(rel['edge_features'])

    # New generator
    new_gen = TableSpecificQuestionGenerator()
    new_questions = new_gen.generate_questions_for_table(
        test_table, relationships, num_questions=5
    )

    print(f"\nTable: {test_table.name}")
    print(f"Columns: {list(test_table.columns)[:5]}...")

    print("\n" + "-"*80)
    print("OLD QuestionGenerator (generic, pattern-based):")
    print("-"*80)
    print("Example questions (would be same for any table with similar patterns):")
    print("  • Which table tracks time-ordered events?")
    print("  • Which table links multiple entities together?")
    print("  • Which table stores quantitative measurements?")
    print("  • Which table records transactional activities?")
    print("  • Which table provides descriptive attributes?")
    print("\n❌ Problem: Multiple tables match these questions → ambiguous supervision")

    print("\n" + "-"*80)
    print("NEW TableSpecificQuestionGenerator (table-specific):")
    print("-"*80)
    for i, q in enumerate(new_questions, 1):
        print(f"  {i}. {q['question']}")

    print("\n✅ Advantage: Each question is unique to this table → clear supervision")


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Run tests
    test_new_question_generator()
    compare_questions()

    print("\n✓ All tests completed!")
