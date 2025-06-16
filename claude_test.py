import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import re

class DataFrameToGraph:
    """
    Convert pandas DataFrame to NetworkX graph with multiple encoding strategies.
    Preserves semantic relationships and table structure.
    """
    
    def __init__(self, df: pd.DataFrame, strategy: str = "hybrid"):
        """
        Initialize converter with DataFrame and encoding strategy.
        
        Args:
            df: Input DataFrame
            strategy: Graph encoding strategy
                - "entity_centric": Focus on entity relationships
                - "schema_aware": Emphasize column relationships and types
                - "value_based": Connect similar values across columns
                - "hybrid": Combine multiple strategies
        """
        self.df = df
        self.strategy = strategy
        self.graph = nx.Graph()
        self.node_attributes = {}
        self.edge_attributes = {}
        
    def create_graph(self) -> nx.Graph:
        """Create graph based on selected strategy."""
        if self.strategy == "entity_centric":
            return self._create_entity_centric_graph()
        elif self.strategy == "schema_aware":
            return self._create_schema_aware_graph()
        elif self.strategy == "value_based":
            return self._create_value_based_graph()
        elif self.strategy == "hybrid":
            return self._create_hybrid_graph()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _create_entity_centric_graph(self) -> nx.Graph:
        """
        Create graph where each row is an entity, columns are attributes.
        Edges connect entities with shared attribute values.
        """
        G = nx.Graph()
        
        # Add row nodes (entities)
        for idx, row in self.df.iterrows():
            node_id = f"row_{idx}"
            G.add_node(node_id, 
                      node_type="entity",
                      row_index=idx,
                      attributes=row.to_dict())
        
        # Add column nodes (attributes)
        for col in self.df.columns:
            col_node = f"col_{col}"
            G.add_node(col_node,
                      node_type="attribute",
                      column_name=col,
                      data_type=str(self.df[col].dtype),
                      unique_values=len(self.df[col].unique()),
                      null_count=self.df[col].isnull().sum())
        
        # Connect entities to their attributes
        for idx, row in self.df.iterrows():
            row_node = f"row_{idx}"
            for col, value in row.items():
                col_node = f"col_{col}"
                G.add_edge(row_node, col_node,
                          edge_type="has_attribute",
                          value=value,
                          is_null=pd.isna(value))
        
        # Connect entities with shared values
        for col in self.df.columns:
            value_groups = self.df.groupby(col).groups
            for value, indices in value_groups.items():
                if len(indices) > 1 and not pd.isna(value):
                    # Connect entities sharing this value
                    indices_list = list(indices)
                    for i in range(len(indices_list)):
                        for j in range(i + 1, len(indices_list)):
                            row1, row2 = f"row_{indices_list[i]}", f"row_{indices_list[j]}"
                            if G.has_edge(row1, row2):
                                G[row1][row2]['shared_attributes'].append(col)
                                G[row1][row2]['shared_values'].append(value)
                            else:
                                G.add_edge(row1, row2,
                                          edge_type="shares_value",
                                          shared_attributes=[col],
                                          shared_values=[value])
        
        return G
    
    def _create_schema_aware_graph(self) -> nx.Graph:
        """
        Create graph emphasizing column relationships and data types.
        """
        G = nx.Graph()
        
        # Add column nodes with rich metadata
        for col in self.df.columns:
            col_stats = self._get_column_stats(col)
            G.add_node(f"col_{col}", 
                      node_type="column",
                      **col_stats)
        
        # Add data type nodes
        data_types = set(str(self.df[col].dtype) for col in self.df.columns)
        for dtype in data_types:
            G.add_node(f"type_{dtype}",
                      node_type="data_type",
                      type_name=dtype)
        
        # Connect columns to their data types
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            G.add_edge(f"col_{col}", f"type_{dtype}",
                      edge_type="has_type")
        
        # Connect columns with statistical relationships
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicate edges
                        correlation = corr_matrix.loc[col1, col2]
                        if abs(correlation) > 0.1:  # Threshold for significance
                            G.add_edge(f"col_{col1}", f"col_{col2}",
                                      edge_type="correlation",
                                      correlation=correlation,
                                      strength=abs(correlation))
        
        # Add value nodes for categorical data
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            unique_values = self.df[col].dropna().unique()
            if len(unique_values) <= 20:  # Limit for manageability
                for value in unique_values:
                    value_node = f"val_{col}_{value}"
                    G.add_node(value_node,
                              node_type="categorical_value",
                              column=col,
                              value=value,
                              frequency=sum(self.df[col] == value))
                    G.add_edge(f"col_{col}", value_node,
                              edge_type="contains_value")
        
        return G
    
    def _create_value_based_graph(self) -> nx.Graph:
        """
        Create graph focusing on value relationships across the table.
        """
        G = nx.Graph()
        
        # Create value nodes
        value_to_locations = defaultdict(list)
        
        for col in self.df.columns:
            for idx, value in enumerate(self.df[col]):
                if not pd.isna(value):
                    # Normalize value for comparison
                    normalized_value = self._normalize_value(value)
                    value_to_locations[normalized_value].append((col, idx, value))
        
        # Add nodes for values that appear in multiple locations
        for norm_value, locations in value_to_locations.items():
            if len(locations) > 1:
                value_node = f"value_{hash(norm_value) % 10000}"
                G.add_node(value_node,
                          node_type="shared_value",
                          normalized_value=norm_value,
                          occurrences=len(locations),
                          locations=locations)
                
                # Connect to position nodes
                for col, row_idx, original_value in locations:
                    pos_node = f"pos_{col}_{row_idx}"
                    if not G.has_node(pos_node):
                        G.add_node(pos_node,
                                  node_type="position",
                                  column=col,
                                  row=row_idx,
                                  value=original_value)
                    
                    G.add_edge(value_node, pos_node,
                              edge_type="occurs_at")
        
        return G
    
    def _create_hybrid_graph(self) -> nx.Graph:
        """
        Create comprehensive graph combining multiple strategies.
        """
        G = nx.Graph()
        
        # Start with entity-centric base
        entity_graph = self._create_entity_centric_graph()
        G = nx.compose(G, entity_graph)
        
        # Add schema information
        schema_graph = self._create_schema_aware_graph()
        # Only add schema nodes that don't conflict
        for node, attrs in schema_graph.nodes(data=True):
            if not G.has_node(node):
                G.add_node(node, **attrs)
        
        for u, v, attrs in schema_graph.edges(data=True):
            if G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
                G.add_edge(u, v, **attrs)
        
        # Add hierarchical structure
        self._add_hierarchical_structure(G)
        
        return G
    
    def _get_column_stats(self, col: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a column."""
        series = self.df[col]
        stats = {
            'column_name': col,
            'data_type': str(series.dtype),
            'unique_count': len(series.unique()),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().mean(),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median()
            })
        elif pd.api.types.is_string_dtype(series):
            stats.update({
                'avg_length': series.str.len().mean(),
                'max_length': series.str.len().max(),
                'contains_numbers': series.str.contains(r'\d').any(),
                'contains_special_chars': series.str.contains(r'[^a-zA-Z0-9\s]').any()
            })
        
        return stats
    
    def _normalize_value(self, value: Any) -> str:
        """Normalize values for comparison across columns."""
        if isinstance(value, (int, float)):
            return f"num_{value}"
        elif isinstance(value, str):
            # Basic normalization
            return value.lower().strip()
        else:
            return str(value)
    
    def _add_hierarchical_structure(self, G: nx.Graph):
        """Add hierarchical relationships to the graph."""
        # Add table-level node
        table_node = "table_root"
        G.add_node(table_node,
                  node_type="table",
                  shape=self.df.shape,
                  columns=list(self.df.columns),
                  dtypes=dict(self.df.dtypes.astype(str)))
        
        # Connect table to all columns
        for col in self.df.columns:
            col_node = f"col_{col}"
            if G.has_node(col_node):
                G.add_edge(table_node, col_node,
                          edge_type="contains_column")
    
    def get_graph_summary(self, G: nx.Graph) -> Dict[str, Any]:
        """Get summary statistics of the created graph."""
        node_types = {}
        edge_types = {}
        
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for u, v, attrs in G.edges(data=True):
            edge_type = attrs.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'num_components': nx.number_connected_components(G)
        }

# Example usage and testing
def demo_conversion():
    """Demonstrate the conversion with sample data."""
    # Create sample DataFrame
    sample_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing'],
        'salary': [75000, 65000, 80000, 60000, 70000],
        'years_experience': [3, 5, 7, 2, 4],
        'location': ['NYC', 'SF', 'NYC', 'LA', 'SF']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Test different strategies
    strategies = ["entity_centric", "schema_aware", "value_based", "hybrid"]
    
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        converter = DataFrameToGraph(df, strategy=strategy)
        graph = converter.create_graph()
        summary = converter.get_graph_summary(graph)
        
        print(f"Graph Summary: {summary}")
        print(f"Sample nodes: {list(graph.nodes())[:5]}")
        print(f"Sample edges: {list(graph.edges())[:5]}")
        print("\n" + "-"*30 + "\n")
    
    return df, converter, graph

if __name__ == "__main__":
    demo_conversion()