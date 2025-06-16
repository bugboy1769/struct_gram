import pandas as pd
import networkx as nx
import numpy as np
import re as r

#Schema Aware Representation
class DF2G:

    def __init__(self, pd: pd.DataFrame):
        self.df = pd
        self.graph = nx.Graph()
        self.node_atts = {}
        self.edge_atts = {}

    def schema_graph(self) -> nx.Graph:

        G = nx.Graph()
        #Adding Column Nodes
        for col in self.df.columns:
            col_stats = self._get_col_stats(col)
            G.add_node(f"col_{col}", node_type = "column", **col_stats)
        
        #Adding Data Type Nodes       
        data_types = set(str(self.df[col].dtype) for col in self.df.columns)
        for d_type in data_types:
            G.add_node(f"d_type_{d_type}", node_type = "data_type", type_name = d_type)
        
        #Connecting Column to Data Types
        for col in self.df.columns:
            d_type = str(self.df[col].dtype)
            G.add_edge(f"col_{col}", f"d_type_{d_type}", edge_type = "has_type")
        
        #Connect Columns with Statistical Relationships
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.1: #significance
                            G.add_edge(f"col_{col1}", f"col_{col2}", edge_type = "corr", corr = corr, strength = abs(corr))
        
        #Add Value Nodes for Categorical Data
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            unique_values = self.df[col].unique()
            if len(unique_values) <= 20: #Manageability Limit
                for value in unique_values:
                    value_node = f"val_{col}_{value}"
                    G.add_node(value_node, node_type = "categorical_value", column = col, value = value, frequency = sum(self.df[col] == value))
                    G.add_edge(f"col_{col}", value_node, edge_type = "contains_value")
        return G
    
    def _create_value_based_graph(self) -> nx.Graph:
        """
        Create graph focusing on value relationships across the table.
        """
        G = nx.Graph()
        
        # Create value nodes
        value_to_locations = {}
        
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

    
    def _get_col_stats(self, col):
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
                'median': series.median(),
                'std_dev': series.std(),
                'min': series.min(),
                'max': series.max()
            })
        if pd.api.types.is_string_dtype(series):
            stats.update({
                'avg_length': series.str.len().mean(),
                'max_length': series.str.len().max(),
                'contains_numbers': series.str.contains(r'\d').any(),
                'contains_sp_chars': series.str.contains(r'[^a-zA-Z0-9\s]').any()
            })
        return stats
    
    def get_graph_summary(self, G: nx.Graph) -> dict[str, any]:
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
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'num_components': nx.number_connected_components(G)
        }

    




        
