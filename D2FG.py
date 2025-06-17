import pandas as pd
import networkx as nx
import numpy as np
import re as r
from collections import defaultdict
import fasttext as ft
import fasttext.util as ftu
import math
#Schema Aware Representation
class DF2G:

    def __init__(self, pd: pd.DataFrame):
        self.df = pd
        self.graph = nx.Graph()
        self.node_atts = {}
        self.edge_atts = {}
        self.ft_embedder = ft.load_model('cc.en.300.bin')

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


    def create_column_graph(self) -> nx.Graph:

        G = nx.Graph()

        for col in self.df.columns:
            col_stats = self._get_col_stats(col)
            G.add_node(f"col_{col}", node_type = "column", **col_stats)
        
        #Add Edges based on stat relns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.1: #significance
                            G.add_edge(f"col_{col1}", f"col_{col2}", edge_type = "corr", corr = corr, strength = abs(corr))
        #Add Edges for dtype relns
        dtype_groups = self.df.dtypes.groupby(self.df.dtypes).groups
        for dtype, cols in dtype_groups.items():
            if len(cols) > 1:
                col_list = list(cols)
                for i in range(len(col_list)):
                    for j in range(i + 1, len(col_list)):
                        col1, col2 = col_list[i], col_list[j]
                        if not G.has_edge(f"col_{col1}", f"col_{col2}"):
                            G.add_edge(f"col_{col1}", f"col_{col2}", edge_type = "has_dtype", dtype = str(dtype), weight = 0.5)
        G = self.homogenise_column_features(G)
        return G

    def homogenise_column_features(self, G:nx.Graph) -> nx.Graph:

        all_features = set()
        for node, attrs in G.nodes(data=True):
            all_features.update(attrs.keys())
        
        exclude_keys = {'node_types', 'column_names'}
        feature_keys = sorted(all_features - exclude_keys)

        for node, attrs in G.nodes(data=True):
            feature_vector = []
            for key in feature_keys:
                value = attrs.get(key, None)

                if value is None:
                    feature_vector.append(0.0)
                elif isinstance(value, (int, float)):
                    if pd.isna(value):
                        feature_vector.append(0.0)
                    else:
                        feature_vector.append(float(value))
                elif isinstance(value, bool):
                    feature_vector.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    feature_vector.append(self.ft_embedder.get_word_vector(value).tolist())
                else:
                    feature_vector.append(self.ft_embedder.get_word_vector(value).tolist())
        
        G.nodes[node]['features'] = feature_vector

        if feature_keys:
            G.graph['feature_dim'] = len(feature_keys)
            G.graph['feature_keys'] = feature_keys

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
    
    def _normalize_value(self, value: any) -> str:
        """Normalize values for comparison across columns."""
        if isinstance(value, (int, float)):
            return f"num_{value}"
        elif isinstance(value, str):
            # Basic normalization
            return value.lower().strip()
        else:
            return str(value)
    
    def apply_force_directed_layout(self, G: nx.Graph, iterations: int = 50, k: float = None) -> nx.Graph:
        """
        Apply spring/force-directed layout to compute node positions.
        This uses NetworkX's spring layout algorithm.
        """
        # Compute spring layout positions
        pos = nx.spring_layout(
            G, 
            iterations=iterations,
            k=k,  # Optimal distance between nodes
            weight='weight',  # Use edge weights if available
            seed=42  # For reproducibility
        )
        
        # Add position coordinates as node features
        for node in G.nodes():
            x, y = pos[node]
            G.nodes[node]['pos_x'] = x
            G.nodes[node]['pos_y'] = y
            
            # # Optionally add position to feature vector
            # original_features = G.nodes[node]['features']
            # enhanced_features = original_features + [x, y]
            # G.nodes[node]['features'] = enhanced_features
        
        # # Update feature dimension
        # self.feature_dim = len(enhanced_features)
        
        return G
    
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
    def homogenise_column_features(self, G: nx.Graph) -> nx.Graph:
    
        all_features = set()
        for node, attrs in G.nodes(data=True):
            all_features.update(attrs.keys())
    
        exclude_keys = {'node_type', 'column_name'}
        feature_keys = sorted(all_features - exclude_keys)
    
        # Separate features by type
        text_features = ['data_type']
        numerical_features = ['unique_count', 'null_count', 'null_percentage', 'mean', 'median', 'std_dev', 'min', 'max', 'avg_length', 'max_length']
        boolean_features = ['contains_numbers', 'contains_sp_chars']
        
        # Collect all numerical values for normalization
        all_numerical_values = {}
        for feat in numerical_features:
            values = []
            for node, attrs in G.nodes(data=True):
                if feat in attrs and attrs[feat] is not None and not pd.isna(attrs[feat]):
                    values.append(float(attrs[feat]))
            if values:
                all_numerical_values[feat] = {'min': min(values), 'max': max(values), 'mean': sum(values)/len(values)}
        
        def positional_encoding(value, feat_name, d_model=300):
            """Apply sinusoidal positional encoding to normalized numerical values"""
            if feat_name not in all_numerical_values:
                return [0.0] * d_model
                
            # Normalize value to [0, 1] range
            feat_stats = all_numerical_values[feat_name]
            if feat_stats['max'] == feat_stats['min']:
                normalized = 0.5
            else:
                normalized = (value - feat_stats['min']) / (feat_stats['max'] - feat_stats['min'])
            
            # Apply sinusoidal encoding
            encoding = []
            for i in range(d_model):
                if i % 2 == 0:
                    encoding.append(math.sin(normalized * (10000 ** (i / d_model))))
                else:
                    encoding.append(math.cos(normalized * (10000 ** (i / d_model))))
            return encoding

        for node, attrs in G.nodes(data=True):
            feature_vector = []
            
            # Text embeddings (300d)
            text_embeddings = []
            for feat in text_features:
                value = attrs.get(feat, None)
                if value is not None:
                    # FastText embedding for textual data
                    text_embeddings.extend(self.ft_embedder.get_word_vector(str(value)).tolist())
                else:
                    text_embeddings.extend([0.0] * 300)
            feature_vector.extend(text_embeddings)
        
        # Numerical embeddings (300d each)
        numerical_embeddings = []
        for feat in numerical_features:
            value = attrs.get(feat, None)
            if value is not None and not pd.isna(value):
                # Positional encoding for numerical data
                numerical_embeddings.extend(positional_encoding(float(value), feat))
            else:
                numerical_embeddings.extend([0.0] * 300)
        feature_vector.extend(numerical_embeddings)
        
        # Boolean embeddings (300d each)
        boolean_embeddings = []
        for feat in boolean_features:
            value = attrs.get(feat, None)
            if value is not None:
                # Convert boolean to descriptive text and embed
                text_repr = f"{feat}_{'yes' if value else 'no'}"
                boolean_embeddings.extend(self.ft_embedder.get_word_vector(text_repr).tolist())
            else:
                boolean_embeddings.extend([0.0] * 300)
        feature_vector.extend(boolean_embeddings)
        
        G.nodes[node]['features'] = feature_vector

        # Update graph metadata
        total_dim = len(text_features) * 300 + len(numerical_features) * 300 + len(boolean_features) * 300
        G.graph['feature_dim'] = total_dim
        G.graph['feature_composition'] = {
            'text_features': text_features,
            'numerical_features': numerical_features, 
            'boolean_features': boolean_features,
            'embedding_dims': {'text': 300, 'numerical': 300, 'boolean': 300}
        }

        return G

    




        
