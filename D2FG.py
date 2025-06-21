import pandas as pd
import networkx as nx
import numpy as np
import re as r
from collections import defaultdict
import fasttext as ft
import fasttext.util as ftu
import math
from sentence_transformers import SentenceTransformer
#Schema Aware Representation
class DF2G:

    def __init__(self, pd: pd.DataFrame):
        self.df = pd
        self.graph = nx.Graph()
        self.node_atts = {}
        self.edge_atts = {}
        # self.ft_embedder = ft.load_model('cc.en.300.bin')
        self.st_embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
        
        G = self.homogenise_column_features_claude(G)

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
                    feature_vector.append(self.st_embedder.encode(value).tolist())
                else:
                    feature_vector.append(self.st_embedder.encode(value).tolist())
        
        G.nodes[node]['features'] = feature_vector

        if feature_keys:
            G.graph['feature_dim'] = len(feature_keys)
            G.graph['feature_keys'] = feature_keys

        return G

    def _get_col_stats(self, col):
        series = self.df[col]
        stats = {
            'column_name': f"Column Name_: {col}",
            'data_type': f"Column Data Type_: {str(series.dtype)}",
            'unique_count': f"Unique Values in Column_: {len(series.unique())}",
            'null_count': f"Null Values in Column_: {series.isnull().sum()}",
            'null_percentage': f"Null Percentage in Column_: {series.isnull().mean()}",
            }
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': f"Mean of Column_: {series.mean()}",
                'median': f"Median of Column_: {series.median()}",
                'std_dev': f"Standard Deviation of Column_: {series.std()}",
                'min': f"Minimum of Column_: {series.min()}",
                'max': f"Maximum of Column_: {series.max()}",
            })
        if pd.api.types.is_string_dtype(series):

            digit_pattern = r'\d'
            special_char_pattern = r'[^a-zA-Z0-9\s]'

            stats.update({
                'avg_length': f"Average Length of Column_: {series.str.len().mean()}",
                'max_length': f"Maximum Length of Column_: {series.str.len().max()}",
                'contains_numbers': f"Contains Numbers: {series.str.contains(digit_pattern).any()}",
                'contains_sp_chars': f"Contains Special Characters: {series.str.contains(special_char_pattern).any()}"
            })
        return stats
    
    # def _normalize_value(self, value: any) -> str:
    #     """Normalize values for comparison across columns."""
    #     if isinstance(value, (int, float)):
    #         return f"num_{value}"
    #     elif isinstance(value, str):
    #         # Basic normalization
    #         return value.lower().strip()
    #     else:
    #         return str(value)
    
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

    def homogenise_column_features_claude(self, G: nx.Graph) -> nx.Graph:
    
        all_features = set()
        for node, attrs in G.nodes(data=True):
            all_features.update(attrs.keys())

        exclude_keys = {'node_type', 'column_name'}
        feature_keys = sorted(all_features - exclude_keys)

        # Define ALL possible features that could exist
        text_features = ['data_type', 'unique_count', 'null_count', 'null_percentage', 
            'mean', 'median', 'std_dev', 'min', 'max',  # numeric-only features
            'avg_length', 'max_length', 'contains_numbers', 'contains_sp_chars']
        
        # numerical_features = [
        #     'unique_count', 'null_count', 'null_percentage', 
        #     'mean', 'median', 'std_dev', 'min', 'max',  # numeric-only features
        #     'avg_length', 'max_length'  # string-only features
        # ]

        # boolean_features = ['contains_numbers', 'contains_sp_chars']  # string-only features
        
        # #Collect all numerical values for normalization
        # all_numerical_values = {}
        # for feat in numerical_features:
        #     values = []
        #     for node, attrs in G.nodes(data=True):
        #         if feat in attrs and attrs[feat] is not None and not pd.isna(attrs[feat]):
        #             values.append(float(attrs[feat]))
        #     if values:
        #         all_numerical_values[feat] = {
        #             'min': min(values), 
        #             'max': max(values), 
        #             'mean': sum(values)/len(values)
        #         }

        # def positional_encoding(value, feat_name, d_model=300):
        #     """Apply sinusoidal positional encoding to normalized numerical values"""
        #     if feat_name not in all_numerical_values:
        #         return [0.0] * d_model
                
        #     # Normalize value to [0, 1] range
        #     feat_stats = all_numerical_values[feat_name]
        #     if feat_stats['max'] == feat_stats['min']:
        #         normalized = 0.5
        #     else:
        #         normalized = (value - feat_stats['min']) / (feat_stats['max'] - feat_stats['min'])
            
        #     # Apply sinusoidal encoding
        #     encoding = []
        #     for i in range(d_model):
        #         if i % 2 == 0:
        #             encoding.append(math.sin(normalized * (10000 ** (i / d_model))))
        #         else:
        #             encoding.append(math.cos(normalized * (10000 ** (i / d_model))))
        #     return encoding

        # FIXED: Process each node within the loop
        for node, attrs in G.nodes(data=True):
            feature_vector = []
            
            # Text embeddings (300d each)
            for feat in text_features:
                value = attrs.get(feat, "unknown")  # Default value for missing text features
                text_embedding = self.st_embedder.encode(str(value)).tolist()
                feature_vector.extend(text_embedding)
            
            # # Numerical embeddings (300d each) - FIXED: Now inside the node loop
            # for feat in numerical_features:
            #     value = attrs.get(feat, None)
            #     if value is not None and not pd.isna(value):
            #         # Positional encoding for numerical data
            #         numerical_embedding = positional_encoding(float(value), feat)
            #     else:
            #         # Default zero embedding for missing numerical features
            #         numerical_embedding = [0.0] * 300
            #     feature_vector.extend(numerical_embedding)
            
            # # Boolean embeddings (300d each) - FIXED: Now inside the node loop
            # for feat in boolean_features:
            #     value = attrs.get(feat, None)
            #     if value is not None:
            #         # Convert boolean to descriptive text and embed
            #         text_repr = f"{feat}_{'yes' if value else 'no'}"
            #         boolean_embedding = self.st_embedder.encode(text_repr).tolist()
            #     else:
            #         # Default embedding for missing boolean features
            #         text_repr = f"{feat}_unknown"
            #         boolean_embedding = self.st_embedder.encode(text_repr).tolist()
            #     feature_vector.extend(boolean_embedding)
            
            # Assign the complete feature vector to the node
            G.nodes[node]['features'] = feature_vector
        
        # Clean up inconsistent attributes
        def clean_attributes(G):
            """Remove all attributes except features and essential metadata"""
            essential_attrs = {'node_type', 'features'}
            for node in G.nodes():
                current_attrs = dict(G.nodes[node])
                for attr in current_attrs:
                    if attr not in essential_attrs:
                        del G.nodes[node][attr]

            # Clean edge attributes
            essential_edge_attrs = {'edge_type', 'edge_features'}
            for u, v in G.edges():
                current_attrs = dict(G[u][v])
                for attr in current_attrs:
                    if attr not in essential_edge_attrs:
                        del G[u][v][attr]
            return G

        def homogenize_edge_features(G):
            """Create consistent edge feature vectors"""
            # Define all possible edge attributes
            edge_types = ['corr', 'has_dtype']
            edge_attrs = ['corr', 'strength', 'dtype', 'weight']
            
            for u, v, attrs in G.edges(data=True):
                edge_vector = []
                
                # Edge type embedding (300d)
                edge_type = attrs.get('edge_type', 'unknown')
                type_embedding = self.st_embedder.encode(f"edge_type_{edge_type}").tolist()
                edge_vector.extend(type_embedding)
                
                # Numerical edge attributes (300d each)
                for attr in ['corr', 'strength', 'weight']:
                    value = attrs.get(attr, 0.0)
                    if value is None or pd.isna(value):
                        value = 0.0
                    # Simple numerical encoding
                    attr_embedding = [float(value)] * 300
                    edge_vector.extend(attr_embedding)
                
                # Categorical edge attributes (300d)
                dtype_value = attrs.get('dtype', 'unknown')
                dtype_embedding = self.st_embedder.encode(f"dtype_{dtype_value}").tolist()
                edge_vector.extend(dtype_embedding)
                
                G[u][v]['edge_features'] = edge_vector
            
            return G

        G = homogenize_edge_features(G)
        G = clean_attributes(G)

        # Update graph metadata
        total_dim = len(text_features) * 300 # + len(numerical_features) * 300 + len(boolean_features) * 300
        G.graph['feature_dim'] = total_dim
        G.graph['feature_composition'] = {
            'text_features': text_features,
            # 'numerical_features': numerical_features, 
            # 'boolean_features': boolean_features,
            'embedding_dims': {'text': 300}
        }

        return G

    




        
