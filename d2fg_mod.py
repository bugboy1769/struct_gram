import pandas as pd
import networkx as nx
import numpy as np
import re as r
from collections import defaultdict
import fasttext as ft
import fasttext.util as ftu
import math
from sentence_transformers import SentenceTransformer


class DF2G_Mod:

    def __init__(self, pd: pd.DataFrame):
        self.df = pd
        self.graph = nx.Graph()
        self.node_atts = {}
        self.edge_atts = {}
        # self.ft_embedder = ft.load_model('cc.en.300.bin')
        self.st_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_column_graph(self) -> nx.Graph:

        G = nx.Graph()

        for col in self.df.columns:
            col_stats = self._get_col_stats(col)
            G.add_node(f"col_{col}", node_type = "column", **col_stats)
        
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            #Correlation Matrix but it really isn't a given, for example rows with employee ids doesn't mean anything
            corr_matrix = self.df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.1:
                            G.add_edge(f"col_{col1}", f"col_{col2}", edge_type = "corr", corr = corr, strenght = abs(corr))
        #Edges that connect same data types, again, can be really redundant, feature engineering is really important and poses question about automating them
        dtype_groups = self.df.dtypes.groupby(self.df.dtypes).groups
        for dtype, cols in dtype_groups.items():
            if len(cols) > 1:
                col_list = list(cols)
                for i in range(len(col_list)):
                    for j in range(i + 1, len(col_list)):
                        col1, col2 = col_list[i], col_list[j]
                        if not G.has_edge(f"col_{col1}", f"col_{col2}"):
                            G.add_edge(f"col_{col1}", f"col_{col2}", edge_type = "dtype", dtype = str(dtype), weight = 0.5)

        G = self._homogenise_features(G)
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
    
    def _normalize_value(self, value: any) -> str:
        """Normalize values for comparison across columns."""
        if isinstance(value, (int, float)):
            return f"num_{value}"
        elif isinstance(value, str):
            # Basic normalization
            return value.lower().strip()
        else:
            return str(value)
    
    def _homogenise_features(self, G:nx.Graph) -> nx.Graph:
        all_features = set()
        for node, attrs in G.nodes(data = True):
            all_features.update(attrs.keys())
        
        exclude_keys = {'node_type', 'column_name'}
        feature_keys = sorted(all_features - exclude_keys)

        text_features = ['data_type', 'unique_count', 'null_count', 'null_percentage', 
            'mean', 'median', 'std_dev', 'min', 'max',  # numeric-only features
            'avg_length', 'max_length', 'contains_numbers', 'contains_sp_chars']

    
        # FIXED: Process each node within the loop
        for node, attrs in G.nodes(data=True):
            feature_vector = []
            
            # Text embeddings (300d each)
            for feat in text_features:
                value = attrs.get(feat, "unknown")  # Default value for missing text features
                text_embedding = self.st_embedder.encode(str(value)).tolist()
                feature_vector.extend(text_embedding)
            
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
                
                
                # Categorical edge attributes (300d)
                dtype_value = attrs.get('dtype', 'unknown')
                dtype_embedding = self.st_embedder.encode(f"dtype_{dtype_value}").tolist()
                edge_vector.extend(dtype_embedding)
                
                G[u][v]['edge_features'] = edge_vector
            
            return G

        G = homogenize_edge_features(G)
        G = clean_attributes(G)

        # Update graph metadata
        total_dim = len(text_features) * 300
        G.graph['feature_dim'] = total_dim
        G.graph['feature_composition'] = {
            'text_features': text_features,
            'embedding_dims': {'text': 300, 'numerical': 300, 'boolean': 300}
        }

        return G

                            
