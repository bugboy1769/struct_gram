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
            col_stats = self._get_col_stats(self.df[col])
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
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
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

        text_features = ['data_type']
        numerical_features = [key for key in feature_keys if key not in text_features]
        boolean_features = ['contains_numbers']

        all_numerical_values = {}
        for feature in numerical_features:
            values = []
            for node, attrs in G.nodes(data=True):
                if feature in attrs and attrs[feature] is not None and not pd.isna(attrs[feature]):
                    values.append(float(attrs[feature]))
            if values:
                all_numerical_values[feature] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                }
        def _positional_encoding(value, feature_name, d_model = 300): #d_model will change to fit the projection layer
            if feature_name not in all_numerical_values:
                return [0.0] * d_model
            
            feature_stats = all_numerical_values[feature_name]
            if feature_stats['max'] == feature_stats['min']:
                normalised = 0.5
            else:
                normalised = (value -feature_stats['min'])/ (feature_stats['max' - feature_stats['min']]) #can use sigmoid etc?
            
            encoding = []
            for i in range(d_model):
                if i % 2 == 0:
                    encoding.append(math.sin(normalised * (10000 ** (i/d_model)))) #reminiscent of pos encoding in transformers
                else:
                    encoding.append(math.cos(normalised * (10000 ** (i/d_model))))
            return encoding
            
        # FIXED: Process each node within the loop
        for node, attrs in G.nodes(data=True):
            feature_vector = []
            
            # Text embeddings (300d each)
            for feat in text_features:
                value = attrs.get(feat, "unknown")  # Default value for missing text features
                text_embedding = self.st_embedder.encode(str(value)).tolist()
                feature_vector.extend(text_embedding)
            
            # Numerical embeddings (300d each) - FIXED: Now inside the node loop
            for feat in numerical_features:
                value = attrs.get(feat, None)
                if value is not None and not pd.isna(value):
                    # Positional encoding for numerical data
                    numerical_embedding = _positional_encoding(float(value), feat)
                else:
                    # Default zero embedding for missing numerical features
                    numerical_embedding = [0.0] * 300
                feature_vector.extend(numerical_embedding)
            
            # Boolean embeddings (300d each) - FIXED: Now inside the node loop
            for feat in boolean_features:
                value = attrs.get(feat, None)
                if value is not None:
                    # Convert boolean to descriptive text and embed
                    text_repr = f"{feat}_{'yes' if value else 'no'}"
                    boolean_embedding = self.st_embedder.encode(text_repr).tolist()
                else:
                    # Default embedding for missing boolean features
                    text_repr = f"{feat}_unknown"
                    boolean_embedding = self.st_embedder.encode(text_repr).tolist()
                feature_vector.extend(boolean_embedding)
            
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
        total_dim = len(text_features) * 300 + len(numerical_features) * 300 + len(boolean_features) * 300
        G.graph['feature_dim'] = total_dim
        G.graph['feature_composition'] = {
            'text_features': text_features,
            'numerical_features': numerical_features, 
            'boolean_features': boolean_features,
            'embedding_dims': {'text': 300, 'numerical': 300, 'boolean': 300}
        }

        return G

                            
