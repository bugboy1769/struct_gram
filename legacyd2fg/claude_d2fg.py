import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.utils import from_networkx

class SimpleDF2G:
    """
    Simple DataFrame to Graph converter that creates homogeneous node representations
    suitable for GCN encoders and downstream LLM integration.
    """
    
    def __init__(self, df: pd.DataFrame, similarity_threshold: float = 0.3):
        self.df = df
        self.similarity_threshold = similarity_threshold
        self.node_features = {}
        self.feature_dim = None
        
    def create_simple_graph(self) -> nx.Graph:
        """
        Creates a simple force-directed graph where:
        - Each column becomes a node
        - Edges connect columns based on statistical similarity
        - All nodes have homogeneous feature vectors
        """
        G = nx.Graph()
        
        # Step 1: Create homogeneous feature vectors for each column
        column_features = self._create_column_features()
        
        # Step 2: Add nodes with homogeneous attributes
        for col_idx, col_name in enumerate(self.df.columns):
            G.add_node(
                col_idx,  # Use integer indices for cleaner processing
                name=col_name,
                features=column_features[col_name].tolist(),
                node_type="column"
            )
        
        # Step 3: Add edges based on feature similarity
        self._add_similarity_edges(G, column_features)
        
        return G
    
    def _create_column_features(self) -> Dict[str, np.ndarray]:
        """
        Create homogeneous feature vectors for each column.
        All columns get the same feature structure.
        """
        features = {}
        all_features = []
        
        for col in self.df.columns:
            col_features = self._extract_column_features(col)
            features[col] = col_features
            all_features.append(col_features)
        
        # Standardize features across all columns
        all_features = np.array(all_features)
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(all_features)
        
        # Update features dictionary with standardized values
        for i, col in enumerate(self.df.columns):
            features[col] = standardized_features[i]
        
        self.feature_dim = standardized_features.shape[1]
        return features
    
    def _extract_column_features(self, col: str) -> np.ndarray:
        """
        Extract a fixed-size feature vector for any column type.
        This ensures homogeneity across all nodes.
        """
        series = self.df[col]
        
        # Basic statistical features (always computed)
        basic_features = [
            len(series),                    # length
            series.isnull().sum(),          # null count
            series.isnull().mean(),         # null percentage
            len(series.unique()),           # unique count
            len(series.unique()) / len(series),  # uniqueness ratio
        ]
        
        # Numeric features (0 if not numeric)
        if pd.api.types.is_numeric_dtype(series):
            numeric_features = [
                series.mean() if not series.empty else 0,
                series.median() if not series.empty else 0,
                series.std() if not series.empty else 0,
                series.min() if not series.empty else 0,
                series.max() if not series.empty else 0,
                series.skew() if not series.empty else 0,
                series.kurt() if not series.empty else 0,
            ]
        else:
            numeric_features = [0] * 7
        
        # String/Categorical features (0 if not string)
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            try:
                str_series = series.astype(str)
                string_features = [
                    str_series.str.len().mean(),        # avg length
                    str_series.str.len().std(),         # length std
                    str_series.str.len().max(),         # max length
                    str_series.str.contains(r'\d').sum(),    # contains numbers
                    str_series.str.contains(r'[A-Z]').sum(), # contains uppercase
                    str_series.str.contains(r'[^a-zA-Z0-9\s]').sum(), # special chars
                ]
                # Handle NaN values
                string_features = [x if not pd.isna(x) else 0 for x in string_features]
            except:
                string_features = [0] * 6
        else:
            string_features = [0] * 6
        
        # Data type encoding (one-hot style)
        dtype_features = [
            1 if pd.api.types.is_numeric_dtype(series) else 0,
            1 if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) else 0,
            1 if pd.api.types.is_datetime64_any_dtype(series) else 0,
            1 if pd.api.types.is_bool_dtype(series) else 0,
        ]
        
        # Combine all features
        all_features = basic_features + numeric_features + string_features + dtype_features
        
        # Ensure no NaN or infinite values
        all_features = [x if np.isfinite(x) else 0 for x in all_features]
        
        return np.array(all_features, dtype=np.float32)
    
    def _add_similarity_edges(self, G: nx.Graph, column_features: Dict[str, np.ndarray]):
        """
        Add edges between columns based on feature similarity.
        """
        columns = list(self.df.columns)
        feature_matrix = np.array([column_features[col] for col in columns])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # Add edges for similar columns
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                similarity = similarity_matrix[i, j]
                if similarity > self.similarity_threshold:
                    G.add_edge(
                        i, j,
                        weight=similarity,
                        edge_type="similarity"
                    )
    
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
            
            # Optionally add position to feature vector
            original_features = G.nodes[node]['features']
            enhanced_features = original_features + [x, y]
            G.nodes[node]['features'] = enhanced_features
        
        # Update feature dimension
        self.feature_dim = len(enhanced_features)
        
        return G
    
    def apply_custom_spring_forces(self, G: nx.Graph, iterations: int = 50, 
                                  spring_strength: float = 1.0, repulsion_strength: float = 1.0) -> nx.Graph:
        """
        Apply custom spring forces for more control over the layout.
        This implements a basic spring-mass system.
        """
        import random
        random.seed(42)
        
        # Initialize random positions
        positions = {}
        velocities = {}
        for node in G.nodes():
            positions[node] = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            velocities[node] = np.array([0.0, 0.0])
        
        dt = 0.01  # Time step
        damping = 0.9  # Velocity damping
        
        for iteration in range(iterations):
            forces = {node: np.array([0.0, 0.0]) for node in G.nodes()}
            
            # Spring forces (attractive) between connected nodes
            for u, v, attrs in G.edges(data=True):
                pos_u, pos_v = positions[u], positions[v]
                distance_vec = pos_v - pos_u
                distance = np.linalg.norm(distance_vec)
                
                if distance > 0:
                    # Spring force proportional to distance
                    spring_force = spring_strength * distance_vec
                    edge_weight = attrs.get('weight', 1.0)
                    spring_force *= edge_weight
                    
                    forces[u] += spring_force
                    forces[v] -= spring_force
            
            # Repulsive forces between all node pairs
            nodes = list(G.nodes())
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    pos_u, pos_v = positions[u], positions[v]
                    distance_vec = pos_v - pos_u
                    distance = np.linalg.norm(distance_vec)
                    
                    if distance > 0:
                        # Coulomb-like repulsive force
                        repulsive_force = repulsion_strength / (distance ** 2)
                        force_direction = distance_vec / distance
                        force_vec = repulsive_force * force_direction
                        
                        forces[u] -= force_vec
                        forces[v] += force_vec
            
            # Update positions using Verlet integration
            for node in G.nodes():
                velocities[node] = velocities[node] * damping + forces[node] * dt
                positions[node] += velocities[node] * dt
        
        # Add final positions to graph
        for node in G.nodes():
            x, y = positions[node]
            G.nodes[node]['pos_x'] = x
            G.nodes[node]['pos_y'] = y
            
            # Add position to feature vector
            original_features = G.nodes[node]['features']
            enhanced_features = original_features + [x, y]
            G.nodes[node]['features'] = enhanced_features
        
        # Update feature dimension
        self.feature_dim = len(enhanced_features)
        
        return G
    
    def to_torch_geometric(self, G: nx.Graph):
        """
        Convert NetworkX graph to PyTorch Geometric format.
        """
        # Ensure all nodes have the 'features' attribute
        for node in G.nodes():
            if 'features' not in G.nodes[node]:
                raise ValueError(f"Node {node} missing 'features' attribute")
        
        # Convert to PyTorch Geometric
        data = from_networkx(G)
        
        # Extract features and convert to tensor
        features = []
        for node in sorted(G.nodes()):
            features.append(G.nodes[node]['features'])
        
        data.x = torch.tensor(features, dtype=torch.float32)
        
        # Add edge weights if they exist
        if G.edges() and 'weight' in next(iter(G.edges(data=True)))[2]:
            edge_weights = []
            for _, _, attrs in G.edges(data=True):
                edge_weights.append(attrs.get('weight', 1.0))
            data.edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        return data
    
    def get_graph_info(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Get information about the created graph.
        """
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'feature_dim': self.feature_dim,
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'avg_clustering': nx.average_clustering(G),
            'node_names': [G.nodes[i]['name'] for i in sorted(G.nodes())],
        }

# Example usage and testing
def test_simple_df2g():
    """
    Test the SimpleDF2G class with a sample DataFrame.
    """
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_col1': np.random.normal(0, 1, 100),
        'numeric_col2': np.random.normal(0, 1, 100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'string_col': [f"text_{i}" for i in range(100)],
        'mixed_col': np.random.choice([1, 'two', 3.0, None], 100)
    })
    
    # Create graph
    converter = SimpleDF2G(df, similarity_threshold=0.2)
    G = converter.create_simple_graph()
    
    print("Original graph (before force-directed layout):")
    info = converter.get_graph_info(G)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Apply force-directed layout
    print(f"\nOriginal feature dimension: {converter.feature_dim}")
    G_with_layout = converter.apply_force_directed_layout(G, iterations=100)
    print(f"Feature dimension after layout: {converter.feature_dim}")
    
    # You can also try the custom spring implementation:
    # G_with_layout = converter.apply_custom_spring_forces(G, iterations=50)
    
    # Show positions
    print("\nNode positions after force-directed layout:")
    for node in sorted(G_with_layout.nodes()):
        node_data = G_with_layout.nodes[node]
        print(f"  {node_data['name']}: ({node_data['pos_x']:.3f}, {node_data['pos_y']:.3f})")
    
    # Convert to PyTorch Geometric
    try:
        data = converter.to_torch_geometric(G_with_layout)
        print(f"\nPyTorch Geometric conversion successful!")
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        if hasattr(data, 'edge_attr'):
            print(f"Edge attributes shape: {data.edge_attr.shape}")
            
        # Show that positions are now included in features
        print(f"\nSample node features (last 2 values are x,y positions):")
        print(f"Node 0 features: {data.x[0].tolist()}")
        
    except Exception as e:
        print(f"Error in conversion: {e}")
    
    return converter, G_with_layout, data

if __name__ == "__main__":
    test_simple_df2g()