from D2FG import DF2G
from viz import GraphVisualizer
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
from GCN import GCNEncoder, embed_graph, networkx_to_torch_geometric, encode_het_nodes
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

sample_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing'],
        'salary': [75000, 65000, 80000, 60000, 70000],
        'years_experience': [3, 5, 7, 2, 4],
        'location': ['NYC', 'SF', 'NYC', 'LA', 'SF']
    }

df = pd.read_csv('customer.csv')
dfs = df[:10]
    
print("Original DataFrame:")
print(dfs)
print("\n" + "="*50 + "\n")

converter = DF2G(dfs)
graph = converter.create_column_graph()
f_graph = converter.apply_force_directed_layout(graph, iterations=20)
summary = converter.get_graph_summary(graph)

print(f"Graph Summary:\n{summary}")
print("\n" + "="*50 + "\n")
print(f"Sample nodes: {list(graph.nodes())[:5]}")
print(f"Sample edges: {list(graph.edges())[:5]}")
print("\n" + "-"*30 + "\n")


"""
Complete example showing how to visualize graphs created from DataFrames.
"""

# Assuming you have the DataFrameToGraph class from the previous artifact

# Create visualizer
viz = GraphVisualizer(f_graph)

# print("1. Basic Matplotlib Visualization:")
# viz.basic_matplotlib_viz()

print("\n2. Hierarchical Layout:")
viz.hierarchical_viz()

print("\n3. Interactive Plotly Visualization:")
viz.interactive_plotly_viz()

print("\n4. Adjacency Matrix:")
viz.adjacency_matrix_viz()

adj_mat = nx.adjacency_matrix(f_graph)
ftr_mat = nx.attr_matrix(f_graph)
print(adj_mat)
print("\n" + "="*50 + "\n")
print(ftr_mat)

# het_g = encode_het_nodes(graph)
# # torch_graph = networkx_to_torch_geometric(graph)
# print('\n' + "="*50 + '\n')
# print(f"Sample nodes: {list(het_g.nodes())[:5]}")
# print(f"Sample edges: {list(het_g.edges())[:5]}")


