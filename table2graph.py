import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
import itertools
from llm_call import generate_batch_without_decode
import matplotlib.pyplot as plt
from torch_geometric.data import Data

path = "customer.csv"
df = pd.read_csv(path)
truncated_df = df [:15]
print(f"Original DataFrame: {truncated_df}")

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
    "m3": "meta-llama/Llama-3.1-8B",
    "m4": "google-t5/t5-small",
}


model = AutoModelForCausalLM.from_pretrained(model_dict["m2"]) #, local_files_only = True)
tokenizer = AutoTokenizer.from_pretrained(model_dict["m2"]) #, local_files_only = True)
tokenizer.pad_token = tokenizer.eos_token
device = ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

def table_to_graph(df):

    G = nx.Graph()
    node_features = []
    for col in df.columns:
        col_stats = get_col_stats(df, col)
        G.add_node(f"col_{col}", node_type = "column", **col_stats)
        tensor_features = feature_tokenizer(col_stats)
        node_features.append(tensor_features)
    #Connecting Data Types with Similar DataTypes
    create_and_connect_edges (G, df)
    node_features_tensor = torch.stack(node_features)

    return G, node_features_tensor

def get_col_stats(df, col):

    series = df[col]
    stats = {
        "column_name": f"Column Name: {col}",
        "column_data_type": f"Column Data Type: {str(series.dtype)}",
        "unique_count": f"Unique Values in the Column: {len(series.unique())}",
        "null_count:": f"Null Values in the Column: {series.isnull().sum()}"
    }
    #Numeric DataType Stats: However, there are multiple types of numeric data, not every numeric data has statistical meaning, for example customer_id. however, here, we do it anyway, ideally, we need to bifurcate numerical columns further.
    if pd.api.types.is_numeric_dtype(series):
        stats.update(
            {
                "mean": f"Mean of Column: {series.mean()}",
                "standard_deviation": f"Standard Deviation of Column: {series.std()}",
                "median": f"Median of Column: {series.median()}",
                "min": f"Minimum Value in Column: {series.min()}",
                "max": f"Maximum Value in Column: {series.max()}"
            }
        )
    if pd.api.types.is_string_dtype(series):
        #ideally we want llm calls within here to create proper classification of string type columns, but sigh, we proceed anyway
        digit_pattern = r'\d'
        special_char_pattern = r'[^a-zA-Z0-9\s]'
        stats.update(
            {
                "avg_length": f"Average Length of Column Elements: {series.str.len().mean()}",
                "max_length": f"Maximum Length of Column Elements: {series.str.len().max()}",
                "contains_numbers": f"Contains Numbers: {series.str.contains(digit_pattern).any()}",
                "contains_spl_chars": f"Contains Special Characters: {series.str.contains(special_char_pattern).any()}"
            }
        )
    #stats = feature_tokenizer(stats)
    #print(f"Tokenized Stats: {stats}")
    return stats

def create_and_connect_edges(graph: nx.Graph, df):

    column_names = df.columns.tolist()
    unique_tuples = list(itertools.combinations(column_names, 2))
    prompt_list = []
    for tuple in unique_tuples:
        prompt = f"Complete the sentence, the relationship of {str(tuple[0])} and {str(tuple[1])} is "
        prompt_list.append(prompt)
    col_relns = generate_batch_without_decode(prompt_list)
    print(f"Column Relationships: {col_relns}")
    for i, tuple in enumerate(unique_tuples):
        graph.add_edge(f"col_{tuple[0]}", f"col_{tuple[1]}", edge_type = tokenizer("relationship", padding = True, return_tensors = 'pt'), relationship = col_relns[i])
    
def get_graph_summary(G: nx.Graph) -> dict[str, any]:
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

def feature_tokenizer(stat_dict):
    all_text = "|".join([f"{k}: {v}" for k, v in stat_dict.items()])
    tokens = tokenizer(
        all_text,
        padding = 'max_length',
        truncation = True,
        max_length = 128,
        return_tensors = 'pt'
    )
    tokens = tokens.to(device)
    #This is essentially my projection layer for now
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(tokens['input_ids'])
        pooled = embeddings.mean(dim = 1).squeeze(0)
    return pooled

def nx_to_pyg(graph, node_features):

    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

    edge_list = []
    edge_features_list = []

    for u, v, data in graph.edges(data=True):
        src = node_to_idx[u]
        dst = node_to_idx[v]

        if 'relationship' in data:
            edge_feat = data['relationship']

            if edge_feat.dim() == 0:
                edge_feat = edge_feat.unsqueeze(0)
            
            with torch.no_grad():
                embeddings = model.get_input_embeddings()(edge_feat.unsqueeze(0))
                edge_feat = embeddings.mean(dim=1).squeeze(0)
        else:
            edge_feat = torch.zeros(786).to(device)
        
        edge_list.append([src, dst])
        edge_list.append([dst, src])
        edge_features_list.append(edge_feat)
        edge_features_list.append(edge_feat)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features_list) if edge_features_list else None

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data, node_to_idx


graph, feature_tensor = table_to_graph(truncated_df)
pyg, node_idx = nx_to_pyg(graph, feature_tensor)
print(f"Pytorch Geometric Graph: {pyg}")
print(f"Node Index Mapping: {node_idx}")
#print(get_graph_summary(graph))
print(f"Feature Tensor: {feature_tensor}")

pos = nx.circular_layout(graph)  # or spring_layout, shell_layout, etc.
nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=8, font_weight='bold', 
        edge_color='gray', width=2)
plt.show()