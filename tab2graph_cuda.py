import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
import pandas as pd
import numpy as np
import itertools
from llm_call import generate_batch_without_decode
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from gcn_conv import TableGCN
from projection_layer import LLMProjector
import time
from vllm import LLM, SamplingParams


path = "Project.csv"
df = pd.read_csv(path)
truncated_df = df [:15]
# print(f"Original DataFrame: {truncated_df}")
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
    "m3": "meta-llama/Llama-3.1-8B",
    "m4": "google-t5/t5-small",
    "m5": "meta-llama/Llama-3.2-3B",
}


model = AutoModelForCausalLM.from_pretrained(model_dict["m1"]) #, local_files_only = True)
tokenizer = AutoTokenizer.from_pretrained(model_dict["m1"]) #, local_files_only = True)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

separator_string = "\n ===================================================================================== \n"

sampling_params=SamplingParams(
    temperature=0.2,
    top_p=0.7,
    top_k=50,
    max_tokens=100,
)

# Move LLM initialization inside main block to avoid multiprocessing issues
llm = None

def generate_vllm(prompt):
    return llm.generate(prompt, sampling_params)

def table_to_graph(df):

    G = nx.Graph()
    node_features = []
    max_length = 0
    for col in df.columns:
        col_stats = get_col_stats(df, col)
        G.add_node(f"col_{col}", node_type = "column", **col_stats)
        tensor_features = feature_tokenizer(col_stats)
        print(separator_string + f"Tensor Features Shape: {tensor_features.shape}" + separator_string)
        node_features.append(tensor_features) #a list of tensors of shape [1, seq_len, embedding_dim]
        max_length = max(max_length, tensor_features.shape[1])
    
    padded_features = []
    for features in node_features:
        current_length = features.shape[1]
        if current_length < max_length:
            padding = torch.zeros(1, max_length - current_length, 768, device=device).to(device)
            padded = torch.cat([features, padding], dim = 1)
        else:
            padded = features
        padded_features.append(padded.squeeze(0))

    #Connecting Data Types with Similar DataTypes
    create_and_connect_edges (G, df)
    node_features_tensor = torch.stack(padded_features)   

    return G, node_features_tensor, node_features

def get_col_stats(df, col):

    series = df[col]
    stats = {
        "column_name": f" {col}",
        "column_data_type": f" {str(series.dtype)}",
        "unique_count": f" {len(series.unique())}",
        "null_count:": f" {series.isnull().sum()}"
    }
    #Numeric DataType Stats: However, there are multiple types of numeric data, not every numeric data has statistical meaning, for example customer_id. however, here, we do it anyway, ideally, we need to bifurcate numerical columns further.
    if pd.api.types.is_numeric_dtype(series):
        stats.update(
            {
                "mean": f" {series.mean()}",
                "standard_deviation": f" {series.std()}",
                "median": f" {series.median()}",
                "min": f" {series.min()}",
                "max": f" {series.max()}"
            }
        )
    if pd.api.types.is_string_dtype(series):
        #ideally we want llm calls within here to create proper classification of string type columns, but sigh, we proceed anyway
        digit_pattern = r'\d'
        special_char_pattern = r'[^a-zA-Z0-9\s]'
        stats.update(
            {
                "avg_length_elements": f" {series.str.len().mean()}",
                "max_length_elements": f" {series.str.len().max()}",
                "contains_numbers": f" {series.str.contains(digit_pattern).any()}",
                "contains_spl_chars": f" {series.str.contains(special_char_pattern).any()}"
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
    col_relns = generate_vllm(prompt_list)
    #print(f"Column Relationships: {col_relns}")
    for i, tuple in enumerate(unique_tuples):
        edge_tokens = tokenizer("relationship", padding=True, return_tensors='pt')
        edge_tokens = {k: v.to(device) for k, v in edge_tokens.items()}
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
    all_text = "|| Is a Column Node Connected To ||".join([f"{k}: {v}" for k, v in stat_dict.items()])
    print(separator_string + f"all text:{all_text}" + separator_string)
    tokens = tokenizer(
        all_text,
        padding = True,
        return_tensors = 'pt'
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    #This is essentially my projection layer for now
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(tokens['input_ids'])
        #pooled = embeddings.mean(dim = 1).squeeze(0)
    return embeddings

def nx_to_pyg(graph, node_features):

    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

    edge_list = []
    edge_features_list = []

    for u, v, data in graph.edges(data=True):
        src = node_to_idx[u]
        dst = node_to_idx[v]

        if 'relationship' in data:
            edge_feat = data['relationship']
            #print(edge_feat)

            if edge_feat.dim() == 0: #changed dim to shape
                edge_feat = edge_feat.unsqueeze(0)
            
            edge_feat = edge_feat.to(device)

            with torch.no_grad():
                embeddings = model.get_input_embeddings()(edge_feat.unsqueeze(0)).to(device)
                edge_feat = embeddings.mean(dim=1).squeeze(0)
        else:
            embeddimg_dim = model.get_input_embeddings().embedding_dim
            edge_feat = torch.zeros(embeddimg_dim).to(device)
        
        edge_list.append([src, dst])
        edge_list.append([dst, src])
        edge_features_list.append(edge_feat)
        edge_features_list.append(edge_feat)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features_list) if edge_features_list else None

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data, node_to_idx


if __name__ == '__main__':
    # Initialize vLLM here to avoid multiprocessing issues
    llm = LLM(
        model="meta-llama/Llama-3.2-3B",
        gpu_memory_utilization=0.8,
        disable_log_stats=True
    )

    graph, feature_tensor, test_node = table_to_graph(truncated_df)
    pyg, node_idx = nx_to_pyg(graph, feature_tensor)
    print(separator_string + f"PYG: {pyg}" + separator_string)
    print(separator_string + f"NODE IDX: {node_idx}" + separator_string)
    #print(get_graph_summary(graph))
    print(separator_string + f"Feature Tensor Shape: {feature_tensor.shape}" + separator_string)

    # pos = nx.circular_layout(graph)  # or spring_layout, shell_layout, etc.
    # nx.draw(graph, pos, with_labels=True, node_color='lightblue',
    #         node_size=2000, font_size=8, font_weight='bold',
    #         edge_color='gray', width=2)
    # plt.show()


    #initialise and run GCN
    embeddimg_dim = model.get_input_embeddings().embedding_dim
    gcn_model = TableGCN(input_dim=embeddimg_dim, hidden_dim=embeddimg_dim, output_dim=embeddimg_dim)
    gcn_model = gcn_model.to(device)
    gcn_model.eval()

    pyg.x = pyg.x.to(device)
    pyg.edge_index = pyg.edge_index.to(device)

    with torch.no_grad():
        graph_embedding, node_embeddings = gcn_model(pyg.x, pyg.edge_index)

    print(separator_string+f" {graph_embedding.shape}"+separator_string)
    print(separator_string+f" {node_embeddings.shape}"+separator_string)

    #initialise and run projection layer
    projector = LLMProjector(gcn_dim=embeddimg_dim, llm_dim=embeddimg_dim)
    projector = projector.to(device)
    projector.eval()

    table_contexts = []
    for node in node_embeddings:
        with torch.no_grad():
            projection = projector(node.unsqueeze(0))
            table_contexts.append(projection)
    table_context = torch.cat(table_contexts, dim = 0)

    print(separator_string+f" {table_context.shape}"+separator_string)

    #appending table context to tokenised query

    query = "You are being presented an aggregated sentence which represents the aggregation of a csv table, which was converted to a graph where the nodes are the columns. Your job is to look at these text node features and give me a total summary of the table: "
    query_tokens = tokenizer(query, padding = True, return_tensors = 'pt').to(device)
    query_embeds = model.get_input_embeddings()(query_tokens['input_ids']).to(device)
    print(separator_string+f"QE Shape {query_embeds.shape}"+separator_string)
    #print(f"Query Embeds: {query_embeds}")

    def generate_tokens(inputs):
        with torch.no_grad():
            outputs = model(inputs_embeds = inputs, use_cache = False) # Because feedforward, we don't need pytorch to update or store grads
            logits = outputs.logits #Storing all the logits that the model produces
            #past_kv = outputs.past_key_values
            last_logits = logits[0, -1, :] #Pick out the final logit as we are doing autoregressive generation
            next_token_id = last_logits.argmax() #Applying argmax to sample out the most likely token_id

        return next_token_id

    def generate_output(feature_tensor):
        generated_tokens = []
        duration_s = []
        past_kv = None
        current_embeds = feature_tensor.unsqueeze(0)
        for _ in range(100):
            next_token_id = generate_tokens(current_embeds) # Generate next_token_id
            generated_tokens.append(next_token_id.item())
            next_embed = model.get_input_embeddings()(next_token_id.unsqueeze(0))
            next_embed = next_embed.unsqueeze(0)
            current_embeds = torch.cat([current_embeds, next_embed], dim = 1)

        return generated_tokens, tokenizer.decode(generated_tokens)

    print(separator_string+f" {test_node[0].shape}"+separator_string)
    combined_embeds = torch.concat([query_embeds, test_node[0]], dim = 1).squeeze(0).to(device)
    print(separator_string+f" {combined_embeds.shape}"+separator_string)

    print(separator_string+f" {feature_tensor.squeeze(0).shape}"+separator_string)

    def feature_concat (feature_tensor):
        num_nodes, seq_len, embeddimg_dim = feature_tensor.shape
        concat_features = feature_tensor.reshape(-1, embeddimg_dim)

        return concat_features

    concat_features = feature_concat (feature_tensor)
    print(separator_string+f"concat_features: {concat_features.shape}"+separator_string)
    generated_tokens, generated_words = generate_output(concat_features) #input dim = [seq_len, embedding_dim]
    print(separator_string+f"Gen Word: {generated_words}"+separator_string)