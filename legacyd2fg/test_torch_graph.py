from D2FG import DF2G
from d2fg_mod import DF2G_Mod
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from GCN import GCNEncoder, gcn_forward_pass, networkx_to_torch_geometric, encode_het_nodes
from langchain_ollama import OllamaLLM
from gcn_decoder import T5Decoder, SemanticMLP

local_model = OllamaLLM(model="llama3.2:latest")


df = pd.read_csv('customer.csv')
dfs = df[:15]

print("Original DataFrame:")
print(dfs)
print("\n" + "="*50 + "\n")

converter = DF2G_Mod(dfs)
graph = converter.create_column_graph()

print("\n" + "="*50 + "\n")
# summary = converter.get_graph_summary(graph)
# print(f"Graph Summary:\n{summary}")
print("\n" + "="*50 + "\n")
print(f"Sample nodes: {list(graph.nodes())[:5]}")
print(f"Sample edges: {list(graph.edges())[:5]}")
print("\n" + "-"*30 + "\n")

adj_mat = nx.adjacency_matrix(graph)
ftr_mat = nx.attr_matrix(graph)
print(adj_mat)
print("\n" + "="*50 + "\n")
print(ftr_mat)
print("\n" + "-"*30 + "\n")
# het_g = encode_het_nodes(graph)
torch_graph = networkx_to_torch_geometric(graph)
print(torch_graph.x)
print("\n" + "="*50 + "\n")

#Training!
# model = GCNEncoder(input_dim=torch_graph.x.shape[1], hidden_dim=50, output_dim=100)
embeds, model = gcn_forward_pass(torch_graph)
print("\n" + "="*50 + "\n")
print(embeds)

mid_decoder = T5Decoder()
mid_decoder_output = mid_decoder.forward(embeds)
print("\n" + "="*50 + "\n")
print(mid_decoder_output)

# projection_layer = SemanticMLP()
# projection_layer_output = projection_layer.forward(embeds)
# print("\n" + "="*50 + "\n")

# decoder = T5Decoder()
# decoder_output = decoder.forward(projection_layer_output)
# print(decoder_output)


# response = local_model.invoke(f"Look at these embeddings{decoder_output} and tell me what you understand")
# print(response)
# print(response)
# respone = local_model.invoke("What is the capital of Sierra Leone?")