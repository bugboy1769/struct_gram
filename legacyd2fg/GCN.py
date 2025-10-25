import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from sentence_transformers import SentenceTransformer

class TableGCN(nn.Module):
    def __init__(self, input_dim=786, hidden_dim=786, output_dim=786, num_layers=3, dropout=0.1):
        super(TableGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def get_graph_embeddings(self, x, edge_index, batch=None):
        node_embeddings = self.forward(x, edge_index, batch)
        if batch is None:
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
        else:
            graph_embedding = global_mean_pool(x=node_embeddings, batch=batch)
        return graph_embedding

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 2):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        #First Layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        #Hidden Layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        #Output Layer
        self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        #Final Layer without activation
        x = self.convs[-1](x, edge_index)
        return x

def networkx_to_torch_geometric(G):
    
    data = from_networkx(G)

    #Extract Node Features
    features = []
    for node in sorted(G.nodes()):
        features.append(G.nodes[node]['features'])
    #Convert to Tensor
    data.x = torch.tensor(features, dtype = torch.float)

    return data

def gcn_forward_pass(G, embedding_dim = 32):

    data = G

    num_nodes = data.x.shape[0]
    input_dim = data.x.shape[1]
    
    print(f"Graph info:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Input feature dim: {input_dim}")
    print(f"  Output embedding dim: {embedding_dim}")

    #Init Model
    model = GCNEncoder(input_dim=input_dim, hidden_dim=64, output_dim=embedding_dim, num_layers=2)
    #Simple run
    model.eval()

    #Get embeds
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    return embeddings, model

def encode_het_nodes(G, encoder_model = 'all-MiniLM-L6-v2'):
    encoder = SentenceTransformer(encoder_model)
    for node in G.nodes():
        node_text = str(G.nodes[node].get('text', f"node_{node}"))
        embedding = encoder.encode(node_text)
        G.nodes[node]['features'] = embedding.tolist()
    return G