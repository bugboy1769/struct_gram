from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class TableGCN(nn.Module):
    def __init__(self, input_dim=786, hidden_dim=786, output_dim=786, num_layers=3, dropout=0.1):
        super(TableGCN, self).__init__()
        self.num_layers=num_layers
        self.dropout=dropout
        self.convs=nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x=conv(x, edge_index)
            if i<self.num_layers-1:
                x=F.relu(x)
                x=F.dropout(x, p=self.dropout, training=self.training)
        return x
    def get_graph_embeddings(self, x, edge_index, batch=None):
        node_embeddings=self.forward(x, edge_index, batch)
        #Pool to graph level
        if batch is None:
            graph_embedding=node_embeddings.mean(dim=0, keepdim=True)
        else:
            graph_embedding=global_mean_pool(x=node_embeddings, batch=batch)
        return graph_embedding
