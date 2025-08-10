from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class TableGCN(nn.Module):
    def __init__(self, input_dim = 786, hidden_dim = 786, output_dim = 786):
        super(TableGCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch = None):
        #first layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #second layer
        x = self.conv2(x, edge_index)

        if batch == None:
            graph_embedding = x.mean(dim = 0, keepdim = True)
        else:
            graph_embedding = global_mean_pool(x=x, batch=batch)
        
        return graph_embedding, x
