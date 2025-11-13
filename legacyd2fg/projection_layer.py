import torch.nn as nn

class LLMProjector(nn.Module):
    
    def __init__(self, gcn_dim = 786, llm_dim = 786):
        super(LLMProjector, self).__init__()

        self.proj = nn.Linear(gcn_dim, llm_dim)
    
    def forward(self, graph_embedding):
        return self.proj(graph_embedding)