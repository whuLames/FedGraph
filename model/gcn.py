from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from torch_geometric.data import Data 

class GCN(torch.nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM) :
        super().__init__()
        self.conv1 = GCNConv(INPUT_DIM, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, OUTPUT_DIM)
    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,0.5,training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,0.5,training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)