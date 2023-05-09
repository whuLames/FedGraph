from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch
from torch_geometric.data import Data 

class GATLP(torch.nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM) :
        super().__init__()
        self.conv1 = GATConv(INPUT_DIM, 8, heads = 8, concat = True, dropout = 0.6)
        self.conv2 = GATConv(8*8, OUTPUT_DIM, dropout = 0.6)  

    def encode(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
             dim=-1
         )  # product of a pair of nodes on each edge
    def decode_all(self, z, edge_label_index):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)