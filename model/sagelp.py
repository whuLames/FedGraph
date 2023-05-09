import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
class SageLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels1,out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels1)
        self.conv3 = SAGEConv(hidden_channels1, out_channels)

    def encode(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        return self.conv3(x, edge_index)
    
    def decode(self, z, edge_label_index):
         return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
             dim=-1
         )  # product of a pair of nodes on each edge
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()