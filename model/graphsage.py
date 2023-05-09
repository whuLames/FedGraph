import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
import torch.optim as optim
from torch_geometric.utils import subgraph, k_hop_subgraph
ROOT = '/home/lames/code/Gcode/FedGraph/data/subgraphdata'
sys.path.append('/home/lames/code/Gcode')
from FedGraph.utils.utils import get_client_data_without_link, get_client_data_with_extend
from FedGraph.data.subgraphdata.data_loader import load_partition_data

class Sage(nn.Module):
    def __init__(self, INPUT_DIM, OUT_DIM) -> None:
        super(Sage, self).__init__()
        self.conv1 = SAGEConv(INPUT_DIM, 64, neighborhood_size=20)
        self.conv2 = SAGEConv(64, OUT_DIM, neighborhood_size=20)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x
    
if __name__ == '__main__':
    dataset = Planetoid(root=ROOT, name='Cora')
    data = dataset[0]
    # model = Sage(data.num_node_features, dataset.num_classes)
    
    _, node_lists = load_partition_data(5, data)

    data1 = get_client_data_without_link(data, 1, node_lists)

    data2 = get_client_data_with_extend(data, 1, node_lists)

    print(data1)

    print(data2)