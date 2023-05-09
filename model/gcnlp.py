from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch import tensor
from torch import Tensor
import sys
sys.path.append('/home/lames/code')
from Gcode.FedGraph.data.subgraphdata.data_loader import get_data
from Gcode.FedGraph.model.graphsage import Sage
from Gcode.FedGraph.model.gcn import GCN
from sklearn.metrics import f1_score
root = '/home/lames/code/data/subgraphdata'
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

# 数据存储路径为 data/Cora/raw
# class GCN(torch.nn.Module):
#     def __init__(self) :
#         super().__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 32)
#         self.conv2 = GCNConv(32, 16)
#         self.conv3 = GCNConv(16, dataset.num_classes)
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=1)

class GCNLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_label_index):
         return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
             dim=-1
         )  # product of a pair of nodes on each edge
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100):
     for epoch in range(1, n_epochs + 1):
 
         model.train()
         optimizer.zero_grad()
         z = model.encode(train_data)
 
         # sampling training negatives for every training epoch
         neg_edge_index = negative_sampling(
             edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
             num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
 
         edge_label_index = torch.cat(
             [train_data.edge_label_index, neg_edge_index],
             dim=-1,
         )
         edge_label = torch.cat([
             train_data.edge_label,
             train_data.edge_label.new_zeros(neg_edge_index.size(1))
         ], dim=0)
 
         out = model.decode(z, edge_label_index).view(-1)
         loss = criterion(out, edge_label)
         loss.backward()
         optimizer.step()
 
         val_auc = eval_link_predictor(model, val_data)
 
         if epoch % 10 == 0:
             print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
 
     return model 

@torch.no_grad()
def eval_link_predictor(model, data):
    model.eval()
    z = model.encode(data)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def test():
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    # 从github下载数据
    
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # batch_size = 4
    
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = get_data('Cora')
    device = 'cuda'
    data.to(device)
    model = GCNLP(data.num_node_features, 128, 64).to(device)

    split = T.RandomLinkSplit(
         num_val=0.1,
         num_test=0.1,
         is_undirected=True,
         add_negative_train_samples=False,
         neg_sampling_ratio=1.0
    )

    train_data, val_data, test_data = split(data)

    print(train_data)
    print(val_data)
    print(test_data)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100)

    test_auc = eval_link_predictor(model, test_data)

    print(f"Test: {test_auc:.3f}")




















    


    
