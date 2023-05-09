"""
get_data() : 获取SubGraph设置下的图数据

load_partition_data() : 获取切分后的图数据
"""
from torch_geometric.datasets import Planetoid
root = '/home/lames/code/data'
import os.path as osp
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull, Coauthor
from torch_geometric.data import Data
import networkx as nx
import nxmetis
ROOT = osp.dirname(__file__)  # 数据加载根路径
DATA_NAME = ['Cora', 'CiteSeer', 'PubMed']

def get_data(name):
    """
    Subgraph setting 下的数据集, 只需要制定数据集名字即可返回数据
    
    Return :
        dataset
    """
    # if name not in DATA_NAME :
    #     raise Exception("没有此数据集, 请仔细检查输入的数据集名称")
    
    import logging

    if name in DATA_NAME:
        dataset = Planetoid(root=ROOT, name=name)
    elif name == "DBLP":
        dataset = CitationFull(root=ROOT, name=name)
    elif name == 'CS':
        dataset = Coauthor(root=ROOT, name=name)
    elif name == 'Physics':
        dataset = Coauthor(root=ROOT, name=name)
        
    data = dataset[0]
    # train_num = len(data.x) * 0.8 if name != 'Cora' else len(data.x) * 0.4
    train_num = len(data.x) * 0.9
    all_train_mask = []
    all_test_mask = []

    for i in range(len(data.x)):
        if i <= train_num:
            all_train_mask.append(True)
            all_test_mask.append(False)
        else:
            all_train_mask.append(False)
            all_test_mask.append(True)
    import torch
    data.train_mask = torch.Tensor(all_train_mask).bool()
    data.test_mask = torch.Tensor(all_test_mask).bool()
    return data

def load_partition_data(num_clients: int, data : Data, isImpaired=False) :
    """
    切割图数据集(一整张图切分为多个子图)
    Params:
        num_clients: 客户端数量
        data: 数据集
        isImpaired: 是否考虑损失的link
    Returns:
        nodes_list: ths list of nodes 
    """
    # 将 PyG data 转换为 networkx graph
    G = nx.Graph()
    G.add_nodes_from([i for i in range(data.x.shape[0])])
    edges = np.array(data.edge_index.T, dtype=int)
    G.add_edges_from(edges)
    x, nodes_list = nxmetis.partition(G, num_clients)
    return x, nodes_list

if __name__ == '__main__':
    # dataset = get_data('Cora')
    # data = dataset[0]
    # #拆分的数据集
    # nodes_list = load_partition_data(10, data=dataset)
    # #根据每个
    # print(nodes_list)
    print(ROOT)
    data = get_data('Physics')
    print(data)
    