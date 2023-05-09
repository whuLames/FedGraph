# 一些工具包
import torch
from torch_geometric.utils import to_dense_adj
import sys
sys.path.append('/home/lames/code/Gcode')
from FedGraph.data.clientdata import ClientData
from torch_geometric.data import Data
root = '/home/lames/code/data'
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, subgraph, k_hop_subgraph

def get_adj(edge_index, add_loop_self=True):
    """
    Paras:
        edge_index: data.edge_index
        add_loop_self: 是否加入自环
    Returns:
        返回拉普拉斯矩阵
    """
    a = edge_index[0].tolist()
    b = edge_index[1].tolist()
    N = int(edge_index.max().item()) + 1 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index_re = torch.Tensor([b, a]).long()
    edge_index = torch.cat((edge_index, edge_index_re), dim=1).long()
    
    if add_loop_self:
        d = [i for i in range(N)]
        edge_index_dia = torch.Tensor([d, d]).long()# 加入对角信息
        edge_index = torch.cat((edge_index, edge_index_dia), dim=1).long()

    adj = to_dense_adj(edge_index)
    
    adj = adj[0]
    degrees = adj.sum(dim=1)

    D_sqrt_inv = torch.diag(torch.sqrt(1.0 / (degrees)))
    return D_sqrt_inv@adj@D_sqrt_inv

    
def get_client_data(data :Data, process_id, node_lists) -> ClientData:
    """
    初始化 ClientData
    Paras:
        data: PyG data
        process_id : 进程id
        node_lists: 数据的切割信息 [[], []]
    """
    nodes = node_lists[process_id - 1]

    node_feature = {}
    node_label = {}
    train_mask = []
    x = data.x
    y = data.y
    for node in nodes:
        node_feature[node] = x[node]
        node_label[node] = y[node] 
        if data.train_mask[node]:
            train_mask.append(True)
        else :
            train_mask.append(False)

    train_mask = torch.Tensor(train_mask).bool()

    # 存储 node -> client 的对应关系
    node2client_dict = {}
    for i in range(len(node_lists)):
        for node in node_lists[i]:
            node2client_dict[node] = i

    edge_index = data.edge_index

    # 邻接表
    edge_dict = {}
    for i in range(data.num_nodes):
        edge_dict[i] = []

    for i in range(len(edge_index[0])):
            a = edge_index[0][i].item()
            b = edge_index[1][i].item()
            # if a not in edge_dict.keys():
            #     edge_dict[a] = []
            edge_dict[a].append(b)

            # if b not in edge_dict.keys():
            #     edge_dict[b] = []
            edge_dict[b].append(a)

    # 可能存在一个node需要发送到多个client的情况，所以采用dict不合适，需要采用[[], [], []]
    receive_info = [] 
    send_info = []

    for node in nodes:
        for n in edge_dict[node] : # 遍历相邻节点 
            if n not in nodes: # 如果相邻节点不nodes_list中
                s = [node, node2client_dict[n] + 1]
                r = [n, node2client_dict[n] + 1]
                send_info.append(s)
                receive_info.append(r)
                # send_info[node] = node2client_dict[n] + 1 # process_id = client_id + 1
                # receive_info[n] = node2client_dict[n] + 1 

    node_neighbor = {}

    for node in nodes:
        node_neighbor[node] = edge_dict[node]
    
    sub = subgraph(subset=torch.Tensor(nodes), edge_index=edge_index)
    edge_index_new = sub[0]

    for i in range(len(edge_index_new[0])) :
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        new_a = node2client_dict[a]
        new_b = node2client_dict[b]
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)
        
    clientdata = ClientData(node_list=nodes, node_feature=node_feature, node_label=node_label,
                            send_info=send_info, receive_info=receive_info,
                            node_neighbor=node_neighbor, edge_index=edge_index_new,process_id=process_id,
                            train_mask=train_mask)

    print('send_node:  {}  receive_node: {}'.format(len(send_info), len(receive_info)))
    return clientdata


def get_client_data_without_link(data :Data, process_id, node_lists) -> Data:
    """
    得到客户端数据, 但不包含跨节点信息
    
    不包含跨节点信息的话，不需要返回 `ClientData` 数据类型 根据分割的节点, 直接生成Data对象

    """
    nodes = node_lists[process_id - 1]  # 得到节点集
    edge_index = data.edge_index
    
    node2idx = {}
    train_mask = []

    
    # 重新分配
    for i in range(len(nodes)):
        node2idx[nodes[i]] = i
        if data.train_mask[nodes[i]]:
            train_mask.append(True)
        else :
            train_mask.append(False)
             

    # 重新给节点编序号
    graph = subgraph(nodes, edge_index)
    
    # edge_index
    edge_index_new = graph[0]

    x = None
    y = None

    for node in nodes:
        if x == None:
            x = data.x[node].unsqueeze(0)
        else :
            x = torch.cat((x, data.x[node].unsqueeze(0)), dim=0)
        
        if y == None:
            y = data.y[node].unsqueeze(0)
        else :
            y = torch.cat((y, data.y[node].unsqueeze(0)), dim=0)

    for i in range(len(edge_index_new[0])) :
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        new_a = node2idx[a]
        new_b = node2idx[b]
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)
    
    train_mask = torch.Tensor(train_mask).bool()

    graph_data = Data(x=x, edge_index=edge_index_new, y=y, train_mask=train_mask)

    return graph_data

def get_client_data_with_extend(data :Data, process_id, node_lists) -> Data:
    """
    Args:
        
    Returns:
        Data: 根据nodes进行k-hop (一般取 k = 2)拓展后的图数据
    """
    nodes = node_lists[process_id - 1]  # 得到节点集
    edge_index = data.edge_index
    
    subset, edge_index_new, mapping, edge_mask = k_hop_subgraph(
    nodes, 2, edge_index, relabel_nodes=False)

    subset = list(subset) # tensor 转为 list
    #print(subset)
    for i in range(len(subset)):
        subset[i] = subset[i].item()

    nodes = subset

    node2idx = {}
    train_mask = []

    
    # 重新分配

    # 现在的nodes是原本的nodes经过2-hop扩展后的nodes

    # 我们希望只有原先的nodes才能在本地当作训练集，因为只有原本的nodes才能够完美聚合
    for i in range(len(nodes)):
        node2idx[nodes[i]] = i
        if data.train_mask[nodes[i]] and nodes[i] in node_lists[process_id - 1]:
            train_mask.append(True)
        else :
            train_mask.append(False)
    
   
    # print('nodes {}  true {}'.format(len(node_lists[process_id - 1]), train_mask.count(True)))
    # 重新给节点编序号
    # graph = subgraph(nodes, edge_index)
    # edge_index
    # edge_index_new = graph[0]

    # 重新生成节点的 node 的 特征 与 标签
    x = None
    y = None

    for node in nodes:
        if x == None:
            x = data.x[node].unsqueeze(0)
        else :
            x = torch.cat((x, data.x[node].unsqueeze(0)), dim=0)
        
        if y == None:
            y = data.y[node].unsqueeze(0)
        else :
            y = torch.cat((y, data.y[node].unsqueeze(0)), dim=0)

    for i in range(len(edge_index_new[0])) :
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        new_a = node2idx[a]
        new_b = node2idx[b]
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)
    
    train_mask = torch.Tensor(train_mask).bool()

    graph_data = Data(x=x, edge_index=edge_index_new, y=y, train_mask=train_mask)

    return graph_data


def get_Laplace(data : Data, node_lists):
    """
    Laplace = [{}, {}, {}]

    Tensor 转为 int 类型

    """
    Laplance = []
    for i in range(len(node_lists)):
        Laplance.append({})
    
    # 得到  Laplance 矩阵
    L = get_adj(data.edge_index)

    edge_index = data.edge_index
    
    # 邻接表
    edge_dict = {}
    for i in range(data.num_nodes):
        edge_dict[i] = []

    for i in range(len(edge_index[0])):
            a = edge_index[0][i].item()
            b = edge_index[1][i].item()
            edge_dict[a].append(b)
            edge_dict[b].append(a)

    print(data.num_nodes)
    print(len(list(edge_dict)))
    # 基础数据类型为tensor类型
    for i in (range(len(node_lists))):
        for node in node_lists[i]:
            Laplance[i][node] = {}  # dict存储每个节点邻居节点的权重
            for neighbor in edge_dict[node]:  # 存储邻居节点的权重
                Laplance[i][node][neighbor] = L[node][neighbor]
            Laplance[i][node][node] = L[node][node]  # 存储自身节点权重(聚合时也要聚合自身节点)

    # tensor 2 float
    for i in range(len(node_lists)):
        nodes = Laplance[i].keys()
        tmp = {}
        for node in nodes:
            tmp[node] = {}
            for key in Laplance[i][node].keys():
                tmp[node][key] = Laplance[i][node][key].item()
        Laplance[i] = tmp
    
    # 这个传输一整行数据 会有很多0 加大通信开销
    # for i in (range(len(node_lists))):
    #     for node in node_lists[i]:
    #         Laplance[i][node] = L[node]

    return Laplance


# 加入链接预测

def get_data_lp_withoutlink(data, train_data, process_id, nodes_lists):
    """
    data: 测试数据集
    train_data: 训练数据集
    """
    nodes = nodes_lists[process_id - 1]
    edge_index = data.edge_index
    graph = subgraph(nodes, edge_index) # 此时没有重新编序号
    node2idx = {} # 新旧节点的对应
    
    for i in range(len(nodes)):
        node2idx[nodes[i]] = i

    
    edge_index_new = graph[0]

    x = None
    y = None
    # 重建 x y
    for node in nodes:
        if x == None:
            x = data.x[node].unsqueeze(0)
        else :
            x = torch.cat((x, data.x[node].unsqueeze(0)), dim=0)
        
        if y == None:
            y = data.y[node].unsqueeze(0)
        else :
            y = torch.cat((y, data.y[node].unsqueeze(0)), dim=0)

    edge_label_index = train_data.edge_label_index
    new_lebel_index1 = []
    new_label_index2 = []
    mydict = {}
    for i in range(len(edge_label_index[0])):
        a = edge_label_index[0][i].item()
        b = edge_label_index[1][i].item()
        mydict[(a, b)] = 1
        mydict[(b, a)] = 1

    for i in range(len(edge_index_new[0])) :
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        new_a = node2idx[a]
        new_b = node2idx[b]
        #if mydict[(a, b)] == 1 or mydict[(b, a)] == 1:
        if ((a,b) in mydict.keys() and mydict[(a, b)] == 1) or ((b,a) in mydict.keys() and mydict[(b,a)] == 1):
            new_lebel_index1.append(new_a)
            new_label_index2.append(new_b)
            mydict[(a,b)] = 0
            mydict[(b,a)] = 0
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)

    new_edge_label_index = [new_lebel_index1, new_label_index2]
    new_edge_label_index = torch.Tensor(new_edge_label_index).long()
    
    length = len(new_edge_label_index[0])
    
    edge_label = [1] * length
    edge_label = torch.Tensor(edge_label)
    

    clientdata = Data(x=x, edge_index=edge_index_new,y=y,
                      edge_label=edge_label,edge_label_index=new_edge_label_index)

    return clientdata
    

def get_data_lp_withextend(data, train_data, process_id, nodes_lists):
    """
    data: 测试数据集
    train_data: 训练数据集
    """
    nodes = nodes_lists[process_id - 1]
    edge_index = data.edge_index
    # graph = subgraph(nodes, edge_index) # 此时没有重新编序号

    subset, edge_index_new, mapping, edge_mask = k_hop_subgraph(
    nodes, 2, edge_index, relabel_nodes=False)

    node2idx = {} # 新旧节点的对应
    subset = list(subset)
    for i in range(len(subset)):
        subset[i] = subset[i].item()
    nodes = list(subset)
    

    for i in range(len(nodes)):
        node2idx[nodes[i]] = i

    #print(node2idx)
    
    x = None
    y = None
    # 重建 x y
    for node in nodes:
        if x == None:
            x = data.x[node].unsqueeze(0)
        else :
            x = torch.cat((x, data.x[node].unsqueeze(0)), dim=0)
        
        if y == None:
            y = data.y[node].unsqueeze(0)
        else :
            y = torch.cat((y, data.y[node].unsqueeze(0)), dim=0)

    edge_label_index = train_data.edge_label_index
    new_lebel_index1 = []
    new_label_index2 = []
    mydict = {}
    for i in range(len(edge_label_index[0])):
        a = edge_label_index[0][i].item()
        b = edge_label_index[1][i].item()
        mydict[(a, b)] = 1
        mydict[(b, a)] = 1

    for i in range(len(edge_index_new[0])) :
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        new_a = node2idx[a]
        new_b = node2idx[b]
        #if mydict[(a, b)] == 1 or mydict[(b, a)] == 1:
        if ((a,b) in mydict.keys() and mydict[(a, b)] == 1) or ((b,a) in mydict.keys() and mydict[(b,a)] == 1):
            new_lebel_index1.append(new_a)
            new_label_index2.append(new_b)
            mydict[(a,b)] = 0
            mydict[(b,a)] = 0
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)

    new_edge_label_index = [new_lebel_index1, new_label_index2]
    new_edge_label_index = torch.Tensor(new_edge_label_index).long()
    
    length = len(new_edge_label_index[0])
    
    edge_label = [1] * length
    edge_label = torch.Tensor(edge_label)
    

    clientdata = Data(x=x, edge_index=edge_index_new,y=y,
                      edge_label=edge_label,edge_label_index=new_edge_label_index)

    return clientdata









if __name__ == '__main__':
    # edge_index = torch.Tensor([[0, 0, 0], [1, 2, 3]]).long()
    # A = get_adj(edge_index)
    # print(type(A))
    import sys
    A = {}
    B = {}

    A[1] = torch.Tensor(10)

    A[0] = torch.Tensor(10)

    B[0] = [i for i in range(10)]

    B[1] = [i for i in range(10)]

    A[2] = torch.Tensor(10)

    C = {}

    print(sys.getsizeof(A))

    print(sys.getsizeof(B))

    print(sys.getsizeof(C))


