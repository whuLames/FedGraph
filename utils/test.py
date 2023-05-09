import sys
sys.path.append('/home/lames/code/Gcode')
from FedGraph.data.subgraphdata.data_loader import get_data, load_partition_data
from mpi4py import MPI
from utils import get_client_data, get_adj
import torch

if __name__ == '__main__':
    dataset = get_data('Cora')
    data = dataset[0]
    print(data.y[data.train_mask])
    A = torch.Tensor(1)
    B = torch.Tensor(1)
    B = torch.cat((A, B), dim=0)
    print(B)
    # # 数据切割
    # node_lists = load_partition_data(10, data)
    
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    # #nodes = node_lists[rank - 1]
    # clientdata = get_client_data(data=data, process_id=rank, node_lists=node_lists)
    
    # if rank == 0:
    #     print('this is server')
    # else :
    #     print('the client {} has nodes {}  type {}'.format(rank - 1, clientdata.node_list, type(clientdata.x[0])))
    # edge_index = torch.Tensor([[0, 0, 0], [1, 2, 3]])
    # 0 1 2 在一个client  3在另一个client
    # x = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    # A = get_adj(edge_index)
    # node_neighbor = {}

    # node_neighbor[0] = [1, 2, 3]
    # node_neighbor[1] = [0]
    # node_neighbor[2] = [0]
    
    # nodes = [0, 1, 2]
    
    # node_index_2_client_index = {}

    # node_index_2_client_index[0] = 0
    # node_index_2_client_index[1] = 1
    # node_index_2_client_index[2] = 2

    # nodes_feature = None

    # Laplace_dict = {}
    # Laplace_dict[0] = A[0]
    # Laplace_dict[1] = A[1]
    # Laplace_dict[2] = A[2]
    # receive_node_info = {}
    # receive_node_info[3] = torch.Tensor([3, 3, 3])
    # print(Laplace_dict)
    # lenx = len(x[0]) #特征向量维度
    # print(A)
    # for i in range(len(x)):
    #     # 得到本地 第 i 个 node 的 特征向量
    #     feature = None
    #     # 第 j 个属性
    #     for j in range(lenx):
    #         neighbors = node_neighbor[nodes[i]]
    #         val = torch.zeros(1)
    #         weights = Laplace_dict[nodes[i]]
    #         val += weights[nodes[i]] * x[i][j]  # 聚合当前节点信息
    #         for neighbor in neighbors:
    #             # 聚合邻居节点
    #             if neighbor in nodes: # 聚合本地节点
    #                 val += weights[neighbor] * x[node_index_2_client_index[neighbor]][j]
    #             else :  # 聚合其他client发送来的信息
    #                 val += weights[neighbor] * receive_node_info[neighbor][j]
    #         if feature == None:
    #             feature = val
    #         else :
    #             feature = torch.cat((feature, val), dim=0)
    #         print('第 {} 个属性为 {}'.format(j, val))
    #     print('第 {} 个节点的属性为 {}'.format(i, feature))
    #     if nodes_feature == None:
    #         nodes_feature = feature.unsqueeze(0)
    #     else :
    #         nodes_feature = torch.cat((nodes_feature, feature.unsqueeze(0)), dim=0)
    
    # print(nodes_feature)

