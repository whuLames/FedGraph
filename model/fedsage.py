# 自己写一个Layer 类
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch import Tensor
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from typing import Optional
import sys
import pickle
import time
from torch_geometric.nn import SAGEConv
# 添加环境变量路径
sys.path.append('/home/lames/code/Gcode')
from FedGraph.data.clientdata import ClientData 
from FedGraph.utils.utils import get_adj
import logging
OptTensor = Optional[Tensor]
from mpi4py import MPI
from FedGraph.distributed.Message import Message
from FedGraph.distributed.CommThread import ReceiveThread, ReceiveNodeThread
import queue
import torch
class FedSageLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,layer_num: int,
                 bias: bool = True):
        """
        Paras:
            in_channls: 输入维度
            out_channls: 输出维度
            bias: 是否添加偏置项
            layer_num: 当前的layer
        """
        super().__init__()
        self.in_channls = in_channels
        self.out_channls = out_channels
        self.layer_num = layer_num
        self.lin = Linear(in_channels, out_channels, bias=True,
                          weight_initializer='glorot')
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def pass_info(self, data: ClientData, x, new_comm):
        """
        进行消息的传递
        data: 本地数据集
        x: 本地特征
        new_comm: 用于进程同步
        Returns:
            receive_node_info = {}: key:node_index val:node_feature
        """
        # 真实节点对应本地索引，便于查找 features
        node_index_2_client_index = data.node_index_2_client_index
        process_id = data.process_id

        send_info = data.send_info
        receive_info = data.receive_info
        logging.info('client {} 需要接受信息 {} 个'.format(MPI.COMM_WORLD.Get_rank(),len(receive_info)))
        logging.info('client {} 需要发送信息 {} 个'.format(MPI.COMM_WORLD.Get_rank(),len(send_info)))
        # 存放接受的节点信息
        receive_node_queue = queue.Queue(0)
        comm = MPI.COMM_WORLD
        receive_thread = ReceiveNodeThread(comm=comm, receive_node_info=receive_node_queue, receive_num=len(receive_info))

        receive_thread.start()

        
        # 先 send 
        for i in range(len(send_info)):
            msg = Message(sender_id=process_id, receiver_id=send_info[i][1], type=Message.MSG_TYPE_C2C_NODE_INFO)
            msg.add_content(Message.MSG_ARG_KEY_NODE_FEATURE, x[node_index_2_client_index[send_info[i][0]]])
            msg.add_content(Message.MSG_ARG_KEY_NODE_INDEX, send_info[i][0])
            comm.send(msg, dest=send_info[i][1])

            # logging.info('client {} 发送消息到 {}'.format(MPI.COMM_WORLD.Get_rank(), send_info[i][1]))
            #logging.info('process {} send msg to process {}'.format(process_id, send_info[key]))
       
        # 再 receive
        """
            这里可以是死循环然后判断收到的数量
            也可在在知道数量的情况下进行receive 后者更为合适
        """
        
        receive_node_num = len(receive_info)

        # 持续接收消息
        while  receive_node_queue.qsize() < receive_node_num:
            pass

        
        logging.info('client {} 消息接收结束'.format(MPI.COMM_WORLD.Get_rank()))
        
        # 结束接受线程
        receive_thread.stop_thread()

        receive_thread.join()

        logging.info('client {}  new_group_size {}'.format(MPI.COMM_WORLD.Get_rank(), new_comm.Get_group().Get_size()))

        new_comm.Barrier()
        logging.info('client {} 线程结束'.format(MPI.COMM_WORLD.Get_rank()))
        

        receive_node_info = {}
        while not receive_node_queue.empty():
            msg = receive_node_queue.get()
            key = msg.get_one_content(Message.MSG_ARG_KEY_NODE_INDEX)
            val = msg.get_one_content(Message.MSG_ARG_KEY_NODE_FEATURE)
            receive_node_info[key] = val

        
        return receive_node_info

    def forward(self, data, x, new_comm) -> Tensor:
        """
        Paras:
            data: ClientData
            Laplace: dict : 本地节点的所对应的拉普拉斯矩阵中的行向量 (用于邻居的聚合)
                        # 发送初始化信息时 发送Laplace 矩阵的行向量
            x: 上一层中的特征
        """
        # Laplace_dict == None， 说明是server端在测试数据集
        # if Laplace_dict == None:
        #     x = self.lin(x)
        #     if self.bias is True:
        #         x = x + self.bias
        #     L = get_adj(data.edge_index)
        #     out = L @ x
        #     return out
        
        # 整图的邻接矩阵
        # x = data.x
        if self.layer_num == 0:
            #  此时不用pass_info
            layer = SAGEConv(self.in_channls, self.out_channls)
            x = layer(data.x, data.edge_index)
            return x
        
        # 本地线性变换
        x = self.lin(x)

        if self.bias is True:
            x = x + self.bias

        
        # 消息传递与接收，得到 receive_node_info
        receive_node_info = self.pass_info(data, x, new_comm)


        #邻居聚合
        """
        如何根据本地特征信息 和 receive到的信息进行 gcn聚合
        1. 将邻居节点的信息进行加权聚合
        2. 向server发送请求拉取 归一化拉普拉斯矩阵 中的信息

        q: 归一化拉普拉斯矩阵的参数到底有何意义 (横向 纵向标准化)
        """

        # x : N * feature_num  type: torch.Tensor
        # node_neighbor = data.node_neighbor

        # nodes = data.node_list  # 本地节点集合

        # lenx = len(x[0]) # 得到特征向量的维度

        # logging.info('client {} 特征向量维度为 {}'.format(MPI.COMM_WORLD.Get_rank(), lenx))
        # node_index_2_client_index = data.node_index_2_client_index

        # nodes_feature = None
        # for i in range(len(x)):
        #     # 得到本地 第 i 个 node 的 特征向量
        #     feature = None
        #     # 第 j 个属性
        #     for j in range(lenx):
        #         neighbors = node_neighbor[nodes[i]]
        #         val = torch.Tensor([0])
        #         #logging.info('val1 {}'.format(val))
        #         weights = Laplace_dict[nodes[i]] # 目前weights是list，由于数据传输问题我们希望他是dict，只存储邻接节点的weight，以降低数据传输量
        #         val += weights[nodes[i]] * x[i][j]  # 聚合当前节点信息
        #         #logging.info('val2 {}'.format(val))
        #         for neighbor in neighbors:
        #             # 聚合邻居节点
        #             if neighbor in nodes: # 聚合本地节点
        #                 val += weights[neighbor] * x[node_index_2_client_index[neighbor]][j]
        #             else :  # 聚合其他client发送来的信息
        #                 val += weights[neighbor] * receive_node_info[neighbor][j]
        #         # logging.info('属性 {} 为 {}'.format(j, val))
        #         if feature == None:
        #             feature = val
        #         else :
        #             feature = torch.cat((feature, val), dim=0)
        #     #logging.info('feature shape: {}'.format(feature.shape))
        #     if nodes_feature == None:
        #         nodes_feature = feature.unsqueeze(0)
        #     else :
        #         nodes_feature = torch.cat((nodes_feature, feature.unsqueeze(0)), dim=0)
        

        # out = nodes_feature
        
        # return out
        # node_index_2_client_index = data.node_index_2_client_index
        node_neighbor = data.node_neighbor
        edge_index = data.edge_index
        now_index = len(x)
        for key in receive_node_info.keys():
            # 遍历得到的receive信息
            torch.cat((x, receive_node_info[key].unsqueeze(0)),dim=0)
            for idx in node_neighbor[key]:
                a = torch.Tensor([[now_index], [idx]]).long()
                torch.cat((a, edge_index), dim=1)
                now_index += 1
        layer = SAGEConv(self.in_channls, self.out_channls)
        x = layer(x, edge_index)
        return x
    
class FedSage(nn.Module):
    def __init__(self, INPUT_DIM, OUT_DIM, new_comm) -> None:
        super().__init__()
        
        self.conv1 = FedSageLayer(INPUT_DIM, 32)
        self.conv2 = FedSageLayer(32, OUT_DIM)
        self.new_comm = new_comm #新的通信组
    def forward(self, data, Laplanc_dict=None):
        """
            data: ClientData(Client) or Data(Server)
            Laplance
        """
        x = data.x
        new_comm = self.new_comm
        # logging.info('初始形状为： {}'.format(x.shape))
        x = self.conv1(data, x, new_comm)
        # logging.info('client {} 经过第一层之后的形状为： {}'.format(MPI.COMM_WORLD.Get_rank(), x.shape))
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(data, x, new_comm)
        # logging.info('client {} 经过第二层之后的形状为： {}'.format(MPI.COMM_WORLD.Get_rank(), x.shape))
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(data, x, new_comm, Laplanc_dict)
        # logging.info('client {} 经过第三层之后的形状为： {}'.format(MPI.COMM_WORLD.Get_rank(), x.shape))
        return F.log_softmax(x, dim=1)
    
def test():
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    # root = '/home/lames/code/data'
    # dataset = Planetoid(root, name='Cora')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # data = dataset[0]
    # model = FedGCN(data.num_node_features, dataset.num_classes)
    # # Para = model.state_dict()
    # # print(Para)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # model.train()
    # for i in range(1000):
    #     optimizer.zero_grad()
    #     out = model(data)
    #     # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    #     test()
    a = torch.Tensor([[0, 1, 2], [1, 2, 0]]).long()
    b = torch.Tensor([[0], [2]])
    a = torch.cat((a, b), dim=1)
    print(a)
    
   


    
    
