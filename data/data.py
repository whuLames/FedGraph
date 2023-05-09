# 存储数据类型
from torch_geometric.data import Data

"""
    
"""
from typing import List
class SimulationClientData():
    """
    封装client上的数据
    """
    def __init__(self, data: Data, node_list, isPull=False) -> None:
        """
        params:
            nodes_list: 切分后的node_list [[], [], []]
            data: 全局图数据
            isPull: 是否需要聚合不在本client上的数据
        """
        self.data = data
        self.node_list = node_list

    def get_comm_info(process_id, nodes_list, edge_index):
        """
        Paras:
            client_indexes: [] 每轮选出训练的客户端索引, client_indexes[process_id - 1] 即为process_id进程对应的客户端
            process_id: 进程号
            nodes_list: nodes_list[i] 为第i个客户端所对应的节点集
            edge_list: 边信息
            获取process_id 需要发送的信息
        Return:
            receive_info
            send_info

        """
        nodes = nodes_list[process_id - 1]  # 当前进程持有的节点集合 默认所有进程都参  与训练
        
        # 邻接表存储边信息
        edge_dict = {}
        # 存储node在哪个client上 client + 1 为 process_id
        node2client_dict = {}
        for i in range(len(nodes_list)):
            for node in nodes_list[i]:
                node2client_dict[node] = i
        
        for i in len(edge_index[0]):
            a = edge_index[0][i]
            b = edge_index[1][i]
            if a not in edge_dict.keys():
                edge_dict[a] = []
            a.append(b)

            if b not in edge_dict.keys():
                edge_dict[b] = []
            b.append(a)
        

        # 查找缺失node
        """
            遍历每个node, 如果某个邻居node缺失, 我们需要把自己send出去 同时 receive该node

        """
        # receive and send   dict[node] = process_id
        receive_info = {}
        send_info = {}
        for node in range(nodes):
            for n in edge_dict[node] : # 遍历相邻节点n
                if n not in nodes: # 如果相邻节点不nodes_list中
                    send_info[node] = node2client_dict[n] + 1 # process_id = client_id + 1
                    receive_info[n] = node2client_dict[n] + 1 
        
        return send_info, receive_info
            

