# 自定义数据类型

import torch
from typing import List, Dict
import sys
sys.path.append('/home/lames/code/Gcode')

class ClientData():
    """
    数据封装类
    """
    def __init__(self,node_list: List, node_feature: Dict, node_label: Dict, send_info:Dict, receive_info: Dict, node_neighbor,edge_index, process_id, **kwargs) -> None:
        """
        Paras:
            node_list: 本地持有节点数据
            node_feature: 本地节点特征 Dict {node : feature}
            send_info: 节点发送信息 Dict {node : dest}
            receive_info: 节点消息接收 Dict {node : source}
            node_neighbor: 节点邻居列表 Dict {node : []}
                每个本地节点的邻居信息(包括在其他节点的邻居信息)
            
            process_id: 进程id (数据与client相绑定)
        """
        self.node_list = node_list
        self.node_feature = node_feature
        self.send_info = send_info
        self.receive_info = receive_info
        self.node_neighbor = node_neighbor
        self.process_id = process_id
        self.node_label = node_label
        self.edge_index = edge_index
        self.node_index_2_client_index, self.x = self.get_x()
        self.y = self.get_y()
        for key, value in kwargs.items():
            setattr(self, key, value)
    def get_y(self):
        y = None
        nodes = self.node_list
        node_label = self.node_label

        for node in nodes:
            if y == None:
                y = node_label[node].unsqueeze(dim=0)
                import logging
                # logging.info('y {}  shape : {}  type {}'.format(y, y.shape, type(y)))
            else :
                y = torch.cat((y, node_label[node].unsqueeze(dim=0)), dim=0)
        
        return y
            
    def get_x(self):
        """
        根据node_feature (dict) -->  矩阵形式的node_feature
        Returns:
            dict: 全局索引 与 本地索引的对应
            x: 本地的数据矩阵
        """
        x = None
        nodes = self.node_list
        node_feature = self.node_feature
        
        node_index_2_client_index = {}

        id = 0
        for node in nodes:
            if x == None:
                x = node_feature[node].unsqueeze(0) # 添加一个维度
            else: # 特征拼接
                x = torch.cat((x, node_feature[node].unsqueeze(0)), dim=0)
            
            node_index_2_client_index[node] = id
            id += 1

        return node_index_2_client_index, x
    
