import numpy as np
import logging
import sys
sys.path.append('/home/lames/code/Gcode')
from GraphSageAndFedAvg.Code import sampling
import torch
from torch_geometric.data import Data
class Aggregator(object):
    def __init__(self, args, worker_num, model, data, agg_type="FedAvg"):
        """
            完成server端模型聚合等操作
            args: 一些参数信息
            model: 训练模型
            data: 数据集 Data
        """

        self.args = args
        self.worker_num = worker_num
        self.model = model
        self.agg_type = agg_type
        self.data = data
        # self.test_indexes = test_indexes

        # 客户端上传的参数列表
        self.client_model_params = {}
        # 客户端采样列表
        self.client_sample_num = {}
        # 每个客户端是否上传
        self.client_is_upload = {}

        self.acc = []

        for i in range (self.worker_num - 1):
            self.client_is_upload[i] = False

        logging.info('浅浅测试一下未训练的模型')
        self.test()
    def get_global_model_params(self):
        """
            获取模型参数
        """
        # 测试阶段 模型简化为 numpy数组
        # return self.model
        return self.model.state_dict()
    
    def set_global_model_params(self, model_params):
        """
            更新全局参数
        """
        self.model.load_state_dict(model_params)

        # 测试阶段 模型简化为 numpy 数组

    def add_client_result(self, index, client_params, client_sample_nums):
        logging.info('客户端上传参数: 索引 {}'.format(index))
        # logging.info('客户端上传参数:  索引{}  参数 {}'.format(index, client_params))
        self.client_model_params[index] = client_params
        self.client_sample_num[index] = client_sample_nums
        self.client_is_upload[index] = True

    def client_sampling(self, round_now_index, client_num_in_total, client_num_per_round):
        """
            进行邻居节点的采样
            round_now_index: 当前轮次
            client_num_in_total: 全部节点数
            client_num_pre_round: 每轮采样节点数
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [i for i in range(client_num_in_total)]
        else:
            num_client = min(client_num_in_total, client_num_per_round)
            client_indexes = np.random.choice([i for i in range(client_num_in_total)], num_client, replace=False) #不重复选取 num_client
        
        return client_indexes


    def check_whether_all_receive(self):
        """
            判断该迭代轮次中, 是否所有的客户端的训练结果均已上传
        """

        # print(self.client_is_upload)
        logging.info('client_is_upload 内容为 {}'.format(self.client_is_upload))
        for i in range(self.worker_num - 1): # 映射从0开始， 0 - worker_num - 2 即对应所有的client
            """
            ex: worker_num = 4 则client的 process_id 分别为: 1, 2 , 3
            对应的下标为 0, 1, 2 即 rang(self.worker_num - 1)
            """
            
            if not self.client_is_upload[i]:
                return False
        
        for i in range(self.worker_num - 1):
            self.client_is_upload[i] = False
        
        return True;

    def test(self):
        """
            测试聚合后模型的 性能
        """
        logging.info('准备测试')
        model = self.model
        data = self.data
        model.eval()
        pred = model(data).argmax(dim=1) # 只传入data,代表Laplace = None，即可认为传递过程中不进行消息传递
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        self.acc.append(acc)
        logging.info('All Accurary : {}'.format(self.acc))
        logging.info(f'Model Accuracy: {acc:.4f}')
        
    def aggregate(self):
        """
            完成模型的聚合
        """
        logging.info('参数聚合方式为: {}'.format(self.agg_type))
        if(self.agg_type == "FedAvg"):
            return self.FedAvg_Aggregate()
        
    def FedAvg_Aggregate(self):
        """
        客户端上传的参数列表 self.client_model_params
        客户端采样列表      self.client_sample_num
        客户端状态列表      self.client_is_upload

        采用FedAvg 对模型进行聚合
        """ 
        sum = 0  # 总采样数
        for i in range(len(self.client_sample_num)):
            sum += self.client_sample_num[i]
        
        agg_model_params = self.client_model_params[0]
        

        for key in agg_model_params.keys():
            for i in range(len(self.client_model_params)):
                model_params = self.client_model_params[i]
                if i == 0: agg_model_params[key] = model_params[key] * self.client_sample_num[i] / sum
                else : agg_model_params[key] += model_params[key] * self.client_sample_num[i] / sum

        return agg_model_params            




