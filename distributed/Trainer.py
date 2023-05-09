from Message import Message
import logging
from torch import nn
from torch import optim
import numpy as np
import torch
import sys
sys.path.append('/home/lames/code/Gcode')
from GraphSageAndFedAvg.Code import sampling
from FedGraph.data.clientdata import ClientData
from FedGraph.distributed.CommManager import CommManager
import time
class Trainer(object):
    def __init__(self, args, model, data: ClientData):
        """
            data: clientdata
        """
        self.args = args
        # Data = namedtuple('Data', ['x', 'y', 'adjacency_dict']) 
        self.data = data
        self.model = model
        # Simulation下默认所有client都持有完整数据的dict，只取自己需要的即可

        # 本地不需要测试集, 在server端模型聚合之后进行测试即可, 本身关注的即为server端联邦优化后的复杂度
        # self.local_data_dict = local_data_dict
        # self.local_sample_num_dict = local_sample_num_dict

        # 训练数据
        # self.local_train_data = None
        # self.local_sample_num = None
    
    def set_comm_manager(self, val: CommManager):
        self.comm_manager = val


    def train(self, Laplace_dict):
        """
        根据本地模型、以及训练数据进行训练
        Laplance_dict: 本地节点的全局邻接信息,  用于GCN
        Return: 
            model_params: 本地训练之后的模型参数
            local_sample_num : 本地数据数量
        """
        # # 模拟本地训练
        # return self.local_train_data, self.local_sample_num
        # model = self.model
        # Data = namedtuple('Data', ['x', 'y', 'adjacency_dict'])
        # data = self.data
        # device = self.args.device
        # lr = self.args.lr
        # criterion = nn.CrossEntropyLoss().to(device)
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        # epoch = self.args.epoch
        # x = data.x / data.x.sum(1, keepdims=True)
        # train_label = data.y
        # num_neighbors_list = self.args.num_neighbors_list
        # for i in range(epoch):
        #     batch_num, all_batch_index = self.dividing()
        #     for j in range(batch_num):
        #         batch_src_index = all_batch_index[j]
        #         batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(device)
        #         batch_sampling_result = sampling.multihop_sampling(batch_src_index, num_neighbors_list, data.adjacency_dict)
        #         batch_sampling_x = [torch.from_numpy(x[idx]).float().to(device) for idx in batch_sampling_result]
        #         batch_train_logits = model(batch_sampling_x)
        #         loss = criterion(batch_train_logits, batch_src_label)  # 损失函数
        #         # print(batch_train_logits)
        #         # print(batch_src_label)
        #         optimizer.zero_grad()  # 梯度清空
        #         loss.backward()  # 反向传播计算参数的梯度
        #         optimizer.step()  # 使用优化方法进行梯度更新
        #         logging.info("Client: {} Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(self.args.process_id, i + 1, j + 1, loss.item()))
        #         # print("Client: {} Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(self.args.process_id, i + 1, j + 1, loss.item()))
        # return model.state_dict(), self.local_sample_num
        
        """
            本地训练会涉及client - client之间的通信, 我们需要在本地训练之前, 暂停掉本进程下的receiver_thread
            同时挂起 2 s 来等待其他进程均关闭掉receiver_thread
        """
        if Laplace_dict is not None:  # 当Laplace_dict 不为 None时，代表不需要节点之间通信，即不给线程上锁
            self.comm_manager.receive_thread.pause_thread()
            logging.info('进程 {} 关闭接受线程 准备开始训练'.format(self.args.process_id))
            time.sleep(2)

        model = self.model
        data =  self.data
        device = self.args.device
        epoches = self.args.epoch
        batch_size = self.args.batch_size
        lr = self.args.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.NLLLoss()

        for epoch in range(epoches):
            optimizer.zero_grad()
            if Laplace_dict is not None:
                out = model(data, Laplace_dict)
            else :
                out = model(data)
            #logging.info('输出结果为 {} 形状为 {}'.format(out, out.shape))
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            #loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            from mpi4py import MPI
            process_id = MPI.COMM_WORLD.Get_rank()
            logging.info('client {} has completed {} epoch '.format(process_id - 1, epoch))
        
        if Laplace_dict is not None:
            self.comm_manager.receive_thread.resume_thread()
            logging.info('进程 {} 开启接受线程 '.format(self.args.process_id))
            time.sleep(2)

        return model.state_dict(), len(self.data.x)
    
    def dividing(self):
        """
            将本地数据按照batch_size 切分
        
        Returns:
            batch_num: batch 数目
            all_batch_index: list, 每个batch的数据索引
        """
        batch_size = self.args.batch_size
        data_index = self.local_train_data
        batch_num = int(len(data_index) / batch_size)
        all_batch_index = []
        for i in range(batch_num - 1):
            batch_index = np.random.choice(data_index, batch_size, replace=False)
            data_index = list(set(data_index) - set(batch_index))
            all_batch_index.append(batch_index)
        
        all_batch_index.append(data_index)
        return batch_num, all_batch_index
    
    def set_model(self, model_params):
        """
            model_params: 模型参数
            更新模型参数
        """
        # np 数组模拟
        # self.model = model_params

        # 真实模型下的模型参数更新
        self.model.load_state_dict(model_params)

        
    def get_model(self):
        """
            获得模型参数
        """
        return self.model.state_dict()
    
    def set_data(self, client_index):
        """
            client_index : 客户端索引
            根据客户端索引更新数据
        """
        # logging.info('根据 {}  索引 {} 数据设置参数'.format(self.local_data_dict, client_index))

        # 看一下图数据的存在形式
        self.local_train_data = self.local_data_dict[client_index]
        self.local_sample_num = self.local_sample_num_dict[client_index]
        