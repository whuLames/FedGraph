import torch
from Message import Message
import numpy as np
from Aggregator import Aggregator
from ServerManager import ServerManager
from ClientManager import ClientManager
from Trainer import Trainer
from Arguments import Arguments
import logging
import sys
# 添加环境变量路径
sys.path.append('/home/lames/code/Gcode')
from GraphSageAndFedAvg.Code import dividing
from GraphSageAndFedAvg.Code.data import CoraData
from GraphSageAndFedAvg.Code.net import GraphSage

# logging.basicConfig(filename='/home/lames/code/Gcode/FedGraph/log/Cora_Sage1.log',
#                     filemode="a",
#                     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                     level=logging.ERROR)

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

from FedGraph.data.subgraphdata.data_loader import get_data, load_partition_data
from FedGraph.data.data import SimulationClientData
from FedGraph.model.fedgcn import FedGCN
from FedGraph.model.gcn import GCN
from FedGraph.model.graphsage import Sage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.nn.functional as F
from FedGraph.utils.utils import get_Laplace, get_client_data, get_client_data_without_link, get_client_data_with_extend
from mpi4py import MPI
import time
import argparse
IS_CONCENRN_LINK = False

def init_args(dataset):
    # 加入device lr epoch batch_size data num_neighbors_list
    args = {}
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('-n', '--clientnum', type=int)
    parser.add_argument('-r', '--round', type=int)
    parser.add_argument('-e', '--epoch', type=int)
    x = parser.parse_args()
    client_num = x.clientnum
    round = x.round
    epoch = x.epoch
    args["client_num_in_total"] = client_num
    args["client_num_per_round"] = client_num
    args["comm_round"] = round

    # model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
    #               num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
    
    comm = MPI.COMM_WORLD
    group = comm.Get_group()
    new_group = group.Excl([0])
    new_comm = comm.Create(new_group)
    #model = FedGCN(data.num_node_features, int(max(data.y)) + 1, new_comm)
    # model = GCN(data.num_node_features, int(max(data.y)) + 1)
    # model = Sage(data.num_node_features, int(max(data.y)) + 1, new_comm) if IS_CONCENRN_LINK else GCN(
    #     data.num_node_features, int(max(data.y)) + 1
    # )
    model = Sage(data.num_node_features, int(max(data.y)) + 1)
    args['model'] = model    
    args['comm'] = comm
    args['process_id'] = comm.Get_rank()
    args['worker_num'] = comm.Get_size()
    args['device'] =  DEVICE
    args['lr'] = 0.01
    args['batch_size'] = 16
    args['epoch'] = epoch
    # args['num_neighbors_list'] = NUM_NEIGHBORS_LIST
    Args = Arguments(args=args)
    
    return Args

def init_manager(args, data, node_lists):
    """
    args: 模型训练相关参数
    data: 数据data
    node_lists: 节点划分
    """
    
    manager = None
    if args.process_id == 0:
        manager = init_server(args, data, node_lists)
    else :
        manager = init_client(args, data, node_lists)

    return manager

def init_server(args, data, node_lists):
    logging.info('This is server. process_id = {} '.format(args.process_id))
    

    aggregator = Aggregator(
        args=args,
        worker_num=args.worker_num,
        model=args.model,
        data=data,  # Cora  Data  用于数据集测试
    )

    # 得到每个client上的邻接信息
    Laplance = get_Laplace(data, node_lists) if IS_CONCENRN_LINK else None
    
    server_manager = ServerManager(
        args=args,
        comm=args.comm,
        rank=args.process_id,
        size=args.worker_num,
        aggregator=aggregator,
        Laplance=Laplance
    )
    server_manager.send_init_msg()

    return server_manager

    
def init_client(args, data, node_lists):
    # local_sample_num_dict = {}
    # # 遍历dividing 之后的数据, 得到每个client上的数据量
    # for key in local_data_dict.keys():
    #     local_sample_num_dict[key] = len(local_data_dict[key])
    logging.info("client {} 初始化".format(args.process_id - 1))
    clientdata = get_client_data(data, args.process_id, node_lists) if IS_CONCENRN_LINK else get_client_data_without_link(
        data, args.process_id, node_lists
    ) 

    # clientdata = get_client_data_with_extend(data, args.process_id, node_lists) if IS_CONCENRN_LINK else get_client_data_without_link(
    #     data, args.process_id, node_lists
    # ) 
    logging.info("client {} 初始化完成".format(args.process_id - 1))
    trainer = Trainer(
        args=args,
        model=args.model,
        data=clientdata
    )
    
    client_manager = ClientManager(
        args=args,
        comm=args.comm,
        rank=args.process_id,
        size=args.worker_num,
        trainer=trainer
    )
    logging.info("client {} 初始化完成".format(args.process_id - 1))
    logging.info("client {}".format(clientdata.x))
    return client_manager

    


    
if __name__ == "__main__":
    """
        模拟这样一种情况：
        每个client的数据是list, 每次训练过程是对list的每个数 + 某个数
        
        server聚合过程就是取平均值
        
    server: 持有全局data(x, y, adencent list)  test_indexes 测试数据集
    client: 持有local_data_dict 和 全局data(x, y, adencent list)
    
    """
    
    start = time.time()

    logging.info('开始时间 {}'.format(start))

    data = get_data('Cora')

    #torch.set_printoptions(profile="full")

    args = init_args(data)

    x, node_lists = load_partition_data(args.client_num_in_total, data=data)  # 数据切分

    if args.process_id == 0:
        logging.error(x)
    
    manager = init_manager(args, data, node_lists)

    logging.info('准备运行')

    manager.run()

    






























    

    
   


    


