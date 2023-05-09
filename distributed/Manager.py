from mpi4py import MPI
import logging
from abc import abstractmethod
from CommManager import CommManager
"""
    Manager: ServerManager 和 ClientManager的父类
    抽象为Server 和 Client 的实体        
"""
class Manager(object):
    def __init__(self, args, comm=None, rank=0, size=0):
        """
            args: 参数信息
            comm: MPI.COMM_WORLD
            rank: process_id  comm.Get_rank()
            size: worker_num  comm.Get_size()
        """
        self.args = args
        self.size = size
        self.rank = rank
        self.comm = comm
        # comm_manager 调用init_comm_manager 方法进行初始化操作
        self.comm_manager = None
        # 记录消息类型 与 处理函数 的映射关系
        self.message_handler_dict = {}

        # 初始化 comm_manager 底层通信接口
        self.init_comm_manager() 

    # 消息发送，调用comm_manager 的消息传递接口
    def send_message(self, message):
        self.comm_manager.send_message(message)

    def receive_message(self, message):
        """
        收到消息

            message: Massage 实体
        """
        # print('--------------msg_type : {}'.format(message.get_message_type()))
        # print(self.message_handler_dict)
        callback_func = self.message_handler_dict[message.get_message_type()]
        callback_func(message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, callback_func):
        """
            注册登记回调函数, 通过dict 将 消息类型 和 回调函数进行绑定
            msg_type: 消息类型
            callback_func: 回调函数
        """
        # try:
        #     self.check_msg_type()
        #     self.message_handler_dict[msg_type] = callback_func
        # except KeyError:
        #     raise Exception(
        #         "Error. msg_type = {}. The msg_type is Not Valid"
        #     )

        self.message_handler_dict[msg_type] = callback_func

    def run(self):
        # 注册回调函数
        self.register_message_receive_handlers()
        logging.info('回调函数注册完成.....')
        
        # 处理消息
        self.comm_manager.handle_receive_message() 

    def finish(self):
        print("训练结束")
        MPI.COMM_WORLD.Abort()
    
    def init_comm_manager(self):
        """
            初始化 comm_manager
            comm_manager 管理底层通信
        """
        # MPI 初始化
        self.comm_manager = CommManager(self.args, self.comm, self.rank, self.size)
        # 将ServerManager or ClientManager 加入 观察者列表
        self.comm_manager.add_observer(self)
