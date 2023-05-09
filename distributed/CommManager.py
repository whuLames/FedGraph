import queue
from CommThread import ReceiveThread
import time
from Message import Message
import logging


class CommManager(object):
    def __init__(self, args, comm, rank, size) :
        """
        Params:
            comm : MPI.COMM_WORLD
            rank : comm.Get_rank  process_id
            size : comm.Get_size()
        """
        self.args = args
        self.comm = comm
        self.rank = rank
        self.size = size
        self.role = "server" if self.rank == 0 else "client"

        self.observers = []

        # 通信队列
        self.send_queue = None
        self.receive_queue = None

        # 初始化通信线程 包括消息接受队列和消息接受线程
        self.init_communication()


        # logging.info('type : {}'.format(type(self.receive_thread)))

        # 通信线程
        # self.receive_thread = None
        # self.send_thread = None

        
    def send_message(self, message):
        dest = message.get_one_content(Message.MSG_ARG_KEY_RECEIVER)
        logging.info('目的地址为 {}'.format(dest))
        logging.info('消息为 {}'.format(message))
        # print("接受者为 {} 号进程".format(dest))
        self.comm.send(message, dest=dest)
        # logging.info('发送成功')

    def init_communication(self):
        # print('初始化{}端通信, process_id = {}'.format(self.role, self.rank))
        if self.role == 'server':
            # 初始化server端的通信
            self.send_queue , self.receive_queue = self.init_server_comm()
        if self.role == 'client':
            # 初始化client端的通信
            self.send_queue , self.receive_queue = self.init_client_comm()
    
    def add_observer(self, observer):
        self.observers.append(observer)

    def init_server_comm(self):
        """
        初始化server端通信

        Returns:
            server_send_queue
            server_receive_queue

        """
        send_queue = queue.Queue(0)
        receive_queue = queue.Queue(0)
        self.receive_thread = ReceiveThread(
            self.comm,
            self.rank,
            self.size,
            receive_queue
        )
        self.receive_thread.start()
        
        return send_queue, receive_queue
    
    def init_client_comm(self):
        """
        初始化client端通信

        Returns:
            client_send_queue
            client_receive_queue
        """
        send_queue = queue.Queue(0)
        receive_queue = queue.Queue(0)
        self.receive_thread = ReceiveThread(
            self.comm,
            self.rank,
            self.size,
            receive_queue
        )
        self.receive_thread.start()
        return send_queue, receive_queue
    
    def handle_receive_message(self):
        """
        处理接收信息
        """
        self.is_Running = True

        while self.is_Running:
            if self.receive_queue.qsize() > 0:
                # print(1)
                message = self.receive_queue.get()
                # logging.info('进程 {} 接收到消息  消息类型: {} '.format(self.rank, message.get_message_type()))
                self.notify(message)
            
            time.sleep(0.001)

    
    def notify(self, message):
        for ob in self.observers:
            ob.receive_message(message)
    
