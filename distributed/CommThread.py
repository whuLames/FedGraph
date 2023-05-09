from threading import Thread
import traceback
from FedGraph.distributed.Message import Message
import logging
import threading
import time
import mpi4py

mpi4py.MPI.COMM_WORLD

class ReceiveThread(Thread):
    def __init__(self, comm, rank, size, receive_queue):
        """
        Params:
            comm : MPI.COMM_WORLD
            rank : comm.Get_rank  process_id
            size : comm.Get_size()
            receive_queue: 消息接受队列
        """
        super(ReceiveThread, self).__init__()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.receive_queue = receive_queue
        self.role = 'server' if self.rank == 0 else 'client'
        self.lock = threading.Lock()

    def run(self):
        """
        进程 start 时的执行函数
        """
        # print('Receiver Thread Start from {} with process_id {}'.format(self.role, self.rank))
        while True:
            with self.lock:
                try:
                    msg = self.comm.recv()
                    # 获取 messgae type
                    type = msg.get_message_type()
                    # 如果消息类型是 client2client 则不处理(我们希望在client train的过程中，我们可以关闭此接受线程)
                    # if type != Message.MSG_TYPE_C2C_NODE_INFO:
                    #     logging.info('放入消息队列, 消息类型: {}'.format(type))
                    if type == Message.MSG_TYPE_C2C_NODE_INFO:
                        logging.info('receiver the msg that not belong me')
                    self.receive_queue.put(msg)
                except Exception:
                    traceback.print_exc()
                    raise Exception("MPI failed!")
            logging.info('线程上锁')
            time.sleep(1)

    def pause_thread(self):
        """
        暂停线程执行
        """
        self.lock.acquire()

    def resume_thread(self):
        """
        继续线程执行
        """
        self.lock.release()

# class SendThread(Thread):

class ReceiveNodeThread(Thread):
    def __init__(self, comm, receive_node_info, receive_num):
        """
        comm:
        receive_node_info: 消息接收
        """
        super(ReceiveNodeThread, self).__init__()
        self.comm = comm
        self.receive_node_info = receive_node_info
        self.receive_num = receive_num
        self._stop_event = threading.Event()

    def run(self):
        count = 0
        from mpi4py import MPI
        while not self._stop_event.is_set() :
            if count < self.receive_num:
                # 接受消息不够时会才允许继续接受
                count += 1
                # logging.info('client {} 接受第 {} 个 消息'.format(MPI.COMM_WORLD.Get_rank(), count))
                msg = self.comm.recv()
                self.receive_node_info.put(msg)
            time.sleep(0.01)
        logging.info('client {} 线程退出'.format(MPI.COMM_WORLD.Get_rank()))

    def stop_thread(self):
        self._stop_event.set()    
        from mpi4py import MPI
        logging.info('client {} 准备关闭线程'.format(MPI.COMM_WORLD.Get_rank()))