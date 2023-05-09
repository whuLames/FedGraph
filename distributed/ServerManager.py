import logging
from Manager import Manager
from Message import Message
import sys
import time
class ServerManager(Manager):
    def __init__(self, args, comm=None, rank=0, size=0, aggregator=None, Laplance=None):
        """
        Laplance: 全局的邻接信息 在考虑缺失的情况下需要server端持有邻接信息
        Laplance: [{}, {}, {}]
        """
        super().__init__(args, comm, rank, size)
        self.args = args
        self.Laplace = Laplance
        self.aggregator = aggregator
        #客户端持有训练轮次来决定什么时候 finish
        # 训练轮次
        self.rounds = args.comm_round
        # 当前训练轮次
        self.args.round_now_index = 0
        
    def run(self):
        """
        执行函数
        """
        super().run()
    

    def send_message(self, message):
        """
            调用comm_manager 发送 message
        """
        self.comm_manager.send_message(message)

    def send_init_msg(self):
        """
            向 采样的 clients 发送初始化信息 准备开始训练
        """
        logging.info('server 开始发送初始化信息')
        # 客户端采样
        # client_indexes = self.aggregator.client_sampling(
        # self.args.round_now_index, 
        # self.args.client_num_in_total,
        # self.args.client_num_per_round)

        global_model_params = self.aggregator.get_global_model_params()
       
        # 向每一个 client 进程
        # logging.info('size : {}'.format(self.size))
        for process_id in range(1, self.size):
            message = Message(
                receiver_id=process_id,
                type=Message.MSG_TYPE_S2C_INIT
            )
            # 在消息中加入模型全局参数，以及client序号
            message.add_content(key=Message.MSG_ARG_KEY_MODEL_PARAMS, value=global_model_params)
            message.add_content(key=Message.MSG_ARG_KEY_CLIENT_INDEX, value=process_id - 1)
            if self.Laplace is not None:
                laplace_dict = self.Laplace[process_id - 1]
                message.add_content(key=Message.MSG_ARG_KEY_Laplance_DICT, value=laplace_dict) 
            else:
                message.add_content(key=Message.MSG_ARG_KEY_Laplance_DICT, value=None)
            logging.info("server 向 {} 号进程发送初始化消息".format(process_id))
            self.send_message(message)
            

    def register_message_receive_handlers(self):
        """
            实现 父类中的抽象方法
            注册服务器端收到客户端消息的回调函数
        """
        self.register_message_receive_handler(Message.MSG_TYPE_C2S_SEND_MODEL, self.handle_message_receive_model_from_client)
        
    def handle_message_receive_model_from_client(self, message):
        """
            回调函数, 收到来自客户端模型时的处理函数
            message: Message类实体
        """
        sender_id = message.get_sender_id()
        client_model_params = message.get_one_content(key=Message.MSG_ARG_KEY_MODEL_PARAMS)
        client_sample_nums = message.get_one_content(key=Message.MSG_ARG_KEY_NUM_SAMPLES)

        # logging.info('接收到来自客户端 {} 的 信息 为 {}'.format(sender_id, client_model_params))
        logging.info('server 接收到来自客户端 {} 的信息'.format(sender_id))
        # 添加客户端训练参数
        self.aggregator.add_client_result(
            index=sender_id - 1, # 映射到从0开始
            client_params=client_model_params,
            client_sample_nums=client_sample_nums
        )
        is_all_receive = self.aggregator.check_whether_all_receive()

        if is_all_receive:
            # 聚合客户端模型参数
            global_model_params = self.aggregator.aggregate()
            self.aggregator.set_global_model_params(global_model_params)

            # 测试全局模型
            self.aggregator.test()
            # 
            #测试结果 self.aggregator.test()
            # logging.info('第 {} 轮的server端聚合结果为 : {}'.format(self.args.round_now_index, global_model_params))
            logging.info('第 {} 轮server端聚合完成'.format(self.args.round_now_index))
            self.aggregator.test()
            self.args.round_now_index += 1
            if self.args.round_now_index == self.rounds :
                end = time.time()
                logging.info('最后一轮聚合结束 训练停止, 结束时间 {}'.format(end))
                max_acc = max(self.aggregator.acc)
                logging.error('client_num: {} round: {} epoch: {}  acc: {}  max_acc: {}'.format(self.args.client_num_in_total,
                                                                                   self.args.comm_round,
                                                                                   self.args.epoch,
                                                                                   self.aggregator.acc, max_acc))
                self.finish()
                return

            client_indexes = self.aggregator.client_sampling(
                self.args.round_now_index, 
                self.args.client_num_in_total,
                self.args.client_num_per_round)
            logging.info('开始发送第 {} 轮次消息'.format(self.args.round_now_index))
            for receive_id in range(1, self.size):
                message = Message(
                    receiver_id=receive_id,
                    type=Message.MSG_TYPE_S2C_SEND_MODEL
                )
                # 添加聚合后的全局参数
                message.add_content(key=Message.MSG_ARG_KEY_MODEL_PARAMS, value=global_model_params)
                message.add_content(key=Message.MSG_ARG_KEY_CLIENT_INDEX, value=client_indexes[receive_id - 1])

                self.send_message(message)
                

    

            

