from Manager import Manager
from Message import Message
import logging
class ClientManager(Manager):
    def __init__(self, args, comm=None, rank=0, size=0, trainer=None):
        super().__init__(args, comm, rank, size)
        self.trainer = trainer

        # 给 trainer 传入comm_manager, 用于在需要跨节点信息时给线程上锁
        self.trainer.set_comm_manager(self.comm_manager)

        self.rounds = args.comm_round
        
        self.laplace_dict = None
        # 训练轮次初始化为 0
        self.round_now_index = 0
    

    def run(self):
        super().run()

    def train(self, Laplance_dict=None):
        """
            开始本地训练
        """
        # 得到训练结果
        model_params, local_sample_num = self.trainer.train(Laplance_dict)
        logging.info('{} 号client 本地训练完成 '.format(self.rank))
        # logging.info('{}  号client 本地训练后的结果 model_params: {}  local_sample_num: {}'.format(self.rank, model_params, local_sample_num))
        # 封装Message
        message = Message(sender_id=self.args.process_id, type=Message.MSG_TYPE_C2S_SEND_MODEL)
        message.add_content(key=Message.MSG_ARG_KEY_MODEL_PARAMS, value=model_params)
        message.add_content(key=Message.MSG_ARG_KEY_NUM_SAMPLES, value=local_sample_num)
        self.send_message(message=message)
    
    def register_message_receive_handlers(self):
        """
            注册回调函数：
            1. 收到Server端 初始化信息时的回调函数
            2. 收到Server端 聚合模型时的回调函数
        """

        self.register_message_receive_handler(Message.MSG_TYPE_S2C_INIT, self.handle_message_init)
        self.register_message_receive_handler(Message.MSG_TYPE_S2C_SEND_MODEL, self.handle_message_receive_model_form_server)
        

    def handle_message_init(self, message):
        """
            回调函数: 当收到Server端的初始化信息时执行的操作

            message: 发送的信息
        """
        model_params = message.get_one_content(key=Message.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = message.get_one_content(key=Message.MSG_ARG_KEY_CLIENT_INDEX)
        laplace_dict = message.get_one_content(key=Message.MSG_ARG_KEY_Laplance_DICT)

        self.laplace_dict = laplace_dict

        #logging.info('.........{}........'.format(laplace_dict))
        
        # print('client {}  receive message from server. message is : {}'.format(self.rank, model_params))

        # 根据参数更新训练模型
        self.trainer.set_model(model_params)
        # 根据client_index 更新训练数据
        # 每个client初始化时带有data，无须根据client_index 更新(更改训练策略)
        #self.trainer.set_data(client_index)
        # 开始训练
        self.train(laplace_dict)


    def handle_message_receive_model_form_server(self, message):
        """
            回调函数: 当收到Server端发送的聚合模型时执行的操作
        """
        self.round_now_index += 1

        if self.round_now_index == self.rounds:
            # 训练结束
            self.finish()
        else:
            model_params = message.get_one_content(key=Message.MSG_ARG_KEY_MODEL_PARAMS)
            client_index = message.get_one_content(key=Message.MSG_ARG_KEY_CLIENT_INDEX)
            self.trainer.set_model(model_params)
            # self.trainer.set_data(client_index)
            self.train(self.laplace_dict)
