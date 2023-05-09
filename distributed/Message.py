# 封装消息传递内容

class Message(object):
    def __init__(self, sender_id=0, receiver_id=0, type=None):
        """
            sender_id: 消息发送者id
            receiver_id: 消息接收者id

            msg_content: 字典 存储message内容
        """
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.type = type
        self.msg_messages = dict()
        self.msg_messages["sender"] = sender_id
        self.msg_messages["receiver"] = receiver_id
    
    def set_msg_messages(self, msg_messages):
        """
            根据参数定义 msg_content
        """
        self.msg_messages = msg_messages
    
    def get_msg_messages(self):
        """
            获取整个 msg_content 元组 即全部内容
        """
        return self.msg_messages
    
    def get_one_content(self, key):
        """
            根据key 得到相应的 content
        """
        if key not in self.msg_messages.keys():
            return None
        return self.msg_messages[key]

    def add_content(self, key, value):
        """
            向msg_content 中添加内容
        """
        self.msg_messages[key] = value

    def get_sender_id(self):
        return self.sender_id
    
    def get_message_type(self):
        return self.type

    def get_receiver_id(self):
        return self.receiver_id

    
    # 服务器端 -> 客户端
    MSG_TYPE_S2C_INIT = 1
    MSG_TYPE_S2C_SEND_MODEL = 2

    # 客户端 -> 服务器端
    MSG_TYPE_C2S_SEND_MODEL = 3

    # 客户端 -> 客户端
    MSG_TYPE_C2C_NODE_INFO = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_NODE_FEATURE = "node_feature"
    MSG_ARG_KEY_NODE_INDEX = "node_index"
    MSG_ARG_KEY_Laplance_DICT = "laplance_dict"

    