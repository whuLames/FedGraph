import numpy as np
import logging
import sys
sys.path.append('/home/lames/code/Gcode')

class AggregatorSP(object):
    def __init__(self, args, model, data, agg_type='FedAvg') -> None:
        self.args = args
        self.model = model
        self.worker_num = args.worker_num

        # 客户端参数列表
        self.client_model_params = {}

        # 客户端采样列表
        self.client_sample_num = {}

        self.data = data
    
    def test(self):
        logging.info('测试初始化模型性能')
        model = self.model
        data = self.data
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

        logging.info(f'Model Accuracy: {acc:.4f}')
    
    def aggregate(self):
        """
            自定义模型的聚合
            默认为 FedAvg 聚合
        """
        sum = 0 # 总采样数

        for i in range(len(self.client_sample_num)):
            sum += self.client_sample_num[i]
        
        agg_model_params = self.client_model_params[0]
        

        for key in agg_model_params.keys():
            for i in range(len(self.client_model_params)):
                model_params = self.client_model_params[i]
                if i == 0: agg_model_params[key] = model_params[key] * self.client_sample_num[i] / sum
                else : agg_model_params[key] += model_params[key] * self.client_sample_num[i] / sum

        return agg_model_params
