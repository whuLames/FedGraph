import torch
import logging
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
device = 'cuda'
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__
        self.data = data
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data

class TrainerSP(object):
    def __init__(self, args, model, data) -> None:
        self.args = args

        self.model = model

        self.data = data
    
    def set_model(self, model_params):
        """
            model_params: 模型参数
            更新模型参数
        """
        self.model.load_state_dict(model_params)

    def set_data(self, clientdata):
        self.data = clientdata

    def train(self):
        """
        用于自定义设置训练过程
        """
        
        
        model = self.model
        data = self.data

        batch_size = len(data.x)
        epoch = self.args.epoch
        dataset = MyDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # batch_size = self.args.batch_size
        lr = self.args.lr
        # rank = self.args.rank
        data.to(device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.NLLLoss().to(device)
        
        logging.info('开始训练')
        # logging.error(model.device)
        # logging.error(data.device)
        for e in range(epoch):
            # for datas in dataloader:
            # optimizer.zero_grad()
            # datas = datas.to(device)
            optimizer.zero_grad()
            out = model(data)
            # print(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
           

            logging.info('client has completed {} epoch '.format(epoch))

        return model.state_dict(), len(self.data.x)  # 返回模型参数 和 数据长度
    
    def trainLP(self):
        model = self.model
        data = self.data
        epoch = self.args.epoch
        lr = self.args.lr

        data.to(device)
        model.to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        logging.info('开始训练')

        for _ in range(epoch):
            model.train()
            optimizer.zero_grad()
            z = model.encode(data) # data即为train_data

            # sampling training negatives for every training epoch
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index, num_nodes=data.num_nodes,
                num_neg_samples=data.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                data.edge_label,
                data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            out = model.decode(z, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            loss.backward()

            optimizer.step()

        return model.state_dict(), len(self.data.x)