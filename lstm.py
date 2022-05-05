import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class netLSTM(nn.Module):  # 配合predict 函数，因为有out = out[:, -config.predict_len:, :]，所以是输出一段（天）的数据预测结果
    def __init__(self, input_dim, hid_dim, output_dim, num_layer, drop_out, pred_len=1):
        super(netLSTM, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_dim, hid_dim,
                            num_layer, batch_first=True, dropout=drop_out)
        # 全连接至预测的测井曲线
        self.fc2 = nn.Linear(pred_len * hid_dim, int(hid_dim / 2))
        self.fc3 = nn.Linear(int(hid_dim / 2), output_dim * pred_len)
        # self.fc4 = nn.Linear(int(config.hid_dim/2), int(config.hid_dim/2))
        self.bn = nn.BatchNorm1d(int(hid_dim / 2))

    def forward(self, x, hs=None, use_gpu=True, full_output=False):
        batch_size = x.size(0)
        # 不能用batch_size = config.batch_size，因为从第二个epoch开始，
        # dataloder导入的数据batch_size变为了2，如果用config.batch_size,
        # 那么hs维度和输入的x会不匹配。
        if hs is None:
            h = Variable(torch.zeros(self.num_layer, batch_size, self.hid_dim))
            c = Variable(torch.zeros(self.num_layer, batch_size, self.hid_dim))
            hs = (h, c)
        if use_gpu:
            hs = (hs[0].cuda(), hs[1].cuda())
        out, hs_0 = self.lstm(x, hs)  # 输入：batch_size * train_len * input_dim；输出：batch_size * train_len * hid_dim
        if not full_output:
            out = out[:, -self.pred_len:, :]
        out = out.contiguous()
        out = out.view(-1, self.hid_dim * self.pred_len)  # 相当于reshape成(batch_size * train_len) * hid_dim的二维矩阵
        #         print(x.shape)
        #         print(out.shape)
        # normal net
        out = F.relu(self.bn(self.fc2(out)))
        #         out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out, hs_0