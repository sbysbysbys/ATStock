import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from datasets.akshareutils import get_stock_name
import math

# 设置随机种子
def random_seed():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


config_path = ".//models//models.yaml"
# 取数据
def get_stock_data():
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    daily_dir = config["daily"]["dir"]
    config_single = config['daily']['single_stock']
    s_symbol = config_single['symbol']
    s_name = get_stock_name(s_symbol)
    s_path = os.path.join(daily_dir, s_symbol + s_name[:-1] + ".csv")
    s_data = pd.read_csv(s_path, encoding="gbk")
    s_data["日期"] = s_data["日期"].str.replace("/", "-")
    s_data["日期"] = pd.to_datetime(s_data["日期"], format="%Y-%m-%d").dt.weekday
    s_data = s_data.values
    # print(s_data)
    return s_data

# print("start data preparation")
# 构建自定义数据集
class ATSingleDataset(Dataset):
    def __init__(self, data):
        self.data = data
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        self.x_len = config['daily']['single_stock']['x_length']
        self.y_len = config['daily']['single_stock']['y_length']

    def __len__(self):
        if len(self.data) < self.x_len + self.y_len:
            return 0
        return len(self.data)-self.x_len-self.y_len-1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.x_len, 0:11]
        y = self.data[idx+self.x_len:idx+self.x_len+self.y_len, 0:11]
        return x,y

# 划分训练集和测试集
def get_train_and_test_data(data):
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    x_len = config['daily']['single_stock']['x_length']
    y_len = config['daily']['single_stock']['y_length']
    if_nomalized = config['daily']['single_stock']['if_normalized']
    length = len(data)

    train_size = int(length * 0.9)
    test_size = max(length - train_size, x_len + y_len)
    train_size = length - test_size
    train_data = data[:train_size+x_len+y_len-1]
    test_data = data[train_size:]
    if if_nomalized:
        train_data, test_data = get_normalized_data(train_data, test_data)
    # print("train data = ", train_data)
    # print("test data = ", test_data)

    return train_data, test_data

# 归一化 分别归一化训练集和测试集，但是使用的标准还是训练集的标准 原因是什么？
def get_normalized_data(train_data, test_data):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data

# 构建数据加载器
def get_train_and_test_loader(data):
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    batch_size = config['daily']['single_stock']['batch_size']
    train_data, test_data = get_train_and_test_data(data)
    train_dataset = ATSingleDataset(train_data)
    test_dataset = ATSingleDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# GRU作为编码器
class Encoder(nn.Module):
    def __init__(self, hidden_size = 0, num_layers = 0):
        super(Encoder, self).__init__()
        
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily']['single_stock']
        kernel_size = config['kernel_size']
        conv_nums = config['conv_nums']
        self.conv_nums = conv_nums
        conv_channels = config['conv_channels']
        conv_channels_times = config['conv_channels_times']
        if hidden_size == 0:
            hidden_size = config['hidden_size']
        if num_layers == 0:
            num_layers = config['num_layers']
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if conv_nums == 0:
            input_size = 11
        else:
            input_size = conv_channels * (conv_channels_times ** (conv_nums-1))
        for i in range(conv_nums):
            if i == 0:
                setattr(self, 'conv'+str(i+1), nn.Conv1d(11, conv_channels, kernel_size=kernel_size, padding=kernel_size//2))
            else:
                setattr(self, 'conv'+str(i+1), nn.Conv1d(conv_channels, conv_channels*conv_channels_times, kernel_size=kernel_size, padding=kernel_size//2))
                conv_channels *= conv_channels_times
            setattr(self, 'maxpool'+str(i+1), nn.MaxPool1d(kernel_size=kernel_size))
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x, check_size = False):
        if check_size:
            print("---------------start encoder-----------------")
            print("x.shape = ", x.shape)
        x = x.permute(0,2,1).float()
        if check_size:
            print("x.first_permute = ", x.shape)
        for i in range(self.conv_nums):
            x = getattr(self, 'conv'+str(i+1))(x)
            if check_size:
                print("conv", i, ", x.shape = ", x.shape)
            x = getattr(self, 'maxpool'+str(i+1))(x)
            if check_size:
                print("maxpool", i, ", x.shape = ", x.shape)
        x = x.permute(0,2,1)
        if check_size:
            print("x.second_permute = ", x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # AUTO
        torch.nn.init.xavier_normal_(h0)
        if check_size:
            print("h0.shape = ", h0.shape)
            print("---------------end encoder-----------------")
        return self.rnn(x, h0)

# 注意力机制
class Attention(nn.Module):
    def __init__(self, attention_size = 0):
        super(Attention, self).__init__()
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily']['single_stock']
        if attention_size == 0:
            attention_size = config['attention_size']
        self.num_layers = config['num_layers']
        self.attention_size = attention_size
        self.attn = nn.Linear(self.attention_size*2, attention_size)
        # v是用来计算注意力权重的参数
        # !!!这个v是全局的还是在循环中分别学习的？
        self.v = nn.Parameter(torch.rand(attention_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
    
    def forward(self, hidden, encoder_outputs, check_size = False):
        timestep = encoder_outputs.size(1)
        if check_size:
            print("---------------start attention-----------------")
            print("hidden.shape = ", hidden.shape)
            print("encoder_outputs.shape = ", encoder_outputs.shape)
        h = hidden.repeat(timestep, 1, 1).transpose(0,1)
        # AUTO:不知道这样线性连接会不会出问题,num_layers变一下试试！！
        encoder_outputs = encoder_outputs.repeat(1, self.num_layers,1)
        if check_size:
            print("h_repeat.shape = ", h.shape)
            print("encoder_outputs_repeat.shape = ", encoder_outputs.shape)
        attn_energies = self.score(h, encoder_outputs, check_size = check_size)
        # 最后输出的是注意力的权重
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    def score(self, hidden, encoder_outputs, check_size = False):
        # AUTO 是否换成加性注意力？在时间上的损耗又是多少
        energy = F.leaky_relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        if check_size:
            print("energy.shape = ", energy.shape)
        energy = energy.transpose(1,2)
        if check_size:
            print("energy_transpose.shape = ", energy.shape)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        if check_size:
            print("v.shape = ", v.shape)
        energy = torch.bmm(v, energy)
        energy = energy.sum(axis=2, padding=encoder_outputs.size(1))#####################################
        if check_size:
            print("energy_bmm.shape = ", energy.shape)
            print("---------------end attention-----------------")
        return energy.squeeze(1)
    
# GRU作为解码器
class Decoder(nn.Module):
    def __init__(self, hidden_size = 0, output_size = 1,  num_layers = 0):
        super(Decoder, self).__init__()
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily']['single_stock']
        if hidden_size == 0:
            # 这里和encoder保持一致
            hidden_size = config['hidden_size']
        if num_layers == 0:
            # 与encoder保持一致
            num_layers = config['num_layers']
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 这里的attion_size和hidden_size保持一致,因为要连接
        self.attn = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size*2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    # AUTO:是否要使用上一个的输出值做输入？
    def forward(self, hidden, encoder_outputs, check_size = False):
        if check_size:
            print("---------------start decoder---------------")
            print("hidden.shape = ", hidden.shape)
            print("encoder_outputs.shape = ", encoder_outputs.shape)
        attn_weights = self.attn(hidden, encoder_outputs, check_size = check_size)
        if check_size:
            print("attn_weights.shape = ", attn_weights.shape)
        context = attn_weights.bmm(encoder_outputs)
        if check_size:
            print("context.shape = ", context.shape)
        rnn_input = torch.cat((context, hidden), 2)
        if check_size:
            print("rnn_input.shape = ", rnn_input.shape)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        if check_size:
            print("output.shape = ", output.shape)
            print("---------------end decoder---------------")
        return self.fc(output), hidden
    
    def begin_state(self, enc_hidden):
        return enc_hidden
