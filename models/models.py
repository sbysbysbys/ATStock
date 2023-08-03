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
        x = x.permute(0,2,1)
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
        for i in range(self.num_layers):
            if check_size:
                print("hidden[", i, "].shape = ", hidden[i].shape)
            h = hidden[i].repeat(timestep, 1, 1).transpose(0,1)
            # AUTO:不知道这样线性连接会不会出问题,num_layers变一下试试！！
            if check_size:
                print("h_repeat", i, ".shape = ", h.shape)
            attn_energies_layers = self.score(h, encoder_outputs, check_size = check_size).unsqueeze(1)
            # 最后输出的是注意力的权重
            if i == 0:
                attn_energies = attn_energies_layers
            else:
                attn_energies = torch.cat((attn_energies, attn_energies_layers), dim=1)
            if check_size:
                print("attn_energies", i, ".shape = ", attn_energies.shape)
        # 如果是上一个的输出值输入的话就要有下面的这句话
        # attn_energies = torch.sum(attn_energies, dim=1).unsqueeze(1)
        attn_weights = F.softmax(attn_energies, dim=1)
        if check_size:
            print("attn_weights.shape = ", attn_weights.shape)
            print("---------------end attention-----------------")
        return attn_weights
    
    def score(self, hidden, encoder_outputs, check_size = False):
        # AUTO 是否换成加性注意力？在时间上的损耗又是多少
        energy = F.leaky_relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        if check_size:
            print("---------------start score-----------------")
            print("energy.shape = ", energy.shape)
        energy = energy.transpose(1,2)
        if check_size:
            print("energy_transpose.shape = ", energy.shape)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        if check_size:
            print("v.shape = ", v.shape)
        energy = torch.bmm(v, energy)
        if check_size:
            print("energy_bmm.shape = ", energy.shape)
            print("---------------end score-----------------")
        return energy.squeeze(1)
    
# GRU作为解码器
class Decoder(nn.Module):
    # AUTO：选择只输出一个值还是输出全11个值
    def __init__(self, hidden_size = 0, output_size = 2,  num_layers = 0): # add confidence
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
        self.rnn = nn.GRU(hidden_size*(2*num_layers+1), hidden_size, num_layers, batch_first=True) # change last_input
        self.fc = nn.Linear(hidden_size, output_size)
    
    # AUTO:是否要使用上一个的输出值做输入？
    def forward(self, hidden, encoder_outputs, last_input, check_size = False):
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
        rnn_input = torch.cat((context, hidden.transpose(0,1)), 2)
        rnn_input = torch.reshape(rnn_input, (rnn_input.shape[0], 1, -1))
        last_input = last_input.unsqueeze(1)
        rnn_input = torch.cat((rnn_input, last_input), 2)
        if check_size:
            print("rnn_input.shape = ", rnn_input.shape)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        if check_size:
            print("output.shape = ", output.shape)
            print("---------------end decoder---------------")
        return self.fc(output), hidden, output
    
    def begin_state(self, enc_hidden):
        return enc_hidden

# Seq2Seq模型+损失函数  
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, y, check_size = False):
        # x_end = x[:, -1, 2]
        y_length = y.shape[1]
        enc_output,enc_hidden = self.encoder(x, check_size=check_size)
        enc_end = enc_output[:, -1, :]
        if check_size == True:
            print("enc_output.shape = ", enc_output.shape)
            print("enc_state.shape = ",enc_hidden.shape)

        dec_hidden = self.decoder.begin_state(enc_hidden)
        # last_input = x_end.unsqueeze(1)
        last_input = enc_end
        # confidence = torch.ones((last_input.shape[0], 1))/last_input.shape[0]  # add confidence
        # last_input = torch.cat((last_input, confidence), dim=1)  # add confidence
        tstep_check_size = check_size
        for tstep in range(y_length):
            dec_output, dec_hidden, this_output = self.decoder(dec_hidden, enc_output, last_input, check_size=tstep_check_size)
            last_input = this_output # change last_input
            if tstep_check_size == True:
                print("dec_output.size = ", dec_output.shape)
                print("dec_hidden.size = ", dec_hidden.shape)
                tstep_check_size = False
            if tstep == 0:
                dec_output_tstep = dec_output
            elif tstep == y_length-1:
                dec_output_tstep = torch.cat((dec_output_tstep, dec_output), dim=1)
            else:
                dec_output_tstep = torch.cat((dec_output_tstep, dec_output[:,0].unsqueeze(1)), dim=1)
        trend_loss, mse_loss, up_diff, up_pred = self.criterion(dec_output_tstep, y, check_size=check_size)
        # mse_loss = nn.MSELoss()(dec_output_tstep[:,:-2], y[:,:,2])   # add confidence
        # trend_loss = torch.zeros((1))
        return trend_loss, mse_loss, up_diff, up_pred
    
    def criterion(self, dec_output, y, check_size = False):
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily']['single_stock']
        wt_up = config['wt_up']
        wt_down = config['wt_down']
        wt_avg = (wt_up+wt_down)/2
        wt_off = abs(wt_up-wt_down)/2
        # 我们更关心涨的，而且需要置信度高的
        output = dec_output[:,0:y.shape[1]]
        # 用relu还是abs还是square？rule不行
        confidence = F.softmax(torch.square(dec_output[:,y.shape[1]]), dim=0)
        pred_trend = output[:,-1] - output[:,0]
        pred_trend_sign = torch.sign(pred_trend)
        y_trend = y[:,-1,2] - y[:,0,2]
        y_trend_sign = torch.sign(y_trend)
        diff_sign = (pred_trend_sign - y_trend_sign)/2
        diff_sign_weight = (torch.ones(diff_sign.shape)*wt_avg + diff_sign*wt_off)* diff_sign
        trend_loss = (pred_trend - y_trend)*diff_sign_weight
        trend_loss = trend_loss*confidence
        mse_loss = torch.mean(torch.square(output-y[:,:,2]), dim=1)*confidence
        up_diff = torch.sum(torch.eq(diff_sign, 1))
        up_pred = torch.sum(torch.eq(pred_trend_sign, 1))
        if check_size:
            print("confidence = ", confidence)
            print("pred_trend = ", pred_trend)
            print("pred_trend_sign = ", pred_trend_sign)
            print("y_trend = ", y_trend)
            print("y_trend_sign = ", y_trend_sign)
            print("diff_sign = ", diff_sign)
            print("diff_sign_weight = ", diff_sign_weight)
            print("trend_loss = ", trend_loss)
            print("mse_loss = ", mse_loss)
        trend_loss = torch.sum(trend_loss)
        mse_loss = torch.sum(mse_loss)
        if check_size:
            print("trend_loss = ", trend_loss)
            print("mse_loss = ", mse_loss)
            print("up_diff = ", up_diff)
            print("up_pred = ", up_pred)
        return trend_loss, mse_loss, up_diff, up_pred
'''
---------------start encoder-----------------
x.shape =  torch.Size([32, 81, 11])
x.first_permute =  torch.Size([32, 11, 81])
conv 0 , x.shape =  torch.Size([32, 4, 81])
maxpool 0 , x.shape =  torch.Size([32, 4, 27])
conv 1 , x.shape =  torch.Size([32, 8, 27])
maxpool 1 , x.shape =  torch.Size([32, 8, 9])
x.second_permute =  torch.Size([32, 9, 8])
h0.shape =  torch.Size([1, 32, 64])
---------------end encoder-----------------
enc_output.shape =  torch.Size([32, 9, 64])
enc_state.shape =  torch.Size([1, 32, 64])
---------------start decoder---------------
hidden.shape =  torch.Size([1, 32, 64])
encoder_outputs.shape =  torch.Size([32, 9, 64])
---------------start attention-----------------
hidden.shape =  torch.Size([1, 32, 64])
encoder_outputs.shape =  torch.Size([32, 9, 64])
hidden[ 0 ].shape =  torch.Size([32, 64])
h_repeat 0 .shape =  torch.Size([32, 9, 64])
---------------start score-----------------
energy.shape =  torch.Size([32, 9, 64])
energy_transpose.shape =  torch.Size([32, 64, 9])
v.shape =  torch.Size([32, 1, 64])
energy_bmm.shape =  torch.Size([32, 1, 9])
---------------end score-----------------
attn_energies 0 .shape =  torch.Size([32, 1, 9])
attn_weights.shape =  torch.Size([32, 1, 9])
---------------end attention-----------------
attn_weights.shape =  torch.Size([32, 1, 9])
context.shape =  torch.Size([32, 1, 64])
rnn_input.shape =  torch.Size([32, 1, 128])
output.shape =  torch.Size([32, 64])
---------------end decoder---------------
dec_output.size =  torch.Size([32, 1])
dec_hidden.size =  torch.Size([1, 32, 64])
'''
# loss function
    # trend_loss ?
    # mse_loss √
# werther use last_input as a input to decoder GRU?
    # only use hidden?
    # add last_input (this_output:dim64)(lt = 0.48, lm = 0.47) or (dec_output:dim2)
    # only use this_output?