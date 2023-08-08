import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = ".//models//models.yaml"
with open(config_path, 'r')as f:
    config = yaml.unsafe_load(f)
train_all = config['daily']['train_all']
if train_all:
    train_task = 'all_stocks'
else:
    train_task = 'single_stock'

# 设置随机种子
def random_seed():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 取数据
def get_stock_data():
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    daily_dir = config["daily"]["dir"]
    config_single = config['daily'][train_task]
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
        self.x_len = config['daily'][train_task]['x_length']
        self.y_len = config['daily'][train_task]['y_length']

    def __len__(self):
        if len(self.data) < self.x_len + self.y_len:
            return 0
        return len(self.data)-self.x_len-self.y_len+1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.x_len, 0:6]
        y = self.data[idx+self.x_len:idx+self.x_len+self.y_len, 0:6]
        # if abs(y[0,2]/x[-1,2]-1) > 0.3:
        #     print("idx = ", idx+self.x_len-1)
        #     print("x = ", x[-5:,:])
        #     print("y = ", y)
        #     print("self = ", self.data[idx+self.x_len-3:idx+self.x_len+3])
        return x,y
    
# 构建自定义数据集(all_stock)
class ATAllDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        self.x_len = config['daily'][train_task]['x_length']
        self.y_len = config['daily'][train_task]['y_length']
        length = 0
        begin_lens = []
        begin_lens.append(0)
        for data in self.datas:
            if len(data) >= self.x_len + self.y_len:
                length += len(data)-self.x_len-self.y_len+1
                begin_lens.append(length)
        self.begin_lens = begin_lens
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_idx = find_first_lower(self.begin_lens, idx, 0, len(self.begin_lens)-1)
        data_begin_len = self.datas[data_idx]
        data = self.datas[data_idx]
        idx = idx - data_begin_len
        x = data[idx:idx+self.x_len, 0:6]
        y = data[idx+self.x_len:idx+self.x_len+self.y_len, 0:6]
        return x,y

# 找到第一个大于某个数再数列中的值
def find_first_lower(nums, target, low, high):
    mid = (low + high) // 2
    if low <= mid:
        if nums[mid] > target:
            return find_first_lower(nums, target, low, mid - 1)
        elif nums[mid] <= target:
            return find_first_lower(nums, target, mid, high)
    else:
        return low 

# 划分训练集和测试集
def get_train_and_test_data(data):
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    x_len = config['daily'][train_task]['x_length']
    y_len = config['daily'][train_task]['y_length']
    if_nomalized = config['daily'][train_task]['if_normalized']
    length = len(data)

    train_size = int(length * 0.9)
    test_size = max(length - train_size, x_len + y_len)
    train_size = length - test_size
    train_data = data[:train_size+x_len+y_len-1]
    test_data = data[train_size:]
    if if_nomalized:
        train_data, test_data = get_normalized_data(train_data, test_data, data)
    
    return train_data, test_data

# 归一化 分别归一化训练集和测试集，但是使用的标准还是训练集的标准 原因是什么？
def get_normalized_data(train_data, test_data, data):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    scaler = MinMaxScaler()
    scaler.fit(data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data

# 构建数据加载器
def get_train_and_test_loader(data):
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    batch_size = config['daily'][train_task]['batch_size']
    train_data, test_data = get_train_and_test_data(data)
    train_dataset = ATSingleDataset(train_data)
    test_dataset = ATSingleDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# all_daily train_before_test数据加载器


# 构建数据加载器2:随机分配
def get_train_and_test_loader2(data):
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    batch_size = config['daily'][train_task]['batch_size']
    if_nomalized = config['daily'][train_task]['if_normalized']

    if if_nomalized:
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
    
    dataset = ATSingleDataset(data)
    train_size = int(len(dataset)*0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# all_daily随机分配
def get_train_and_test_loader2_all(datas):
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    batch_size = config['daily'][train_task]['batch_size']
    if_nomalized = config['daily'][train_task]['if_normalized']

    if if_nomalized:
        for data in datas:
            scaler = MinMaxScaler()
            scaler.fit(data)
            data = scaler.transform(data)
    
    dataset = ATAllDataset(datas)
    train_size = int(len(dataset)*0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# GRU作为编码器
class Encoder(nn.Module):
    def __init__(self, hidden_size = 0, num_layers = 0):
        super(Encoder, self).__init__()
        
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily'][train_task]
        kernel_size = config['kernel_size']
        conv_nums = config['conv_nums']
        self.conv_nums = conv_nums
        conv_channels = config['conv_channels']
        conv_channels_times = config['conv_channels_times']
        self.if_pooling = config['if_pooling']
        if hidden_size == 0:
            hidden_size = config['hidden_size']
        if num_layers == 0:
            num_layers = config['num_layers']
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if conv_nums == 0:
            input_size = 6   # change input_size
        else:
            input_size = conv_channels * (conv_channels_times ** (conv_nums-1))
        for i in range(conv_nums):
            if i == 0:
                setattr(self, 'conv'+str(i+1), nn.Conv1d(6, conv_channels, kernel_size=kernel_size, padding=kernel_size//2)) # change input_size
            else:
                setattr(self, 'conv'+str(i+1), nn.Conv1d(conv_channels, conv_channels*conv_channels_times, kernel_size=kernel_size, padding=kernel_size//2))
                conv_channels *= conv_channels_times
            if self.if_pooling:
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
            if self.if_pooling:
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
        config = config['daily'][train_task]
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
    def __init__(self, hidden_size = 0, output_size = 1,  num_layers = 0): # add confidence
        super(Decoder, self).__init__()
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily'][train_task]
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
        self.rnn = nn.GRU(hidden_size*(num_layers + 1), hidden_size, num_layers, batch_first=True) # change last_input
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
        # rnn_input = context     # change last_input
        last_input = last_input.unsqueeze(1)

        # rnn_input = torch.cat((context, hidden.transpose(0,1)), 2)
        # rnn_input = torch.reshape(rnn_input, (rnn_input.shape[0], 1, -1))     # mode1
        
        # rnn_input = torch.cat((rnn_input, last_input), 2)      # mode2

        context = torch.reshape(context, (context.shape[0], 1, -1))     # mode3
        rnn_input = torch.cat((context, last_input), 2)

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
        x_end = x[:, -1, 2]
        y_length = y.shape[1]
        enc_output,enc_hidden = self.encoder(x, check_size=check_size)
        enc_end = enc_output[:, -1, :]
        if check_size == True:
            print("enc_output.shape = ", enc_output.shape)
            print("enc_state.shape = ",enc_hidden.shape)

        dec_hidden = self.decoder.begin_state(enc_hidden)
        # last_input = x_end.unsqueeze(1)   # change last_input
        last_input = enc_end
        if check_size == True:
            print("last_input = ", last_input)
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
                dec_output_tstep = dec_output[:,0].unsqueeze(1)
            elif tstep == y_length-1:
                dec_output_tstep = torch.cat((dec_output_tstep, dec_output), dim=1)
            else:
                dec_output_tstep = torch.cat((dec_output_tstep, dec_output[:,0].unsqueeze(1)), dim=1)
        if check_size == True:
            print("dec_output_tstep = ", dec_output_tstep)
            print("y = ", y[:,:,2])
        trend_loss, mse_loss, up_diff, up_pred = self.criterion(dec_output_tstep, y, x, check_size=check_size)
        # mse_loss = nn.MSELoss()(dec_output_tstep[:,:-2], y[:,:,2])   # add confidence
        # trend_loss = torch.zeros((1))
        return trend_loss, mse_loss, up_diff, up_pred
    
    def criterion(self, dec_output, y, x, check_size = False):
        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily'][train_task]
        wt_up = config['wt_up']
        wt_down = config['wt_down']
        wt_avg = (wt_up+wt_down)/2
        wt_off = abs(wt_up-wt_down)/2
        # 我们更关心涨的，而且需要置信度高的
        output = dec_output[:,0:y.shape[1]]
        # 用relu还是abs还是square？rule不行
        # confidence = F.softmax(torch.square(dec_output[:,y.shape[1]]), dim=0)
        x_end = x[:, -1, 2]
        pred_trend = output[:,-1] - x_end
        pred_trend_sign = torch.sign(pred_trend)
        y_trend = y[:,-1,2] - x_end
        y_trend_sign = torch.sign(y_trend)
        diff_sign = (pred_trend_sign - y_trend_sign)/2
        diff_sign_weight = (torch.ones(diff_sign.shape)*wt_avg + diff_sign*wt_off)* diff_sign
        trend_loss = (pred_trend - y_trend)*diff_sign_weight      # way1
        # trend_loss = torch.abs(diff_sign_weight)                   # way2
        trend_loss = trend_loss                                                 # add confidence here
        mse_loss = torch.mean(torch.square((output-y[:,:,2])), dim=1)  # add confidence here
        up_diff = torch.sum(torch.eq(diff_sign, 1)) + torch.sum(torch.eq(diff_sign, -1))   # correctness change
        up_pred = torch.sum(torch.eq(pred_trend_sign, 1)) + torch.sum(torch.eq(pred_trend_sign, -1))
        if check_size:
            # print("confidence = ", confidence)
            # print("pred_trend = ", pred_trend)
            print("pred_trend_sign = ", pred_trend_sign)
            # print("y_trend = ", y_trend)
            print("y_trend_sign = ", y_trend_sign)
            print("diff_sign = ", diff_sign)
            print("trend_loss = ", trend_loss)
            print("mse_loss = ", mse_loss)
            # print("diff_sign_weight = ", diff_sign_weight)
        # trend_loss = torch.sum(trend_loss*confidence)  # add confidence
        # mse_loss = torch.sum(mse_loss*confidence)   # add confidence
        trend_loss = torch.mean(trend_loss)
        mse_loss = torch.mean(mse_loss)
        # mse_loss = nn.MSELoss()(output, y[:,:,2])
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



"""
if_single_normalized: (变动幅度太小)
  trend_loss =  0.00033726659797442455 mse_loss =  0.0006289897894021124 correctness =  0.5042918454935622 time =  0.5494203567504883
  epoch = 20+20: 
      model20: trend_loss =  0.0002907167664185787 mse_loss =  0.0006192489565970997 correctness =  0.5729613733905579 time =  0.42800140380859375 
      model30: trend_loss =  0.0002892682983656414 mse_loss =  0.000627826782874763 correctness =  0.5729613733905579 time =  0.4690060615539551
      model40: trend_loss =  0.0003037602233234793 mse_loss =  0.0006267268967349082 correctness =  0.5450643776824035 time =  0.6093623638153076
      train_loss(去掉泛化误差): trend_loss =  0.0002292540926055443 mse_loss =  0.00032754416046790175 correctness =  0.5059059059059059 time =  3.875549077987671
      所以是为啥呢？  理解不了
    + conv_num = 1: time-->16
        model20: trend_loss =  0.00043319146304080885 mse_loss =  0.0006252593729489793 correctness =  0.42703862660944203 time =  0.6519975662231445
        (不行，无脑升)
  epoch = 15+15:
      trend_loss =  0.0002990877951863998 mse_loss =  0.0006194724235683679 correctness =  0.5686695278969958 time =  0.44699788093566895 
    + conv_num = 0: time-->27 无脑transformer, 时间比较长
        trend_loss =  0.0002712245118649056 mse_loss =  0.0005735191109124572 correctness =  0.5708154506437768 time =  1.0989899635314941 (这里应该是最佳)
        + x_length = 27: 能否抓取关键信息？

去掉池化层: trend_loss =  0.0002396833917979772 mse_loss =  0.0006104801238204042 correctness =  0.6051502145922747 time =  1.0780136585235596(正确率最高，值接近最佳)


if_normalized:
   epoch = 20+20:
      model20: trend_loss =  0.0004307566540470968 mse_loss =  0.0007441111064205567 correctness =  0.4334763948497854 time =  0.4120001792907715
      model30: trend_loss =  0.0004441321128979325 mse_loss =  0.0007680826548797389 correctness =  0.4291845493562232 time =  0.4009890556335449
      model40: trend_loss =  0.00038059445505496116 mse_loss =  0.0008375487950009604 correctness =  0.502145922746781 time =  0.41099023818969727
      train_loss : trend_loss =  0.00017316875491119998 mse_loss =  0.00039703551184250323 correctness =  0.5565565565565566 time =  3.8409900665283203
    + conv_num = 1:(原来是2)时间9-->17
        model20:trend_loss =  0.00030712548080676544 mse_loss =  0.0007713655417319387 correctness =  0.5708154506437768 time =  0.6459994316101074
        model30:trend_loss =  0.000456053720942388 mse_loss =  0.0009147254168055952 correctness =  0.42703862660944203 time =  0.6359972953796387
        model40:trend_loss =  0.00043708745506592094 mse_loss =  0.0008684455611122151 correctness =  0.42703862660944203 time =  0.601996898651123

 - train_before_test = false:
    - if_normalized:
        conv_num = 2, 20+20, model40: trend_loss =  0.0002990986112207692 mse_loss =  0.00039118630714559305 correctness =  0.5484460694698354 time =  1.4149975776672363
        30+30, model60: trend_loss =  0.0002523203147575259 mse_loss =  0.0003646051748849762 correctness =  0.5338208409506399 time =  1.6169962882995605 
        + convchannels = 18: 再测：后面
        cov_num = 0, 20+20, model40: trend_loss =  0.00026803724257560034 mse_loss =  0.0003838968243346446 correctness =  0.5283363802559415 time =  1.3149983882904053  
        30+30: model60: trend_loss =  0.00024785491324210953 mse_loss =  0.0003717678199690353 correctness =  0.5155393053016453 time =  1.4519994258880615
改变注意力与上一步的连接方式:+ convchannels = 18:
    原: hidden+output,
    hidden: 30+30, model60: trend_loss =  0.00025639754231734615 mse_loss =  0.0003632977928241922 correctness =  0.5411334552102376 time =  1.536001205444336
    hidden+output64: 30+30, model60: trend_loss =  0.00022280434859567322 mse_loss =  0.0004059511540819787 correctness =  0.5667276051188299 time =  1.4069960117340088 (不知道为啥，收敛的很好，但是泛化误差比想象中要大)
    output64: 30+30, model60: trend_loss =  0.0001958307758387592 mse_loss =  0.00032505861276553734 correctness =  0.586837294332724 time =  1.653007984161377 (best)
num_layers = 1: 18s
    hidden:trend_loss =  0.0002159205460985605 mse_loss =  0.00034897817466925416 correctness =  0.5557586837294333
    output64:trend_loss =  0.0002214294703056415 mse_loss =  0.0003462854470449707 correctness =  0.5484460694698354 time =  0.9219973087310791
"""
