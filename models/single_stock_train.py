import os
import sys
import torch
import torch.nn as nn
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from models.models import *
# get_stock_data, get_train_and_test_loader, Encoder, Decoder, Attention, random_seed

random_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device = ", device)

if __name__ == '__main__':
    config_path = ".//models//models.yaml"
    print("start data preparation")
    s_data = get_stock_data()
    train_dataloader, test_dataloader = get_train_and_test_loader(s_data)

    print("start training")
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    config = config['daily']['single_stock']
    num_epochs = config['num_epochs']
    num_epochs_decay = config['num_epochs_decay']
    lr = config['learning_rate']
    conv_nums = config['conv_nums']
    conv_channels = config['conv_channels']
    conv_channels_times = config['conv_channels_times']
    model_save_freq = config['model_save_freq']
    if_check_size = config['if_check_size']
    check_size_position = config['check_size_position']
    model_path = config['model_path']
    x_length = config['x_length']
    y_length = config['y_length']
    model_path = os.path.join(model_path, str(x_length)+"to"+str(y_length))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    encoder = Encoder()
    encoder.to(device)
    decoder = Decoder()
    decoder.to(device)
    model = Seq2Seq(encoder, decoder)
    model.to(device)

    train_trend_loss = []
    train_mse_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for i in range(num_epochs+num_epochs_decay):
        check_size = if_check_size * (i == check_size_position)
        for j, (x, y) in enumerate(train_dataloader):
            x = x.float().to(device)
            y = y.float().to(device)
            trend_loss, mse_loss = model(x, y, check_size = check_size)
            loss = trend_loss + mse_loss
            check_size = False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_trend_loss.append(trend_loss.item())
            train_mse_loss.append(mse_loss.item())
        print(train_trend_loss)
        print(train_mse_loss)
        avg_trend_loss = sum(train_trend_loss)/len(train_dataloader)
        avg_mse_loss = sum(train_mse_loss)/len(train_dataloader)
        print("epoch = ", i+1, "trend_loss = ", trend_loss, "mse_loss = ", mse_loss)

        if (i+1) % model_save_freq == 0:
            save_path = os.path.join(model_path, "model"+str(i+1)+".pth")
            torch.save(model.state_dict(), save_path)
            print("model saved")

        if (i+1) > num_epochs:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] - lr/num_epochs_decay