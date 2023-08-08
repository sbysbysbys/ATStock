import os
import sys
import torch
import torch.nn as nn
import yaml
import time
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from models.models import *
# get_stock_data, get_train_and_test_loader, Encoder, Decoder, Attention, random_seed

random_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device = ", device)

def parser():
    parser = argparse.ArgumentParser(description="single stock daily prediction")
    parser.add_argument("--train", action="store_true", 
                        help="train: train the model")
    parser.add_argument("--test", action="store_true",
                        help="test: test the model")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    config_path = ".//models//models.yaml"
    start_data_preparation = time.time()
    print("start data preparation")
    
    with open(config_path, 'r')as f:
        config = yaml.unsafe_load(f)
    config = config['daily']['single_stock']
    x_length = config['x_length']
    y_length = config['y_length']
    num_epochs = config['num_epochs']
    symbol = config['symbol']
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
    if_trend_loss = config['if_trend_loss']
    if_single_normalized = config['if_single_normalized']
    train_before_test = config['train_before_test']
    model_path = os.path.join(model_path, str(x_length)+"to"+str(y_length))
    model_path = os.path.join(model_path, str(symbol))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    s_data = get_stock_data()
    if train_before_test:
        train_dataloader, test_dataloader = get_train_and_test_loader(s_data)
    else:
        train_dataloader, test_dataloader = get_train_and_test_loader2(s_data)

    encoder = Encoder()
    encoder.to(device)
    decoder = Decoder()
    decoder.to(device)
    model = Seq2Seq(encoder, decoder)
    if device.type == "cuda":
        model = nn.DataParallel(model)
        model.to(device)
    
    end_data_preparation = time.time()
    print("data preparation time = ", end_data_preparation - start_data_preparation)

    if args.train:
        print("start training")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        for i in range(num_epochs+num_epochs_decay):
            start_time = time.time()
            train_trend_loss = []
            train_mse_loss = []
            train_up_diff = []
            train_up_pred = []
            check_size = if_check_size * (i == check_size_position)
            for j, (x, y) in enumerate(train_dataloader):
                x = x.float().to(device)
                y = y.float().to(device)
                if if_single_normalized:
                    x_end = x[:, -1, 2]
                    x_trade_end = x[:, -1, 5]
                    x_y = torch.cat((x, y), dim=1)
                    x_y[:, :, 1:5] = x_y[:, :, 1:5] - x_end.unsqueeze(1).unsqueeze(2).repeat(1, x_y.shape[1], 4)
                    x_y[:, :, 5] = x_y[:, :, 5] - x_end.unsqueeze(1).repeat(1, x_y.shape[1])
                    x = x_y[:, :x.shape[1], :]
                    y = x_y[:, x.shape[1]:, :]
                    # for j in range(x.shape[0]):
                    #     if abs(y[j,0,2].item()) > 0.1:
                    #         print("x = ", x[j, -5:,:])
                    #         print("y = ", y[j, :, :])
                trend_loss, mse_loss, up_diff, up_pred = model(x, y, check_size = check_size)
                loss = trend_loss*if_trend_loss + mse_loss
                check_size = False
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_trend_loss.append(trend_loss.item())
                train_mse_loss.append(mse_loss.item())
                train_up_diff.append(up_diff.item())
                train_up_pred.append(up_pred.item())
            # print(train_trend_loss)
            # print(train_mse_loss)
            avg_trend_loss = sum(train_trend_loss)/len(train_trend_loss)
            avg_mse_loss = sum(train_mse_loss)/len(train_mse_loss)
            avg_correctness = 1 - sum(train_up_diff)/sum(train_up_pred)
            end_time = time.time()
            print("epoch = ", i+1, "trend_loss = ", avg_trend_loss, "mse_loss = ", avg_mse_loss, "correctness = ", avg_correctness, "time = ", end_time - start_time)

            if (i+1) % model_save_freq == 0:
                save_path = os.path.join(model_path, "model"+str(i+1)+".pth")
                torch.save(model.state_dict(), save_path)
                print("model saved")

            if (i+1) > num_epochs:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] - lr/num_epochs_decay
    
    if args.test:
        print("start testing")
        start_time = time.time()
        test_trend_loss = []
        test_mse_loss = []
        test_up_diff = []
        test_up_pred = []

        encoder = Encoder()
        encoder.to(device)
        decoder = Decoder()
        decoder.to(device)
        model = Seq2Seq(encoder, decoder).to(device)

        with open(config_path, 'r')as f:
            config = yaml.unsafe_load(f)
        config = config['daily']['single_stock']
        test_use_model = config['test_use_model']
        num_epochs = config['num_epochs']
        num_epochs_decay = config['num_epochs_decay']
        if check_size_position == num_epochs+num_epochs_decay:
            print("check size ")
            check_size = True
        else:
            check_size = False
        
        if test_use_model == 0:
            files = os.listdir(model_path)
            if not files:
                print("no model")
                exit()
            last_file = max(files, key = lambda x: int(x[5:-4]))
            use_model =  os.path.join(model_path, last_file)
            # use_model = last_file
        else:
            use_model = os.path.join(model_path, "model"+str(test_use_model)+".pth")
        print("use model = ", use_model)
        model.load_state_dict(torch.load(use_model))
        model.eval()

        for i, (x, y) in enumerate(test_dataloader):
            x = x.float().to(device)
            y = y.float().to(device)
            if if_single_normalized:
                x_end = x[:, -1, 2]
                x_trade_end = x[:, -1, 5]
                x_y = torch.cat((x, y), dim=1)
                x_y[:, :, 1:5] = x_y[:, :, 1:5] - x_end.unsqueeze(1).unsqueeze(2).repeat(1, x_y.shape[1], 4)
                x_y[:, :, 5] = x_y[:, :, 5] - x_end.unsqueeze(1).repeat(1, x_y.shape[1])
                x = x_y[:, :x.shape[1], :]
                y = x_y[:, x.shape[1]:, :]
            with torch.no_grad():
                trend_loss, mse_loss, up_diff, up_pred = model(x, y, check_size = check_size)
                check_size = False
                test_trend_loss.append(trend_loss.item())
                test_mse_loss.append(mse_loss.item())
                test_up_diff.append(up_diff.item())
                test_up_pred.append(up_pred.item())
        avg_trend_loss = sum(test_trend_loss)/len(test_trend_loss)
        avg_mse_loss = sum(test_mse_loss)/len(test_mse_loss)
        if sum(test_up_pred) == 0:
            avg_correctness = "no up prediction"
        else:
            avg_correctness = 1 - sum(test_up_diff)/sum(test_up_pred)
        end_time = time.time()
        print("trend_loss = ", avg_trend_loss, "mse_loss = ", avg_mse_loss, "correctness = ", avg_correctness, "time = ", end_time - start_time)