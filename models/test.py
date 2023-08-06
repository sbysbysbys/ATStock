import os
import sys
import torch
import torch.nn as nn
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from models.models111 import *
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
    y_length = config['y_length']
    if_check_size = config['if_check_size']
    check_size_position = config['check_size_position']

    encoder = Encoder()
    encoder.to(device)
    decoder = Decoder()
    decoder.to(device)

    train_loss = []
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for i in range(num_epochs+num_epochs_decay):
        check_size = if_check_size*(i==check_size_position)
        for j, (x, y) in enumerate(train_dataloader):
            x = x.float().to(device)
            y = y.float().to(device)
            loss = critertion(encoder, decoder, x, y, check_size = check_size).float()
            check_size = False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        avg_loss = sum(train_loss)/len(train_loss)
        print("epoch = ", i, "loss = ", avg_loss)

            # # x_end = x[:, -1, :].to(device)
            # enc_output,enc_hidden = encoder(x, check_size=check_size)
            # if check_size == True:
            #     print("enc_output.shape = ", enc_output.shape)
            #     print("enc_state.shape = ",enc_hidden.shape)

            # dec_hidden = decoder.begin_state(enc_hidden)
            # # dec_input = x_end
            # dec_output_tstep = 0
            # for y in range(y_length):
            #     dec_output, dec_hidden = decoder(dec_hidden, enc_output, check_size=check_size)
            #     dec_input = dec_output
            #     if check_size == True:
            #         print("dec_output.size = ", dec_output.shape)
            #         print("dec_hidden.size = ", dec_hidden.shape)
            #         check_size = False
            #     if dec_output_tstep == 0:
            #         dec_output_tstep = dec_output
            #     else:
            #         dec_output_tstep = torch.cat((dec_output_tstep, dec_output), dim=1)
            
            # loss = critertion(dec_output_tstep, y)
            # train_loss.append(loss.item())
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # print(f'Epoch [{epoch+1}/{num_epochs+num_epochs_decay}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')




    