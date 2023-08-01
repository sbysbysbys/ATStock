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
    y_length = config['y_length']

    encoder = Encoder()
    encoder.to(device)
    decoder = Decoder()
    decoder.to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs_decay, gamma=0.1)

    train_loss = []
    test_loss = []
    check_size = True
    for epoch in range(num_epochs+num_epochs_decay):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            x_end = x[:, -1, :].to(device)
            # x.size = [32, x_length=81, 11]
            # y.size = [32, y_length=8, 11]

            enc_output,enc_hidden = encoder(x, check_size=check_size)
            if check_size == True:
                print("enc_output.shape = ", enc_output.shape)
                print("enc_state.shape = ",enc_hidden.shape)
            # encoder: conv_num = 2
            # x.shape =  torch.Size([32, 81, 11])
            # x.first_permute =  torch.Size([32, 11, 81])
            # conv 0 , x.shape =  torch.Size([32, 4, 81])
            # maxpool 0 , x.shape =  torch.Size([32, 4, 27])
            # conv 1 , x.shape =  torch.Size([32, 8, 27])
            # maxpool 1 , x.shape =  torch.Size([32, 8, 9])
            # x.second_permute =  torch.Size([32, 9, 8])
            # h0.shape =  torch.Size([2, 32, 64])
            # enc_output.size =  torch.Size([32, 9, 64])
            # enc_state,size =  torch.Size([2, 32, 64])

            dec_hidden = decoder.begin_state(enc_hidden)
            dec_input = x_end
            for y in range(y_length):
                dec_output, dec_hidden = decoder(dec_hidden, enc_output, check_size=check_size)
                dec_input = dec_output
                if check_size == True:
                    print("dec_output.size = ", dec_output.shape)
                    print("dec_hidden.size = ", dec_hidden.shape)
                    check_size = False


            # optimizer.zero_grad()
            # outputs = encoder(x)
            # loss = criterion(outputs, y)
            # loss.backward()
            # optimizer.step()

            # print(f'Epoch [{epoch+1}/{num_epochs+num_epochs_decay}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')




    