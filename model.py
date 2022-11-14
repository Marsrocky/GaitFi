import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# size
def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


class CRNN(nn.Module):
    def __init__(self, img_x, img_y, mat_x, mat_y, fc_hidden1, fc_hidden2, CNN_embed_dim,
                 h_RNN_layers, h_RNN, h_FC_dim, drop_p, num_classes, input_type):
        super().__init__()
        assert input_type in ['mat', 'image', 'both'], 'please choose the right type as: mat, image, both'
        self.input_type = input_type
        if input_type == 'image' or input_type == 'both':
            self.image_CNN = CNN(img_x, img_y, 1, fc_hidden1, fc_hidden2, drop_p, CNN_embed_dim, input_type='image', fc_in_dim=256)
            self.image_RNN = RNN(CNN_embed_dim, h_RNN_layers, h_RNN, h_FC_dim, drop_p, num_classes) #h_RNN, h_FC_dim

        if input_type == 'mat' or input_type == 'both':
            self.mat_CNN = CNN(mat_x, mat_y, 3, fc_hidden1, fc_hidden2, drop_p, CNN_embed_dim, input_type='mat', fc_in_dim=512)

        self.fc = nn.Linear(2 * h_FC_dim if input_type == 'both' else h_FC_dim, num_classes)

    def forward(self, image, mat):
        if self.input_type == 'image' or self.input_type == 'both':
            cnn_emb = self.image_CNN(image)
            rnn_emb = self.image_RNN(cnn_emb)
        if self.input_type == 'mat' or self.input_type == 'both':
            mat_emb = self.mat_CNN(mat)

        if self.input_type == 'both':
            # concatenate rnn_emb with mat
            hidden = torch.cat((rnn_emb, mat_emb), dim=1)
            # hidden = rnn_emb + mat_emb
        elif self.input_type == 'image':
            hidden = rnn_emb
        elif self.input_type == 'mat':
            hidden = mat_emb
        output = self.fc(hidden)
        return hidden, output

#一个残差模块
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class CNN(nn.Module):
    def __init__(self, high, wide, in_channel, fc_hidden1, fc_hidden2, drop_p, CNN_embed_dim, input_type, fc_in_dim): #CNN_embed_dim参数设置中为64
        super().__init__()
        self.high = high
        self.wide = wide
        self.input_type = input_type
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3 = 8, 16, 32
        self.k1, self.k2, self.k3 = (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3 = (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.high, self.wide), self.pd1, self.k1,
                                                 self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            #
        )
        self.layer1 = self._make_layer(self.ch1, self.ch1, 2, stride=2)  # res

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),

        )
        self.layer2 = self._make_layer(self.ch2, self.ch2, 2, stride=2)  # res

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.layer3 = self._make_layer(self.ch3, self.ch3, 2, stride=2) # res

        self.fc1 = nn.Linear(fc_in_dim, self.CNN_embed_dim)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        layers = []
        if stride != 1:
            layers.append(Block(in_channel, out_channel, stride, same_shape=False))
        else:
            layers.append(Block(in_channel, out_channel, stride))

        for i in range(1, block_num):
            layers.append(Block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x_2d):
        if self.input_type == 'image':
            cnn_embed_seq = []
            for t in range(x_2d.size(1)): 
                # CNNs
                x = x_2d[:, t, :, :, :]
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.conv2(x)
                x = self.layer2(x)
                x = x.view(x.size(0), -1)  # flatten the output of conv
                x = F.dropout(x, p=self.drop_p, training=self.training)

                # FC layers
                x = F.relu(self.fc1(x))
                cnn_embed_seq.append(x)

            # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
            output = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        else:
            x = self.conv1(x_2d)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv
            x = F.dropout(x, p=self.drop_p, training=self.training)

            # FC layers
            output = F.relu(self.fc1(x))
        return output


class RNN(nn.Module):
    def __init__(self, CNN_embed_dim, h_RNN_layers, h_RNN, h_FC_dim, drop_p, num_classes):
        super().__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)

    def forward(self, x_RNN):
        rnn_out, (_, _) = self.LSTM(x_RNN, None)

        # FC layers
        x = self.fc1(rnn_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return x



