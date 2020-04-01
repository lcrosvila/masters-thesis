import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from stft import Spectrogram, LogmelFilterBank
from augmentation import SpecAugmentation


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5)):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class CNN_two(nn.Module):
    def __init__(self, classes_num, input_length, order1_length, order2_length):

        super(CNN_two, self).__init__()

        depth1 = min(
            math.floor(math.log(order1_length) / math.log(2)) - 1,
            math.floor(math.log(input_length) / math.log(2)) - 1,
        )
        depth2 = min(
            math.floor(math.log(order2_length) / math.log(2)) - 1,
            math.floor(math.log(input_length) / math.log(2)) - 1,
        )

        self.conv_block1 = ConvBlock5x5(
            in_channels=1, out_channels=32, kernel_size=(5, 5)
        )

        channels1 = [32*2**i for i in range(depth1 - 1)]
        for i in range(len(channels1)):
            if channels1[i]>256:
                channels1[i]=256

        self.conv_list1 = nn.ModuleList(
            [
                ConvBlock5x5(
                    in_channels=channels1[i],
                    out_channels=channels1[i + 1],
                    kernel_size=(5, 5),
                )
                for i in range(len(channels1)-1)
            ]
        )

        self.conv_block2 = ConvBlock5x5(
            in_channels=1, out_channels=32, kernel_size=(1, 5)
        )

        channels2 = [32*2**i for i in range(depth2 - 1)]
        for i in range(len(channels2)):
            if channels2[i]>256:
                channels2[i]=256

        self.conv_list2 = nn.ModuleList(
            [
                ConvBlock5x5(
                    in_channels=channels2[i],
                    out_channels=channels2[i + 1],
                    kernel_size=(1, 5),
                )
                for i in range(len(channels2)-1)
            ]
        )

        self.fc1 = nn.Linear(
            channels1[-1] + channels2[-1], 256, bias=True
        )
        self.bn0 = nn.BatchNorm1d(256)
        self.fc_medley = nn.Linear(256, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_medley)

    def forward(self, input1, input2, plot=False):
        """
        Input: (batch_size, data_length)"""

        x1 = input1.view(input1.shape[0], 1, input1.shape[1], input1.shape[2])
        # x1 = input1[:, None, ...]

        x2 = input2.view(input2.shape[0], 1, input2.shape[1], input2.shape[2])

        if plot:
            plt.title("First order input")
            plt.imshow(x1[0, 0, :, :].cpu().detach().numpy(), aspect="auto")
            plt.show()
        x1 = self.conv_block1(x1, pool_size=(2, 2), pool_type="avg")

        if plot:
            plt.title("First order after conv block 1")
            plt.imshow(x1[0, 0, :, :].cpu().detach().numpy(), aspect="auto")
            plt.show()
        for i, c in enumerate(self.conv_list1):
            x1 = c(x1, pool_size=(2, 2), pool_type="avg")
            if plot:
                title = "First order after conv block " + str(i + 2)
                plt.title(title)
                plt.imshow(x1[0, 0, :, :].cpu().detach().numpy(), aspect="auto")
                plt.show()

        if plot:
            plt.title("Second order input")
            plt.imshow(x2[0, 0, :, :].cpu().detach().numpy(), aspect="auto")
            plt.show()

        x2 = self.conv_block2(x2, pool_size=(2, 2), pool_type="avg")
        if plot:
            plt.title("Second order after conv block 1")
            plt.imshow(x2[0, 0, :, :].cpu().detach().numpy(), aspect="auto")
            plt.show()

        for i, c in enumerate(self.conv_list2):
            x2 = c(x2, pool_size=(2, 2), pool_type="avg")
            if plot:
                title = "Second order after conv block " + str(i + 2)
                plt.title(title)
                plt.imshow(x2[0, 0, :, :].cpu().detach().numpy(), aspect="auto")
                plt.show()

        x1 = torch.mean(x1, dim=3)
        (x11, _) = torch.max(x1, dim=2)
        x12 = torch.mean(x1, dim=2)
        x1 = x11 + x12

        x2 = torch.mean(x2, dim=3)
        (x21, _) = torch.max(x2, dim=2)
        x22 = torch.mean(x2, dim=2)
        x2 = x21 + x22

        x = torch.cat((x1, x2), 1)

        embedding = F.relu(self.fc1(x))

        x = self.bn0(embedding)

        clipwise_output = torch.sigmoid(self.fc_medley(x))

        output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}

        return output_dict


class Cnn6(nn.Module):
    def __init__(self, classes_num, time_steps, freq_bins, spec_aug=True):

        super(Cnn6, self).__init__()

        # Spec augmenter
        self.spec_aug = spec_aug
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(freq_bins)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_openmic = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_openmic)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, 0, :, :]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.spec_aug:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_openmic(x))

        output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}

        return output_dict
