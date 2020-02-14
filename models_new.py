import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from kymatio import Scattering1D

from stft import Spectrogram, LogmelFilterBank

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class Scatter(nn.Module):
    def __init__(self, J, Q, audio_length):
        
        super(Scatter, self).__init__()
        
        self.J = J
        self.Q = Q
        self.T = audio_length
        self.meta = Scattering1D.compute_meta_scattering(self.J, self.Q)
                              
        self.order0_indices = (self.meta['order'] == 0)
        self.order1_indices = (self.meta['order'] == 1)
        self.order2_indices = (self.meta['order'] == 2)
        
        self.scattering = Scattering1D(self.J, self.T, self.Q).cuda()
        self.output_size = self.scattering.output_size()
        
    def forward(self, input):

        x = self.scattering.forward(input)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))

        return x

class CNN_scatter(nn.Module):
    def __init__(self, classes_num, J, Q, audio_length):
        
        super(CNN_scatter, self).__init__()
        
        self.scatter = Scatter(J, Q, audio_length)
        
        self.bn0 = nn.BatchNorm2d(math.ceil(audio_length/2**J))

        self.conv1 = nn.Conv2d(in_channels=1, 
                              out_channels=16,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, 
                              out_channels=32,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(2, 2), bias=True)

        self.fc1 = nn.Linear(32, 16, bias=True)
        self.fc_monophonic = nn.Linear(16, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.fc1)
        init_layer(self.fc_monophonic)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        x = self.scatter(input)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        embedding = F.relu(self.fc1(x))
  
        clipwise_output = torch.sigmoid(self.fc_monophonic(embedding))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict