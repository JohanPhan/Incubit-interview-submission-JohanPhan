import math

import torch
import torch.nn as nn


class ResCNN_4layers_with_tanh(nn.Module):
    def __init__(self, num_channels):
        super(ResCNN_4layers_with_tanh, self).__init__()
        self.base_channel = 64
      
        
        
        
        self.input_conv = nn.Conv2d(num_channels, 256, kernel_size=9, stride=1, padding=4, bias=True)         
        self.input_conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.input_conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()         
        #self.input_conv4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=3)
        self.output_conv3 = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)        
        self.ReLU = nn.ReLU()    
        self.output_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)      
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        top_mean_split = torch.split(x, 4, dim=1)        
        top_mean = torch.mean(top_mean_split[0], dim = 1).unsqueeze(1)           

        residual = top_mean
        x = self.ReLU(self.input_conv(x) )
        x = self.ReLU(self.input_conv2(x) )
        x = (self.input_conv3(x) )        
        x = (self.output_conv3(x) )
        x = torch.add(x, residual)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

