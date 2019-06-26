import math

import torch
import torch.nn as nn


class ResCNN_5(nn.Module):
    def __init__(self, num_channels, base_channel, upscale_factor, num_residuals):
        super(ResCNN_5, self).__init__()
        self.base_channel = 64
      
        
        
        
        self.input_conv = nn.Conv2d(9, 64, kernel_size=9, stride=1, padding=4, bias=True)         
        self.input_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.input_conv2b = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.input_conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
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
        x = self.ReLU(self.input_conv2b(x) )
        x = self.ReLU(self.input_conv3(x) )        
        x = (self.output_conv3(x) )
        x = torch.add(x, residual)
        #x = self.output_conv(x)
        return x
        #return x
        #top_mean_split = torch.split(x, 4, dim=1)        
        #top_mean = torch.mean(top_mean_split[0], dim = 1).unsqueeze(1)        
        #residual = top_mean    
        #x = self.ReLU(self.input_conv(x) )

        #x = self.ReLU(self.input_conv2(x) )
        #x = (self.input_conv3(x) )        

        #x = (self.output_conv3(x) )
        #x = torch.add(x, residual)
        
        

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(num_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.bn(self.conv1(x))
        x = self.activation(x)
        x = self.bn(self.conv2(x))
        x = torch.add(x, residual)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.ps(self.conv(x))
        return x