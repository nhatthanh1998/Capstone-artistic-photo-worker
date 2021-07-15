from torch import nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        else:
            conv_block += [nn.ReplicationPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), nn.InstanceNorm2d(dim), nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        else:
            conv_block += [nn.ReplicationPad2d(1)]
            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator2(nn.Module):
    def __init__(self, use_dropout=False, num_residual_block=6):
        super(Generator2, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 32, kernel_size=7, padding=0, stride=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        ]
        for i in range(num_residual_block):
            model += [ResnetBlock(128, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout, use_bias=True)]
        
        model += [
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        ]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(32, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, alpha=1.0):
        super(Generator, self).__init__()
        a = alpha
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, int(a*32), 9, 1),
            nn.ReLU(),
            ConvLayer(int(a*32), int(a*32), 3, 2),
            nn.ReLU(),
            ConvLayer(int(a*32), int(a*32), 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResNextLayer(int(a*32), [int(a*16), int(a*16), int(a*32)], kernel_size=3),
            ResNextLayer(int(a*32), [int(a*16), int(a*16), int(a*32)], kernel_size=3),
            ResNextLayer(int(a*32), [int(a*16), int(a*16), int(a*32)], kernel_size=3),
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(int(a*32), int(a*32), 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(int(a*32), int(a*32), 3, 2, 1),
            nn.ReLU(),
            ConvLayer(int(a*32), 3, 9, 1, norm="None"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out




class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution 
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class ResNextLayer(nn.Module):
    def __init__(self, in_ch=128, channels=[64, 64, 128], kernel_size=3):
        super(ResNextLayer, self).__init__()
        ch1, ch2, ch3 = channels
        self.conv1 = ConvLayer(in_ch, ch1, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvLayer(ch1, ch2, kernel_size=kernel_size, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = ConvLayer(ch2, ch3, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        out = out + identity
        return out
