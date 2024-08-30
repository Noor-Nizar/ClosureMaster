import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_block1_full = EncoderBlock(7, 64, kernel_size=3, padding=1)
        self.encoder_block2_half = EncoderBlock(64, 128, kernel_size=3, padding=1)
        self.encoder_block3_quarter = EncoderBlock(128, 256, kernel_size=3, padding=1)
        self.encoder_block4 = EncoderBlock(256, 512, kernel_size=3, padding=1)
        
        self.pre_half = nn.Conv2d(7, 64, kernel_size=3, padding=1)
        self.pre_quarter = nn.Conv2d(7, 128, kernel_size=3, padding=1)
        

        # Bottleneck
        self.conv1_bottleneck = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn1_bottleneck = nn.BatchNorm2d(1024)
        self.conv2_bottleneck = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2_bottleneck = nn.BatchNorm2d(1024)

    def forward(self, x_full, x_half, x_quarter):
        # Encoder
        x_full = self.encoder_block1_full(x_full)

        x_half = self.pre_half(x_half)
        x_half = x_full + x_half
        x_half = self.encoder_block2_half(x_half)
        
        x_quarter = self.pre_quarter(x_quarter)
        x_quarter = x_quarter + x_half
        x_quarter = self.encoder_block3_quarter(x_quarter)
        
        x_prebottle = self.encoder_block4(x_quarter)

        # Bottleneck
        x = F.relu(self.bn2_bottleneck(self.conv2_bottleneck(F.relu(self.bn1_bottleneck(self.conv1_bottleneck(x_prebottle))))))

        return x, x_full, x_half, x_quarter, x_prebottle  # Return bottleneck features and encoder outputs