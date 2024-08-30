import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import logger


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2)
        self.reduce = nn.Conv2d(out_channels, out_channels//2, kernel_size=1)  # Channel-wise averaging

    def forward(self, x, skip_connection):
        x = self.conv2(F.relu(self.bn1(self.conv1(torch.cat([x, skip_connection], dim=1)))))
        x = F.relu(self.bn2(x))
        x_up = self.reduce(x)  # Reduce depth by 2
        x_up = self.upsample(x_up)  # Upsample after convolutions
        return x, x_up

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_pre_decoder = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_pre_decoder = nn.BatchNorm2d(512)
        
        self.decoder_block0 = DecoderBlock(1024, 512, kernel_size=3, padding=1)
        self.decoder_block1_quarter = DecoderBlock(512, 256, kernel_size=3, padding=1)
        self.decoder_block2_half = DecoderBlock(256, 128, kernel_size=3, padding=1)
        self.decoder_block3_full = DecoderBlock(128, 64, kernel_size=3, padding=1)
        
        # Output layers (multi-scale outputs)
        self.conv_out_full = nn.Conv2d(64, 7, kernel_size=1)

        self.conv_out_half = nn.Conv2d(128, 7, kernel_size=1)

        self.conv_out_quarter = nn.Conv2d(256, 7, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, x_full, x_half, x_quarter, x_prebottle):
        # Pre-decoder layer
        logger.debug(f'xshaoe {x.shape}')
        logger.debug(f'xfull {x_full.shape}')
        logger.debug(f'xhalf {x_half.shape}')
        logger.debug(f'xquarter {x_quarter.shape}')
        logger.debug(f'xpre {x_prebottle.shape}')
        x = F.relu(self.bn_pre_decoder(self.conv_pre_decoder(x)))
        logger.debug(f"after predec {x.shape}")
        # Decoder (using concatenation)
        _, x = self.decoder_block0(x, x_prebottle)
        logger.debug(f"after block0 {x.shape}")
        x_quarter_fm, x = self.decoder_block1_quarter(x, x_quarter) ## x_quarter_fm feature map that we will will squeeze to get reconstructed version
        logger.debug(x.shape)
        x_half_fm, x = self.decoder_block2_half(x, x_half)
        logger.debug(x.shape)
        x_full_fm, x = self.decoder_block3_full(x, x_full)
        logger.debug(x.shape)
#         x_final, x = self.decoder_block3_full(x, x_full)
        
        # Output layers (multi-scale outputs)
        recon_full = self.upsample(self.conv_out_full(x_full))

        recon_half = self.upsample(self.conv_out_half(x_half))

        recon_quarter = self.upsample(self.conv_out_quarter(x_quarter))

        return recon_full, recon_half, recon_quarter