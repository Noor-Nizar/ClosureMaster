from .encoder import Encoder
from .decoder import Decoder

import torch.nn as nn

class PlaceNet(nn.Module):
    def __init__(self):
        super(PlaceNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_full, x_half, x_quarter):
        x, x_full, x_half, x_quarter, x_prebottle = self.encoder(x_full, x_half, x_quarter)
        recon_full, recon_half, recon_quarter = self.decoder(x, x_full, x_half, x_quarter, x_prebottle)
        return recon_full, recon_half, recon_quarter