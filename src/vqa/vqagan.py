# src/vqa/vqagan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantize import VectorQuantizer
from .discriminator import NLayerDiscriminator, weights_init
from .perceptual import VQLPIPSWithDiscriminator

class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(32, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, n_res_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks with residual connections
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels*2, 4, 2, 1),
                nn.GroupNorm(32, hidden_channels*2),
                nn.ReLU(),
                *[ResnetBlock(hidden_channels*2) for _ in range(n_res_blocks)]
            ),
            nn.Sequential(
                nn.Conv2d(hidden_channels*2, hidden_channels*4, 4, 2, 1),
                nn.GroupNorm(32, hidden_channels*4),
                nn.ReLU(),
                *[ResnetBlock(hidden_channels*4) for _ in range(n_res_blocks)]
            )
        ])

    def forward(self, x):
        h = self.conv_in(x)
        for down_block in self.down:
            h = down_block(h)
        return h

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, n_res_blocks=2):
        super().__init__()
        current_channels = hidden_channels * 4
        
        # Upsampling blocks with residual connections
        self.up = nn.ModuleList([
            nn.Sequential(
                *[ResnetBlock(current_channels) for _ in range(n_res_blocks)],
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(current_channels, current_channels//2, 3, 1, 1),
                nn.GroupNorm(32, current_channels//2),
                nn.ReLU()
            ),
            nn.Sequential(
                *[ResnetBlock(current_channels//2) for _ in range(n_res_blocks)],
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(current_channels//2, hidden_channels, 3, 1, 1),
                nn.GroupNorm(32, hidden_channels),
                nn.ReLU()
            )
        ])
        
        # Final convolution layer (separate from activation)
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for block in self.up:
            x = block(x)
        x = self.final_conv(x)
        return torch.tanh(x)

    def get_last_layer(self):
        return self.final_conv.weight

class VQGAN(nn.Module):
    def __init__(self, 
                 n_embed, 
                 embed_dim, 
                 hidden_channels=128,
                 n_res_blocks=2,
                 disc_start=10000,
                 disc_weight=0.8,
                 perceptual_weight=1.0,
                 codebook_weight=1.0):
        super().__init__()
        
        self.encoder = Encoder(hidden_channels=hidden_channels, n_res_blocks=n_res_blocks)
        self.decoder = Decoder(hidden_channels=hidden_channels, n_res_blocks=n_res_blocks)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(hidden_channels*4, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, hidden_channels*4, 1)

        # Initialize discriminator and loss
        self.discriminator = NLayerDiscriminator(input_nc=3).apply(weights_init)
        self.loss = VQLPIPSWithDiscriminator(
            disc_start= 500,
            codebook_weight=codebook_weight,
            pixel_weight=1.0,
            perceptual_weight=perceptual_weight,
            disc_weight=disc_weight
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x, optimizer_idx=None, global_step=None):
        quant, codebook_loss, (_, _, indices) = self.encode(x)
        dec = self.decode(quant)
        
        if optimizer_idx is None:
            return dec, codebook_loss, indices
            
        if optimizer_idx == 0:
            # Generator loss
            if global_step >= self.loss.disc_start:
                logits_fake = self.discriminator(dec)
                g_loss = -torch.mean(logits_fake)
            else:
                g_loss = torch.tensor(0.0, device=x.device)
                
            loss, log_dict = self.loss(
                codebook_loss=codebook_loss,
                inputs=x,
                reconstructions=dec,
                g_loss=g_loss,
                d_loss=None,  # No discriminator loss for generator
                optimizer_idx=optimizer_idx,
                global_step=global_step,
                last_layer=self.decoder.get_last_layer()
            )
            return loss, log_dict
                    
        if optimizer_idx == 1:
            # Discriminator loss
            if global_step < self.loss.disc_start:
                return None, None  # Indicate that discriminator loss is inactive
            else:
                logits_real = self.discriminator(x.detach())
                logits_fake = self.discriminator(dec.detach())
                
                d_loss = torch.mean(F.softplus(-logits_real)) + \
                        torch.mean(F.softplus(logits_fake))
            
            loss, log_dict = self.loss(
                codebook_loss=codebook_loss,
                inputs=x,
                reconstructions=dec,
                g_loss=None,
                d_loss=d_loss,
                optimizer_idx=optimizer_idx,
                global_step=global_step,
                last_layer=self.decoder.get_last_layer()
            )
            return loss, log_dict
