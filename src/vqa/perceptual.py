# src/vqa/perceptual.py

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# dumb down veriosn froom tamming-trasnformers

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
    
        vgg = models.vgg16(pretrained=True)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        blocks.append(vgg.features[23:30].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        # Expect input in range [-1, 1]
        x = (x + 1) / 2  # Convert to [0, 1]
        x = (x - self.mean) / self.std
        return x

    def forward(self, input, target, normalize=True):
        if normalize:
            input = self.preprocess(input)
            target = self.preprocess(target)
            
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixel_weight=1.0,
                 perceptual_weight=1.0, disc_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start

        self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, codebook_loss, inputs, reconstructions, g_loss, d_loss, 
                optimizer_idx, global_step, last_layer=None):
        # origially this was not used in tamming transformers
        rec_loss = torch.abs(inputs - reconstructions).mean()  
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss  # Both are scalars


        if optimizer_idx == 0:
            # disc start is the criminal here, this needs to be fixed, corroupting genrator loss
            if global_step >= self.disc_start:
                loss = (rec_loss + 
                       self.codebook_weight * codebook_loss + 
                       self.disc_weight * g_loss)
            else:
                loss = rec_loss + self.codebook_weight * codebook_loss
                g_loss = torch.tensor(0.0, device=rec_loss.device)

            log = {
                "total_loss": loss,             # this is different from tamming transformers, this is scalar
                "rec_loss": rec_loss,           
                "g_loss": g_loss,              
                "codebook_loss": codebook_loss, 
            }
            return loss, log

        
        if optimizer_idx == 1:
            if global_step >= self.disc_start:
                log = {
                    "d_loss": d_loss, 
                }
                return d_loss, log
            else:
                # No discriminator training before disc_start steps
                d_loss = torch.tensor(0.0, device=rec_loss.device)
                log = {
                    "d_loss": d_loss,  
                }
                return d_loss, log
