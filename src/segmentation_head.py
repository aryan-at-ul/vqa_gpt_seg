import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SegmentationHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # MLP to process GPT features
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x comes from transformer: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Apply MLP
        x = self.mlp(x)  # [batch_size, seq_len, num_classes]
        
        # Calculate spatial dimensions
        h = w = int(math.sqrt(seq_len))
        
        # Reshape to spatial dimensions
        x = x.permute(0, 2, 1).view(batch_size, self.num_classes, h, w)
        
        # Upsample to match input resolution (4x upscaling)
        x = F.interpolate(x, size=(h*4, w*4), mode='bilinear', align_corners=False)
        
        return x