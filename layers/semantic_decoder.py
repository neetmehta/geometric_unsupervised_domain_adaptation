import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from .layers import Conv3x3, ConvBlock, upsample

class SemanticDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_classes):
        """Initialize SemanticDecoder.
        
        Args:
            num_ch_enc (list): List of encoder channels, e.g. [64, 64, 128, 256, 512].
            num_classes (int): Number of semantic classes for output segmentation.
        """
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.final_channel_in = 0

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
            if i != 4:
                self.final_channel_in += self.num_ch_dec[i]
                
        self.convs[("final_conv", 0)] = Conv3x3(self.final_channel_in, num_classes)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, features):
        """Generate semantic logits from encoder features.
        
        Args:
            features (list): List of encoder feature maps.
        
        Returns:
            torch.Tensor: Semantic segmentation logits of shape [B, num_classes, H, W].
        """
        x = features[-1]
        intermediate_features = []
        
        for i in range(4,-1,-1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            
            if i>0:
                x += [features[i-1]]
                
            x = torch.cat(x,1)
            x = self.convs[("upconv", i, 1)](x)
            if i!=4:
                intermediate_features.append(F.interpolate(x, scale_factor=2**i, mode="nearest"))
            
        logits = self.convs[("final_conv", 0)](torch.cat(intermediate_features, 1))

        return logits
