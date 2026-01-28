import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvELUupsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.elu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

class SemanticDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_classes):
        """
        num_ch_enc: list of encoder channels, e.g. [64, 64, 128, 256, 512]
        num_classes: number of semantic classes
        """
        super().__init__()

        self.scales = [1, 2, 3, 4]  # use 4 deepest scales
        self.conv_elu_upsample = nn.ModuleList()
        self.convs = nn.ModuleList()
        final_ch = 0
        conv_elu_upsample_in_ch = num_ch_enc[-1]
        for i, _ in enumerate(num_ch_enc[::-1]):
            
            self.conv_elu_upsample.append(
                ConvELUupsample(conv_elu_upsample_in_ch, int(256/2**i))
            )
            if i < len(num_ch_enc) - 1:
                self.convs.append(
                    nn.Conv2d(int(256/2**i) + num_ch_enc[::-1][i+1], int(256/2**i), kernel_size=3, padding=1)
                )
            conv_elu_upsample_in_ch = int(256/2**i)
            
            if i == len(num_ch_enc) - 1:
                self.convs.append(
                    nn.Conv2d(int(256/2**i), int(256/2**i), kernel_size=3, padding=1)
                )
                
            if i>0:
                final_ch += int(256/2**i)
        

        self.final_conv = nn.Conv2d(
            final_ch,
            num_classes,
            kernel_size=3,
            padding=1
        )


    def forward(self, features):
        """
        features: list of encoder feature maps
        """
        features_list = []
        for i in range(5):
            if i == 0:
                x = features[-1]
            x = self.conv_elu_upsample[i](x)
                
            if i < 4:
                x = torch.cat([x, features[-(i+2)]], dim=1)
                x = self.convs[i](x)
                
            if i==4:
                x = self.convs[i](x)
                
            if i>0:
                features_list.append(F.interpolate(x, scale_factor=2**(4-i), mode='bilinear', align_corners=False))

        x = torch.cat(features_list, dim=1)
        x = self.final_conv(x)
        return x
