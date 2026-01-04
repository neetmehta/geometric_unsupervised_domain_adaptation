import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_classes):
        """
        num_ch_enc: list of encoder channels, e.g. [64, 64, 128, 256, 512]
        num_classes: number of semantic classes
        """
        super().__init__()

        self.scales = [1, 2, 3, 4]  # use 4 deepest scales

        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(num_ch_enc[i], 128, kernel_size=1)
            for i in self.scales
        ])

        self.final_conv = nn.Conv2d(
            128 * len(self.scales),
            num_classes,
            kernel_size=3,
            padding=1
        )

    def forward(self, features):
        """
        features: list of encoder feature maps
        """
        target_size = features[self.scales[0]].shape[2:]  # highest resolution

        upsampled_feats = []

        for i, scale in enumerate(self.scales):
            x = self.reduce_convs[i](features[scale])

            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )

            upsampled_feats.append(x)

        x = torch.cat(upsampled_feats, dim=1)
        logits = self.final_conv(x)

        return logits
