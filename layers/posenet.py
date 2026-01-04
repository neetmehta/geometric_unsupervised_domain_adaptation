import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models

class PoseEncoder(nn.Module):
    def __init__(self, num_input_images=2, pretrained=True):
        super().__init__()
        
        # Load standard ResNet
        resnet = models.resnet18(pretrained=pretrained)
        
        # 1. Modify input layer to accept stacked images (e.g., 6 channels for 2 images)
        self.num_input_images = num_input_images
        orig_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=3 * num_input_images,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=(orig_conv.bias is not None)
        )

        # 2. Smart Weight Initialization (Critical for Monodepth2)
        # We repeat the weights across the new channels and divide by num_images 
        # to keep the initial activation magnitude similar to standard ResNet.
        if pretrained:
            self.conv1.weight.data = (
                orig_conv.weight.data.repeat(1, num_input_images, 1, 1) 
                / num_input_images
            )
        
        # 3. Create Encoder Sequence (removing FC layers)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.num_ch_enc = 512

    def forward(self, x):
        return self.encoder(x)


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        
        # Default: if input is 2 frames, we predict pose for the 1 other frame (target->source)
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        # Use ModuleDict for cleaner registration
        self.convs = nn.ModuleDict()
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc, 256, 1)
        
        # Monodepth2 default "pose" convs (2 layers of 3x3)
        self.convs["pose_0"] = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.convs["pose_1"] = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        # Final projection: 6 DOF per frame (3 rot + 3 trans)
        self.convs["pose_out"] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_features):
        # 1. Reduce channels
        out = self.relu(self.convs["squeeze"](input_features))
        
        # 2. Intermediate processing
        out = self.relu(self.convs["pose_0"](out))
        out = self.relu(self.convs["pose_1"](out))
        
        # 3. Final projection
        out = self.convs["pose_out"](out)

        # 4. Global Average Pooling
        out = out.mean(3).mean(2)

        # 5. Scale output (Monodepth2 trick for better convergence)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        # 6. Split into Axis-Angle and Translation
        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseNet(nn.Module):
    def __init__(self, num_input_images=2, pretrained=True):
        super().__init__()
        self.encoder = PoseEncoder(num_input_images=num_input_images, pretrained=pretrained)
        self.decoder = PoseDecoder(
            num_ch_enc=self.encoder.num_ch_enc, 
            num_input_features=num_input_images
        )

    def forward(self, input_images):
        # FIX: Concatenate list of images into a single tensor
        if isinstance(input_images, list):
            input_images = torch.cat(input_images, 1)

        features = self.encoder(input_images)
        axisangle, translation = self.decoder(features)
        return axisangle, translation