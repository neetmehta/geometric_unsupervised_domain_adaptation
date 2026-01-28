import torch
import torch.nn as nn
from .encoder import ResnetEncoder
from .pose_decoder import PoseDecoder

class PoseNet(nn.Module):
    def __init__(self, num_layers=18, num_input_images=2, pretrained=True):
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=num_input_images)
        self.decoder = PoseDecoder(
            num_ch_enc=self.encoder.num_ch_enc, 
            num_input_features=1,
            num_frames_to_predict_for=2
        )

    def forward(self, input_images):
        # FIX: Concatenate list of images into a single tensor
        if isinstance(input_images, list):
            input_images = torch.cat(input_images, 1)

        features = self.encoder(input_images)
        axisangle, translation = self.decoder([features])
        return axisangle, translation