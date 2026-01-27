import torch
import torch.nn as nn
from .encoder import ResnetEncoder
from .pose_decoder import PoseDecoder

class PoseNet(nn.Module):
    """Network for predicting relative 6-DOF pose between two frames.
    
    Uses a ResNet encoder to extract features from concatenated image pairs,
    then decodes to rotation (axis-angle) and translation vectors.
    """
    
    def __init__(self, num_layers=18, num_input_images=2, pretrained=True):
        """Initialize PoseNet.
        
        Args:
            num_layers (int): ResNet depth (18, 34, 50, 101, or 152). Default: 18.
            num_input_images (int): Number of images to concatenate. Default: 2.
            pretrained (bool): Load ImageNet pretrained weights. Default: True.
        """
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=num_input_images)
        self.decoder = PoseDecoder(
            num_ch_enc=self.encoder.num_ch_enc, 
            num_input_features=1,
            num_frames_to_predict_for=2
        )

    def forward(self, input_images):
        """Predict pose from concatenated image pair.
        
        Args:
            input_images (torch.Tensor or list): Concatenated images of shape [B, 2*C, H, W]
                                                  or list of image tensors.
        
        Returns:
            tuple: (axisangle [B, 1, 3], translation [B, 1, 3]) tensors.
        """

        features = self.encoder(input_images)
        axisangle, translation = self.decoder([features])
        return axisangle, translation