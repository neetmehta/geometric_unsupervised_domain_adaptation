from torch.utils.data import Dataset
from .carla_dataset import CarlaDataset
from .cityscapes_dataset import CityscapesDataset
import random

class GudaDataset(Dataset):
    """Combined dataset loader for GUDA framework.
    
    This dataset loader integrates both the CARLA synthetic dataset and the
    Cityscapes real-world dataset for unsupervised depth and semantic segmentation.
    """
    
    def __init__(self, cfg):
        """Initialize GUDA dataset.
        
        Args:
            cfg (SimpleNamespace): Configuration object with virtual_dataset,
                target_dataset, geometry, and other training settings.
        """
        self.carla_dataset = CarlaDataset(cfg)
        self.cityscapes_dataset = CityscapesDataset(cfg)
        self.total_length = min(len(self.carla_dataset), len(self.cityscapes_dataset))
        
        if len(self.carla_dataset) < len(self.cityscapes_dataset):
            self.primarty_dataset = 'carla'
        else:
            self.primarty_dataset = 'cityscapes'

    def __len__(self):
        """Return the total number of samples in the combined dataset."""
        return self.total_length

    def __getitem__(self, idx):
        """Retrieve a sample from either CARLA or Cityscapes dataset based on index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Sample containing images, depth maps, semantic masks, and calibration data.
        """
        if self.primarty_dataset == 'carla':
            virtual_sample = self.carla_dataset[idx]
            target_idx = random.randint(0, len(self.cityscapes_dataset) - 1)
            target_sample = self.cityscapes_dataset[target_idx]
            
        else:
            target_sample = self.cityscapes_dataset[idx]
            virtual_idx = random.randint(0, len(self.carla_dataset) - 1)
            virtual_sample = self.carla_dataset[virtual_idx]
        
        return virtual_sample, target_sample