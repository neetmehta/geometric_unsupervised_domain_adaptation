import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision.transforms import transforms
import os
import json
import numpy as np

class CityscapesDataset(Dataset):
    
    def __init__(self, cfg):
        """
        Initializes the CityscapesDataset.

        Args:
            images (list): List of image file paths.
            labels (list): List of label file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cfg = cfg
        root_dir = cfg.target_dataset.root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((cfg.target_dataset.img_height, cfg.target_dataset.img_width)),
            transforms.ToTensor(),
        ])
        self.scales = cfg.geometry.scales
        self.resize_transform = [transforms.Compose([transforms.Resize((cfg.virtual_dataset.img_height // (2 ** s), cfg.virtual_dataset.img_width // (2 ** s))), transforms.ToTensor()]) for s in self.scales]
        self.cities = glob.glob(os.path.join(root_dir, "rightImg8bit", "train_extra", "*"))
        self.samples = []
        for city in self.cities:
            self.samples.extend(self.parse_scenes(city))
        
    def parse_scenes(self, scene):
        """Parse scene directory and extract valid frame triplets.
        
        Identifies consecutive frames (t-1, t, t+1) that exist in the dataset.
        
        Args:
            scene (str): Path to scene directory.
        
        Returns:
            list: List of dictionaries with keys 'base_name', 'frame_id', and 'scene'.
        """
        image_paths = [i for i in glob.glob(os.path.join(scene, "*.png"))]

        samples = []
        sorted_images = sorted(image_paths, key=lambda x: int(os.path.normpath(x).split(os.path.sep)[-1].split("_")[2]))
        for image_path in sorted_images:
            image_dir = os.path.dirname(image_path)
            base_name = os.path.dirname(image_dir)
            city, scene_id, frame, camera = os.path.normpath(image_path).split(os.path.sep)[-1].split("_")
            id_t = int(frame)
            image_frame_t_minus_1 = os.path.join(image_dir, f"{city}_{scene_id}_{id_t - 1:06}_{camera}")
            image_frame_t_plus_1 = os.path.join(image_dir, f"{city}_{scene_id}_{id_t + 1:06}_{camera}")

            if os.path.exists(image_frame_t_plus_1) and os.path.exists(image_frame_t_minus_1):
                samples.append({"base_name": base_name, "frame_id": id_t, "city": city, "camera": camera, "scene": scene})
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = {}
        paths = self.samples[idx]
        city, frame, camera = paths["city"], paths["frame_id"], paths["camera"]
        frame_t = f"{city}_000000_{frame:06}_{camera}"
        frame_t_minus_1 = f"{city}_000000_{frame-1:06}_{camera}"
        frame_t_plus_1 = f"{city}_000000_{frame+1:06}_{camera}"
        intrinsics_path = os.path.join(paths["base_name"], city, frame_t.replace("rightImg8bit.png", "camera.json"))
        # RGB image
        image_t = Image.open(os.path.join(paths["base_name"], city, frame_t)).convert("RGB")
        ori_W, ori_H = image_t.size
        for s in self.scales:
            sample[("t", 0, s)] = self.resize_transform[s](image_t)
        sample[("t", -1, 0)] = self.transforms(Image.open(os.path.join(paths["base_name"], city, frame_t_minus_1)).convert("RGB"))
        sample[("t", 1, 0)] = self.transforms(Image.open(os.path.join(paths["base_name"], city, frame_t_plus_1)).convert("RGB"))
        
        with open(intrinsics_path, 'r') as f:
            intrinsic = json.load(f)
            
        fx = intrinsic['intrinsic']['fx']
        fy = intrinsic['intrinsic']['fy']
        u0 = intrinsic['intrinsic']['u0']
        v0 = intrinsic['intrinsic']['v0']

        K = np.array([[fx, 0, u0, 0],
                      [0, fy, v0, 0],
                      [0,  0,  1, 0],
                      [0,  0,  0, 1]])
        
        K[0, :] *= (self.cfg.target_dataset.img_width / ori_W)
        K[1, :] *= (self.cfg.target_dataset.img_height / ori_H)
        
        sample["K"] = torch.from_numpy(K).float()
        sample["inv_K"] = torch.from_numpy(np.linalg.pinv(K)).float()
        return sample