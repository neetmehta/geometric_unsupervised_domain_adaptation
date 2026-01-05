import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, cfg):
        root_dir = cfg.dataset.root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((cfg.dataset.img_height, cfg.dataset.img_width)),
            transforms.ToTensor(),
        ])
        self.calib = {}
        self.samples = []
        self.scales = cfg.model.depth_decoder.scales
        self.resize_transform = [transforms.Compose([transforms.Resize((cfg.dataset.img_height // (2 ** s), cfg.dataset.img_width // (2 ** s))), transforms.ToTensor()]) for s in self.scales]
        for scene in glob.glob(os.path.join(root_dir, "run_*")):
            self.samples.extend(self.parse_scenes(scene))
            self.calib[scene] = self.load_K(scene, "image_2", cfg.dataset.img_height, cfg.dataset.img_width)

    def parse_scenes(self, scene):
        image_paths = [i for i in glob.glob(os.path.join(scene, "image_2", "rgb_images", "*.jpg"))]
        samples = []
        sorted_images = sorted(image_paths, key=lambda x: int(os.path.normpath(x).split(os.path.sep)[-1].split(".")[0]))
        for image_path in sorted_images:
            base_name = os.path.dirname(image_path)
            id_t = int(os.path.normpath(image_path).split(os.path.sep)[-1].split(".")[0])
            frame_t = image_path
            frame_t_minus_1 = os.path.join(base_name, f"{id_t - 1}.jpg")
            frame_t_plus_1 = os.path.join(base_name, f"{id_t + 1}.jpg")
            
            if os.path.exists(frame_t_plus_1) and os.path.exists(frame_t_minus_1):
                samples.append({"scene": scene, "t-1": frame_t_minus_1, "t": frame_t, "t+1": frame_t_plus_1})
        return samples
    
    def load_K(self, path, camera_name, H, W):
        calib_path = os.path.join(path, camera_name, "calib.txt")
        k = np.loadtxt(calib_path, dtype=np.float32)
        K = np.eye(4)
        K[0, :3] = k[0, :3]
        K[1, :3] = k[1, :3]
        K[0, :] *= W
        K[1, :] *= H
        return K

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        for s in self.scales:
            sample[("t", 0, s)] = self.resize_transform[s](Image.open(sample["t"]).convert("RGB"))
        sample[("t", 0, 0)] = self.transforms(Image.open(sample["t"]).convert("RGB"))
        sample[("t", -1, 0)] = self.transforms(Image.open(sample["t-1"]).convert("RGB"))
        sample[("t", 1, 0)] = self.transforms(Image.open(sample["t+1"]).convert("RGB"))
        
        sample['K'] = torch.from_numpy(self.calib[sample["scene"]]).to(torch.float32)
        sample['inv_K'] = torch.from_numpy(np.linalg.inv(sample['K'])).to(torch.float32)
        return sample