import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from .cityscapes_color_palette import CITYSCAPES_COLORS

IGNORE_LABEL = 255

class CarlaDataset(Dataset):
    """CARLA synthetic dataset loader for unsupervised depth and semantic segmentation.
    
    Loads RGB images from multiple temporal frames, depth maps, semantic masks,
    and camera calibration matrices. Supports multi-scale processing.
    """
    
    def __init__(self, cfg):
        """Initialize CARLA dataset.
        
        Args:
            cfg (SimpleNamespace): Configuration object with virtual_dataset,
                geometry, and other training settings.
        """
        self.cfg = cfg
        root_dir = cfg.virtual_dataset.root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((cfg.virtual_dataset.img_height, cfg.virtual_dataset.img_width)),
            transforms.ToTensor(),
        ])
        self.calib = {}
        self.samples = []
        self.scales = cfg.geometry.scales
        self.resize_transform = [transforms.Compose([transforms.Resize((cfg.virtual_dataset.img_height // (2 ** s), cfg.virtual_dataset.img_width // (2 ** s))), transforms.ToTensor()]) for s in self.scales]
        for scene in glob.glob(os.path.join(root_dir, "run_*")):
            self.samples.extend(self.parse_scenes(scene))
            self.calib[scene] = self.load_K(scene, "image_2", cfg.virtual_dataset.img_height, cfg.virtual_dataset.img_width)

    def parse_scenes(self, scene):
        """Parse scene directory and extract valid frame triplets.
        
        Identifies consecutive frames (t-1, t, t+1) that exist in the dataset.
        
        Args:
            scene (str): Path to scene directory.
        
        Returns:
            list: List of dictionaries with keys 'base_name', 'frame_id', and 'scene'.
        """
        image_paths = [i for i in glob.glob(os.path.join(scene, "image_2", "rgb_images", "*.jpg"))]

        samples = []
        sorted_images = sorted(image_paths, key=lambda x: int(os.path.normpath(x).split(os.path.sep)[-1].split(".")[0]))
        for image_path in sorted_images:
            image_dir = os.path.dirname(image_path)
            base_name = os.path.dirname(image_dir)
            id_t = int(os.path.normpath(image_path).split(os.path.sep)[-1].split(".")[0])
            image_frame_t_minus_1 = os.path.join(image_dir, f"{id_t - 1}.jpg")
            image_frame_t_plus_1 = os.path.join(image_dir, f"{id_t + 1}.jpg")
            
            if os.path.exists(image_frame_t_plus_1) and os.path.exists(image_frame_t_minus_1):
                samples.append({"base_name": base_name, "frame_id": id_t, "scene": scene})
        return samples
    
    def load_depth(self, path):
        """Load depth map from numpy file.
        
        Args:
            path (str): Path to .npy depth file.
        
        Returns:
            numpy.ndarray: Loaded depth map.
        """
        depth = np.load(path)
        return depth
    
    def cityscapes_color_to_mask(self, label_img):
        """Convert Cityscapes color-encoded labels to semantic class mask.
        
        Maps RGB color values to semantic class IDs using predefined palette.
        
        Args:
            label_img (torch.Tensor): RGB label image of shape (H, W, 3).
        
        Returns:
            torch.Tensor: Semantic mask of shape (H, W) with class IDs.
        """
        h, w, _ = label_img.shape
        mask = torch.full((h, w), IGNORE_LABEL, dtype=torch.uint8)

        for color, class_id in CITYSCAPES_COLORS.items():
            color = torch.tensor(color)
            matches = torch.all(label_img == color, dim=-1)
            mask[matches] = class_id

        return mask
    
    def load_K(self, path, camera_name, H, W):
        """Load and scale camera intrinsic matrix from calibration file.
        
        Reads calibration file and constructs 4x4 intrinsic matrix scaled to image size.
        
        Args:
            path (str): Path to camera directory.
            camera_name (str): Camera name (e.g., 'image_2').
            H (int): Image height for scaling.
            W (int): Image width for scaling.
        
        Returns:
            numpy.ndarray: 4x4 camera intrinsic matrix.
        """
        calib_path = os.path.join(path, camera_name, "calib.txt")
        k = np.loadtxt(calib_path, dtype=np.float32)
        K = np.eye(4)
        K[0, :3] = k[0, :3]
        K[1, :3] = k[1, :3]
        K[0, :] *= W
        K[1, :] *= H
        return K

    def __len__(self):
        """Return number of valid samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a data sample containing RGB images, depth, semantic labels, and calibration.
        
        Args:
            idx (int): Sample index.
        
        Returns:
            dict: Dictionary with keys for RGB frames at different scales,
                  depth maps, semantic labels, and calibration matrices.
        """
        sample = {}
        paths = self.samples[idx]
        
        # RGB image
        for s in self.scales:
            sample[("t", 0, s)] = self.resize_transform[s](Image.open(os.path.join(paths["base_name"], "rgb_images", f"{paths['frame_id']}.jpg")).convert("RGB"))
        sample[("t", -1, 0)] = self.transforms(Image.open(os.path.join(paths["base_name"], "rgb_images", f"{paths['frame_id']-1}.jpg")).convert("RGB"))
        sample[("t", 1, 0)] = self.transforms(Image.open(os.path.join(paths["base_name"], "rgb_images", f"{paths['frame_id']+1}.jpg")).convert("RGB"))
        
        if self.cfg.virtual_dataset.depth:
            depth = self.load_depth(os.path.join(paths["base_name"], "depth", f"{paths['frame_id']}.npy"))
            depth = torch.from_numpy(depth)
            sample[("depth", 0, 0)] = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(self.cfg.virtual_dataset.img_height, self.cfg.virtual_dataset.img_width), mode='bilinear', align_corners=False).squeeze(0)   
            
        if self.cfg.virtual_dataset.semantic:
            semantic_image = np.load(os.path.join(paths["base_name"], "semantic_mask", f"{paths['frame_id']}.npy"))
            semantic_image = cv2.resize(semantic_image, (self.cfg.virtual_dataset.img_width, self.cfg.virtual_dataset.img_height), 0, 0, interpolation=cv2.INTER_NEAREST)
            semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2RGB)
            sample[("semantic", 0, 0)] = self.cityscapes_color_to_mask(semantic_image).to(torch.int64)
        
        sample['K'] = torch.from_numpy(self.calib[paths["scene"]]).to(torch.float32)
        sample['inv_K'] = torch.linalg.pinv(sample['K']).to(torch.float32)
        return sample