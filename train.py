import torch
from dataset.kitti import KittiOdom
from dataset.carla_dataset import CarlaDataset
from torch.utils.data import DataLoader
from layers.encoder import ResnetEncoder
from layers.depth_decoder import DepthDecoder
from layers.posenet import PoseNet
from utils import load_config
from layers import transformation_from_parameters, Project3D, BackprojectDepth, disp_to_depth, get_smooth_loss, SSIM
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
import time
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

class Trainer:
    
    def __init__(self, cfg_path):
        cfg = load_config(cfg_path)
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        device = cfg.runtime.device
        
        # --- Model Initialization ---
        self.encoder = ResnetEncoder(num_layers=cfg.model.encoder.num_layers, pretrained=cfg.model.encoder.pretrained).to(device)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=cfg.geometry.scales).to(device)
        self.posenet = PoseNet(num_layers=cfg.model.pose_net.num_layers, num_input_images=cfg.model.pose_net.num_input_images, pretrained=cfg.model.pose_net.pretrained).to(device)
        self.scales = cfg.geometry.scales
        
        # --- Geometry Tools ---
        self.backproject_depth = BackprojectDepth(self.batch_size, cfg.geometry.image_height, cfg.geometry.image_width).to(device)
        self.project_3d = Project3D(self.batch_size, cfg.geometry.image_height, cfg.geometry.image_width).to(device)
        self.ssim = SSIM().to(device)
        
        # --- Optimizer ---
        self.parameters_to_train = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.posenet.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, lr=cfg.training.learning_rate)
        self.lr_scheduler =  optim.lr_scheduler.StepLR(
            self.optimizer, self.cfg.training.scheduler_step_size, 0.1)
        
        # --- Data ---
        self.device = device
        dataset = CarlaDataset(cfg)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=cfg.training.shuffle, num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, drop_last=cfg.training.drop_last)
        self.epoch = cfg.training.epochs

        # --- TensorBoard Setup ---
        # Defaults to 'runs/current_time' if not specified in config
        log_dir = getattr(cfg.training, 'log_dir', None) 
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_out = self.writer.log_dir
        
        # Frequencies
        self.log_loss_freq = cfg.tensorboard.log_loss_freq   # Log loss every 50 steps
        self.log_img_freq = cfg.tensorboard.log_img_freq   # Log images every 500 steps

    def train(self):
        num_batches = len(self.dataloader)
        global_step = 0
        self.lr_scheduler.step()
        
        for epoch in range(self.epoch):
            self.encoder.train()
            self.decoder.train()
            self.posenet.train()

            epoch_loss = 0.0
            start_time = time.time()
            
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch [{epoch+1}/{self.epoch}]",
                dynamic_ncols=True
            )
            previous_loss = float('inf')
            
            for batch_idx, data in enumerate(pbar):
                iter_start = time.time()
                
                # Move data to device
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(self.device)
                
                # --- Forward Pass ---
                input_image = data[("t", 0, 0)]
                features = self.encoder(input_image)
                outputs = self.decoder(features)
                
                pose = {}
                reprojection_loss = {}
                identity_reprojection_loss = {}
                
                # Pose 1: t -> t-1
                pose[("axisangle", -1)], pose[("translation", -1)] = self.posenet(
                    torch.cat([data[("t", -1, 0)], data[("t", 0, 0)]], dim=1)
                )

                # Pose 2: t -> t+1
                pose[("axisangle", 1)], pose[("translation", 1)] = self.posenet(
                    torch.cat([data[("t", 0, 0)], data[("t", 1, 0)]], dim=1)
                )

                total_loss = 0
                scales = self.scales
                for i in [-1, 1]:
                    pose[("T", i)] = transformation_from_parameters(
                        pose[("axisangle", i)][:,0], 
                        pose[("translation", i)][:,0], 
                        invert=(i == -1)
                    )
                
                for s in scales:
                    # Upsample disparity to input resolution
                    disp = outputs[("disp", s)]
                    disp = F.interpolate(
                        disp, (input_image.size(2), input_image.size(3)), mode="bilinear", align_corners=False
                    )
                    _, depth = disp_to_depth(disp, 0.1, 100)
                    outputs[("depth", s)] = depth

                    for i in [-1, 1]:
                        
                        # Geometry: Backproject -> Rotate/Translate -> Project
                        cam_points = self.backproject_depth(depth, data[("inv_K")])
                        pix_coords = self.project_3d(cam_points, data[("K")], pose[("T", i)])
                        
                        # Sampling: Corrected Source Selection
                        # FIX: Using 't-1' for i=-1 and 't+1' for i=1
                        source_key = ("t", -1, 0) if i == -1 else ("t", 1, 0)
                        outputs[("recons", i, s)] = F.grid_sample(
                            data[source_key], pix_coords, padding_mode="border"
                        )
                        
                        reprojection_loss[(i, s)] = self.compute_reprojection_loss(
                            outputs[("recons", i, s)], data[("t", 0, 0)]
                        )
                        
                        identity_reprojection_loss[(i, s)] = self.compute_reprojection_loss(
                            data[("t", i, 0)], data[("t", 0, 0)]
                        )            
                        
                    # Combine Losses
                    reprojection_losses = torch.cat(
                        [reprojection_loss[(-1, s)], reprojection_loss[(1, s)]], dim=1
                    )
                    
                    identity_reprojection_losses = torch.cat(
                        [identity_reprojection_loss[(-1, s)], identity_reprojection_loss[(1,s)]], dim=1
                    )
                    
                    identity_reprojection_losses += torch.randn(
                        identity_reprojection_losses.shape, device=self.device) * 0.00001
                    
                    combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
                    
                    to_optimise, idxs = torch.min(combined, dim=1)
                    
                    
                    # Automasking/Minimum logic usually goes here (omitted for brevity as per your snippet)
                    total_loss += to_optimise.mean()
                    
                    # Smoothness Loss
                    mean_disp = outputs[("disp", s)].mean(2, True).mean(3, True)
                    norm_disp = outputs[("disp", s)] / (mean_disp + 1e-7)
                    smoothness_loss = get_smooth_loss(norm_disp, data[("t", 0, s)])
                    total_loss += self.cfg.loss.smoothness_weight * smoothness_loss / (2 ** s)
                
                total_loss /= len(scales)
                
                # --- Backward Pass ---
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # --- Stats ---
                epoch_loss += total_loss.item()
                iter_time = time.time() - iter_start
                samples_per_sec = self.batch_size / iter_time
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                    "s/s": f"{samples_per_sec:.1f}"
                })
                
                # --- TensorBoard Logging ---
                if global_step % self.log_loss_freq == 0:
                    self.writer.add_scalar("Train/Total_Loss", total_loss.item(), global_step)
                    
                if global_step % self.log_img_freq == 0:
                    self.log_visuals(data, outputs, global_step)

            # End of Epoch
            epoch_time = time.time() - start_time
            print(
                f"\nEpoch {epoch+1}/{self.epoch} | "
                f"Avg Loss: {epoch_loss / num_batches:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            if (epoch_loss/num_batches) < previous_loss:
                previous_loss = epoch_loss/num_batches
                state_dict = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict(), "posenet": self.posenet.state_dict()}
                torch.save(state_dict, os.path.join(self.model_out, f"model_{epoch}.pth"))
            
        self.writer.close()

    def log_visuals(self, data, outputs, step):
        """Logs RGB, Reconstructions, and Colormapped Depth to TensorBoard"""
        
        # 1. Input RGB Image (Batch index 0)
        input_rgb = data[("t",0, 0)][0] # [3, H, W]
        
        # 2. Reconstructed Image (Using t-1 -> t, Scale 0)
        recon_rgb = outputs[("recons", -1, 0)][0] # [3, H, W]
        
        # 3. Depth Map (Scale 0)
        disp = outputs[("disp", 0)][0] # [1, H, W]
        # Normalize disparity for visualization (Min-Max scaling)
        disp_max = disp.max()
        disp_min = disp.min()
        disp_norm = (disp - disp_min) / (disp_max - disp_min + 1e-6)
        
        # Apply 'magma' colormap: [1, H, W] -> [H, W] -> colormap -> [H, W, 4] -> [3, H, W]
        cm = plt.get_cmap('magma')
        disp_colored = cm(disp_norm.squeeze().cpu().detach().numpy())[..., :3] # Drop Alpha
        disp_colored = torch.from_numpy(disp_colored).permute(2, 0, 1).to(self.device).float()
        
        # Stack images vertically: Input | Reconstruction | Depth
        combined_img = torch.cat((input_rgb, recon_rgb, disp_colored), dim=1) # Concatenate along Height
        
        # Create a grid (useful if you want to log multiple batch items, currently just 1)
        grid = vutils.make_grid(combined_img, normalize=False)
        
        self.writer.add_image("Train/Input_Recons_Depth", grid, step)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        l1_loss = torch.abs(target - pred).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss
    
def main():
    parser = argparse.ArgumentParser(description="Monodepth2-style Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    trainer = Trainer(cfg_path=args.config)
    trainer.train()

if __name__ == "__main__":
    main()