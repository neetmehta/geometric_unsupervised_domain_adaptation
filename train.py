import torch
from dataset.kitti import KittiOdom
from dataset.carla_dataset import CarlaDataset
from torch.utils.data import DataLoader
from layers.encoder import ResnetEncoder
from layers.depth_decoder import DepthDecoder
from layers.semantic_decoder import SemanticDecoder
from layers.posenet import PoseNet
from utils import load_config
from layers import transformation_from_parameters, Project3D, BackprojectDepth, disp_to_depth, get_smooth_loss, SSIM, BootstrappedCrossEntropyLoss, SupervisedSurfaceNormalLoss
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
    """Main training class for unsupervised domain adaptation using geometric constraints.
    
    Handles model initialization, training loop, loss computation, and visualization.
    Supports depth, semantic segmentation, and pose estimation tasks.
    """
    
    def __init__(self, cfg_path):
        """Initialize the Trainer with configuration.
        
        Args:
            cfg_path (str): Path to YAML configuration file.
        """
        cfg = load_config(cfg_path)
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        device = cfg.runtime.device
        
        # --- Model Initialization ---
        self.encoder = ResnetEncoder(num_layers=cfg.model.encoder.num_layers, pretrained=cfg.model.encoder.pretrained).to(device)
        self.depth_decoder = DepthDecoder(self.encoder.num_ch_enc, scales=cfg.geometry.scales).to(device)
        self.semantic_decoder = SemanticDecoder(self.encoder.num_ch_enc, cfg.model.semantic_decoder.num_classes).to(device)
        self.posenet = PoseNet(num_layers=cfg.model.pose_net.num_layers, num_input_images=cfg.model.pose_net.num_input_images, pretrained=cfg.model.pose_net.pretrained).to(device)
        self.scales = cfg.geometry.scales
        
        # --- Geometry Tools ---
        self.backproject_depth = BackprojectDepth(self.batch_size, cfg.geometry.image_height, cfg.geometry.image_width).to(device)
        self.project_3d = Project3D(self.batch_size, cfg.geometry.image_height, cfg.geometry.image_width).to(device)
        self.ssim = SSIM().to(device)
        self.surface_normal_loss = SupervisedSurfaceNormalLoss(cfg.geometry.image_height, cfg.geometry.image_width, self.batch_size).to(device)
        
        # TODO: explore ignore index
        self.bootstraped_cross_entropy_loss = BootstrappedCrossEntropyLoss()
        self.frames = cfg.geometry.frames
        
        # --- Optimizer ---
        self.parameters_to_train = list(self.encoder.parameters()) + list(self.depth_decoder.parameters()) + list(self.semantic_decoder.parameters())+ list(self.posenet.parameters())
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
        """Execute the main training loop for all epochs.
        
        Performs forward pass, computes losses, backward pass, and logs metrics.
        Saves checkpoint if validation loss improves.
        """
        num_batches = len(self.dataloader)
        global_step = 0
        self.lr_scheduler.step()
        
        for epoch in range(self.epoch):
            self.encoder.train()
            self.depth_decoder.train()
            self.semantic_decoder.train()
            self.posenet.train()

            epoch_loss = 0.0
            start_time = time.time()
            
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch [{epoch+1}/{self.epoch}]",
                dynamic_ncols=True
            )
            previous_loss = float('inf')
            
            for batch_idx, inputs in enumerate(pbar):
                iter_start = time.time()
                losses = {}
                
                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                # --- Forward Pass ---
                input_image = inputs[("t", 0, 0)]
                features = self.encoder(input_image)
                depth_outputs = self.depth_decoder(features)
                semantic_outputs = self.semantic_decoder(features)
                
                pose = self.predict_pose(inputs)
                self.reconstruct_image(inputs, depth_outputs, pose)
                losses.update(self.compute_reconstruction_loss(inputs, depth_outputs))

                losses.update(self.compute_semantic_loss(semantic_outputs, inputs[('semantic', 0,0)]))
                
                losses.update(self.compute_gt_depth_loss(depth_outputs, inputs[("depth", 0, 0)]))
                
                losses.update(self.compute_surface_normal_loss(depth_outputs, inputs[("depth", 0, 0)], inputs["inv_K"]))
                
                virtual_loss = losses["gt_depth_loss"] + 0.001*losses["semantic_loss"] + 0.01*losses["surface_normal_loss"] + 0.05*losses["reconstruction_loss"]
                # real_loss = losses["reconstruction_loss"]
                
                total_loss = virtual_loss
                
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
                    self.log_visuals(inputs, depth_outputs, global_step)

            # End of Epoch
            epoch_time = time.time() - start_time
            print(
                f"\nEpoch {epoch+1}/{self.epoch} | "
                f"Avg Loss: {epoch_loss / num_batches:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            if (epoch_loss/num_batches) < previous_loss:
                previous_loss = epoch_loss/num_batches
                state_dict = {"encoder": self.encoder.state_dict(), "decoder": self.depth_decoder.state_dict(), "posenet": self.posenet.state_dict()}
                torch.save(state_dict, os.path.join(self.model_out, f"model_{epoch}.pth"))
            
        self.writer.close()
        
    def predict_pose(self, inputs):
        """Predict relative poses between current frame and adjacent frames.
        
        Args:
            inputs (dict): Dictionary containing input image tensors with keys like ('t', frame_id, scale).
        
        Returns:
            dict: Pose dictionary with keys ('axisangle', frame_id), ('translation', frame_id), and ('T', frame_id).
        """
        pose = {}
        # Pose 1: t -> t-1
        pose[("axisangle", -1)], pose[("translation", -1)] = self.posenet(
            torch.cat([inputs[("t", -1, 0)], inputs[("t", 0, 0)]], dim=1)
        )

        # Pose 2: t -> t+1
        pose[("axisangle", 1)], pose[("translation", 1)] = self.posenet(
            torch.cat([inputs[("t", 0, 0)], inputs[("t", 1, 0)]], dim=1)
        )
        
        for i in self.frames[1:]:
            pose[("T", i)] = transformation_from_parameters(
                pose[("axisangle", i)][:,0], 
                pose[("translation", i)][:,0], 
                invert=(i == -1)
            )
        return pose
    
    def reconstruct_image(self, inputs, outputs, pose):
        """Reconstruct source frames into current frame using geometry and pose.
        
        Performs 3D backprojection, transformation based on pose, and reprojection.
        
        Args:
            inputs (dict): Input tensors including RGB and calibration matrices.
            outputs (dict): Output dictionary to store reconstructed frames and depth.
            pose (dict): Pose transformations between frames.
        """
        
        input_image = inputs[("t", 0, 0)]
        for s in self.scales:
            # Upsample disparity to input resolution
            disp = outputs[("disp", s)]
            disp = F.interpolate(
                disp, (input_image.size(2), input_image.size(3)), mode="bilinear", align_corners=False
            )
            _, depth = disp_to_depth(disp, 0.1, 100)
            outputs[("depth", s)] = depth

            for i in self.frames[1:]:
                
                # Geometry: Backproject -> Rotate/Translate -> Project
                cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                pix_coords = self.project_3d(cam_points, inputs[("K")], pose[("T", i)])
                
                # Sampling: Corrected Source Selection
                # FIX: Using 't-1' for i=-1 and 't+1' for i=1
                source_key = ("t", -1, 0) if i == -1 else ("t", 1, 0)
                outputs[("recons", i, s)] = F.grid_sample(
                    inputs[source_key], pix_coords, padding_mode="border"
                )
                
    def compute_reconstruction_loss(self, inputs, outputs):
        """Compute photometric reconstruction loss for unsupervised depth learning.
        
        Combines reprojection loss with identity consistency and smoothness regularization.
        
        Returns:
            dict: Dictionary containing total reconstruction loss.
        """
        
        losses = {}
        reprojection_loss = {}
        identity_reprojection_loss = {}
        total_loss = 0
        for s in self.scales:
            
            for i in self.frames[1:]:
                
                reprojection_loss[(i, s)] = self.compute_reprojection_loss(
                    outputs[("recons", i, s)], inputs[("t", 0, 0)]
                )
                
                identity_reprojection_loss[(i, s)] = self.compute_reprojection_loss(
                    inputs[("t", i, 0)], inputs[("t", 0, 0)]
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
            
            to_optimise, _ = torch.min(combined, dim=1)
            
            # Automasking/Minimum logic usually goes here (omitted for brevity as per your snippet)
            total_loss += to_optimise.mean()
            
            smoothness_loss = self.compute_smoothness_loss(outputs[("disp", s)], inputs[("t", 0, s)])
            total_loss += self.cfg.loss.smoothness_weight * smoothness_loss / (2 ** s) 
        
        losses["reconstruction_loss"] = total_loss / len(self.scales)
        return losses
            
    def compute_smoothness_loss(self, disp, scaled_image):
        """Compute smoothness regularization loss for disparity maps.
        
        Args:
            disp (torch.Tensor): Disparity map of shape [B, 1, H, W].
            scaled_image (torch.Tensor): Image tensor for computing gradients.
        
        Returns:
            torch.Tensor: Smoothness loss value.
        """
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smoothness_loss = get_smooth_loss(norm_disp, scaled_image)
        return smoothness_loss             
    
    def compute_semantic_loss(self, outputs, gt):
        """Compute semantic segmentation loss using bootstrapped cross-entropy.
        
        Args:
            outputs (torch.Tensor): Semantic predictions of shape [B, num_classes, H, W].
            gt (torch.Tensor): Ground truth semantic labels of shape [B, H, W].
        
        Returns:
            dict: Dictionary containing semantic loss.
        """
        return {"semantic_loss": self.bootstraped_cross_entropy_loss(outputs, gt)} 
    
    def compute_gt_depth_loss(self, disps, gt_depth, lam=0.85):
        """Compute supervised depth loss using log-scale consistency.
        
        Uses ground truth depth for computing scale-invariant loss.
        
        Args:
            disps (dict): Dictionary containing disparity predictions with key ('disp', 0).
            gt_depth (torch.Tensor): Ground truth depth map of shape [B, 1, H, W].
            lam (float): Weight for scale ambiguity term. Default: 0.85.
        
        Returns:
            dict: Dictionary containing gt_depth_loss.
        """
        eps = 1e-6
        min_depth = self.cfg.virtual_dataset.min_depth
        max_depth = self.cfg.virtual_dataset.max_depth
        disp = disps[("disp", 0)]
        
        _, pred_depth = disp_to_depth(disp, max_depth=max_depth, min_depth=min_depth)

        # valid depth mask
        mask = (gt_depth > min_depth) & (gt_depth < max_depth)
        pred = pred_depth[mask]
        gt = gt_depth[mask]

        log_diff = torch.log(pred + eps) - torch.log(gt + eps)

        mean_sq = torch.mean(log_diff ** 2)
        sq_mean = torch.mean(log_diff) ** 2

        loss = mean_sq - lam * sq_mean
        return {"gt_depth_loss": loss}
    
    def compute_surface_normal_loss(self, disps, gt_depth, inv_k):
        """Compute surface normal consistency loss.
        
        Args:
            disps (dict): Dictionary containing disparity predictions.
            gt_depth (torch.Tensor): Ground truth depth map.
            inv_k (torch.Tensor): Inverse camera intrinsic matrix.
        
        Returns:
            dict: Dictionary containing surface_normal_loss.
        """
        
        min_depth = self.cfg.virtual_dataset.min_depth
        max_depth = self.cfg.virtual_dataset.max_depth
        disp = disps[("disp", 0)]
        
        _, pred_depth = disp_to_depth(disp, max_depth=max_depth, min_depth=min_depth)
        loss = self.surface_normal_loss(pred_depth, gt_depth, inv_k)
        return {"surface_normal_loss": loss}
    
    def compute_partial_photometric_loss(self, inputs, depth_outputs, pose, gt_depth):
        """Compute partial photometric loss for domain-specific adaptation.
        
        Args:
            inputs (dict): Input tensors.
            depth_outputs (dict): Depth predictions.
            pose (dict): Pose transformations.
            gt_depth (torch.Tensor): Ground truth depth.
        
        Returns:
            dict: Dictionary containing partial photometric loss.
        """
        pass
  

    def log_visuals(self, data, outputs, step):
        """Log RGB, reconstructions, and depth visualizations to TensorBoard.
        
        Creates a combined visualization showing input RGB, reconstructed image, and depth.
        
        Args:
            data (dict): Input data dictionary containing RGB images.
            outputs (dict): Model outputs containing reconstructions and depth.
            step (int): Global training step for logging.
        """
        
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
        """Compute photometric reprojection loss using SSIM and L1 loss.
        
        Args:
            pred (torch.Tensor): Predicted/reconstructed image of shape [B, 3, H, W].
            target (torch.Tensor): Target image of shape [B, 3, H, W].
        
        Returns:
            torch.Tensor: Per-pixel reprojection loss of shape [B, 1, H, W].
        """
        l1_loss = torch.abs(target - pred).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss
    
def main():
    """Main entry point for training script.
    
    Parses command-line arguments and initializes the Trainer.
    """
    parser = argparse.ArgumentParser(description="Monodepth2-style Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    trainer = Trainer(cfg_path=args.config)
    trainer.train()

if __name__ == "__main__":
    main()