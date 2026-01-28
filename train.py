import torch
from dataset.guda_dataset import GudaDataset
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
import numpy as np
import os

class Trainer:
    
    def __init__(self, cfg_path):
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
        
        self.bootstraped_cross_entropy_loss = BootstrappedCrossEntropyLoss()
        self.frames = cfg.geometry.frames
        
        # --- Optimizer ---
        self.parameters_to_train = list(self.encoder.parameters()) + list(self.depth_decoder.parameters()) + list(self.semantic_decoder.parameters())+ list(self.posenet.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, lr=cfg.training.learning_rate)
        self.lr_scheduler =  optim.lr_scheduler.StepLR(
            self.optimizer, self.cfg.training.scheduler_step_size, 0.1)
        
        # --- Data ---
        self.device = device
        dataset = GudaDataset(cfg)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=cfg.training.shuffle, num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, drop_last=cfg.training.drop_last)
        self.epoch = cfg.training.epochs

        # --- TensorBoard Setup ---
        log_dir = getattr(cfg.training, 'log_dir', None) 
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_out = self.writer.log_dir
        
        # Frequencies
        self.log_loss_freq = cfg.tensorboard.log_loss_freq   
        self.log_img_freq = cfg.tensorboard.log_img_freq   

        # Create color map for semantic segmentation visualization
        self.num_classes = cfg.model.semantic_decoder.num_classes
        self.semantic_cmap = self.create_semantic_colormap(self.num_classes)

    def create_semantic_colormap(self, num_classes):
        """Creates a fixed colormap for semantic classes"""
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20)[:3] for i in range(num_classes)]
        return torch.tensor(colors, device=self.device).float()

    def train(self):
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
            
            for batch_idx, (virtuals_inputs, targets_inputs) in enumerate(pbar):
                iter_start = time.time()
                
                # Move inputs to device
                for key in virtuals_inputs:
                    if isinstance(virtuals_inputs[key], torch.Tensor):
                        virtuals_inputs[key] = virtuals_inputs[key].to(self.device)
                        
                for key in targets_inputs:
                    if isinstance(targets_inputs[key], torch.Tensor):
                        targets_inputs[key] = targets_inputs[key].to(self.device)
                
                # --- Forward Pass & Loss Calculation ---
                
                # 1. Virtual Batch Processing
                virtual_loss_dict, virtual_outputs = self.process_virtual_batch(virtuals_inputs)
                
                virtual_weighted_loss = (
                    self.cfg.loss.semantic_weight * virtual_loss_dict["semantic_loss"] + 
                    self.cfg.loss.supervised_depth_weight * virtual_loss_dict["gt_depth_loss"] + 
                    self.cfg.loss.surface_normal_weight * virtual_loss_dict["surface_normal_loss"] + 
                    self.cfg.loss.partial_photometric_weight * virtual_loss_dict["partial_photometric_loss"]
                )
                
                # 2. Real (Target) Batch Processing
                target_loss_dict, target_outputs = self.process_target_batch(targets_inputs)
                target_weighted_loss = target_loss_dict["reconstruction_loss"]
                
                # Total Loss
                total_loss = virtual_weighted_loss + target_weighted_loss
                
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
                    # Log Total Loss
                    self.writer.add_scalar("Train/Total_Loss", total_loss.item(), global_step)
                    
                    # Log Individual Virtual Losses
                    self.writer.add_scalar("Train/Losses/Virtual_Semantic", virtual_loss_dict["semantic_loss"].item(), global_step)
                    self.writer.add_scalar("Train/Losses/Virtual_GT_Depth", virtual_loss_dict["gt_depth_loss"].item(), global_step)
                    self.writer.add_scalar("Train/Losses/Virtual_Surface_Normal", virtual_loss_dict["surface_normal_loss"].item(), global_step)
                    self.writer.add_scalar("Train/Losses/Virtual_Partial_Photometric", virtual_loss_dict["partial_photometric_loss"].item(), global_step)
                    
                    # Log Real Losses
                    self.writer.add_scalar("Train/Losses/Real_Reconstruction", target_loss_dict["reconstruction_loss"].item(), global_step)
                    
                if global_step % self.log_img_freq == 0:
                    self.log_visuals(virtuals_inputs, virtual_outputs, targets_inputs, target_outputs, global_step)

            # End of Epoch
            epoch_time = time.time() - start_time
            print(
                f"\nEpoch {epoch+1}/{self.epoch} | "
                f"Avg Loss: {epoch_loss / num_batches:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            if (epoch_loss/num_batches) < previous_loss:
                previous_loss = epoch_loss/num_batches
                state_dict = {"encoder": self.encoder.state_dict(), "decoder": self.depth_decoder.state_dict(), "posenet": self.posenet.state_dict(), "semantic": self.semantic_decoder.state_dict()}
                torch.save(state_dict, os.path.join(self.model_out, f"model_{epoch}.pth"))
            
        self.writer.close()
        
    def process_target_batch(self, inputs):
        losses = {}
        input_image = inputs[("t", 0, 0)]
        features = self.encoder(input_image)
        depth_outputs = self.depth_decoder(features)
        semantic_outputs = self.semantic_decoder(features)
        
        # Store semantic output for visualization (taking argmax)
        outputs = {**depth_outputs}
        outputs["semantic_logits"] = semantic_outputs
        
        pose = self.predict_pose(inputs)
        self.reconstruct_image(inputs, outputs, pose)
        losses.update(self.compute_reconstruction_loss(inputs, outputs))
        
        return losses, outputs
        
    def process_virtual_batch(self, inputs):
        losses = {}
        input_image = inputs[("t", 0, 0)]
        features = self.encoder(input_image)
        depth_outputs = self.depth_decoder(features)
        semantic_outputs = self.semantic_decoder(features)
        
        outputs = {**depth_outputs}
        outputs["semantic_logits"] = semantic_outputs

        pose = self.predict_pose(inputs)
        self.reconstruct_image(inputs, outputs, pose)

        losses.update(self.compute_semantic_loss(semantic_outputs, inputs[('semantic', 0,0)]))
        losses.update(self.compute_gt_depth_loss(depth_outputs, inputs[("depth", 0, 0)]))
        losses.update(self.compute_surface_normal_loss(depth_outputs, inputs[("depth", 0, 0)], inputs["inv_K"]))
        losses.update(self.compute_partial_photometric_loss(inputs, depth_outputs, pose))
        
        return losses, outputs
        
    def predict_pose(self, inputs):
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
    
    def reconstruct_image_from_depth(self, inputs, outputs, pose, scales):
        
        resonstructed_image = {}
        for s in scales:
            
            for i in self.frames[1:]:
                # Backproject depth to 3D points
                cam_points = self.backproject_depth(outputs[("depth", 0, s)], inputs[("inv_K")])
                # Project 3D points into the other view
                pix_coords = self.project_3d(cam_points, inputs[("K")], pose[("T", i)])
                # Sampling
                source_key = ("t", -1, 0) if i == -1 else ("t", 1, 0)
                resonstructed_image[("recons", i, s)] = F.grid_sample(
                        inputs[source_key], pix_coords, padding_mode="border"
                    )
        return resonstructed_image
    
    def reconstruct_image(self, inputs, outputs, pose):
        
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
                
                # Sampling
                source_key = ("t", -1, 0) if i == -1 else ("t", 1, 0)
                outputs[("recons", i, s)] = F.grid_sample(
                    inputs[source_key], pix_coords, padding_mode="border"
                )
                
    def compute_reconstruction_loss(self, inputs, outputs, compute_smoothness_loss=True):
        
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
            
            total_loss += to_optimise.mean()
            if compute_smoothness_loss:
                smoothness_loss = self.compute_smoothness_loss(outputs[("disp", s)], inputs[("t", 0, s)])
                total_loss += self.cfg.loss.smoothness_weight * smoothness_loss / (2 ** s) 
        
        losses["reconstruction_loss"] = total_loss / len(self.scales)
        return losses
        
    def compute_smoothness_loss(self, disp, scaled_image):
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smoothness_loss = get_smooth_loss(norm_disp, scaled_image)
        return smoothness_loss             
    
    def compute_semantic_loss(self, outputs, gt):
        return {"semantic_loss": self.bootstraped_cross_entropy_loss(outputs, gt)} 
    
    def compute_gt_depth_loss(self, disps, gt_depth, lam=0.85):
        eps = 1e-6
        min_depth = self.cfg.virtual_dataset.min_depth
        max_depth = self.cfg.virtual_dataset.max_depth
        disp = disps[("disp", 0)]
        
        _, pred_depth = disp_to_depth(disp, max_depth=max_depth, min_depth=min_depth)

        mask = (gt_depth > min_depth) & (gt_depth < max_depth)
        pred = pred_depth[mask]
        gt = gt_depth[mask]

        log_diff = torch.log(pred + eps) - torch.log(gt + eps)

        mean_sq = torch.mean(log_diff ** 2)
        sq_mean = torch.mean(log_diff) ** 2

        loss = mean_sq - lam * sq_mean
        return {"gt_depth_loss": loss}
    
    def compute_surface_normal_loss(self, disps, gt_depth, inv_k):
        min_depth = self.cfg.virtual_dataset.min_depth
        max_depth = self.cfg.virtual_dataset.max_depth
        disp = disps[("disp", 0)]
        
        _, pred_depth = disp_to_depth(disp, max_depth=max_depth, min_depth=min_depth)
        loss = self.surface_normal_loss(pred_depth, gt_depth, inv_k)
        return {"surface_normal_loss": loss}
    
    def compute_partial_photometric_loss(self, inputs, depth_outputs, pose):
        
        self.reconstruct_image(inputs, depth_outputs, pose)
        pred_depth_pred_pose_loss = self.compute_reconstruction_loss(inputs, depth_outputs)
        
        # Reconstruct using ground truth pose
        self.reconstruct_image(inputs, depth_outputs, inputs)
        pred_depth_gt_pose_loss = self.compute_reconstruction_loss(inputs, depth_outputs)
        
        # Reconstruct using ground truth depth
        gt_depth_outputs = self.reconstruct_image_from_depth(inputs, inputs, pose, [0])
        temp_scales = self.scales
        self.scales = [0]
        gt_depth_pred_pose_loss = self.compute_reconstruction_loss(inputs, gt_depth_outputs, False)
        self.scales = temp_scales
        
        total_loss = (pred_depth_pred_pose_loss["reconstruction_loss"] + pred_depth_gt_pose_loss["reconstruction_loss"] + gt_depth_pred_pose_loss["reconstruction_loss"])/3
        
        return {"partial_photometric_loss": total_loss}
  
    def _colorize_depth(self, disp):
        """Colorizes disparity/depth map using 'magma' colormap"""
        disp = disp.detach().cpu()
        disp_max = disp.max()
        disp_min = disp.min()
        disp_norm = (disp - disp_min) / (disp_max - disp_min + 1e-6)
        
        cm = plt.get_cmap('magma')
        # [H, W] -> [H, W, 4] -> [H, W, 3] -> [3, H, W]
        colored = cm(disp_norm.squeeze().numpy())[..., :3]
        return torch.from_numpy(colored).permute(2, 0, 1).float().to(self.device)

    def _colorize_semantic(self, logits):
        """Converts semantic logits to RGB map using the fixed colormap"""
        # logits: [C, H, W] -> argmax -> [H, W]
        labels = torch.argmax(logits, dim=0) # [H, W]
        # Map indices to colors
        # Output [H, W, 3] -> [3, H, W]
        colored = self.semantic_cmap[labels].permute(2, 0, 1)
        return colored
    
    def _colorize_semantic_gt(self, gt_labels):
        """Converts semantic ground truth to RGB map"""
        # gt_labels: [1, H, W] -> [H, W]
        labels = gt_labels.squeeze(0).long()
        colored = self.semantic_cmap[labels].permute(2, 0, 1)
        return colored

    def log_visuals(self, virt_in, virt_out, real_in, real_out, step):
        """Logs separated visuals for Virtual and Real domains"""
        
        with torch.no_grad():
            # --- Virtual Data ---
            v_rgb = virt_in[("t", 0, 0)][0]  # Input RGB
            
            # Depth
            v_disp_pred = virt_out[("disp", 0)][0]
            v_depth_vis = self._colorize_depth(v_disp_pred)
            v_depth_gt = virt_in[("depth", 0, 0)][0] 
            # GT depth is usually absolute, normalize it for viz or use 1/depth for disp-like viz
            v_depth_gt_vis = self._colorize_depth(1.0 / (v_depth_gt + 1e-7))

            # Semantic
            v_sem_pred = virt_out["semantic_logits"][0]
            v_sem_vis = self._colorize_semantic(v_sem_pred)
            v_sem_gt = virt_in[("semantic", 0, 0)][0]
            v_sem_gt_vis = self._colorize_semantic_gt(v_sem_gt)

            # Stack Virtual: Row 1 [RGB, Pred Depth, GT Depth], Row 2 [RGB, Pred Sem, GT Sem]
            virt_row1 = torch.cat([v_rgb, v_depth_vis, v_depth_gt_vis], dim=2)
            virt_row2 = torch.cat([v_rgb, v_sem_vis, v_sem_gt_vis], dim=2)
            
            # --- Real Data ---
            r_rgb = real_in[("t", 0, 0)][0] # Input RGB
            
            # Depth
            r_disp_pred = real_out[("disp", 0)][0]
            r_depth_vis = self._colorize_depth(r_disp_pred)
            
            # Reconstruction
            r_recon = real_out[("recons", -1, 0)][0] # Reconstruction from t-1
            
            # Semantic
            r_sem_pred = real_out["semantic_logits"][0]
            r_sem_vis = self._colorize_semantic(r_sem_pred)
            
            # Stack Real: [RGB, Pred Depth, Pred Semantic, Reconstruction]
            real_row = torch.cat([r_rgb, r_depth_vis, r_sem_vis, r_recon], dim=2)

            # --- Log to TensorBoard ---
            self.writer.add_image("Visuals/Virtual_Depth_Compare", virt_row1, step)
            self.writer.add_image("Visuals/Virtual_Semantic_Compare", virt_row2, step)
            self.writer.add_image("Visuals/Real_Predictions", real_row, step)

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