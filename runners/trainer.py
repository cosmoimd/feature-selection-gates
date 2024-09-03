#!/usr/bin/env python
# This scripts contains all the utility functions used in the toolbox.

# This code refers to the following paper:
# "Feature Selection Gates with Gradient Routing for Endoscopic Image Computing"
# Accepted for publication at MICCAI 2024.

# Please cite both the accepted version and the preprint version if you use this code,
# its methods, ideas, or any part of them.

# Accepted Publication:
# @inproceedings{roffo2024FSG,
#    title={Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
#    author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
#    booktitle={MICCAI 2024, the 27th International Conference on Medical Image Computing and Computer Assisted Intervention, Marrakech, Morocco, October 2024.},
#    year={2024}
#    organization={Springer}
# }
# Preprint version:
# @misc{roffo2024hardattention,
#    title={Hard-Attention Gates with Gradient Routing for Endoscopic Image Computing},
#    author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
#    year={2024},
#    eprint={2407.04400},
#    archivePrefix={arXiv},
#    primaryClass={eess.IV}
# }


# Import built-in modules
import csv
import importlib
import json
import logging
import os
import shutil
import time

# Import 3rd party modules
import numpy as np
# Import PyTorch modules
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tabulate import tabulate
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
# Import custom modules
from modules.analytics import calculate_metrics
from modules.analytics import visualizations
from modules.analytics.calculate_metrics import compute_metrics_testing
from modules.models.fsg_vision_transformers import FSGViT
from modules.models.vision_transformers import ViTBaseline
from modules.models.multi_stream_nets import FSGNetR18, FSGNetEffNet


# Import plot modules

# File information
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2023, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"


class Trainer:
    """
    The Trainer class handles the training and validation of a given model.

    Attributes:
        config: Configuration object providing trainer parameters.
        model: PyTorch model to train.
        loss_fn: The loss function to minimize.
        optimizer: The optimizer to update the model parameters.
        lr_scheduler: Scheduler for learning rate adjustments.
        writer: Tensorboard writer for logging metrics and other training process information.
    """

    def __init__(self, config, model, loss_fn, gr_ofs_loss, optimizer, GROFS_optimizer, lr_scheduler, lr_scheduler_gr):
        """
        Trainer constructor.

        Args:
            config: Configuration object providing trainer parameters.
            model: PyTorch model to train.
            loss_fn: The loss function to minimize.
            optimizer: The optimizer to update the model parameters.
            lr_scheduler: Scheduler for learning rate adjustments.
        """
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.gr_ofs_loss = gr_ofs_loss
        self.optimizer = optimizer
        self.GROFS_optimizer = GROFS_optimizer # Gradient Routing For Online Feature Selection - Optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_gr = lr_scheduler_gr
        self.writer = SummaryWriter()  # Tensorboard writer

        self.grad_scaler_amp = GradScaler()
        self.depth_map_enabled = self.config['model_params']['use_depth']
        self.use_mask = self.config['model_params']['use_mask']

        self.enable_fsag = self.config['FSAG']
        self.include_depth_maps = True
        self.eval_every_N_epochs = self.config["eval_freq"]
        self.train_depth_est_model = self.config['model_params']['train_midas']
        if self.config['model_params']['num_classes'] == 1:
            self.target_scaler = utils.Scaler(polypsizing=True)  # Scales the target values to the feature range
        else:
            self.target_scaler = utils.Scaler(polypsizing=False) # disable scaler

        # Check if CUDA is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # - Depth Maps Mean: 1813.1324462890625
        # - Depth Maps STD: 1074.3480224609375
        self.depth_std = torch.tensor([512.0]).to(self.device)
        self.depth_mean = torch.tensor([940.0]).to(self.device)
        self.num_samples = 0
        self.model.to(self.device)

        if self.include_depth_maps:
            # Load the MiDaS model
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(self.device)
            if self.train_depth_est_model:
                print("Replace optimizer + scheduler (Multiple models +MiDas). ")
                # see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                model_parameters = [param for model in [self.midas, self.model] for param in model.parameters()]
                self.optimizer = torch.optim.Adam(model_parameters, lr=config['optimizer_params']['lr'],
                                                  weight_decay=float(config['optimizer_params']['weight_decay']))

                print("> Setting up new scheduler...")
                scheduler_module = importlib.import_module('modules.schedulers.' + config['lr_scheduler'])
                scheduler_class = getattr(scheduler_module, config['lr_scheduler_class'])
                self.lr_scheduler = scheduler_class(self.optimizer, config['scheduler_params'], config['num_epochs'])

        # check if the mask is enabled and create and save it in self.mask
        self.circ_crop_mask = None
        if 'circular_crop' in self.config['dataset_params']["image_preprocessing"] and self.config['dataset_params']["image_preprocessing"]['circular_crop']:
            self.circ_crop_mask = self.create_circular_cropping_mask(self.config["resize_to"], self.config["resize_to"],
                                                                     crop_radius_ratio=self.config['dataset_params'][
                                                                         "image_preprocessing"]["crop_radius_ratio"],
                                                                     device=self.device)
        # LOGGING
        self.global_step = 0
        self.global_val_loss = 1e5
        self.global_val_acc = -1e5

        # model-saving options
        self.version = 0
        self.save_path = os.path.join(self.config["dataset_params"]["output_folder"], f"version-{self.version}")
        if not self.config['inference_mode'] and not self.config['verbose_validation_mode']:
            while True:
                # save_path: The path to save model checkpoints.
                self.save_path = os.path.join(self.config["dataset_params"]["output_folder"], f"version-{self.version}")
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                    break
                else:
                    self.version += 1
            self.summarywriter = SummaryWriter(self.save_path)
            logging.basicConfig(filename=os.path.join(self.save_path, "experiment.log"), level=logging.INFO,
                format="%(asctime)s > %(message)s", )
            with open(os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8") as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False, allow_unicode=True)

    def create_circular_cropping_mask(self, height, width, crop_radius_ratio=1.0, device='cuda'):
        """
        Create a circular cropping-mask to be applied to an image.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.
            crop_radius_ratio (float, optional): The ratio of the radius to the half side of the square image (default: 0.9).
            device (str, optional): The device on which the tensor calculations will be performed (default: 'cpu').

        Returns:
            torch.Tensor: A circular mask with shape (height, width).
        """
        mask = torch.zeros((height, width)).to(device)
        y, x = torch.meshgrid(torch.linspace(-1, 1, height).to(device), torch.linspace(-1, 1, width).to(device))
        distance_from_center = torch.sqrt(x ** 2 + y ** 2)
        mask[distance_from_center <= crop_radius_ratio] = 1

        return mask

    def apply_circular_crop(self, data, cropping_mask, device='cuda'):
        """
        Apply a circular crop to a batch of images.

        Args:
            data (torch.Tensor): A batch of images with shape (batch_size, num_channels, height, width).
            cropping_mask (torch.Tensor): A circular mask with shape (height, width).
            device (str, optional): The device on which the tensor calculations will be performed (default: 'cpu').

        Returns:
            torch.Tensor: A batch of images with the circular mask applied, same shape as input data.
        """
        # Ensure the data tensor and mask are on the specified device
        data = data.to(device)
        cropping_mask = cropping_mask.to(device)

        # Broadcast the mask to match the shape of the batch
        batch_size = data.size(0)
        num_channels = data.size(1)
        height, width = cropping_mask.shape
        masked_data = cropping_mask.expand(batch_size, num_channels, height, width)

        # Apply the mask to the batch of images
        masked_data = masked_data * data

        return masked_data

    def save_checkpoint(self, model, optimizer, epoch=None, val_acc=None, val_loss=None, tag='', custom_path=None):
        """
        Save model checkpoint at self.save_path.
        The name will contain also the val_acc for that epoch

        Args:
            epoch (int): number of epoch since training started
            val_acc (float): validation accuracy at the given epoch
            model: pytorch model
        """
        if custom_path is not None:
            output_model_path = os.path.join(custom_path, f"epoch-{epoch}_{tag}_acc-{val_acc:.4f}_loss-{val_loss:.4f}_checkpoint.pth")
            print(f'> Saving CheckPoint: {output_model_path}')
        else:
            output_model_path = os.path.join(self.save_path, f"epoch-{epoch}_{tag}_acc-{val_acc:.4f}_loss-{val_loss:.4f}_checkpoint.pth")

            # if this checkpoint it is the best so fare, save it with the best_model suffix
            if val_acc > self.global_val_acc:
                logging.info(f"Val acc increased ({self.global_val_acc:.4f} → {val_acc:.4f}). Saving best model ...")
                # update best val accuracy
                self.global_val_acc = val_acc

        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), },
            output_model_path)


    def print_config_details(self, epoch, loss):
        """
        Prints the configuration details in a tabular format.

        Returns:
            None
        """
        config_table = [["Epoch", epoch], ["Training Loss", loss.cpu().detach().numpy().item()],
            ["Model", self.config['model']], ["Model Class", self.config['model_class']],
            ["Model Parameters", self.config['model_params']], ["Number of Epochs", self.config['num_epochs']],
            ["Learning Rate", self.config['optimizer_params']['lr']], ["Optimizer", self.config['optimizer']],
            ["Loss Function", self.config['loss']], ["Loss Parameters", self.config['loss_params']],
            ["LR Scheduler", self.config['lr_scheduler']], ["Scheduler Parameters", self.config['scheduler_params']],
            ["Transforms", self.config['transforms']], ["Transform Combination", self.config['transform_combo']],
            ["Transform Parameters",
             {'resize_to': self.config['resize_to'], 'rotation_degree': self.config['rotation_degree'],
                 'color_jitter_params': self.config['color_jitter_params'],
                 'noise_factor': self.config['noise_factor']}]]
        print('\n')
        print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt='pretty'))

    def compute_float_accuracy(self, abs_pred, gt, precision_float_accuracy=1.0):
        """
        This function computes the float accuracy based on the given threshold.

        Args:
            self: The instance of the class calling this function.
            abs_pred (torch.Tensor): A 1D tensor representing the predicted values.
            gt (torch.Tensor): A 1D tensor representing the ground truth values.
            precision_float_accuracy (float, optional): The threshold value used for calculating the accuracy. Default is 1.0.

        Returns:
            float: The computed float accuracy as a percentage.
        """
        # Compute the accuracy based on the given threshold
        float_acc = ((torch.abs(abs_pred - gt) <= precision_float_accuracy).to(dtype=torch.float32)).mean()

        return float_acc.item()

    def prepare_input(self, input_image, binary_mask, expand_factor=0.20):
        """
        Prepares the input by cropping the input image to the region indicated by the binary mask
        (2*expand_factor% more), and resizing it to the original dimensions.

        Args:
            input_image (torch.Tensor): Input image (batch_size, channels, height, width).
            binary_mask (torch.Tensor): Binary mask tensor of shape (batch_size, 1, height, width).
        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, channels, height, width).
        """

        batch_size, c, h, w = binary_mask.size()

        # Number of channels in the input image
        num_channels = input_image.size(1)

        # Initialize a tensor to store the processed images
        processed_images = torch.empty((batch_size, num_channels, h, w), device=input_image.device,
                                       dtype=input_image.dtype)

        # Process each image in the batch separately
        for i in range(batch_size):
            # Find bounding box of the binary mask for the current image
            y_indices, x_indices = torch.where(binary_mask[i, 0] > 0)

            # Check if y_indices and x_indices are not empty
            if y_indices.numel() > 0 and x_indices.numel() > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()

                # Expand bounding box by expand_factor%
                y_range = y_max - y_min
                x_range = x_max - x_min

                y_min = max(0, y_min - int(expand_factor * y_range))
                y_max = min(h - 1, y_max + int(expand_factor * y_range))
                x_min = max(0, x_min - int(expand_factor * x_range))
                x_max = min(w - 1, x_max + int(expand_factor * x_range))

                # Crop the input image according to the bounding box
                cropped_input = input_image[i:i + 1, :, y_min:y_max + 1, x_min:x_max + 1]

                # Resize the cropped input
                resized_input = F.interpolate(cropped_input, size=[h, w], mode='bilinear', align_corners=True)

                # Store the processed image
                processed_images[i] = resized_input
            else:
                # Handle the case where the binary_mask is all zeros for the current image
                processed_images[i] = torch.zeros((num_channels, h, w), device=input_image.device,
                                                  dtype=input_image.dtype)

        return processed_images

    def normalize_depth_maps(self, depth_maps):
        min_val = depth_maps.view(depth_maps.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_val = depth_maps.view(depth_maps.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)

        # Ensure max_val - min_val is not too small to avoid division by a very small number
        range_val = (max_val - min_val).clamp(min=1e-5)

        # Normalize each depth map in the batch
        normalized_depth_maps = (depth_maps - min_val) / range_val

        return normalized_depth_maps


    def fit(self, dataloader, epoch, curr_fold):
        """
        Trains the model for one epoch on the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader object to iterate over the training data.
            epoch (int): current epoch
        Returns:
            float: Average gradient of the model parameters over the epoch.
        """
        #  estimate the average speed in images per second
        batch_time = utils.AverageMeter()
        logging.info(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.9f}")
        if epoch == 0:
            self.save_checkpoint(self.model, self.optimizer, epoch, val_acc=0.0, val_loss=0.0,
                                 tag='init_model')

        # Switch model to training mode
        self.model.train()
        if (isinstance(self.model, FSGNetR18) or
                isinstance(self.model, FSGNetEffNet) or isinstance(self.model, ViTBaseline) or isinstance(self.model, FSGViT)):
            if self.train_depth_est_model:
                self.midas.train()
            else:
                self.midas.eval()

        gradients, loss_list, all_targets, all_preds = [], [], [], []
        loss_list_std = []
        gradients_gr = []
        grad_norm_gr = 0
        num_batch_per_epoch = len(dataloader)
        start_time = time.time()
        # Iterate over batches with tqdm progress bar
        for i, (inputs, loc_masks, targets, idx) in enumerate(
                tqdm(dataloader, desc=f"Ep_{epoch}_training_steps", total=num_batch_per_epoch)):

            # Scale targets to the range -1, 1
            targets = self.target_scaler.scale(targets)

            batch_size = len(inputs.squeeze())
            # Ensure all your tensors are Float tensors
            inputs = inputs.float().to(self.device)
            loc_masks = loc_masks.float().to(self.device)
            if self.config['model_params']['num_classes'] == 1:
                targets = targets.float().to(self.device)
            else:
                targets = targets.long().to(self.device)

            # Check if circular masking is enabled
            if self.circ_crop_mask is not None:
                # If inputs is not None, apply the circular mask to inputs using the apply_circular_crop function
                inputs = self.apply_circular_crop(inputs, self.circ_crop_mask, device=self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            gr_steps = 1
            if self.enable_fsag:
                # Zero the gradients for Gradient Routing For Online Feature Selection optimizer
                self.GROFS_optimizer.zero_grad()
                gr_steps = 2


            # Clip the binary_mask values to be between 0 and 1
            binary_mask = torch.clamp(loc_masks, 0, 1)

            # Expand dimensions of binary_mask to be [batch_size, 3, W, H]
            expanded_binary_mask = binary_mask.expand_as(inputs)

            # Element-wise multiplication to mask the RGB image
            masked_rgb = inputs * expanded_binary_mask

            inputs = self.prepare_input(inputs, binary_mask, expand_factor=0.5)

            depth_map = None
            if self.depth_map_enabled:
                # Compute the depth map using the RGB channels
                with autocast():
                    if self.train_depth_est_model:
                        depth_map = self.midas(inputs)
                    else:
                        with torch.no_grad():
                            depth_map = self.midas(inputs)

                depth_map = depth_map.unsqueeze(1)  # Adds a channel dimension, [64, 1, 384, 384]

                # apply circular mask
                depth_map = self.apply_circular_crop(depth_map, self.circ_crop_mask,
                                                                device=self.device)

                # Normalize the depth map
                depth_map = self.normalize_depth_maps(depth_map)

            grad_norm0_gr = 0.0
            grad_norm0 = 0.0
            for grad_routing in range(gr_steps):
                # Forward pass through the MultiModalNetwork
                with autocast():
                    if not (isinstance(self.model, ViTBaseline) or isinstance(self.model, FSGViT)):
                        outputs = self.model(inputs, masked_rgb, depth_map).squeeze()
                    else:
                        if not self.use_mask:
                            outputs = self.model(inputs).squeeze()
                        else:
                            outputs = self.model(masked_rgb).squeeze()


                # Ensure outputs have a consistent shape, e.g., (batch_size,)
                if self.config['model_params']['num_classes'] == 1:
                    outputs = outputs.view(-1)  # Reshape to ensure 1D output regardless of batch size

                targets = targets.view(-1)  # Reshape to ensure 1D targets

                ##########################################################
                # Gradient Routing For Online Feature Selection
                ##########################################################
                # First FSG and After forward
                if gr_steps == 1 or (grad_routing % 2) == 1:
                    # Compute Main loss
                    with autocast():
                        loss = self.loss_fn(outputs, targets)

                    # Backward pass
                    self.grad_scaler_amp.scale(loss).backward()

                    # Unscale the gradients
                    self.grad_scaler_amp.unscale_(self.optimizer)

                    # clip gradient norm

                    if self.config['clip_grad']:
                        grad_norm0 = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])

                    # Check for NaN/Inf gradients and handle them
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

                    grad_norm = utils.get_grad_norm(self.model.parameters())
                    gradients.append(grad_norm)

                    # Update parameters
                    self.grad_scaler_amp.step(self.optimizer)

                    # # Update the scaler
                    self.grad_scaler_amp.update()

                else:
                    # FSG - Compute loss
                    with autocast():
                        gr_loss = self.gr_ofs_loss(outputs, targets)

                    # Prepare for GROFS optimizer update
                    self.grad_scaler_amp.scale(gr_loss).backward()
                    self.grad_scaler_amp.unscale_(self.GROFS_optimizer)

                    # clip gradient norm
                    if self.config['clip_gr_grad']:
                        grad_norm0_gr = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_gr_grad'])

                    # Check for NaN/Inf gradients and handle them
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

                    grad_norm_gr = utils.get_grad_norm(self.model.parameters())
                    gradients_gr.append(grad_norm_gr)
                    # GROFS Scaler FP16
                    self.grad_scaler_amp.step(self.GROFS_optimizer)

                    # Update the scaler
                    self.grad_scaler_amp.update()

            # Append outputs and targets if they are not empty and have consistent dimensions
            if outputs.nelement() != 0 and targets.nelement() != 0:
                # Append targets and predictions to their respective lists
                all_targets.append(self.target_scaler.inverse_scale(targets).detach().cpu().numpy())
                if self.config['model_params']['num_classes'] == 1:
                    all_preds.append(self.target_scaler.inverse_scale(outputs).detach().cpu().numpy())
                else:
                    predictions = torch.argmax(outputs, dim=1)
                    all_preds.append(self.target_scaler.inverse_scale(predictions).detach().cpu().numpy())

            loss_list.append(loss.cpu().item())
            if self.enable_fsag:
                loss_list_std.append(gr_loss.cpu().item())

            if i % 50 == 0:
                self.print_config_details(epoch, loss)
                # Calculate average losses
                avg_loss = np.mean(loss_list)
                if self.enable_fsag:
                    avg_loss_std = np.mean(loss_list_std)
                else:
                    avg_loss_std = -1

                print('-------------------------------')
                print(f'Current Epoch: {epoch}')
                print(f'Current iter: {i}')
                print(f'Current Fold: {curr_fold}')
                print(f'  Average Loss: {avg_loss:.4f}')
                print(f'[GR] Average GR Loss: {avg_loss_std:.4f}')
                avg_gradient = np.mean([np.mean(g) for g in gradients])
                avg_gradient_gr = np.mean([np.mean(g) for g in gradients_gr])

                print(f'> Average gradient at step {i}: {avg_gradient}')
                print(f"> Gradient: {grad_norm0} -> {grad_norm}")

                if self.enable_fsag:
                    print(f'> [GR] Average gradient at step {i}: {avg_gradient_gr}')
                    print(f"> [GR] Gradient: {grad_norm0_gr} -> {grad_norm_gr}")

                if self.depth_map_enabled:
                    print(f"   - Depth Maps Mean: {depth_map.mean().item()} ")
                    print(f"   - Depth Maps STD: {depth_map.std().item()} ")

            ###################################################
            # Estimate the average speed loop
            ###################################################
            if i > 1:
                time_taken = time.time() - start_time
                batch_time.update(time_taken)

            if i > 1 and i % 50 == 0:
                images_per_second = batch_size / batch_time.avg
                print(f"   - (1-Loop) Estimated average speed: {images_per_second} img/sec.")

            # next loop start time
            start_time = time.time()

            # update learning rate scheduler, if used
            self.lr_scheduler.step(epoch + i / num_batch_per_epoch)  # update the learning rate
            if self.lr_scheduler_gr is not None:
                self.lr_scheduler_gr.step(epoch + i / num_batch_per_epoch)  # update the learning rate

        # Return the average gradient over the epoch
        average_loss = np.mean(loss_list)
        if self.enable_fsag:
            avg_loss_std = np.mean(loss_list_std)
        else:
            avg_loss_std = -1

        average_gradient = np.mean([np.mean(g) for g in gradients])
        print(
            f'Epoch {epoch}: Average total loss {average_loss}, Average STD loss {avg_loss_std}, Average gradient {average_gradient}')

        # Calculate Balanced DSL Accuracy
        # Concatenate predictions and targets with checks for empty or inconsistent dimensions
        all_preds = np.concatenate([pred for pred in all_preds if pred.ndim == 1 and pred.size > 0],
                                   axis=0) if all_preds else np.array([])
        all_targets = np.concatenate([target for target in all_targets if target.ndim == 1 and target.size > 0],
                                     axis=0) if all_targets else np.array([])

        if self.config['model_params']['num_classes'] == 1:
            dsl_preds = np.vectorize(utils.map_to_class_gt)(all_preds)
            dsl_targets = np.vectorize(utils.map_to_class_gt)(all_targets)
            balanced_dsl_accuracy = balanced_accuracy_score(dsl_targets, dsl_preds)
            # Calculate Float Accuracy
            float_accuracy = np.sum(np.abs(all_preds - all_targets) < 1.0) / len(all_targets)

            print(f'Regression - Epoch {epoch}: Balanced DSL Accuracy {balanced_dsl_accuracy} - Float Accuracy +/- 1mm: {float_accuracy}')

        else:
            # Compute standard accuracy
            float_accuracy = np.mean(all_preds == all_targets)

            # Compute balanced accuracy
            balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)

            print(
                f'Classification - Epoch {epoch}: Balanced Accuracy {balanced_accuracy} - Accuracy: {float_accuracy}')

        # Save checkpoint every two epochs with the average training loss
        if epoch % 2 == 0:
            self.save_checkpoint(self.model, self.optimizer, epoch, val_acc=float_accuracy, val_loss=average_loss,
                                 tag='trainset')

        return average_loss

    def save_to_csv(self, object_ids, preds, targets, file_path):

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["object_id", "prediction", "ground_truth"])
            for oid, pred, tgt in zip(object_ids, preds, targets):
                writer.writerow([oid, pred, tgt])

    def validate(self, dataloader, epoch, kfold, output_folder, train_loss, phase, save_path=None):
        """
        Validates the model for one epoch on the provided dataloader and logs the metrics.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader object to iterate over the validation data.
            epoch (int): The current epoch number.
            output_folder (str): The folder to store output visualizations.

        Returns:
            metrics (dict): A dictionary containing the validation metrics.
        """
        network_inputs_path = ''
        self.model.eval()
        # Multimodal Neural Network
        if (isinstance(self.model, FSGNetR18) or
                isinstance(self.model, FSGNetEffNet) or isinstance(self.model, ViTBaseline) or isinstance(self.model, FSGViT)):
            self.midas.eval()

            # Define the desired path
            network_inputs_path = os.path.join(output_folder, 'network_inputs', f'{phase}_ep_{epoch}')
            # Check if the directory exists
            if os.path.exists(network_inputs_path):
                # If it exists, delete it
                shutil.rmtree(network_inputs_path)
            # Create the directory
            os.makedirs(network_inputs_path)

        preds_list, targets_list, loss_list, obj_ids_list, frame_id_list = [], [], [], [], []
        num_batch_per_epoch = len(dataloader)
        with torch.no_grad():
            with autocast():
                for i, (inputs, loc_masks, targets, idx) in enumerate(
                        tqdm(dataloader, desc=f"Ep_{epoch}_{phase}_steps", total=num_batch_per_epoch)):
                    # Ensure all your tensors are Float tensors
                    inputs = inputs.float().to(self.device)
                    loc_masks = loc_masks.float().to(self.device)
                    # target are not transformed here. Range 1,25 mm
                    if self.config['model_params']['num_classes'] == 1:
                        targets = targets.float().to(self.device)
                    else:
                        targets = targets.long().to(self.device)

                    # Check if circular masking is enabled
                    if self.circ_crop_mask is not None:
                        # If inputs is not None, apply the circular mask to inputs using the apply_circular_crop function
                        inputs = self.apply_circular_crop(inputs, self.circ_crop_mask, device=self.device)

                    if (isinstance(self.model, FSGNetR18) or
                            isinstance(self.model, FSGNetEffNet) or isinstance(self.model, ViTBaseline) or isinstance(self.model, FSGViT)):
                        # Clip the binary_mask values to be between 0 and 1
                        binary_mask = torch.clamp(loc_masks, 0, 1)

                        # Expand dimensions of binary_mask to be [batch_size, 3, W, H]
                        expanded_binary_mask = binary_mask.expand_as(inputs)

                        # Element-wise multiplication to mask the RGB image
                        masked_rgb = inputs * expanded_binary_mask
                        inputs = self.prepare_input(inputs, binary_mask, expand_factor=0.5)
                        depth_map = None
                        if self.depth_map_enabled:
                            # Compute the depth map using the RGB channels
                            depth_map = self.midas(inputs)

                            depth_map = depth_map.unsqueeze(1)  # Adds a channel dimension, [64, 1, 384, 384]

                            # apply circular mask
                            depth_map = self.apply_circular_crop(depth_map, self.circ_crop_mask, device=self.device)

                            # Normalize the depth map
                            depth_map = self.normalize_depth_maps(depth_map)

                            # depth_map = self.prepare_input(depth_map, binary_mask)  # [64, 1, 384//2, 384//2]

                        if not (isinstance(self.model, ViTBaseline) or isinstance(self.model, FSGViT)):
                            # Forward pass through the MultiModalNetwork
                            output = self.model(inputs, masked_rgb, depth_map).squeeze()
                        else:
                            output = self.model(inputs).squeeze()

                        # Export some examples
                        if i == 1 and epoch == 0:
                            utils.create_and_save_subplots(inputs, masked_rgb, expanded_binary_mask, depth_map,
                                                           network_inputs_path, i)
                    else:
                        # Forward pass
                        output = self.model(inputs)

                    # Compute loss
                    L = self.loss_fn(output, self.target_scaler.scale(targets.squeeze()))
                    loss_list.append(L.detach().cpu().numpy())

                    # Restore predictions to the original range 1, 25 mm
                    output = self.target_scaler.inverse_scale(output)

                    # Ensure output and targets have consistent shapes
                    if self.config['model_params']['num_classes'] == 1:
                        output = output.view(-1).detach().cpu().numpy()  # Reshape to ensure 1D output regardless of batch size
                    else:
                        predictions = torch.argmax(output, dim=1)
                        output = predictions.detach().cpu().numpy()  # Convert to 1D numpy array

                    targets = targets.view(-1).detach().cpu().numpy()  # Convert to 1D numpy array

                    preds_list.append(output.tolist())
                    targets_list.append(targets.tolist())
                    if self.config['model_params']['num_classes'] == 1:
                        obj_ids_list.extend(dataloader.dataset.list_object_ids[idx].tolist())
                        frame_id_list.extend(dataloader.dataset.list_frame_ids[idx].tolist())
                    else:
                        obj_ids_list.extend(idx.detach().numpy().tolist())
                        frame_id_list.extend(idx.detach().numpy().tolist())

        # Concatenate the predictions and targets lists
        preds = np.concatenate(preds_list) if preds_list else np.array([1])
        targets = np.concatenate(targets_list) if targets_list else np.array([-1])
        # Replace NaN or Inf values in preds and targets
        preds = np.nan_to_num(preds, nan=1.0, posinf=1.0, neginf=1.0)
        targets = np.nan_to_num(targets, nan=-1.0, posinf=-1.0, neginf=-1.0)

        loss_list = np.array(loss_list).squeeze()
        obj_ids_list = np.array(obj_ids_list).squeeze()
        frame_id_list = np.array(frame_id_list).squeeze()
        assert len(preds) == len(targets)

        # Calculate Float Accuracy
        if self.config['model_params']['num_classes'] == 1:
            if len(preds) > 0:  # Check to prevent division by zero
                float_accuracy = np.sum(np.abs(preds - targets) < 1.0) / len(preds)
            else:
                float_accuracy = 0  # Default value in case preds is empty

            # Calculate DSL Accuracy
            if len(preds) > 0:
                vfunc = np.vectorize(utils.map_to_class_gt)
                dsl_preds = vfunc(preds)
                dsl_targets = vfunc(targets)
                sensitivity, specificity, balanced_perf, precision, recall, f1, dsl_bal_accuracy, _ = compute_metrics_testing(dsl_targets, dsl_preds)
            else:
                dsl_bal_accuracy = 0  # Default value in case preds is empty
                dsl_preds = np.array([])  # Initialize dsl_preds as an empty array
                dsl_targets = np.array([])  # Initialize dsl_targets as an empty array

            # Compute other metrics
            metrics = calculate_metrics.training_metrics_logger(dsl_preds, dsl_targets, preds, targets)
            metrics['float_accuracy'] = float_accuracy
            metrics['dsl_accuracy'] = dsl_bal_accuracy
            metrics['loss'] = np.mean(loss_list)
        else:
            # Compute accuracy
            sensitivity, specificity, balanced_perf, precision, recall, f1, balanced_accuracy, accuracy = compute_metrics_testing(
                targets, preds)

            metrics = calculate_metrics.training_metrics_logger(preds, targets, preds, targets)
            metrics['float_accuracy'] = accuracy
            metrics['dsl_accuracy'] = balanced_accuracy
            metrics['loss'] = np.mean(loss_list)

        # Scatter Plot
        scatter_plot_path = os.path.join(output_folder,
                                         f'{phase.upper()}_epoch_{epoch}_fold_{kfold}_pe_{np.around(metrics["PE"], 1)}_mae_{np.around(metrics["MAE"], 1)}_scatter_plot.png')
        if epoch > 100:
            visualizations.scatter_plot(targets, preds, f'Histogram of Model Predictions vs GT (Best Valid Accuracy)', 'Targets', 'Predictions',
                                        scatter_plot_path, color='red', grid=True)
        else:
            visualizations.scatter_plot(targets, preds, f'Scatter Plot {epoch}', 'Targets', 'Predictions',
                                        scatter_plot_path, color='red', grid=True)

        # Other Plots
        data = [preds, targets]
        labels = ['Predictions', 'Targets']
        colors = ['green', 'blue']

        # Histogram
        histogram_path = os.path.join(output_folder,
                                      f'{phase.upper()}_epoch_{epoch}_fold_{kfold}_pe_{np.around(metrics["PE"], 1)}_mae_{np.around(metrics["MAE"], 1)}_histogram.png')
        if epoch > 100:
            visualizations.histogram(data, 18, labels, f'Histogram of Model Predictions vs GT (Best Valid Accuracy)', 'Values', 'Frequency', histogram_path, colors,
                                 True)
        else:
            visualizations.histogram(data, 18, labels, f'Valid Histogram of Preds vs GT at Epoch: {epoch}', 'Values', 'Frequency', histogram_path, colors,
                                 True)

        # Time series
        time_series_path = os.path.join(output_folder,
                                        f'{phase.upper()}_epoch_{epoch}_fold_{kfold}_pe_{np.around(metrics["PE"], 1)}_mae_{np.around(metrics["MAE"], 1)}_time_series.png')
        visualizations.time_series(data, labels, f'Time Series {epoch}', 'Time Index', 'Values', time_series_path,
                                   colors, True)

        # Logging
        for name, value in metrics.items():
            self.writer.add_scalar(f'{phase.upper()}/{name}', value, epoch)

        val_acc = metrics['float_accuracy']  # or whichever metric you choose to track
        val_loss = metrics['loss']  # assuming 'loss' is a computed metric
        self.save_checkpoint(self.model, self.optimizer, epoch, val_acc, val_loss, tag=phase, custom_path=save_path)

        # Save the metrics only if it is not in inference mode
        if not self.config['inference_mode'] and not self.config['verbose_validation_mode']:
            # tensorboard writing
            self.summarywriter.add_scalars("start_lr", {"start_lr": self.optimizer.param_groups[0]["lr"]}, epoch)

            self.summarywriter.add_scalars("loss/epoch", {"val": val_loss, "train": train_loss}, epoch)

            self.summarywriter.add_scalars(f"errors/epoch_{phase.upper()}",
                {"PE": metrics['PE'], "MAE": metrics['MAE'], "MSE": metrics['MSE'], "RMSE": metrics['RMSE'], }, epoch)

            self.summarywriter.add_scalars(f"prec_recall/epoch_{phase.upper()}",
                {"accuracy": metrics['accuracy'], "f1": metrics['f1'], "precision": metrics['precision'],
                 "recall": metrics['recall'], }, epoch)

            self.summarywriter.add_scalars("acc/epoch", {f"{phase.upper()}_acc_float": val_acc}, epoch)

            logging.info(
                f"** global step: {self.global_step}, {phase.upper()} loss: {val_loss:.3f}, {phase.upper()}_acc: {val_acc:.2f}%")

        return metrics, preds, targets, obj_ids_list, frame_id_list


    def load_model_for_inference(self, model_path):
        """
        Loads the model for inference from a specified path.

        Args:
            model_path (str): The file path of the model to be loaded.

        Returns:
            None
        """
        # Load the model state
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()  # Set the model to evaluation mode

    def _save_plots_and_metrics(self, metrics, preds, targets, epoch, output_folder, phase):
        """
        Saves plots and metrics.

        Args:
            metrics (dict): Calculated metrics.
            preds (np.array): Predictions array.
            targets (np.array): Targets array.
            epoch (int): Current epoch number.
            output_folder (str): Directory to save outputs.
            phase (str): Phase of the experiment (e.g., 'inference', 'validation').

        Returns:
            None
        """
        # Save metrics to a file
        metrics_file = os.path.join(output_folder, f'{phase}_metrics_epoch_{epoch}.json')
        with open(metrics_file, 'w') as file:
            json.dump(metrics, file, indent=4)

        # Generate and save plots
        scatter_plot_path = os.path.join(output_folder, f'{phase}_scatter_plot_epoch_{epoch}.png')
        histogram_path = os.path.join(output_folder, f'{phase}_histogram_epoch_{epoch}.png')
        time_series_path = os.path.join(output_folder, f'{phase}_time_series_epoch_{epoch}.png')

        visualizations.scatter_plot(targets, preds, f'Scatter Plot {epoch}', 'Targets', 'Predictions',
                                    scatter_plot_path)
        visualizations.histogram([preds, targets], 18, ['Predictions', 'Targets'], f'Histogram {epoch}', 'Values',
                                 'Frequency', histogram_path, ['green', 'blue'], True)
        visualizations.time_series([preds, targets], ['Predictions', 'Targets'], f'Time Series {epoch}', 'Time Index',
                                   'Values', time_series_path, ['green', 'blue'], True)
