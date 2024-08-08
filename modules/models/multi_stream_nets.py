#!/usr/bin/env python
# This code refers to the following paper:
# "Feature Selection Gates with Gradient Routing for Endoscopic Image Computing"
# Accepted for publication at MICCAI 2024.

# Please cite both the accepted version and the preprint version if you use this code,
# its methods, ideas, or any part of them.

# Accepted Publication:
# @inproceedings{roffo2024featureselection,
#    title={Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
#    author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
#    booktitle={MICCAI},
#    year={2024}
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
import timm
# Import PyTorch modules
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.nn import LayerNorm
from torchvision.models import resnet18

# Import plot modules

# Import 3rd party modules

# Import custom modules
# from datasets import dataset

# File information
# Author: Giorgio Roffo, PhD. Senior Research Scientist. Cosmo IMD, Lainate, MI, Italy.
__author__ = "Giorgio Roffo, Dr."
__copyright__ = "Copyright 2024, COSMO IMD"
__credits__ = ["Giorgio Roffo"]
__license__ = "Private"
__version__ = "1.0.1"
__maintainer__ = "groffo"
__email__ = "groffo@cosmoimd.com,giorgio.roffo@gmail.com"
__status__ = "Production"  # can be "Prototype", "Development" or "Production"


class FSGNetR18(nn.Module):
    """
    FSGNetR18 is a neural network model utilizing Feature Selection Attention Gate (FSAG) for each input branch.
    It is designed to process different types of inputs, namely RGB, Masked RGB, and Depth, individually,
    allowing for flexibility in handling various input modalities. The model uses ResNet-18 as the backbone for
    each branch and applies FSAG to scale features in an attention-like, learnable manner.

    Unlike traditional attention mechanisms that use softmax to distribute relative importance among features,
    FSAG operates differently. Each feature's importance is evaluated independently, allowing a feature to be
    considered important (close to 1) or not (close to 0), without the constraint of summing up to 1. This
    independent assessment via FSAG contrasts with the attention mechanism where increasing one feature's importance
    necessarily decreases another's. FSAG thus offers a more flexible and less inter-dependent approach to feature
    selection, suitable for tasks where feature relationships are not competitive or comparative.

    The FSAG parameters are initialized such that their sigmoid activation is centered around 0.50. This balanced
    start avoids the extremes of the sigmoid function, preventing saturation at the early stages of training and
    allowing better gradient flow. This initialization gives each feature a moderate level of importance, enabling
    the model to adjust the relevance of features dynamically based on training data.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - num_classes (int): Number of output classes or units (1 for regression tasks).
            - nodes_fc_layers (list of int): List specifying the number of nodes in each fully connected layer.
            - dropout_rate (float): Dropout rate applied to the fully connected layers.
            - use_rgb (bool): Flag to include or exclude the RGB branch.
            - use_mask (bool): Flag to include or exclude the Masked RGB branch.
            - use_depth (bool): Flag to include or exclude the Depth branch.
            - warmup_epochs (int): Number of initial epochs before applying FSAG.
            - pretrained (bool): Flag to use pretrained weights for EfficientNet B0 backbones.

    The `forward` method of the class accepts the following optional arguments:
        - rgb (torch.Tensor): Input tensor for the RGB branch.
        - mask (torch.Tensor): Input tensor for the Masked RGB branch.
        - depth (torch.Tensor): Input tensor for the Depth branch.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes), representing the model's predictions.
    """
    def __init__(self, config):
        super(FSGNetR18, self).__init__()

        # Extracting values from the config dictionary
        self.fsag_init_method = config.get('fsag_init_method', 'xavier')
        num_classes = config.get('num_classes')
        nodes_fc_layers = config.get('nodes_fc_layers', [512, 512])
        use_rgb = config.get('use_rgb', True)
        use_mask = config.get('use_mask', False)
        use_depth = config.get('use_depth', False)
        warmup_epochs = config.get('warmup_epochs', 5)
        self.pretrained = config.get('pretrained', True)

        print(f'Setting: use_rgb: [{use_rgb}], use_mask: [{use_mask}], use_depth: [{use_depth}]')

        self.use_rgb = use_rgb
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.epoch = 0  # Current epoch
        self.warmup_epochs = warmup_epochs
        input_channels = 512

        # FSAG parameters for each ResNet block in each branch
        for branch in ['rgb', 'mask', 'depth']:
            for layer, dim in zip(['layer1', 'layer2', 'layer3', 'layer4'], [64, 128, 256, 512]):
                key = f'fsag_{branch}_{layer}'
                setattr(self, key, self._init_fsag_params_for_layer(dim))

        # Initialize branches conditionally based on the configuration
        self.branches = nn.ModuleDict()
        if self.use_rgb:
            self.branches['rgb'] = self._init_branch(resnet18, self.pretrained, 3)
        if self.use_mask:
            self.branches['mask'] = self._init_branch(resnet18, self.pretrained, 3)
        if self.use_depth:
            self.branches['depth'] = self._init_branch(resnet18, self.pretrained, 1)

        feature_dim = input_channels * (self.use_rgb + self.use_mask + self.use_depth)

        # Apply the chosen FSAG initialization
        self._init_fsag_params(feature_dim, nodes_fc_layers)

        self.fsag_rectifier = nn.Sigmoid()

        self.fc2 = nn.Linear(feature_dim, nodes_fc_layers[1])
        self.fc3 = nn.Linear(nodes_fc_layers[1], num_classes)

        # Compute and print the FLOPs
        print("\nCompute and print the FLOPs")
        self.print_flops()

    def _init_fsag_params_for_layer(self, feature_dim):
        fsag_init = torch.Tensor(1, feature_dim)
        if self.fsag_init_method == 'he':
            nn.init.kaiming_normal_(fsag_init, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(fsag_init, gain=nn.init.calculate_gain('sigmoid'))
        return nn.Parameter(fsag_init.squeeze(), requires_grad=True)

    def _init_fsag_params(self, feature_dim, nodes_fc_layers):
        # Initialize as 2D tensors for initialization
        fsag_L1_init = torch.Tensor(1, feature_dim)
        fsag_L2_init = torch.Tensor(1, nodes_fc_layers[1])

        if self.fsag_init_method == 'he':
            # Kaiming initialization
            nn.init.kaiming_normal_(fsag_L1_init, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(fsag_L2_init, mode='fan_in', nonlinearity='relu')
        else:
            # Xavier initialization
            nn.init.xavier_normal_(fsag_L1_init, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.xavier_normal_(fsag_L2_init, gain=nn.init.calculate_gain('sigmoid'))

        # Convert back to 1D tensors and wrap in nn.Parameter
        self.fsag_L1 = nn.Parameter(fsag_L1_init.squeeze(), requires_grad=True)
        self.fsag_L2 = nn.Parameter(fsag_L2_init.squeeze(), requires_grad=True)

    def _init_branch(self, backbone_fn, pretrained, in_channels):
        """
        Initializes a branch with the given backbone function, pretrained setting, and input channels.
        """
        branch = backbone_fn(pretrained=pretrained)
        if in_channels != 3:  # Adjusts the first convolutional layer for non-RGB inputs
            branch.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        branch.fc = nn.Identity()  # Removes the final fully connected layer, as it's not used in this model

        # Apply hooks to store outputs after each ResNet layer
        self.feature_maps = {}
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(branch, layer_name)
            layer.register_forward_hook(self._forward_hook(f'{branch}_{layer_name}'))

        return branch

    def _forward_hook(self, branch_layer_name):
        def hook(module, input, output):
            self.feature_maps[branch_layer_name] = output

        return hook
    def print_flops(self):
        # Prepare dummy inputs for each active branch
        input_size = (1, 3, 384, 384)  # Adjusted input size
        dummy_rgb = torch.randn(input_size) if self.use_rgb else None
        dummy_mask = torch.randn(input_size) if self.use_mask else None
        dummy_depth = torch.randn((1, 1, 384, 384)) if self.use_depth else None  # Assuming depth is single-channel

        # Make sure all layers are in eval mode to avoid issues with dropout or batchnorm
        self.eval()

        # Compute FLOPs for the model with the provided inputs
        flops = FlopCountAnalysis(self, (dummy_rgb, dummy_mask, dummy_depth))

        print('*********************************************************************')
        print(f"FLOPs: {flops.total()}")
        print('*********************************************************************')
        print(flop_count_table(flops))

        print(
            "FLOPs, or Floating Point Operations, measure the number of floating-point calculations involved in a model's forward pass. Higher FLOPs indicate greater computational complexity and resource usage. FLOPs are crucial for understanding the efficiency of a model, especially in resource-constrained environments like mobile devices.")

    def get_fsag_weights(self):
        """
        Returns the Feature Selection Attention Gate (FSAG) weights for all layers in all branches.
        """
        fsag_weights = {}

        # Iterate through branches and layers to get FSAG weights
        for branch in ['rgb', 'mask', 'depth']:
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                key = f'fsag_{branch}_{layer}'
                if hasattr(self, key):
                    fsag_param = getattr(self, key)
                    fsag_weights[key] = fsag_param.data.cpu().numpy()

        # Get weights for fsag_L1 and fsag_L2 if they exist
        if hasattr(self, 'fsag_L1'):
            fsag_weights['fsag_L1'] = self.fsag_L1.data.cpu().numpy()
        if hasattr(self, 'fsag_L2'):
            fsag_weights['fsag_L2'] = self.fsag_L2.data.cpu().numpy()

        return fsag_weights

    def forward(self, rgb=None, mask=None, depth=None):
        features_list = []

        # Process inputs through respective branches
        for branch_name, branch in self.branches.items():
            input_tensor = locals().get(branch_name)
            if input_tensor is not None:
                # First, pass the input through the initial layers of the branch (conv1, bn1, relu)
                x = branch.conv1(input_tensor)
                x = branch.bn1(x)
                x = branch.relu(x)
                x = branch.maxpool(x)
                # Then pass through each ResNet layer
                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    layer = getattr(branch, layer_name)

                    x = layer(x)  # Get output from current layer
                    key = f'{branch_name}_{layer_name}'
                    fsag_scores = self.fsag_rectifier(getattr(self, f'fsag_{key}'))

                    # Reshape fsag_scores to match the dimensions of x
                    fsag_scores = fsag_scores.view(1, -1, 1, 1)  # Shape: [1, channels, 1, 1]

                    x = x * fsag_scores  # Apply FSAG scores to the layer output

                # Apply Global Average Pooling
                x = branch.avgpool(x)  # Assuming branch.avgpool is the AdaptiveAvgPool2d layer

                # Flatten the output for the fully connected layers
                x = torch.flatten(x, 1)

                features_list.append(x)

        # Check if features are present and properly batch them
        if features_list:
            embeddings = torch.cat(features_list, dim=1)
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
        else:
            raise ValueError("No input features provided for active branches.")

        # FSAG to the second layer's output
        fsag_scores_L1 = self.fsag_rectifier(self.fsag_L1)
        embeddings = embeddings * fsag_scores_L1

        # fully connected layer
        x = self.fc2(embeddings)
        x = torch.relu(x)

        # FSAG to the second layer's output
        fsag_scores_L2 = self.fsag_rectifier(self.fsag_L2)
        # Use non-in-place multiplication for applying FSAG weights
        x = x * fsag_scores_L2

        # Final output layer
        x = self.fc3(x)

        return x



class FSGNetEffNet(nn.Module):
    """
    FSGNetEffNet is a neural network model utilizing Feature Selection Attention Gate (FSAG) for each input branch.
    It is designed to process different types of inputs, namely RGB, Masked RGB, and Depth, individually, allowing
    for flexibility in handling various input modalities. The model uses EfficientNet B0 as the backbone for each branch
    and applies FSAG to scale features in an attention-like, learnable manner.

    Unlike traditional attention mechanisms that use softmax to distribute relative importance among features,
    FSAG operates differently. Each feature's importance is evaluated independently, allowing a feature to be
    considered important (close to 1) or not (close to 0), without the constraint of summing up to 1. This
    independent assessment via FSAG contrasts with the attention mechanism where increasing one feature's importance
    necessarily decreases another's. FSAG thus offers a more flexible and less inter-dependent approach to feature
    selection, suitable for tasks where feature relationships are not competitive or comparative.

    The FSAG parameters are initialized such that their sigmoid activation is centered around 0.50. This balanced
    start avoids the extremes of the sigmoid function, preventing saturation at the early stages of training and
    allowing better gradient flow. This initialization gives each feature a moderate level of importance, enabling
    the model to adjust the relevance of features dynamically based on training data.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - num_classes (int): Number of output classes or units (1 for regression tasks).
            - nodes_fc_layers (list of int): List specifying the number of nodes in each fully connected layer.
            - dropout_rate (float): Dropout rate applied to the fully connected layers.
            - use_rgb (bool): Flag to include or exclude the RGB branch.
            - use_mask (bool): Flag to include or exclude the Masked RGB branch.
            - use_depth (bool): Flag to include or exclude the Depth branch.
            - warmup_epochs (int): Number of initial epochs before applying FSAG.
            - pretrained (bool): Flag to use pretrained weights for EfficientNet B0 backbones.

    The `forward` method of the class accepts the following optional arguments:
        - rgb (torch.Tensor): Input tensor for the RGB branch.
        - mask (torch.Tensor): Input tensor for the Masked RGB branch.
        - depth (torch.Tensor): Input tensor for the Depth branch.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes), representing the model's predictions.
    """
    def __init__(self, config):
        super(FSGNetEffNet, self).__init__()

        # Extracting values from the config dictionary
        self.fsag_init_method = config.get('fsag_init_method', 'default')
        num_classes = config.get('num_classes')
        nodes_fc_layers = config.get('nodes_fc_layers', [512, 512])
        dropout_rate = config.get('dropout_rate', 0.35)
        self.use_rgb = config.get('use_rgb', True)
        self.use_mask = config.get('use_mask', False)
        self.use_depth = config.get('use_depth', False)
        pretrained = config.get('pretrained', True)

        print(f'Setting: use_rgb: [{self.use_rgb}], use_mask: [{self.use_mask}], use_depth: [{self.use_depth}]')


        input_channels = 1280

        # Initialize branches conditionally
        self.branches = nn.ModuleDict()
        if self.use_rgb:
            self.branches['rgb'] = self._init_branch('efficientnet_b0', pretrained, 3, input_channels)
        if self.use_mask:
            self.branches['mask'] = self._init_branch('efficientnet_b0', pretrained, 3, input_channels)
        if self.use_depth:
            self.branches['depth'] = self._init_branch('efficientnet_b0', pretrained, 1, input_channels)

        feature_dim = input_channels * (self.use_rgb + self.use_mask + self.use_depth)

        # Initialize FSAG parameters for each branch
        self.fsag_params = nn.ModuleDict()
        for branch in self.branches.keys():
            self.fsag_params[f'fsag_{branch}'] = nn.Parameter(torch.randn(input_channels))


        # Apply the chosen FSAG initialization
        self._init_fsag_params(feature_dim, nodes_fc_layers)

        self.fsag_rectifier = nn.Sigmoid()

        # Dropout and Fully Connected layers
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(feature_dim, nodes_fc_layers[1])
        self.fc3 = nn.Linear(nodes_fc_layers[1], num_classes)

        # Compute and print the FLOPs
        print("\nCompute and print the FLOPs")
        self.print_flops()

    def _init_fsag_params(self, feature_dim, nodes_fc_layers):
        # Initialize as 2D tensors for initialization
        fsag_L1_init = torch.Tensor(1, feature_dim)
        fsag_L2_init = torch.Tensor(1, nodes_fc_layers[1])

        if self.fsag_init_method == 'he':
            # Kaiming initialization
            nn.init.kaiming_normal_(fsag_L1_init, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(fsag_L2_init, mode='fan_in', nonlinearity='relu')
        else:
            # Xavier initialization
            nn.init.xavier_normal_(fsag_L1_init, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.xavier_normal_(fsag_L2_init, gain=nn.init.calculate_gain('sigmoid'))

        # Convert back to 1D tensors and wrap in nn.Parameter
        self.fsag_L1 = nn.Parameter(fsag_L1_init.squeeze(), requires_grad=True)
        self.fsag_L2 = nn.Parameter(fsag_L2_init.squeeze(), requires_grad=True)

    def _init_branch(self, backbone, pretrained, in_channels, out_channels):
        # Load the pretrained EfficientNet-B0 model
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        # Adjust the first convolutional layer if not using RGB inputs
        if in_channels != 3:
            original_conv = model.conv_stem
            model.conv_stem = nn.Conv2d(
                in_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            model.conv_stem.weight.data = original_conv.weight.data[:, 0:1, :, :]  # Copy weights for the first channel

        return model

    def forward_branch(self, branch, x):
        x = branch.conv_stem(x)
        x = branch.bn1(x)

        for block in branch.blocks:
            x = block(x)
            # Apply FSAG after each block (if needed)

        x = branch.conv_head(x)
        x = branch.bn2(x)
        x = branch.global_pool(x)
        x = branch.classifier(x)
        return x

    def print_flops(self):
        # Prepare dummy inputs for each active branch
        input_size = (1, 3, 384, 384)  # Adjusted input size
        dummy_rgb = torch.randn(input_size) if self.use_rgb else None
        dummy_mask = torch.randn(input_size) if self.use_mask else None
        dummy_depth = torch.randn((1, 1, 384, 384)) if self.use_depth else None  # Assuming depth is single-channel

        # Make sure all layers are in eval mode to avoid issues with dropout or batchnorm
        self.eval()

        # Compute FLOPs for the model with the provided inputs
        flops = FlopCountAnalysis(self, (dummy_rgb, dummy_mask, dummy_depth))

        print('*********************************************************************')
        print(f"FLOPs: {flops.total()}")
        print('*********************************************************************')
        print(flop_count_table(flops))

        print(
            "FLOPs, or Floating Point Operations, measure the number of floating-point calculations involved in a model's forward pass. Higher FLOPs indicate greater computational complexity and resource usage. FLOPs are crucial for understanding the efficiency of a model, especially in resource-constrained environments like mobile devices.")

    def get_fsag_weights(self):
        """
        Returns the Feature Selection Attention Gate (FSAG) weights.
        """
        return {
            'fsag_L1': self.fsag_L1.data.cpu().numpy(),
            'fsag_L2': self.fsag_L2.data.cpu().numpy()
        }

    def forward(self, rgb=None, mask=None, depth=None):
        features_list = []

        # Process inputs through respective branches
        if self.use_rgb and rgb is not None:
            features_list.append(self.branches['rgb'](rgb))
        if self.use_mask and mask is not None:
            features_list.append(self.branches['mask'](mask))
        if self.use_depth and depth is not None:
            features_list.append(self.branches['depth'](depth).view(depth.size(0), -1))

        # Check if features are present and properly batch them
        if features_list:
            embeddings = torch.cat(features_list, dim=1)
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
        else:
            raise ValueError("No input features provided for active branches.")

        # fully connected layer
        fsag_scores_L1 = self.fsag_rectifier(self.fsag_L1)
        embeddings = embeddings * fsag_scores_L1
        x = self.fc2(embeddings)
        x = torch.relu(x)

        # FSAG to the second layer's output
        fsag_scores_L2 = self.fsag_rectifier(self.fsag_L2)
        # Use non-in-place multiplication for applying FSAG weights
        x = x * fsag_scores_L2

        # Final output layer
        x = self.fc3(x)

        return x


