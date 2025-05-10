'''
ViTwithFSG: Vision Transformer wrapper with Feature Selection Gates (FSG)

This script defines a wrapper class to apply Feature Selection Gates (FSG) to a Vision Transformer (ViT) model.
FSG enhances model generalization by introducing sparse, learnable gates on the residual paths of attention and MLP blocks.
It is a form of architectural regularization designed for vision tasks and applicable to NLP tasks.

The method is introduced in:

@inproceedings{roffo2024FSG,
   title={Feature Selection Gates with Gradient Routing for Endoscopic Image Computing},
   author={Giorgio Roffo and Carlo Biffi and Pietro Salvagnini and Andrea Cherubini},
   booktitle={MICCAI 2024, the 27th International Conference on Medical Image Computing and Computer Assisted Intervention, Marrakech, Morocco, October 2024.},
   year={2024},
   organization={Springer}
}

- Publication: https://papers.miccai.org/miccai-2024/316-Paper0410.html
- Code: https://github.com/cosmoimd/feature-selection-gates
- Contact: giorgio.roffo@gmail.com
- Affiliation: Cosmo Intelligent Medical Devices (IMD), Lainate, Italy
'''

# imports
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

class FSGBlock(nn.Module):
    """
    A Transformer encoder block augmented with Feature Selection Gates (FSG).
    Each residual path (attention and MLP) is weighted element-wise by a learnable sigmoid gate.
    This promotes sparse activation and serves as a regularization mechanism to avoid overfitting.
    """
    def __init__(self, original_block):
        super().__init__()
        self.self_attention = original_block.self_attention  # Multi-head self-attention module
        self.mlp = original_block.mlp                        # Feedforward network (2-layer MLP)
        self.ln_1 = original_block.ln_1                      # LayerNorm before attention
        self.ln_2 = original_block.ln_2                      # LayerNorm before MLP
        self.dropout = original_block.dropout                # Dropout after attention

        dim = self.ln_1.normalized_shape[0]                  # Dimensionality of the model

        # FSG: learnable gates (one per channel), initialized with Xavier normal
        self.fsg_rectifier = nn.Sigmoid()
        self.fsg_rgb_ls1 = nn.Parameter(torch.empty(dim))  # Gate for attention path
        self.fsg_rgb_ls2 = nn.Parameter(torch.empty(dim))  # Gate for MLP path
        nn.init.xavier_normal_(self.fsg_rgb_ls1.unsqueeze(0), gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_normal_(self.fsg_rgb_ls2.unsqueeze(0), gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        # Self-attention + gate
        x_norm = self.ln_1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, need_weights=False)
        attn_output = self.dropout(attn_output)
        fsg_scores_1 = self.fsg_rectifier(self.fsg_rgb_ls1)
        x = x + attn_output * fsg_scores_1  # Residual connection weighted by gate

        # MLP + gate
        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)
        fsg_scores_2 = self.fsg_rectifier(self.fsg_rgb_ls2)
        x = x + mlp_output * fsg_scores_2   # Residual connection weighted by gate

        return x

class ViTwithFSG(nn.Module):
    """
    Wrapper module that injects FSGBlocks into each Transformer encoder block of a given ViT model.
    """
    def __init__(self, vit_backbone: VisionTransformer):
        super().__init__()
        self.vit = vit_backbone
        for i, blk in enumerate(self.vit.encoder.layers):
            self.vit.encoder.layers[i] = FSGBlock(blk)  # Replace original block with FSGBlock

    def forward(self, x):
        return self.vit(x)

def vit_with_fsg(vit_backbone: VisionTransformer):
    """
    Factory function that wraps a torchvision VisionTransformer with FSG-enhanced encoder blocks.
    """
    return ViTwithFSG(vit_backbone)


# === Example Usage ===
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Failed to load image Python extension*")

    from torchvision.models import vit_b_16, ViT_B_16_Weights

    print("\n📥 Loading pretrained ViT_B_16 backbone from torchvision...")
    backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    print("🔧 Wrapping with Feature Selection Gates (FSG)...")
    model = vit_with_fsg(vit_backbone=backbone)

    print("🧪 Running dummy input through FSG-augmented ViT...")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("✅ Inference completed.")
    print("📐 Output shape:", output.shape)
