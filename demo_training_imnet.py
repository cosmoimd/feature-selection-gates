'''
Demo training script for Feature Selection Gates (FSG) with ViT on Imagenette

This script loads the Imagenette dataset (ImageNet-mini),
trains a ViT model augmented with FSG, and saves the model checkpoint.

Paper:
https://papers.miccai.org/miccai-2024/316-Paper0410.html
Code:
https://github.com/cosmoimd/feature-selection-gates
Contact:
giorgio.roffo@gmail.com
'''

import os
import tarfile
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from vit_with_fsg import vit_with_fsg

# System info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")
if device.type == "cuda":
    print(f"🚀 CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU memory total: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
print(f"🧠 System RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

# Dataset path
imagenette_path = "./imagenette2-160/val"
if not os.path.exists(imagenette_path):
    print("📦 Downloading Imagenette...")
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    tgz_path = "imagenette2-160.tgz"
    urllib.request.urlretrieve(url, tgz_path)
    print("📂 Extracting Imagenette dataset...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall()
    os.remove(tgz_path)
    print("✅ Dataset ready.")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset and loader
dataset = ImageFolder(root=imagenette_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
print("\n📥 Loading pretrained ViT backbone from torchvision...")
backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model = vit_with_fsg(backbone).to(device)

# Optimizer with separate LRs for FSG and base ViT
fsg_params, base_params = [], []
for name, param in model.named_parameters():
    if 'fsag_rgb_ls' in name:
        fsg_params.append(param)
    else:
        base_params.append(param)

lr_base = 1e-4
lr_fsg = 5e-4
print(f"\n🔧 Optimizer setup:")
print(f"   🔹 Base ViT parameters LR: {lr_base}")
print(f"   🔸 FSG parameters LR: {lr_fsg}")

optimizer = optim.AdamW([
    {"params": base_params, "lr": lr_base},
    {"params": fsg_params, "lr": lr_fsg}
])
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 3
print(f"\n🚀 Starting demo training for {epochs} epochs...")
model.train()
for epoch in range(epochs):
    steps_demo = 0  # to remove: for demo only
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
    for inputs, targets in pbar:
        if steps_demo > 25: # to remove: for demo only
            break # to remove: for demo only
        steps_demo += 1  # to remove: for demo only
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1e-8)})

print("\n✅ Training complete.")

# Save checkpoint
ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "fsg_vit_imagenette_demo.pth")
torch.save(model.state_dict(), ckpt_path)
print(f"💾 Checkpoint saved to: {ckpt_path}")
