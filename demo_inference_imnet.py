'''
Demo script for applying Feature Selection Gates (FSG) to torchvision Vision Transformers
and running inference on the ImageNet-mini (Imagenette) validation set.

Each image is resized to 224x224 and has 3 RGB channels to be compatible with ViT.

Usage:

demo_inference_imnet.py --checkpoint ./checkpoints/fsg_vit_imagenette_demo.pth

Paper:
https://papers.miccai.org/miccai-2024/316-Paper0410.html
Code:
https://github.com/cosmoimd/feature-selection-gates
Contact:
giorgio.roffo@gmail.com
'''

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import tarfile
import urllib.request
import torch
import psutil
from torchvision.models import vit_b_16, ViT_B_16_Weights
from vit_with_fsg import vit_with_fsg
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="FSG-ViT inference on Imagenette")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth file of trained FSG-ViT model")
args = parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Failed to load image Python extension*")
    wrn = False
    print(f"\n📌 To run this script:\n"
          f"   ▶ Without checkpoint: python {os.path.basename(__file__)}\n"
          f"   ▶ With checkpoint:    python {os.path.basename(__file__)} --checkpoint path/to/model.pth\n")

    # Device and system info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"🚀 CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory total: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"🧠 System RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

    print("\n📥 Loading pretrained ViT backbone from torchvision...")
    backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    print("🔧 Wrapping with Feature Selection Gates (FSG)...")
    model = vit_with_fsg(backbone).to(device)

    if args.checkpoint is not None:
        print(f"📂 Loading model weights from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        wrn = True
        print("\n⚠️  No checkpoint provided. Evaluating randomly initialized model! 🧪\n")
        print("❗ Note: The model has not been trained. Results will reflect a randomly initialized backbone.")

    model.eval()

    print("📚 Loading Imagenette validation set (224x224 RGB)...")
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = ImageFolder(root=imagenette_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []

    print("🧪 Running inference on Imagenette validation set using FSG-ViT-B-16 (code by G. Roffo)...\n\n")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="🔍 Inference progress", ncols=100):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print("✅ Inference completed.")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    if wrn == True:
        print("\n⚠️  No checkpoint provided. Evaluated randomly initialized model! 🧪\n")
        print(f"\n📌 To run this script:\n"
              f"   ▶ With checkpoint:    python {os.path.basename(__file__)} --checkpoint path/to/model.pth\n")

    print(f"📊 Accuracy:  {acc * 100:.2f}%")
    print(f"📊 Precision: {prec * 100:.2f}%")
    print(f"📊 Recall:    {rec * 100:.2f}%")
    print(f"📊 F1 Score:  {f1 * 100:.2f}%")
