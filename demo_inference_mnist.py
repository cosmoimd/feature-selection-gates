'''
Demo script for applying Feature Selection Gates (FSG) to torchvision Vision Transformers
and running inference on the MNIST test set.

Each MNIST image is resized to 224x224 and converted to 3 channels to be compatible with ViT.

Usage:

demo_inference_mnist.py --checkpoint ./checkpoints/fsg_vit_mnist_demo.pth

Paper:
https://papers.miccai.org/miccai-2024/316-Paper0410.html
Code:
https://github.com/cosmoimd/feature-selection-gates
Contact:
giorgio.roffo@gmail.com
'''

import torch
import psutil
import argparse
import warnings
from torchvision.models import vit_b_16, ViT_B_16_Weights
from vit_with_fsg import vit_with_fsg
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="FSG-ViT inference on MNIST")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth file of trained FSG-ViT model")
args = parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Failed to load image Python extension*")
    wrn = False
    print(f"\n📌 To run this script:\n"
          f"   ▶ Without checkpoint: python {os.path.basename(__file__)}\n"
          f"   ▶ With checkpoint:    python {os.path.basename(__file__)} --checkpoint path/to/model.pth\n")

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

    print("📚 Loading MNIST test set (resized to 224x224, 3-channel)...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []

    print("🧪 Running inference on MNIST test set using FSG-ViT-B-16 (code by G. Roffo)...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="🔍 Inference progress", ncols=100):
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
