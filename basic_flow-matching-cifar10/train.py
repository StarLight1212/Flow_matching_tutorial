import argparse, os, math, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from models.unet_flow import UNet32
from utils.ema import EMA

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def beta_sample(bsz, a=0.5, b=0.5, device="cpu"):
    return torch.distributions.Beta(a, b).sample((bsz,)).to(device).clamp(1e-4, 1-1e-4)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--save-dir", type=str, default="./runs/exp")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--t-schedule", type=str, default="uniform", choices=["uniform","beta"])
    p.add_argument("--beta-a", type=float, default=0.5)
    p.add_argument("--beta-b", type=float, default=0.5)
    p.add_argument("--drop-prob", type=float, default=0.1)  # classifier-free guidance drop prob at train
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2-1.0)
    ])
    train_set = datasets.CIFAR10(root=args.data, train=True, download=True, transform=tfm)
    dl = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    model = UNet32(in_ch=3, base=128, t_dim=128, n_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=args.ema)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        loss_avg = 0.0
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            z = torch.randn_like(x)
            bsz = x.size(0)

            if args.t-schedule == "uniform":  # placeholder to avoid syntax highlight issue
                pass
