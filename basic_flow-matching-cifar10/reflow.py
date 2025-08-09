# One-round ReFlow fine-tuning:
# - Load a trained model (ema ckpt recommended).
# - Generate pseudo targets: x_hat from noise z with a *slightly* higher NFE.
# - Fine-tune on synthetic pairs along the rectified path (z -> x_hat).
# This approximates a single rectification pass.

import argparse, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models.unet_flow import UNet32
from utils.sampler import rk4_sampler
from utils.ema import EMA

@torch.no_grad()
def generate_pairs(model, n, nfe=8, device="cuda"):
    B = 256
    xs, zs = [], []
    total = 0
    while total < n:
        b = min(B, n - total)
        z = torch.randn(b,3,32,32, device=device)
        y = torch.randint(0,10,(b,), device=device)
        x_hat = rk4_sampler(model, z, nfe=nfe, class_labels=y, cfg_scale=2.0)
        xs.append(x_hat.cpu()); zs.append(z.cpu()); total += b
    return torch.cat(xs), torch.cat(zs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="./runs/reflow")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nfe-gen", type=int, default=8)
    p.add_argument("--t-schedule", type=str, default="beta", choices=["uniform","beta"])
    p.add_argument("--beta-a", type=float, default=0.5)
    p.add_argument("--beta-b", type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet32(in_ch=3, base=128, t_dim=128, n_classes=10).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # generate synthetic pairs
    xs, zs = generate_pairs(model, n=50000, nfe=args.nfe_gen, device=device)  # ~50k synthetic
    ys = torch.randint(0,10,(xs.size(0),), dtype=torch.long)  # random labels for conditioning
    ds = torch.utils.data.TensorDataset(xs, zs, ys)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    # fine-tune (rectified pairs)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=0.999)

    seen = 0
    pbar = tqdm(total=args.steps, desc="reflow ft")
    while seen < args.steps:
        for x, z, y in dl:
            x = x.to(device); z = z.to(device); y = y.to(device)
            bsz = x.size(0)
            if args.t_schedule == "uniform":
                t = torch.rand(bsz, device=device)
            else:
                t = torch.distributions.Beta(args.beta_a, args.beta_b).sample((bsz,)).to(device).clamp(1e-4, 1-1e-4)

            x_t = (1 - t)[:,None,None,None] * z + t[:,None,None,None] * x
            target_v = x - z
            pred_v = model(x_t, t, y)
            loss = F.mse_loss(pred_v, target_v)
            opt.zero_grad(); loss.backward(); opt.step()
            ema.update(model)

            seen += bsz
            pbar.update(bsz)
            pbar.set_postfix(loss=loss.item())
            if seen >= args.steps:
                break

    # save ema
    ema.apply_shadow(model)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "ema.pt"))
    ema.restore(model)

if __name__ == "__main__":
    main()
