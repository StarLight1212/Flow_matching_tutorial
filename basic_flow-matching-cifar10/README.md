# CIFAR-10 Conditional Rectified Flow (Flow Matching) — Low-NFE Study

This repo contains a *from-scratch* implementation of **Flow Matching** with **Rectified Flow** on CIFAR-10, including:
- Conditional U-Net velocity field `v_θ(x,t,y)`
- U-shaped time sampling (Beta distribution)
- Euler & RK4 ODE samplers with NFE ∈ {1,2,4,8}
- Classifier-Free Guidance (CFG) at sampling time
- EMA weights for stable training
- **One-round ReFlow fine-tuning** (optional)
- FID evaluation via `clean-fid`

> Python ≥3.9, CUDA recommended.


## 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 2) Train (Uniform t)

```bash
python train.py --data ./data --epochs 300 --batch 256 --lr 2e-4 --t-schedule uniform   --save-dir ./runs/base_uniform --ema 0.999 --drop-prob 0.1
```

## 3) Train (U-shaped t, Beta(0.5,0.5))

```bash
python train.py --data ./data --epochs 300 --batch 256 --lr 2e-4 --t-schedule beta --beta-a 0.5 --beta-b 0.5   --save-dir ./runs/base_beta --ema 0.999 --drop-prob 0.1
```

## 4) Sample (Euler/RK4, low NFE)

```bash
# Euler, NFE=4
python sample.py --ckpt ./runs/base_beta/ema.pt --n 5000 --nfe 4 --sampler euler --cfg-scale 2.0   --out ./samples/base_beta_euler4

# RK4, NFE=4
python sample.py --ckpt ./runs/base_beta/ema.pt --n 5000 --nfe 4 --sampler rk4 --cfg-scale 2.0   --out ./samples/base_beta_rk4_4
```

## 5) One-round ReFlow fine-tune

```bash
python reflow.py --ckpt ./runs/base_beta/ema.pt --steps 50_000 --batch 256 --lr 1e-4   --save-dir ./runs/reflow_beta --t-schedule beta --beta-a 0.5 --beta-b 0.5
```

## 6) FID evaluation (vs. CIFAR-10 train stats)

```bash
# Download CIFAR-10 automatically through torchvision (during train) or beforehand.
python fid_eval.py --samples ./samples/base_beta_rk4_4 --ref train
```

## 7) Suggested Experiments (Coursework)

1. **NFE & Sampler**: Evaluate NFE ∈ {1,2,4,8} × {Euler,RK4}; report FID & imgs/sec.
2. **t Schedule**: Uniform vs. Beta(0.5,0.5) under NFE=4.
3. **ReFlow**: With/without one-round ReFlow under NFE=2 & 4.

Fill in your report with curves/tables. The code saves checkpoints and sample grids per epoch.

## Tips
- Images are scaled to `[-1,1]`. Noise is standard Gaussian.
- CFG is applied **only at sampling**; set `--cfg-scale 0` for unconditional.
- To speed up FID, generate 10k images (or 5k for quick iteration).

## Acknowledgement
This is a clean-room educational implementation inspired by the Flow Matching / Rectified Flow literature.
