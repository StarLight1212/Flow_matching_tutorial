# Flow Matching Tutorial

# 1) 这玩意儿到底是什么？
flow-matching（FM）算是近两年生成建模里最“爽脆”的路线之一：不再模拟SDE、也不必做likelihood的昂贵积分，而是直接回归一个**速度场（vector field）**，用简单的ODE求解器就能生成高质量样本。

* **目标**：学一个时间依赖的速度场 $v_\theta(x,t)$，让它把“噪声分布”逐步推到“数据分布”。生成时，只需从噪声出发，沿着 ODE $\dot x = v_\theta(x,t)$ 从 $t=0$ 积分到 $t=1$。

* **训练法则（Flow Matching）**：先选一条“几何上舒服”的**概率路径** $p_t(x)$ 把噪声和数据连接起来，然后最小化

  $\mathbb{E}\big[ \|v_\theta(X_t,t) - u_t(X_t)\|^2 \big],$

  其中 $u_t$ 是这条路径的**真速度**（或其条件期望），
  $X_t\sim p_t$。

* **两个常用路径**

  * **Diffusion 路径**：可把FM当作“更稳的扩散模型训练”看待。
  * **Rectified Flow（直线流）**：最简单的选择，让样本沿**直线**从噪声走向数据，训练、采样都很快，往往**1–8步**就能出图。([arXiv][1])

进一步的系统综述与代码库：Facebook Research 的 FM 指南 + PyTorch 实现，非常适合作为权威参考与对照实现。([facebookresearch.github.io][2], [GitHub][3], [arXiv][4])

---

# 2) 数学到落地：以 Rectified Flow 为例（强烈推荐上手）

取数据样本 $x\sim p_{\text{data}}$、噪声 $z\sim\mathcal N(0,I)$，采样 $t\sim \mathrm{Uniform}(0,1)$。**直线**插值：

$$
x_t = (1-t)z + t x.
$$

这条路径的**目标速度**是常量：

$$
u_t(x_t) = x - z \quad(\text{与 }t\text{ 无关})
$$

训练时回归 $v_\theta(x_t,t)\approx x-z$。
生成时从 $z$ 起步，数值积分 $\dot x=v_\theta(x,t)$ 即可（Euler/RK4）。实践中 1–8 步就能不错。([arXiv][5])

> 近年的改进（采样t的U形分布、单轮ReFlow微调、感知度量等）可进一步提升低NFE（步数很少时）质量。([arXiv][6], [NeurIPS 会议论文集][7])

---

# 3) 最小可跑 PyTorch 例子（MNIST，Rectified Flow）

> 下面是能跑通的**极简**版：CNN 速度场 + 直线路径训练 + Euler 采样（8步）。建议先在 MNIST 试跑、看训练损失与可视化样本，再移植到 CIFAR-10/自定义数据。

```python
# flow_matching_mnist.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import trange
import math, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 1) 速度场网络 v_theta(x,t)
# ----------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):  # t in [0,1], shape (B,)
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.), math.log(1000.), steps=half, device=t.device))
        angles = 2*math.pi * t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(emb)

class SmallUNet(nn.Module):
    def __init__(self, ch=64, time_dim=128):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.in1  = nn.Conv2d(1, ch, 3, padding=1)
        self.in2  = nn.Conv2d(ch, ch, 3, padding=1)
        self.down = nn.Conv2d(ch, 2*ch, 4, stride=2, padding=1)
        self.mid1 = nn.Conv2d(2*ch, 2*ch, 3, padding=1)
        self.up   = nn.ConvTranspose2d(2*ch, ch, 4, stride=2, padding=1)
        self.out1 = nn.Conv2d(2*ch, ch, 3, padding=1)
        self.out2 = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(t)[:, :, None, None]  # (B,C,1,1)
        h1 = F.silu(self.in1(x))
        h1 = F.silu(self.in2(h1))
        d  = F.silu(self.down(h1))
        m  = F.silu(self.mid1(d + temb))           # 简单注入t
        u  = F.silu(self.up(m))
        u  = torch.cat([u, h1], dim=1)
        u  = F.silu(self.out1(u))
        v  = self.out2(u)
        return v

# ----------------------------
# 2) 直线流训练数据管道
# ----------------------------
def get_dataloaders(batch_size=128, num_workers=2):
    tmf = transforms.Compose([
        transforms.ToTensor(),                 # [0,1]
        transforms.Normalize([0.5], [0.5])     # -> [-1,1]  (x-0.5)/0.5 = 2x-1
    ])
    train = datasets.MNIST(root='./data', train=True, download=True, transform=tmf)
    # NOTE: if you still see spawn issues, set num_workers=0 as a fallback.
    return DataLoader(train, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, pin_memory=torch.cuda.is_available())

# ----------------------------
# 3) 训练一步
# ----------------------------
def train_epoch(model, opt, dl):
    model.train()
    total = 0.0
    for x,_ in dl:
        x = x.to(device)                        # (B,1,28,28) in [-1,1]
        z = torch.randn_like(x)
        t = torch.rand(x.size(0), device=device)  # (B,)
        x_t = (1 - t)[:,None,None,None] * z + t[:,None,None,None] * x
        target_v = x - z

        pred_v = model(x_t, t)
        loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / (len(dl.dataset))

# ----------------------------
# 4) 生成
# ----------------------------
@torch.no_grad()
def sample(model, n=64, nfe=8):
    model.eval()
    x = torch.randn(n,1,28,28, device=device)
    for i in range(nfe):
        t0 = torch.full((n,), i/float(nfe), device=device)
        v  = model(x, t0)
        dt = 1.0 / nfe
        x  = x + v * dt
    x = (x.clamp(-1,1)+1)/2.0
    return x


# ----------------------------
# 5) 主程序
# ----------------------------
def main():
    dl = get_dataloaders(256, num_workers=2)
    model = SmallUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    os.makedirs('samples', exist_ok=True)
    for epoch in trange(10):
        loss = train_epoch(model, opt, dl)
        print(f"epoch {epoch} | loss {loss:.4f}")
        imgs = sample(model, n=64, nfe=8)
        utils.save_image(imgs, f'samples/e{epoch:02d}.png', nrow=8)


if __name__ == "__main__":
    main()

```

**你应该看到**：训练几轮后，`samples/` 下会逐渐出现像 MNIST 的数字；`nfe=1` 也能生成可辨数字，但 4–8 步更稳。

**常见坑位**

* 记得把数据映射到 $[-1,1]$（跟噪声对齐）。
* 速度场输出维度要和输入一致。
* 小数据（MNIST）下过拟合并不致命，CIFAR-10 要上 U-Net/残差块、EMA、数据增广。
* 想更快收敛：把采样的 $t$ 改成**U形分布**（比如 Beta(0.5,0.5)），多在 0/1 端训练。([NeurIPS 会议论文集][7])

---

# 4) 进一步优化：把它做成“像扩散一样好用”的工程

* **更强的骨干**：U-Net（多尺度下采样/上采样、残差+SiLU、GroupNorm）、时间嵌入用更高维度并在每个block注入。
* **类条件生成**：在 $v_\theta(x,t,y)$ 里注入类别嵌入，并做**Classifier-Free Guidance**（随机丢失条件）。
* **采样器**：用 RK4 或 DPM-Solver-风格的多步 ODE 求解器，保持低 NFE 的同时提升质量。
* **一次 ReFlow** 微调：用当前模型生成伪“对偶”，再以直线校正一次，常常已足够。([arXiv][6])
* **官方库做对照**：跑一遍 Meta 的 `flow_matching` 示例，与你的实现对比设计。([GitHub][3], [facebookresearch.github.io][2])

---

# 5) 快速参考 &延伸阅读

* **Flow Matching 原始论文**与 PDF。([arXiv][1], [ar5iv][9])
* **Rectified Flow**（直线流，核心直觉与性质）。([arXiv][5], [OpenReview][10])
* **改进训练（U形 t、单轮 ReFlow、感知损失）**。([arXiv][6], [NeurIPS 会议论文集][7])
* **官方库与指南（强推对照）**。([GitHub][3], [facebookresearch.github.io][2])
* **Awesome Flow Matching**（论文清单）。([GitHub][11])

---

想不想我把上面的 MNIST 代码帮你直接**改成 CIFAR-10 类条件版本**，顺手把 U-shape t 与 RK4 也补上？你可以直接开跑、再做作业里的实验。

[1]: https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "Flow Matching for Generative Modeling"
[2]: https://facebookresearch.github.io/flow_matching/?utm_source=chatgpt.com "Flow Matching documentation"
[3]: https://github.com/facebookresearch/flow_matching?utm_source=chatgpt.com "A PyTorch library for implementing flow matching ..."
[4]: https://arxiv.org/abs/2412.06264?utm_source=chatgpt.com "Flow Matching Guide and Code"
[5]: https://arxiv.org/abs/2209.03003?utm_source=chatgpt.com "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
[6]: https://arxiv.org/html/2405.20320v1?utm_source=chatgpt.com "Improving the Training of Rectified Flows"
[7]: https://proceedings.neurips.cc/paper_files/paper/2024/file/7343a5c976f8399880b695267f1f9e9f-Paper-Conference.pdf?utm_source=chatgpt.com "Improving the Training of Rectified Flows"
[8]: https://arxiv.org/html/2407.12718v2?utm_source=chatgpt.com "SlimFlow: Training Smaller One-Step Diffusion Models with ..."
[9]: https://ar5iv.labs.arxiv.org/html/2210.02747?utm_source=chatgpt.com "[2210.02747] Flow Matching for Generative Modeling - ar5iv"
[10]: https://openreview.net/pdf/910c5efa5739a5d2bef83d432da87d3096712ebe.pdf?utm_source=chatgpt.com "FLOW STRAIGHT AND FAST: LEARNING TO GENER"
[11]: https://github.com/dongzhuoyao/awesome-flow-matching?utm_source=chatgpt.com "Awesome Flow Matching ( Stochastic Interpolant )"
