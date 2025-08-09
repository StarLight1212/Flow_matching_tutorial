import torch, torch.nn as nn, torch.nn.functional as F, math

# --- small U-Net for 32x32 CIFAR-10 ---
def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.), math.log(1000.), steps=half, device=t.device))
    angles = 2*math.pi * t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim, cond_dim=0):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.emb = nn.Linear(t_dim + cond_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class UNet32(nn.Module):
    def __init__(self, in_ch=3, base=64, t_dim=128, n_classes=10, drop_prob=0.0):
        super().__init__()
        self.t_dim = t_dim
        self.n_classes = n_classes
        self.drop_prob = drop_prob

        self.cls_embed = nn.Embedding(n_classes + 1, t_dim)  # last index = null token
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )

        # down
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, t_dim*2)
        self.down1 = nn.Conv2d(base, base*2, 4, stride=2, padding=1)
        self.rb2 = ResBlock(base*2, base*2, t_dim*2)
        self.down2 = nn.Conv2d(base*2, base*4, 4, stride=2, padding=1)
        self.rb3 = ResBlock(base*4, base*4, t_dim*2)

        # mid
        self.mid1 = ResBlock(base*4, base*4, t_dim*2)
        self.mid2 = ResBlock(base*4, base*4, t_dim*2)

        # up
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1)
        self.rb4 = ResBlock(base*4, base*2, t_dim*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1)
        self.rb5 = ResBlock(base*2, base, t_dim*2)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t, y=None):
        # t:[B], y: [B] with values in 0..9 or None
        B = x.size(0)
        t_emb = sinusoidal_embedding(t, self.t_dim)
        if y is None:
            y_idx = torch.full((B,), self.n_classes, device=x.device, dtype=torch.long)  # null token
        else:
            y_idx = y
        # (optional) classifier-free dropout at TRAIN time should be handled outside by passing y=None
        y_emb = self.cls_embed(y_idx)
        temb = torch.cat([self.time_mlp(t_emb), y_emb], dim=-1)

        h0 = self.in_conv(x)
        h1 = self.rb1(h0, temb)
        d1 = self.down1(h1)
        h2 = self.rb2(d1, temb)
        d2 = self.down2(h2)
        h3 = self.rb3(d2, temb)

        m = self.mid1(h3, temb)
        m = self.mid2(m, temb)

        u1 = self.up1(m)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.rb4(u1, temb)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.rb5(u2, temb)

        out = self.out_conv(F.silu(self.out_norm(u2)))
        return out  # velocity field

    # helper for CFG at sampling
    @torch.no_grad()
    def forward_with_cfg(self, x, t, y, cfg_scale=0.0):
        if cfg_scale == 0.0 or y is None:
            return self.forward(x, t, y)
        v_cond = self.forward(x, t, y)
        v_uncond = self.forward(x, t, None)
        return v_uncond + cfg_scale * (v_cond - v_uncond)
