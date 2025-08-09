import torch

@torch.no_grad()
def euler_sampler(model, x0, nfe, class_labels=None, cfg_scale=0.0):
    x = x0
    b = x.size(0)
    for i in range(nfe):
        t = torch.full((b,), i/float(nfe), device=x.device)
        v = model.forward_with_cfg(x, t, class_labels, cfg_scale)
        dt = 1.0 / nfe
        x = x + v * dt
    return x

@torch.no_grad()
def rk4_sampler(model, x0, nfe, class_labels=None, cfg_scale=0.0):
    x = x0
    b = x.size(0)
    for i in range(nfe):
        t = torch.full((b,), i/float(nfe), device=x.device)
        dt = 1.0 / nfe
        k1 = model.forward_with_cfg(x, t, class_labels, cfg_scale)
        k2 = model.forward_with_cfg(x + 0.5*dt*k1, t + 0.5/nfe, class_labels, cfg_scale)
        k3 = model.forward_with_cfg(x + 0.5*dt*k2, t + 0.5/nfe, class_labels, cfg_scale)
        k4 = model.forward_with_cfg(x + dt*k3, t + 1.0/nfe, class_labels, cfg_scale)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x
