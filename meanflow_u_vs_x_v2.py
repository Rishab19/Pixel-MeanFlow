"""
Toy projection experiment from the MeanFlow paper.

A 2D Swiss Roll is linearly projected into D-dimensional space via a fixed
column-orthonormal matrix P ∈ R^{D×2}.  We train two MeanFlow variants:

  x-prediction:  net outputs x_hat ∈ R^D;  u = (z - x_hat) / t
  u-prediction:  net outputs u_hat ∈ R^D;  trained directly to predict e - x

Both use the same 7-layer ReLU MLP with 256 hidden units (paper spec).
We compare 1-NFE samples projected back to 2D for visualisation.

D ∈ {2, 8, 16, 512}

Usage:
    python projection_experiment.py
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))   # so we can import from swiss_roll_meanflow

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from swiss_roll_meanflow import (
    make_swiss_roll,
    SwissRollMeanFlowLoss,
    SinusoidalEmbedding,
)


# ---------------------------------------------------------------------------
# Network — paper spec: 7-layer ReLU MLP, 256 hidden units
# ---------------------------------------------------------------------------

class PaperMLP(nn.Module):
    """
    7-layer ReLU MLP, 256 hidden, as specified in the paper.
    Works for any data_dim (the projected dimension D).

    mode='x': outputs x_hat  (x-prediction)
    mode='u': outputs u_hat  (u-prediction / velocity-prediction)
    """
    def __init__(self, data_dim: int, emb_dim: int = 64, hidden: int = 256, mode: str = 'x'):
        super().__init__()
        assert mode in ('x', 'u')
        self.mode = mode
        self.t_emb = SinusoidalEmbedding(emb_dim)
        self.h_emb = SinusoidalEmbedding(emb_dim)

        # 7 layers total: 1 input projection + 5 hidden + 1 output
        dims = [data_dim + 2 * emb_dim] + [hidden] * 6 + [data_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        t_e = self.t_emb(t.view(B))
        h_e = self.h_emb(h.view(B))
        return self.net(torch.cat([z, t_e, h_e], dim=-1))


# ---------------------------------------------------------------------------
# u-prediction loss
# ---------------------------------------------------------------------------

class UPredMeanFlowLoss:
    """
    Original MeanFlow loss (u-prediction), from the earlier paper's Algorithm 1.

    fn(z, r, t) predicts u directly.
    v = e - x  is the TRUE velocity, used as the z-tangent in the JVP.

    u, dudt  = jvp(fn, (z, r, t), (v, 0, 1))
    u_tgt    = v - (t - r) * dudt          # subtract, not add
    error    = u - stopgrad(u_tgt)
    loss     = metric(error)
    """

    def __init__(
        self,
        noise_dist: str        = "uniform",
        data_proportion: float = 0.25,
        t_min: float           = 0.02,
    ):
        self.noise_dist      = noise_dist
        self.data_proportion = data_proportion
        self.t_min           = t_min

    def _sample_time(self, shape, device):
        return self.t_min + (1.0 - self.t_min) * torch.rand(shape, device=device)

    def _sample_t_r(self, B, device):
        shape = (B, 1)
        s1 = self._sample_time(shape, device)
        s2 = self._sample_time(shape, device)
        t  = torch.max(s1, s2)
        r  = torch.min(s1, s2)
        data_mask = (torch.arange(B, device=device) < int(B * self.data_proportion)).unsqueeze(1)
        r = torch.where(data_mask, t, r)
        return t, r

    def __call__(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device

        t, r = self._sample_t_r(B, device)
        e    = torch.randn_like(x)
        z_t  = (1.0 - t) * x + t * e
        v    = e - x                        # true velocity — also the z-tangent

        # fn: network predicts u directly
        def fn(z, r_, t_):
            h = t_ - r_
            return net(z, t_, h)

        # JVP tangents: (v, 0, 1) per Algorithm 1 of the original paper
        primals  = (z_t, r,                   t                  )
        tangents = (v,   torch.zeros_like(r), torch.ones_like(t) )
        u, dudt  = torch.func.jvp(fn, primals, tangents)

        u_tgt = v - (t - r) * dudt          # subtract (original algo)
        error = u - u_tgt.detach()          # stopgrad the whole target

        loss = error.pow(2).sum(dim=1).mean()
        return loss

# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def make_projection_matrix(D: int, seed: int = 0) -> torch.Tensor:
    """
    Random column-orthonormal matrix P ∈ R^{D×2}.
    Projects 2D data into D-dimensional space: x_D = P @ x_2.
    """
    if D == 2:
        # Identity-like: just return a 2×2 orthonormal matrix
        rng = torch.Generator().manual_seed(seed)
        A = torch.randn(2, 2, generator=rng)
        P, _ = torch.linalg.qr(A)
        return P   # [2, 2]
    rng = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=rng)
    Q, _ = torch.linalg.qr(A)        # Q is D×D orthonormal
    return Q[:, :2]                   # take first 2 columns => [D, 2]


def project_up(x2: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """x2: [B,2], P: [D,2]  =>  [B,D]"""
    return x2 @ P.T                   # [B, D]

def project_down(xD: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """xD: [B,D], P: [D,2]  =>  [B,2]  (pseudo-inverse = P.T for orthonormal P)"""
    return xD @ P                     # [B, 2]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    mode: str,          # 'x' or 'u'
    D: int,             # observation dimension
    P: torch.Tensor,    # projection matrix [D, 2], on device
    n_data: int      = 50_000,
    batch_size: int  = 512,
    lr: float        = 1e-3,
    n_steps: int     = 30_000,
    warmup_steps: int = 2_000,
    log_every: int   = 5_000,
    device: str      = "cpu",
    seed: int        = 42,
) -> nn.Module:
    torch.manual_seed(seed)

    data2d = make_swiss_roll(n_data).to(device)
    dataD  = project_up(data2d, P)          # [N, D]

    loader  = DataLoader(TensorDataset(dataD), batch_size=batch_size, shuffle=True, drop_last=True)
    net     = PaperMLP(data_dim=D, mode=mode).to(device)
    opt     = torch.optim.Adam(net.parameters(), lr=lr)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / warmup_steps
        progress = (s - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched   = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = SwissRollMeanFlowLoss() if mode == 'x' else UPredMeanFlowLoss()

    step = 0
    it   = iter(loader)
    while step < n_steps:
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(loader); (x,) = next(it)

        opt.zero_grad()
        loss = loss_fn(net, x)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            print(f"  [{mode}-pred D={D:4d}] step {step:6d}  loss {loss.item():.5f}")
        step += 1

    return net


# ---------------------------------------------------------------------------
# Sampling (1-NFE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one_step(net: nn.Module, D: int, n: int = 1024) -> torch.Tensor:
    """
    x-pred:  x_hat = net(z, t=1, h=1)           [direct]
    u-pred:  x_hat = z - net(z, t=1, h=1) * 1   [u integrates one step]
    """
    net.eval()
    device = next(net.parameters()).device
    z = torch.randn(n, D, device=device)
    t = torch.ones(n, 1, device=device)
    h = torch.ones(n, 1, device=device)
    out = net(z, t, h)
    if net.mode == 'x':
        return out                  # network outputs x directly
    else:
        return z - out * 1.0        # z - u*dt, dt=1 for full step


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    DIMS     = [2, 8, 16, 512]
    N_STEPS  = 30_000
    N_SAMPLE = 1024

    gt2d = make_swiss_roll(N_SAMPLE)   # ground truth in 2D for visualisation

    fig, axes = plt.subplots(
        len(DIMS), 3,
        figsize=(12, 4 * len(DIMS)),
        squeeze=False,
    )

    col_titles = ["Ground truth (2D)", "x-prediction (1 NFE)", "u-prediction (1 NFE)"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight='bold')

    for row, D in enumerate(DIMS):
        print(f"\n{'='*50}")
        print(f"D = {D}")
        print('='*50)

        P = make_projection_matrix(D, seed=0).to(device)

        # --- train both modes ---
        net_x = train_model('x', D, P, n_steps=N_STEPS, device=device, seed=42)
        net_u = train_model('u', D, P, n_steps=N_STEPS, device=device, seed=42)

        # --- sample and project back to 2D ---
        sD_x = sample_one_step(net_x, D, N_SAMPLE)
        sD_u = sample_one_step(net_u, D, N_SAMPLE)

        s2d_x = project_down(sD_x.cpu(), P.cpu()).numpy()
        s2d_u = project_down(sD_u.cpu(), P.cpu()).numpy()
        gt     = gt2d.numpy()

        kw  = dict(s=3, alpha=0.5, rasterized=True)
        lim = (-2.8, 2.8)

        ax_gt = axes[row][0]
        ax_x  = axes[row][1]
        ax_u  = axes[row][2]

        ax_gt.scatter(gt[:, 0],    gt[:, 1],    color="steelblue", **kw)
        ax_x.scatter( s2d_x[:, 0], s2d_x[:, 1], color="tomato",    **kw)
        ax_u.scatter( s2d_u[:, 0], s2d_u[:, 1], color="seagreen",  **kw)

        for ax in [ax_gt, ax_x, ax_u]:
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)

        ax_gt.set_ylabel(f"D = {D}", fontsize=11, fontweight='bold')

    plt.suptitle(
        "MeanFlow: x-pred vs u-pred across projected dimensions\n(1-NFE generation)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("projection_experiment.png", dpi=120, bbox_inches='tight')
    print("\nSaved projection_experiment.png")