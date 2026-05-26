"""
Pixel MeanFlow (Algorithm 1) for a toy 2D Swiss Roll dataset.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MeanFlowMLP(nn.Module):
    """
    x-prediction MLP for 2-D data.
    Inputs : z [B,2], t [B,1], h [B,1]  (h = t - r)
    Output : x_hat [B,2]
    """
    def __init__(self, data_dim: int = 2, hidden: int = 512, depth: int = 5, emb_dim: int = 128):
        super().__init__()
        self.t_emb = SinusoidalEmbedding(emb_dim)
        self.h_emb = SinusoidalEmbedding(emb_dim)

        layers = [nn.Linear(data_dim + 2 * emb_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        t_e = self.t_emb(t.view(z.shape[0]))
        h_e = self.h_emb(h.view(z.shape[0]))
        return self.net(torch.cat([z, t_e, h_e], dim=-1))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SwissRollMeanFlowLoss:
    """
    Pixel MeanFlow loss (Algorithm 1) for 2-D toy data.

    u(z, r, t) = (z - net(z, t, h)) / t        [average velocity]
    v          = u(z_t, t, t)                   [instantaneous velocity, z-tangent]
    u, dudt    = jvp(u_fn, (z,r,t), (v, 0, 1)) [JVP wrt t]
    V          = u + (t-r) * stopgrad(dudt)     [compound target]
    loss       = ||V - (e-x)||^2
    """

    def __init__(
        self,
        noise_dist: str    = "uniform",   # flat coverage; logit-normal starves t≈1
        data_proportion: float = 0.25,    # fraction of batch with r=t (self-prediction)
        t_min: float = 0.02,              # hard floor; keeps 1/t and dudt finite
    ):
        self.noise_dist      = noise_dist
        self.data_proportion = data_proportion
        self.t_min           = t_min

    def _sample_time(self, shape, device):
        if self.noise_dist == "uniform":
            # Sample uniformly but floor at t_min so we never divide by ~0
            return self.t_min + (1.0 - self.t_min) * torch.rand(shape, device=device)
        elif self.noise_dist == "logit_normal":
            rnd = torch.randn(shape, device=device)
            return torch.sigmoid(rnd * 1.0 + (-0.4)).clamp(self.t_min, 1.0)
        raise ValueError(self.noise_dist)

    def _sample_t_r(self, B, device):
        shape = (B, 1)
        s1 = self._sample_time(shape, device)
        s2 = self._sample_time(shape, device)
        t  = torch.max(s1, s2)   # t >= r always
        r  = torch.min(s1, s2)

        # data_proportion slice: r = t  (network predicts x from nearly-clean z)
        data_mask = (torch.arange(B, device=device) < int(B * self.data_proportion)).unsqueeze(1)
        r = torch.where(data_mask, t, r)
        return t, r

    def __call__(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device

        t, r   = self._sample_t_r(B, device)
        e      = torch.randn_like(x)
        z_t    = (1.0 - t) * x + t * e
        v_true = e - x                          # ground-truth flow velocity

        def u_fn(z, r_, t_):
            h = t_ - r_
            x_pred = net(z, t_, h)
            return (z - x_pred) / t_.clamp(min=self.t_min)

        # Instantaneous velocity — z-tangent for JVP (no stopgrad per Algorithm 1)
        v_inst = u_fn(z_t, t, t)               # [B, 2]

        # JVP wrt t (r fixed, z moves at rate v_inst)
        primals  = (z_t,    r,                   t                  )
        tangents = (v_inst, torch.zeros_like(r), torch.ones_like(t) )
        u, dudt  = torch.func.jvp(u_fn, primals, tangents)

        # Compound target — stopgrad only on dudt (Algorithm 1 exactly)
        V = u + (t - r) * dudt.detach()

        loss = (V - v_true).pow(2).sum(dim=1).mean()
        return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_swiss_roll(
    n: int       = 50_000,
    noise: float = 0.1,
    turns: float = 2.5,
    seed: int    = 0,
) -> torch.Tensor:
    rng  = torch.Generator().manual_seed(seed)
    t    = math.pi * (1 + 2 * turns * torch.rand(n, generator=rng))
    x    = t * t.cos()
    z    = t * t.sin()
    data = torch.stack([x, z], dim=1)
    data += noise * torch.randn(n, 2, generator=rng)
    data = (data - data.mean(0)) / data.std(0)
    return data.float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_data: int      = 50_000,
    batch_size: int  = 1024,
    lr: float        = 1e-3,
    n_steps: int     = 50_000,
    warmup_steps: int = 2_000,   # linear warmup — stabilises early JVP tangents
    log_every: int   = 1_000,
    device: str      = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int        = 42,
):
    torch.manual_seed(seed)

    data    = make_swiss_roll(n_data).to(device)
    loader  = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=True)

    net     = MeanFlowMLP().to(device)
    opt     = torch.optim.Adam(net.parameters(), lr=lr)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / warmup_steps          # linear ramp 0 → 1
        # cosine decay from 1 → 0.1 over remaining steps
        progress = (s - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched   = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = SwissRollMeanFlowLoss()

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
            print(f"step {step:6d}  loss {loss.item():.6f}  lr {sched.get_last_lr()[0]:.2e}")
        step += 1

    print("Training done.")
    return net


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one_step(net: nn.Module, n: int = 2048) -> torch.Tensor:
    """
    One-step MeanFlow generation.
    Start at t=1 (pure noise), r=0 => h=1.
    x_hat = net(z, t=1, h=1)
    """
    net.eval()
    device = next(net.parameters()).device
    z = torch.randn(n, 2, device=device)
    t = torch.ones(n, 1, device=device)
    h = torch.ones(n, 1, device=device)
    return net(z, t, h)


@torch.no_grad()
def sample_euler(net: nn.Module, n: int = 2048, steps: int = 100) -> torch.Tensor:
    """
    Multi-step Euler integration of the instantaneous velocity field.
    Integrates from t=1 (noise) to t=0 (data) using:
        u_inst(z, t) = (z - net(z, t, h=0)) / t   [h=0 => r=t, instantaneous]
        z_{t-dt} = z_t - u_inst * dt
    This is purely a diagnostic — it tells us if the flow field is correct
    independent of the one-step generalisation.
    """
    net.eval()
    device = next(net.parameters()).device
    z      = torch.randn(n, 2, device=device)
    ts     = torch.linspace(1.0, 1e-3, steps + 1, device=device)  # t=1 down to ~0
    for i in range(steps):
        t_now = ts[i]
        dt    = ts[i] - ts[i + 1]                              # positive step size
        t_vec = torch.full((n, 1), t_now, device=device)
        h_vec = torch.zeros(n, 1, device=device)               # h=0 => instantaneous
        x_pred = net(z, t_vec, h_vec)
        u_inst = (z - x_pred) / t_now.clamp(min=1e-3)
        z = z - u_inst * dt
    return z


# ---------------------------------------------------------------------------
# Entry point — plots one-step AND multi-step side by side
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    net = train(n_steps=30_000, log_every=1_000)

    s_one  = sample_one_step(net, n=2048)
    s_many = sample_euler(net,    n=2048, steps=100)

    print(f"1-step  | mean {s_one.mean(0).tolist()}  std {s_one.std(0).tolist()}")
    print(f"100-step| mean {s_many.mean(0).tolist()}  std {s_many.std(0).tolist()}")

    try:
        import matplotlib.pyplot as plt
        gt = make_swiss_roll(2048)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        kw = dict(s=2, alpha=0.5)

        axes[0].scatter(gt[:, 0],       gt[:, 1],              **kw)
        axes[0].set_title("Ground truth")

        axes[1].scatter(s_one[:, 0].cpu(), s_one[:, 1].cpu(),  **kw, color="tomato")
        axes[1].set_title("MeanFlow — 1 step")

        axes[2].scatter(s_many[:, 0].cpu(), s_many[:, 1].cpu(), **kw, color="steelblue")
        axes[2].set_title("Euler — 100 steps (flow field check)")

        for ax in axes:
            ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)

        plt.tight_layout()
        plt.savefig("swiss_roll_meanflow.png", dpi=120)
        print("Saved swiss_roll_meanflow.png")
    except ImportError:
        pass