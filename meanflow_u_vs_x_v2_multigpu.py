"""
Toy projection experiment from the MeanFlow paper.

A 2D Swiss Roll is linearly projected into D-dimensional space via a fixed
column-orthonormal matrix P ∈ R^{D×2}.  We train two MeanFlow variants:

  x-prediction:  net outputs x_hat ∈ R^D;  u = (z - x_hat) / t
  u-prediction:  net outputs u_hat ∈ R^D;  original MeanFlow algorithm

Both use a 7-layer SiLU MLP with 256 hidden units.
We compare 1-NFE samples projected back to 2D for visualisation.

D ∈ {2, 8, 16, 512}

Usage (2x GPU):
    python projection_experiment.py
"""

import math
import sys
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from swiss_roll_meanflow import make_swiss_roll, SwissRollMeanFlowLoss, SinusoidalEmbedding


# ---------------------------------------------------------------------------
# Network — 7-layer SiLU MLP, 256 hidden units
# ---------------------------------------------------------------------------

class PaperMLP(nn.Module):
    def __init__(self, data_dim: int, emb_dim: int = 64, hidden: int = 256, mode: str = 'x'):
        super().__init__()
        assert mode in ('x', 'u')
        self.mode = mode
        self.t_emb = SinusoidalEmbedding(emb_dim)
        self.h_emb = SinusoidalEmbedding(emb_dim)
        # 7 layers: input + 5 hidden + output
        dims = [data_dim + 2 * emb_dim] + [hidden] * 6 + [data_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z, t, h):
        B = z.shape[0]
        t_e = self.t_emb(t.view(B))
        h_e = self.h_emb(h.view(B))
        return self.net(torch.cat([z, t_e, h_e], dim=-1))


# ---------------------------------------------------------------------------
# u-prediction loss (original MeanFlow paper, Algorithm 1)
# ---------------------------------------------------------------------------

class UPredMeanFlowLoss:
    """
    Original MeanFlow loss (u-prediction).

    u, dudt  = jvp(fn, (z, r, t), (v, 0, 1))   # v = e - x is the z-tangent
    u_tgt    = v - (t - r) * dudt               # subtract, not add
    error    = u - stopgrad(u_tgt)
    loss     = ||error||^2
    """
    def __init__(self, noise_dist='uniform', data_proportion=0.25, t_min=0.02):
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
        mask = (torch.arange(B, device=device) < int(B * self.data_proportion)).unsqueeze(1)
        r = torch.where(mask, t, r)
        return t, r

    def __call__(self, net, x):
        B, device = x.shape[0], x.device
        t, r = self._sample_t_r(B, device)
        e    = torch.randn_like(x)
        z_t  = (1.0 - t) * x + t * e
        v    = e - x

        def fn(z, r_, t_):
            return net(z, t_, t_ - r_)

        primals  = (z_t, r,                   t                  )
        tangents = (v,   torch.zeros_like(r), torch.ones_like(t) )
        u, dudt  = torch.func.jvp(fn, primals, tangents)

        u_tgt = v - (t - r) * dudt
        error = u - u_tgt.detach()
        return error.pow(2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def make_projection_matrix(D, seed=0):
    """Column-orthonormal P ∈ R^{D×2}."""
    rng = torch.Generator().manual_seed(seed)
    A   = torch.randn(max(D, 2), max(D, 2), generator=rng)
    Q, _ = torch.linalg.qr(A)
    return Q[:D, :2]

def project_up(x2, P):   return x2 @ P.T   # [B,2] -> [B,D]
def project_down(xD, P): return xD @ P     # [B,D] -> [B,2]


# ---------------------------------------------------------------------------
# Training (runs in a worker process)
# ---------------------------------------------------------------------------

def train_model(mode, D, P, device, n_data=50_000, batch_size=1024,
                lr=1e-3, n_steps=50_000, warmup_steps=2_000,
                log_every=5_000, seed=42):
    torch.manual_seed(seed)
    data2d = make_swiss_roll(n_data).to(device)
    dataD  = project_up(data2d, P.to(device))
    loader = DataLoader(TensorDataset(dataD), batch_size=batch_size,
                        shuffle=True, drop_last=True)
    net    = PaperMLP(data_dim=D, mode=mode).to(device)
    opt    = torch.optim.Adam(net.parameters(), lr=lr)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / warmup_steps
        p = (s - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched   = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = SwissRollMeanFlowLoss() if mode == 'x' else UPredMeanFlowLoss()

    step = 0
    it   = iter(loader)
    while step < n_steps:
        try:   (x,) = next(it)
        except StopIteration:
            it = iter(loader); (x,) = next(it)

        opt.zero_grad()
        loss = loss_fn(net, x)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            print(f"  [{mode}-pred D={D:4d} {device}] step {step:6d}  loss {loss.item():.5f}",
                  flush=True)
        step += 1

    return net


@torch.no_grad()
def sample_one_step(net, D, n=1024):
    net.eval()
    device = next(net.parameters()).device
    z = torch.randn(n, D, device=device)
    t = torch.ones(n, 1, device=device)
    h = torch.ones(n, 1, device=device)
    out = net(z, t, h)
    return out if net.mode == 'x' else z - out


# ---------------------------------------------------------------------------
# Worker: trains all jobs assigned to one GPU, writes results to a queue
# ---------------------------------------------------------------------------

def worker(gpu_id, jobs, result_queue):
    """
    jobs: list of (mode, D) tuples assigned to this GPU.
    Puts (mode, D, samples_2d_numpy) into result_queue when done.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    for mode, D in jobs:
        print(f"\n[GPU {gpu_id}] Starting {mode}-pred D={D}", flush=True)
        P   = make_projection_matrix(D, seed=0)
        net = train_model(mode, D, P, device=device)
        sD  = sample_one_step(net, D, n=1024)
        s2d = project_down(sD.cpu(), P.cpu()).numpy()
        result_queue.put((mode, D, s2d))
        print(f"[GPU {gpu_id}] Done {mode}-pred D={D}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DIMS     = [2, 8, 16, 512]
    N_SAMPLE = 1024
    N_GPUS   = min(2, torch.cuda.device_count())

    if N_GPUS < 1:
        print("No CUDA GPUs found — falling back to CPU (will be slow)")
        N_GPUS = 1

    # Build job list: all (mode, D) combinations
    all_jobs = [(mode, D) for D in DIMS for mode in ('x', 'u')]  # 8 jobs

    # Distribute jobs round-robin across GPUs
    gpu_jobs = [[] for _ in range(N_GPUS)]
    for i, job in enumerate(all_jobs):
        gpu_jobs[i % N_GPUS].append(job)

    print(f"Running on {N_GPUS} GPU(s)")
    for g, jobs in enumerate(gpu_jobs):
        print(f"  GPU {g}: {jobs}")

    # Launch one process per GPU
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    for gpu_id in range(N_GPUS):
        if not gpu_jobs[gpu_id]:
            continue
        p = mp.Process(target=worker, args=(gpu_id, gpu_jobs[gpu_id], result_queue))
        p.start()
        processes.append(p)

    # Collect results as they arrive
    results = {}   # (mode, D) -> samples_2d
    total = len(all_jobs)
    for _ in range(total):
        mode, D, s2d = result_queue.get()
        results[(mode, D)] = s2d
        print(f"Collected: {mode}-pred D={D}")

    for p in processes:
        p.join()

    # ------------------------------------------------------------------
    # Plot: rows = D, cols = [ground truth | x-pred | u-pred]
    # ------------------------------------------------------------------
    gt2d = make_swiss_roll(N_SAMPLE).numpy()

    fig, axes = plt.subplots(len(DIMS), 3, figsize=(12, 4 * len(DIMS)), squeeze=False)

    for col, title in enumerate(["Ground truth (2D)", "x-prediction (1 NFE)", "u-prediction (1 NFE)"]):
        axes[0][col].set_title(title, fontsize=11, fontweight='bold')

    kw  = dict(s=3, alpha=0.5, rasterized=True)
    lim = (-2.8, 2.8)

    for row, D in enumerate(DIMS):
        axes[row][0].scatter(gt2d[:, 0], gt2d[:, 1], color="steelblue", **kw)
        axes[row][1].scatter(results[('x', D)][:, 0], results[('x', D)][:, 1], color="tomato",   **kw)
        axes[row][2].scatter(results[('u', D)][:, 0], results[('u', D)][:, 1], color="seagreen", **kw)

        for col in range(3):
            ax = axes[row][col]
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
        axes[row][0].set_ylabel(f"D = {D}", fontsize=11, fontweight='bold')

    plt.suptitle(
        "MeanFlow: x-pred vs u-pred across projected dimensions (1-NFE)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("projection_experiment.png", dpi=120, bbox_inches='tight')
    print("\nSaved projection_experiment.png")