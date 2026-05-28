"""
train_cfg.py
============
DDP training loop + 1-NFE CFG inference for MeanFlowGuidanceMLP
on the ThreeDShapeDataset (Swiss Roll / Möbius Strip / Torus).

Launch on 2× T4s:
    torchrun --nproc_per_node=2 train_cfg.py

Or with a custom config:
    torchrun --nproc_per_node=2 train_cfg.py \
        --batch_size 2048 --n_steps 100000 --cfg_scale 3.0
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── project modules ──────────────────────────────────────────────────────── #
# Assumes loss_cfg.py, net_cfg.py, dataset_3d.py are on PYTHONPATH / cwd.
# from algorithm2    import SwissRollMeanFlowGuidanceLoss   # the loss you provided
from algorithm2 import PixelMeanFlowGuidanceLoss
from alg2_net     import MeanFlowGuidanceMLP             # the net  you provided
from plot_3d  import ThreeDShapeDataset
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def is_main() -> bool:
    """True only on rank-0 (or when DDP is not in use)."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def setup_ddp():
    """Initialise the process group from env-vars set by torchrun."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def build_lr_lambda(warmup_steps: int, n_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)          # linear ramp 0 → 1
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ══════════════════════════════════════════════════════════════════════════ #
#  Training                                                                  #
# ══════════════════════════════════════════════════════════════════════════ #

def train(args):
    # ── DDP setup ────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + dist.get_rank())   # different seed per rank

    # ── Dataset & sampler ────────────────────────────────────────────────
    dataset = ThreeDShapeDataset(
        n_samples = args.n_data,
        noise     = args.noise,
        seed      = args.seed,
        normalize = True,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        sampler     = sampler,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    net = MeanFlowGuidanceMLP(
        data_dim    = ThreeDShapeDataset.DATA_DIM,     # 3
        num_classes = ThreeDShapeDataset.NUM_CLASSES,  # 3
        hidden      = args.hidden,
        depth       = args.depth,
        emb_dim     = args.emb_dim,
    ).to(device)

    # SyncBatchNorm is a no-op here (MLP has none), but good practice
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank])

    # ── Optimiser & scheduler ────────────────────────────────────────────
    opt = torch.optim.AdamW(
            net.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, build_lr_lambda(args.warmup_steps, args.n_steps)
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_fn = PixelMeanFlowGuidanceLoss(
    num_classes         = ThreeDShapeDataset.NUM_CLASSES,
    noise_dist          = args.noise_dist,
    data_proportion     = args.data_proportion,
    class_dropout_prob  = args.class_dropout_prob,
    t_min               = args.t_min,
    w_max               = args.w_max,
)

    # ── Training loop ────────────────────────────────────────────────────
    step = 0
    it   = iter(loader)

    log(
        f"Starting training — "
        f"world_size={dist.get_world_size()}, "
        f"batch/gpu={args.batch_size}, "
        f"effective_batch={args.batch_size * dist.get_world_size()}"
    )

    while step < args.n_steps:
        sampler.set_epoch(step)          # re-shuffle each pseudo-epoch

        try:
            x, c = next(it)
        except StopIteration:
            it = iter(loader)
            x, c = next(it)

        x = x.to(device, non_blocking=True)   # (B, 3)
        c = c.to(device, non_blocking=True)   # (B,)  integer labels

        opt.zero_grad(set_to_none=True)

        # loss_fn expects the raw nn.Module, not the DDP wrapper, so that
        # torch.func.jvp can trace through it cleanly.
        loss = loss_fn(net.module, x, c)

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            # Average loss across ranks for a cleaner scalar
            loss_tensor = loss.detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            log(
                f"step {step:7d} / {args.n_steps}  "
                f"loss {loss_tensor.item():.6f}  "
                f"lr {sched.get_last_lr()[0]:.2e}"
            )

        if is_main() and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_step{step:07d}.pt")
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model": net.module.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                
                    # training args
                    "args": vars(args),
                
                    # explicit architecture metadata
                    "data_dim": ThreeDShapeDataset.DATA_DIM,
                    "num_classes": ThreeDShapeDataset.NUM_CLASSES,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "emb_dim": args.emb_dim,
                
                    # normalization stats
                    "ds_mean": dataset.mean,
                    "ds_std": dataset.std,
                },
                ckpt_path,
            )
            log(f"  ↳ checkpoint saved → {ckpt_path}")

        step += 1

    # ── Final checkpoint ─────────────────────────────────────────────────
    if is_main():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
        torch.save(
            {
                "step": step,
                "model": net.module.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                
                # training args
                "args": vars(args),
                
                # explicit architecture metadata
                "data_dim": ThreeDShapeDataset.DATA_DIM,
                "num_classes": ThreeDShapeDataset.NUM_CLASSES,
                "hidden": args.hidden,
                "depth": args.depth,
                "emb_dim": args.emb_dim,
                
                # normalization stats
                "ds_mean": dataset.mean,
                "ds_std": dataset.std,
                },
            final_path,
        )
        log(f"Training done. Final checkpoint → {final_path}")

    cleanup_ddp()
    return net.module if is_main() else None


# ══════════════════════════════════════════════════════════════════════════ #
#  1-NFE Inference                                                           #
# ══════════════════════════════════════════════════════════════════════════ #

@torch.no_grad()
def sample_one_step(
    net       : nn.Module,
    n         : int   = 2048,
    label     : int   = 0,
    cfg_scale : float = 3.0,
    device    : str   = "cuda",
    t_val     : float = 1.0,
) -> torch.Tensor:

    net.eval()
    net = net.to(device)

    # pure noise start
    z = torch.randn(
        n,
        net.data_dim,
        device=device
    )

    # generation uses r = 0
    r = torch.zeros(
        (n, 1),
        device=device
    )

    t = torch.full(
        (n, 1),
        t_val,
        device=device
    )

    w = torch.ones((n, 1), device=device)

    cond_labels = torch.full(
        (n,),
        label,
        dtype=torch.long,
        device=device
    )

    # unconditional null token
    uncond_labels = torch.full(
        (n,),
        net.num_classes,
        dtype=torch.long,
        device=device
    )

    # x-predictions
    x_cond = net(
        z,
        r,
        t,
        w,
        cond_labels
    )

    x_uncond = net(
        z,
        r,
        t,
        w,
        uncond_labels
    )

    # CFG blend
    x_hat = x_uncond + cfg_scale * (x_cond - x_uncond)

    return x_hat


def visualize_generated_shapes(net, dataset, cfg_scale=3.0, n_samples=2048, save_path="generated_shapes.png"):
    """
    Generates and plots a side-by-side 3D comparison of the 1-step generated shapes.
    
    Args:
        net: The trained MeanFlowGuidanceMLP model.
        dataset: An instance of ThreeDShapeDataset (used to pull class names and normalization constants).
        cfg_scale: Guidance scale (w) for Classifier-Free Guidance.
        n_samples: Number of points to sample per shape class.
        save_path: Filepath where the final plot will be saved.
    """
    import matplotlib.pyplot as plt
    
    device = next(net.parameters()).device
    class_names = dataset.CLASS_NAMES # ['Swiss Roll', 'Möbius Strip', 'Torus']
    num_classes = len(class_names)
    
    # Grab dataset statistics for accurate physical reconstruction
    ds_mean = torch.tensor(dataset.mean, device=device)
    ds_std = torch.tensor(dataset.std, device=device)
    
    # Set up matplotlib 3D Canvas
    fig = plt.figure(figsize=(6 * num_classes, 6))
    
    # Distinct color palette for each geometric flow
    colors = ['#FF4B4B', '#0083B0', '#00B4DB'] 
    
    for label, name in enumerate(class_names):
        # 1. Generate normalized 3D samples using your 1-NFE sampler
        samples_norm = sample_one_step(
            net=net,
            n=n_samples,
            label=label,
            cfg_scale=cfg_scale,
            device=device
        )
        
        # 2. Denormalize samples to recover original physical scales
        samples_orig = (samples_norm * ds_std + ds_mean).cpu().numpy()
        
        # 3. Add a dedicated 3D subplot
        ax = fig.add_subplot(1, num_classes, label + 1, projection='3d')
        
        # Draw the generated point cloud
        ax.scatter(
            samples_orig[:, 0], 
            samples_orig[:, 1], 
            samples_orig[:, 2], 
            c=colors[label % len(colors)], 
            alpha=0.6, 
            s=4, 
            edgecolor='none'
        )
        
        # Aesthetic tuning for 3D visibility
        ax.set_title(f"{name}\n(CFG={cfg_scale})", fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Balance axes ratios evenly to prevent squishing structural shapes
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_tick_params(labelsize=9)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved 3D shape generation plot to: {save_path}")


# ══════════════════════════════════════════════════════════════════════════ #
#  Quick inference smoke-test (single-GPU, loads a checkpoint)              #
# ══════════════════════════════════════════════════════════════════════════ #

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.ckpt}")
    torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct
])

    ckpt = torch.load(
    args.ckpt,
    map_location="cpu",
    weights_only=False
)
    cfg  = ckpt.get("args", {})

    # Reconstruct dataset to fetch normalisation metrics dynamically
    dataset = ThreeDShapeDataset(
        n_samples=100,  # Minimal sample footprint since we only need metadata
        noise=cfg.get("noise", 0.05),
        normalize=True
    )

    net = MeanFlowGuidanceMLP(
    data_dim    = ckpt["data_dim"],
    num_classes = ckpt["num_classes"],
    hidden      = ckpt["hidden"],
    depth       = ckpt["depth"],
    emb_dim     = ckpt["emb_dim"],
).to(device)
    net.load_state_dict(ckpt["model"])

    # ──── NEW: Call the visualizer ────
    visualize_generated_shapes(
        net=net,
        dataset=dataset,
        cfg_scale=args.cfg_scale,
        n_samples=args.n_samples,
        save_path="generated_shapes_output.png"
    )

    print("Inference and plot export done.")


# ══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                       #
# ══════════════════════════════════════════════════════════════════════════ #

def parse_args():
    p = argparse.ArgumentParser()

    # Mode
    p.add_argument("--mode", choices=["train", "infer"], default="train")

    # ── Dataset ──────────────────────────────────────────────────────────
    p.add_argument("--n_data",    type=int,   default=90_000)
    p.add_argument("--noise",     type=float, default=0.05)

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--hidden",    type=int,   default=512)
    p.add_argument("--depth",     type=int,   default=5)
    p.add_argument("--emb_dim",   type=int,   default=128)

    # ── Loss ──────────────────────────────────────────────────────────────
    p.add_argument("--noise_dist",      default="uniform")
    p.add_argument("--data_proportion", type=float, default=0.25)
    p.add_argument("--t_min",           type=float, default=0.02)
    # p.add_argument("--cfg_scale_max",   type=float, default=7.0)
    p.add_argument("--w_max", type=float, default=7.0)
    p.add_argument("--class_dropout_prob", type=float, default=0.1)

    # ── Optimisation ──────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--n_steps",      type=int,   default=50_000)
    p.add_argument("--warmup_steps", type=int,   default=2_000)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)

    # ── Logging / checkpointing ───────────────────────────────────────────
    p.add_argument("--log_every",  type=int, default=500)
    p.add_argument("--save_every", type=int, default=10_000,
                   help="0 to disable intermediate checkpoints")
    p.add_argument("--ckpt_dir",   default="checkpoints")

    # ── Inference ─────────────────────────────────────────────────────────
    p.add_argument("--ckpt",      default="checkpoints/ckpt_final.pt")
    p.add_argument("--n_samples", type=int,   default=4096)
    p.add_argument("--cfg_scale", type=float, default=3.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        run_inference(args)
