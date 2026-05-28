"""
dataset_3d.py
=============
Three-class 3-D point-cloud dataset for MeanFlow CFG training, plus
an interactive / static visualization helper.

Classes
-------
0 : Swiss Roll 3-D
1 : Möbius Strip
2 : Torus

Usage
-----
    from dataset_3d import ThreeDShapeDataset, visualize_dataset

    ds = ThreeDShapeDataset(n_samples=30_000, noise=0.05)
    visualize_dataset(ds)                        # interactive (plotly)
    visualize_dataset(ds, backend="matplotlib")  # static (matplotlib)
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class ThreeDShapeDataset(Dataset):
    """
    Three-class 3-D point-cloud dataset.

    Parameters
    ----------
    n_samples   : total number of points (split equally across 3 classes)
    noise       : additive Gaussian noise std applied to every shape
    seed        : RNG seed for reproducibility
    normalize   : whether to z-score the coordinates
    """

    CLASS_NAMES = ["Swiss Roll", "Möbius Strip", "Torus"]
    # Colour palette (hex) — used by visualize_dataset
    CLASS_COLORS = ["#e05c5c", "#5b9bd5", "#6abf69"]
    NUM_CLASSES  = 3
    DATA_DIM     = 3

    def __init__(
        self,
        n_samples : int   = 30_000,
        noise     : float = 0.05,
        seed      : int   = 42,
        normalize : bool  = True,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.noise     = noise
        self.seed      = seed

        rng         = np.random.default_rng(seed)
        n_per_class = n_samples // self.NUM_CLASSES

        pts, lbls = [], []

        # ── class 0 : Swiss Roll ──────────────────────────────────────────
        sr, _ = make_swiss_roll(n_samples=n_per_class, noise=0.0, random_state=seed)
        sr    = sr.astype(np.float32)
        sr   /= sr.std(axis=0, keepdims=True).clip(1e-6)
        sr   += rng.normal(0, noise, sr.shape).astype(np.float32)
        pts.append(sr)
        lbls.append(np.zeros(n_per_class, dtype=np.int64))

        # ── class 1 : Möbius Strip ────────────────────────────────────────
        pts.append(self._mobius(n_per_class, noise, rng))
        lbls.append(np.ones(n_per_class, dtype=np.int64))

        # ── class 2 : Torus ───────────────────────────────────────────────
        pts.append(self._torus(n_per_class, noise, rng))
        lbls.append(np.full(n_per_class, 2, dtype=np.int64))

        X = np.concatenate(pts,  axis=0)   # (N, 3)
        y = np.concatenate(lbls, axis=0)   # (N,)

        # shuffle
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        if normalize:
            self.mean = X.mean(axis=0)
            self.std  = X.std(axis=0).clip(1e-6)
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(3, dtype=np.float32)
            self.std  = np.ones(3,  dtype=np.float32)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    # ------------------------------------------------------------------ #
    #  Shape generators                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mobius(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
        """
        Möbius strip parametrisation:
            u ∈ [0, 2π),  v ∈ [-1, 1]
            x = (1 + v/2 · cos(u/2)) · cos(u)
            y = (1 + v/2 · cos(u/2)) · sin(u)
            z =      v/2 · sin(u/2)
        """
        u      = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        v      = rng.uniform(-1, 1,          n).astype(np.float32)
        half_u = u / 2.0
        r      = 1.0 + (v / 2.0) * np.cos(half_u)
        x      = r * np.cos(u)
        y      = r * np.sin(u)
        z      = (v / 2.0) * np.sin(half_u)
        pts    = np.stack([x, y, z], axis=-1)
        pts   += rng.normal(0, noise, pts.shape).astype(np.float32)
        return pts

    @staticmethod
    def _torus(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
        """
        Torus (R=1 major radius, r=0.4 tube radius).
        θ is rejection-sampled for uniform surface-area coverage.
        """
        R, r  = 1.0, 0.4
        theta = []
        while len(theta) < n:
            th   = rng.uniform(0, 2 * math.pi, n * 2).astype(np.float32)
            acc  = rng.uniform(0, 1,            n * 2).astype(np.float32)
            keep = acc < (R + r * np.cos(th)) / (R + r)
            theta.extend(th[keep].tolist())
        theta = np.array(theta[:n], dtype=np.float32)
        phi   = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        x     = (R + r * np.cos(theta)) * np.cos(phi)
        y     = (R + r * np.cos(theta)) * np.sin(phi)
        z     =  r * np.sin(theta)
        pts   = np.stack([x, y, z], axis=-1)
        pts  += rng.normal(0, noise, pts.shape).astype(np.float32)
        return pts

    # ------------------------------------------------------------------ #
    #  Dataset protocol                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert z-score normalisation (useful for visualising generated samples)."""
        mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
        std  = torch.tensor(self.std,  device=x.device, dtype=x.dtype)
        return x * std + mean

    def get_class_points(self, label: int, max_pts: int = 5_000) -> np.ndarray:
        """Return up to *max_pts* (N,3) numpy array for a single class."""
        mask = (self.y == label).numpy()
        pts  = self.X.numpy()[mask]
        if len(pts) > max_pts:
            idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
        return pts


# ═══════════════════════════════════════════════════════════════════════════ #
#  Visualization                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def visualize_dataset(
    dataset: ThreeDShapeDataset,
    max_pts_per_class: int = 4_000,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    point_size: float = 2.0,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise all three shapes in a single 3-D scatter plot.

    Parameters
    ----------
    dataset           : ThreeDShapeDataset instance
    max_pts_per_class : points rendered per class (subsampled for speed)
    backend           : "plotly" (interactive, default) or "matplotlib" (static)
    point_size        : marker size
    alpha             : opacity
    save_path         : if given, save figure to this path instead of showing
                        (.html for plotly, .png/.pdf etc. for matplotlib)
    """
    if backend == "plotly":
        _visualize_plotly(dataset, max_pts_per_class, point_size, alpha, save_path)
    elif backend == "matplotlib":
        _visualize_matplotlib(dataset, max_pts_per_class, point_size, alpha, save_path)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'plotly' or 'matplotlib'.")


# ── Plotly ──────────────────────────────────────────────────────────────── #

def _visualize_plotly(
    dataset    : ThreeDShapeDataset,
    max_pts    : int,
    point_size : float,
    alpha      : float,
    save_path  : Optional[str],
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("Install plotly:  pip install plotly") from e

    # ── combined view + one subplot per class ───────────────────────────
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        subplot_titles=["All classes"] + dataset.CLASS_NAMES,
        horizontal_spacing=0.02,
    )

    opacity = alpha

    for label, (name, color) in enumerate(
        zip(dataset.CLASS_NAMES, dataset.CLASS_COLORS)
    ):
        pts = dataset.get_class_points(label, max_pts)

        trace = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            name=name,
            marker=dict(size=point_size, color=color, opacity=opacity),
            showlegend=(True),  # legend only on combined plot
        )

        # Add to combined (col 1) and individual subplot (col label+2)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", name=name,
            marker=dict(size=point_size, color=color, opacity=opacity),
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", name=name,
            marker=dict(size=point_size, color=color, opacity=opacity),
            showlegend=False,
        ), row=1, col=label + 2)

    fig.update_layout(
        title=dict(
            text="ThreeDShapeDataset — point cloud overview",
            font=dict(size=18),
        ),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(color="#e0e0e0"),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(30,30,30,0.7)",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=550,
    )

    # dark axes for every scene
    scene_style = dict(
        xaxis=dict(backgroundcolor="#111", gridcolor="#333", color="#aaa"),
        yaxis=dict(backgroundcolor="#111", gridcolor="#333", color="#aaa"),
        zaxis=dict(backgroundcolor="#111", gridcolor="#333", color="#aaa"),
        bgcolor="#111",
    )
    for i in range(1, 5):
        key = "scene" if i == 1 else f"scene{i}"
        fig.update_layout(**{key: scene_style})

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive plot → {save_path}")
    else:
        fig.show()


# ── Matplotlib ───────────────────────────────────────────────────────────── #

def _visualize_matplotlib(
    dataset    : ThreeDShapeDataset,
    max_pts    : int,
    point_size : float,
    alpha      : float,
    save_path  : Optional[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as e:
        raise ImportError("Install matplotlib:  pip install matplotlib") from e

    fig = plt.figure(figsize=(20, 5), facecolor="#0d0d0d")
    fig.suptitle(
        "ThreeDShapeDataset — point cloud overview",
        color="#e0e0e0", fontsize=15, y=1.01,
    )

    axes = []
    for col in range(4):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        ax.set_facecolor("#111111")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333333")
        ax.tick_params(colors="#666666", labelsize=6)
        axes.append(ax)

    titles = ["All classes"] + dataset.CLASS_NAMES

    for label, (name, color) in enumerate(
        zip(dataset.CLASS_NAMES, dataset.CLASS_COLORS)
    ):
        pts = dataset.get_class_points(label, max_pts)

        # combined plot
        axes[0].scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=point_size, c=color, alpha=alpha, label=name, linewidths=0,
        )

        # individual subplot
        axes[label + 1].scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=point_size, c=color, alpha=alpha, linewidths=0,
        )
        axes[label + 1].set_title(name, color="#e0e0e0", fontsize=11, pad=6)

    axes[0].set_title("All classes", color="#e0e0e0", fontsize=11, pad=6)
    axes[0].legend(
        markerscale=4, framealpha=0.3, facecolor="#222",
        edgecolor="#555", labelcolor="#e0e0e0", fontsize=8,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved static plot → {save_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Quick smoke-test                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset smoke-test + visualisation")
    parser.add_argument("--n",       type=int,   default=30_000)
    parser.add_argument("--noise",   type=float, default=0.05)
    parser.add_argument("--backend", choices=["plotly", "matplotlib"], default="plotly")
    parser.add_argument("--save",    type=str,   default=None,
                        help="Save path (.html for plotly, .png for matplotlib)")
    args = parser.parse_args()

    print("Building dataset …")
    ds = ThreeDShapeDataset(n_samples=args.n, noise=args.noise)

    print(f"  Total samples : {len(ds)}")
    print(f"  Data shape    : {ds.X.shape}")
    print(f"  Labels        : {ds.y.unique().tolist()}")
    print(f"  Mean (norm)   : {ds.X.mean(0).tolist()}")
    print(f"  Std  (norm)   : {ds.X.std(0).tolist()}")

    x_sample, c_sample = ds[0]
    print(f"  Sample x      : {x_sample}  label={c_sample.item()}")

    print(f"\nLaunching {args.backend} visualisation …")
    visualize_dataset(ds, backend=args.backend, save_path=args.save)