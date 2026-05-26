import torch
import matplotlib.pyplot as plt


def plot_spiral_2d(x_high, dataset, title="Recovered 2D Spiral"):
    """
    Takes obs_dim-dimensional points, projects back to 2D, and plots.
    """
    if torch.is_tensor(x_high):
        x_high = x_high.detach().cpu()

    coords_2d = dataset.project_back(x_high).cpu().numpy()

    x_coords = coords_2d[:, 0]
    y_coords = coords_2d[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(x_coords, y_coords, c=x_coords, cmap='plasma', s=4, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Intrinsic X")
    ax.set_ylabel("Intrinsic Y")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    plt.colorbar(sc, ax=ax, label='Gradient position')
    plt.tight_layout()
    plt.show()