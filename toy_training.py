import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jvp
from tqdm import tqdm

import torch
from torch.func import jvp


def sample_tr(batch_size, device, sigma=1.0, data_proportion=0.5):

    def logit_normal(shape):
        x = torch.randn(shape, device=device) * sigma
        return torch.sigmoid(x)

    t = logit_normal((batch_size, 1, 1, 1))
    r = logit_normal((batch_size, 1, 1, 1))

    # enforce ordering
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # FM subset: r = t
    fm_mask = torch.arange(batch_size, device=device) < int(batch_size * data_proportion)
    fm_mask = fm_mask.view(-1, 1, 1, 1)

    r = torch.where(fm_mask, t, r)

    return t, r, fm_mask

def u_fn(net, z, r, t):
    # IMPORTANT: clamp only for denominator stability
    t_safe = t.clamp_min(1e-3)
    return (z - net(z, r, t)) / t_safe

def train_step(net, x, loss_fn):
    batch_size = x.size(0)
    device = x.device

    t, r, _ = sample_tr(batch_size, device)

    e = torch.randn_like(x)

    z = (1 - t) * x + t * e

    # IMPORTANT: NO detach here
    def u_fn(z, r, t):
        t_safe = t.clamp_min(1e-3)
        return (z - net(z, r, t)) / t_safe

    # instantaneous velocity (NO detach)
    v = u_fn(z, t, t)

    # JVP
    u, dudt = torch.func.jvp(
        u_fn,
        (z, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t))
    )

    # ONLY detach here (paper requirement)
    V = u + (t - r) * dudt.detach()

    target = e - x

    loss = loss_fn(V, target)

    return loss


def train(dataset, model, epochs=1000, batch_size=256, lr=1e-4):
    device = next(model.parameters()).device

    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(1, epochs + 1)):

        x = dataset.get_batch(batch_size).to(device)

        optimizer.zero_grad()

        loss = train_step(model, x, loss_fn)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch [{epoch:4d}/{epochs}] "
                f"| Loss: {loss.item():.6f}"
            )

    return model