import torch
import torch.nn as nn
import numpy as np

# Implementation 1 of the Pixel MeanFlow Guidance Loss as described in the reference paper.
class PixelMeanFlowGuidanceLoss:
    """
    Pixel MeanFlow Guidance Loss (Algorithm 2) for conditional generation / CFG.

    u(z, r, t, w, c) = (z - net(z, t, h, w, c)) / t
    v_c              = u(z, t, t, w, c)
    v_u              = u(z, t, t, w, None)
    v_g              = (e - x) + (1 - 1/w) * (v_c - v_u)
    u, dudt          = jvp(u_fn, (z, r, t, w, c), (v_c, 0, 1, 0, 0))
    V                = u + (t - r) * stopgrad(dudt)
    loss             = ||V - stopgrad(v_g)||^2
    """

    def __init__(
        self,
        num_classes: int,
        noise_dist: str = "uniform",
        data_proportion: float = 0.25,
        class_dropout_prob: float = 0.1,
        t_min: float = 0.02,
        w_max: float = 7.0,
    ):
        self.num_classes = num_classes
        self.uncond_class = num_classes  # Usually index 'num_classes' is reserved for null token
        self.noise_dist = noise_dist
        self.data_proportion = data_proportion
        self.class_dropout_prob = class_dropout_prob
        self.t_min = t_min
        self.w_max = w_max

    def _sample_time(self, shape, device):
        if self.noise_dist == "uniform":
            return self.t_min + (1.0 - self.t_min) * torch.rand(shape, device=device)
        elif self.noise_dist == "logit_normal":
            # Matching the P_mean = -0.4, P_std = 1.0 from reference paper
            rnd = torch.randn(shape, device=device)
            return torch.sigmoid(rnd * 1.0 + (-0.4)).clamp(self.t_min, 1.0)
        raise ValueError(self.noise_dist)

    def _sample_t_r_w(self, B, device):
        shape = (B, 1)
        s1 = self._sample_time(shape, device)
        s2 = self._sample_time(shape, device)
        t = torch.max(s1, s2)   # t >= r always
        r = torch.min(s1, s2)

        # Self-prediction slice: r = t
        data_mask = (torch.rand((B, 1), device=device) < self.data_proportion)
        r = torch.where(data_mask, t, r)

        # Sample CFG scale w uniformly between 1.0 and w_max (or using power law distribution)
        # Here we use standard uniform space sampling matching continuous training variants
        w = 1.0 + (self.w_max - 1.0) * torch.rand(shape, device=device)
        return t, r, w

    def __call__(self, net: nn.Module, x: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device

        # 1. Sample dynamics variables
        t, r, w = self._sample_t_r_w(B, device)
        e = torch.randn_like(x)
        z_t = (1.0 - t) * x + t * e
        v_true = e - x  # Ground-truth unguided trajectory velocity

        # Apply standard classification conditional dropout for training stability
        drop_mask = torch.rand(B, device=device) < self.class_dropout_prob
        c_dropped = torch.where(drop_mask, torch.tensor(self.uncond_class, device=device), classes)

        # 2. Define the core average velocity field function u_fn
        # In Pixel MeanFlow: u = (z - net(...)) / t
        def u_fn_functional(z_input, r_input, t_input, w_input, condition_input):
            h_input = t_input - r_input
            # net receives: (noisy_latent, t, h, cfg_scale, condition)
            x_pred = net(z_input, t_input, h_input, w_input, condition_input)
            return (z_input - x_pred) / t_input

        # 3. Calculate v_c and v_u to compute the CFG Target vector (r = t -> h = 0)
        # Note: We wrap evaluations in torch.no_grad() or detach because 
        # Alg 2 defines v_g strictly under a stopgrad operator.
        with torch.no_grad():
            # For v_c and v_u, r is set equal to t, forcing h = 0
            v_c = u_fn_functional(z_t, t, t, w, c_dropped)
            
            # For unconditional path, pass the null token class
            uncond_tokens = torch.full_like(c_dropped, self.uncond_class)
            v_u = u_fn_functional(z_t, t, t, w, uncond_tokens)
            
            # Target generation sequence following Alg 2
            v_g = v_true + (1.0 - 1.0 / w) * (v_c - v_u)

        # 4. Instantaneous velocity conditional evaluation wrapper for JVP tracking 
        # JVP takes floating tensors; we capture the discrete classes inside the closure
        def u_fn_jvp_wrap(z_arg, r_arg, t_arg, w_arg):
            return u_fn_functional(z_arg, r_arg, t_arg, w_arg, c_dropped)

        # Compute instantaneous condition velocity for the primal tracking
        # For the JVP call, r=t maps exactly to the paper's specification
        with torch.no_grad():
            v_c_inst = u_fn_jvp_wrap(z_t, t, t, w)

        # 5. JVP evaluation over time variables 
        # Primal inputs and corresponding tangent velocity vectors 
        primals = (z_t, r, t, w)
        tangents = (
            v_c_inst,                     # dz/dt = v_c
            torch.zeros_like(r),          # dr/dt = 0 (r is constant during t variation)
            torch.ones_like(t),           # dt/dt = 1
            torch.zeros_like(w)           # dw/dt = 0
        )

        u, dudt = torch.func.jvp(u_fn_jvp_wrap, primals, tangents)

        # 6. Compound construction target 
        # detach() applies the `stopgrad` to the derivative tracking path
        V = u + (t - r) * dudt.detach()

        # Compute Mean Squared Error against the guidance vector path
        loss = (V - v_g.detach()).pow(2).flatten(1).sum(dim=1).mean()
        return loss


# Alternate implementation

import math
import torch
import torch.nn as nn

class SwissRollMeanFlowGuidanceLoss:
    """
    Pixel MeanFlow w/ CFG — Algorithm 2.
    
    u(z, r, t, w, c) = (z - net(z, t, h, w, c)) / t    average velocity
    v_c = u(z, t, t, w, c)                            cond instantaneous velocity
    v_u = u(z, t, t, w, None)                         uncond instantaneous velocity
    v_g = (e - x) + (1 - 1/w) * (v_c - v_u)           CFG target  [stopgrad]
    u, dudt = jvp(u_fn, (z, r, t, w), (v_c, 0, 1, 0)) JVP wrt t   [c closed over]
    V   = u + (t - r) * stopgrad(dudt)
    loss = ||V - stopgrad(v_g)||^2
    """

    def __init__(
        self,
        noise_dist: str = "uniform",
        data_proportion: float = 0.25,
        t_min: float = 0.02,
        cfg_scale_max: float = 7.0,
    ):
        self.noise_dist = noise_dist
        self.data_proportion = data_proportion
        self.t_min = t_min
        self.cfg_scale_max = cfg_scale_max

    def _sample_time(self, shape, device):
        if self.noise_dist == "uniform":
            return self.t_min + (1.0 - self.t_min) * torch.rand(shape, device=device)
        elif self.noise_dist == "logit_normal":
            return torch.sigmoid(
                torch.randn(shape, device=device) * 1.0 - 0.4
            ).clamp(self.t_min, 1.0)
        raise ValueError(self.noise_dist)

    def _sample_t_r(self, B, device):
        s1 = self._sample_time((B, 1), device)
        s2 = self._sample_time((B, 1), device)
        t  = torch.max(s1, s2)
        r  = torch.min(s1, s2)
        
        # data_proportion slice: r=t (self-prediction / flow-matching samples)
        mask = (torch.arange(B, device=device) < int(B * self.data_proportion)).unsqueeze(1)
        r = torch.where(mask, t, r)
        return t, r

    def _sample_cfg_scale(self, B, device):
        # w ~ exp(U * log(1 + w_max)), giving log-uniform coverage over [1, w_max]
        u = torch.rand((B, 1), device=device)
        return torch.exp(u * math.log(1.0 + self.cfg_scale_max))

    def __call__(self, net: nn.Module, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net : callable  net(z, t, h, w, c) -> x_pred
                  c=None means unconditional (network handles null condition)
            x   : clean data  (B, D)
            c   : condition   (B,) e.g. integer class labels
        Returns:
            loss : scalar
        """
        B, device = x.shape[0], x.device

        # ── 1. Sample Dynamics Parameters ──────────────────────────────────
        t, r = self._sample_t_r(B, device)       # (B, 1)
        w    = self._sample_cfg_scale(B, device)  # (B, 1)
        e    = torch.randn_like(x)
        z_t  = (1.0 - t) * x + t * e

        # ── 2. Functional Average Velocity Field Definition ────────────────
        def u_fn(z_input, r_input, t_input, w_input, condition_input):
            h_input = t_input - r_input
            x_pred = net(z_input, t_input, h_input, w_input, condition_input)
            return (z_input - x_pred) / t_input.clamp(min=self.t_min)

        # ── 3. Compute CFG Target Vectors via Detach ───────────────────────
        # Evaluated safely on-graph so tracking stays perfect, but isolated via .detach()
        v_c = u_fn(z_t, t, t, w, c)              # Conditional, h=0 (r=t)
        v_u = u_fn(z_t, t, t, w, None)           # Unconditional, h=0 (r=t)

        # The complete stopgrad'd target velocity composition matching Algorithm 2
        v_g = ((e - x) + (1.0 - 1.0 / w) * (v_c - v_u)).detach()

        # ── 4. JVP Evaluation Wrapper Over Time Variables ──────────────────
        def u_fn_t(z_arg, r_arg, t_arg):
            """u_fn with w and c closed over; only (z, r, t) are tracked primals."""
            return u_fn(z_arg, r_arg, t_arg, w, c)

        # Recompute instantaneous conditional velocity on-graph for primal tracking 
        v_c_grad = u_fn_t(z_t, t, t)

        primals  = (z_t,      r,                   t                  )
        tangents = (v_c_grad, torch.zeros_like(r), torch.ones_like(t) )

        # Execute Jacobian-Vector Product across time tracking primals
        u, dudt = torch.func.jvp(u_fn_t, primals, tangents)

        # ── 5. Compound Loss Vector Calculation ────────────────────────────
        # Apply the stopgrad step to the derivative tracking path via .detach()
        V = u + (t - r) * dudt.detach()
        
        # Flatten feature dims to provide clean dimensional invariance (e.g. 1D, 2D, 3D, etc.)
        loss = (V - v_g).pow(2).flatten(1).sum(dim=1).mean()
        return loss

# import math

# import torch
# import torch.nn as nn


# class SwissRollMeanFlowGuidanceLoss:
#     """
#     Pixel MeanFlow w/ CFG — Algorithm 2.

#     u(z,r,t,w,c) = (z - net(z,t,h,w,c)) / t     average velocity
#     v_c = u(z,t,t,w,c)                            cond  instantaneous velocity
#     v_u = u(z,t,t,w,None)                         uncond instantaneous velocity
#     v_g = (e-x) + (1 - 1/w)*(v_c - v_u)          CFG target  [stopgrad]
#     u, dudt = jvp(u_fn, (z,r,t,w), (v_c,0,1,0))  JVP wrt t   [c closed over]
#     V   = u + (t-r)*stopgrad(dudt)
#     loss = ||V - stopgrad(v_g)||^2
#     """

#     def __init__(
#         self,
#         noise_dist: str = "uniform",
#         data_proportion: float = 0.25,
#         t_min: float = 0.02,
#         cfg_scale_max: float = 7.0,
#     ):
#         self.noise_dist = noise_dist
#         self.data_proportion = data_proportion
#         self.t_min = t_min
#         self.cfg_scale_max = cfg_scale_max

#     def _sample_time(self, shape, device):
#         if self.noise_dist == "uniform":
#             return self.t_min + (1.0 - self.t_min) * torch.rand(shape, device=device)
#         elif self.noise_dist == "logit_normal":
#             return torch.sigmoid(
#                 torch.randn(shape, device=device) * 1.0 - 0.4
#             ).clamp(self.t_min, 1.0)
#         raise ValueError(self.noise_dist)

#     def _sample_t_r(self, B, device):
#         s1 = self._sample_time((B, 1), device)
#         s2 = self._sample_time((B, 1), device)
#         t  = torch.max(s1, s2)
#         r  = torch.min(s1, s2)
#         # data_proportion slice: r=t (self-prediction / flow-matching samples)
#         mask = (torch.arange(B, device=device) < int(B * self.data_proportion)).unsqueeze(1)
#         r = torch.where(mask, t, r)
#         return t, r

#     def _sample_cfg_scale(self, B, device):
#         # w ~ exp(U * log(1 + w_max)), giving log-uniform coverage over [1, w_max]
#         u = torch.rand((B, 1), device=device)
#         return torch.exp(u * math.log(1.0 + self.cfg_scale_max))

#     def __call__(self, net: nn.Module, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             net : callable  net(z, t, h, w, c) -> x_pred
#                   c=None means unconditional (network must handle null condition)
#             x   : clean data  (B, D)
#             c   : condition   (B,) e.g. integer class labels
#         Returns:
#             loss : scalar
#         """
#         B, device = x.shape[0], x.device

#         t, r = self._sample_t_r(B, device)       # (B, 1)
#         w    = self._sample_cfg_scale(B, device)  # (B, 1)
#         e    = torch.randn_like(x)
#         z_t  = (1.0 - t) * x + t * e

#         def u_fn(z, r_, t_, w_, cond):
#             x_pred = net(z, t_, t_ - r_, w_, cond)
#             return (z - x_pred) / t_.clamp(min=self.t_min)

#         # ------------------------------------------------------------------
#         # CFG target v_g  (fully stopgrad'd — no grads needed here)
#         # ------------------------------------------------------------------
#         with torch.no_grad():
#             v_c = u_fn(z_t, t, t, w, c)     # conditional,   h=0 (r=t)
#             v_u = u_fn(z_t, t, t, w, None)  # unconditional, h=0 (r=t)

#         v_g = ((e - x) + (1.0 - 1.0 / w) * (v_c - v_u)).detach()  # stopgrad

#         # ------------------------------------------------------------------
#         # JVP wrt t — close over (w, c) so all differentiated primals are float.
#         # v_c is recomputed with grad so the JVP can differentiate through net.
#         # ------------------------------------------------------------------
#         def u_fn_t(z, r_, t_):
#             """u_fn with w and c closed over; only (z, r, t) differentiated."""
#             return u_fn(z, r_, t_, w, c)

#         v_c_grad = u_fn_t(z_t, t, t)  # recompute conditional velocity with grad

#         primals  = (z_t,      r,                   t                  )
#         tangents = (v_c_grad, torch.zeros_like(r), torch.ones_like(t) )

#         u, dudt = torch.func.jvp(u_fn_t, primals, tangents)

#         # ------------------------------------------------------------------
#         # Compound V and loss
#         # ------------------------------------------------------------------
#         V    = u + (t - r) * dudt.detach()        # stopgrad(dudt)
#         loss = (V - v_g).pow(2).sum(dim=1).mean()
#         return loss
