import torch
import torch.nn as nn


class PixelMeanFlowGuidanceLoss:
    """
    Stable Pixel MeanFlow Guidance Loss (Algorithm 2).

    Assumes the network signature is:

        net(z, r, t, w, c)

    returning x-predictions.

    Key stabilizations:
    - explicit r,t parameterization
    - tangent clipping before JVP
    - dudt clipping
    - global mean loss
    - randomized self-prediction slice
    """

    def __init__(
        self,
        num_classes: int,
        noise_dist: str = "uniform",
        data_proportion: float = 0.25,
        class_dropout_prob: float = 0.1,
        t_min: float = 0.15,
        w_max: float = 7.0,
        tangent_clip: float = 5.0,
        dudt_clip: float = 50.0,
    ):
        self.num_classes = num_classes
        self.uncond_class = num_classes

        self.noise_dist = noise_dist
        self.data_proportion = data_proportion
        self.class_dropout_prob = class_dropout_prob

        self.t_min = t_min
        self.w_max = w_max

        self.tangent_clip = tangent_clip
        self.dudt_clip = dudt_clip

    def _sample_time(self, shape, device):

        if self.noise_dist == "uniform":
            return self.t_min + (
                1.0 - self.t_min
            ) * torch.rand(shape, device=device)

        elif self.noise_dist == "logit_normal":
            rnd = torch.randn(shape, device=device)
            return torch.sigmoid(
                rnd * 1.0 - 0.4
            ).clamp(self.t_min, 1.0)

        raise ValueError(self.noise_dist)

    def _sample_t_r_w(self, B, device):

        shape = (B, 1)

        s1 = self._sample_time(shape, device)
        s2 = self._sample_time(shape, device)

        t = torch.maximum(s1, s2)
        r = torch.minimum(s1, s2)

        # randomized self-prediction slice
        data_mask = (
            torch.rand((B, 1), device=device)
            < self.data_proportion
        )

        r = torch.where(data_mask, t, r)

        # uniform CFG scale
        w = 1.0 + (
            self.w_max - 1.0
        ) * torch.rand(shape, device=device)

        return t, r, w

    def __call__(
        self,
        net: nn.Module,
        x: torch.Tensor,
        classes: torch.Tensor,
    ) -> torch.Tensor:

        B = x.shape[0]
        device = x.device

        # ---------------------------------------------------
        # Sample dynamics
        # ---------------------------------------------------

        t, r, w = self._sample_t_r_w(B, device)

        e = torch.randn_like(x)

        z_t = (1.0 - t) * x + t * e

        v_true = e - x

        # ---------------------------------------------------
        # classifier-free dropout
        # ---------------------------------------------------

        drop_mask = (
            torch.rand(B, device=device)
            < self.class_dropout_prob
        )

        null_tokens = torch.full_like(
            classes,
            self.uncond_class
        )

        c_dropped = torch.where(
            drop_mask,
            null_tokens,
            classes
        )

        # ---------------------------------------------------
        # average velocity field
        # ---------------------------------------------------

        def u_fn(
            z_input,
            r_input,
            t_input,
            w_input,
            c_input,
        ):

            x_pred = net(
                z_input,
                r_input,
                t_input,
                w_input,
                c_input,
            )

            return (
                z_input - x_pred
            ) / t_input

        # ---------------------------------------------------
        # CFG target
        # ---------------------------------------------------

        with torch.no_grad():

            # conditional instantaneous velocity
            v_c = u_fn(
                z_t,
                t,
                t,
                w,
                c_dropped,
            )

            # unconditional instantaneous velocity
            v_u = u_fn(
                z_t,
                t,
                t,
                w,
                null_tokens,
            )

            v_g = (
                v_true
                + (1.0 - 1.0 / w) * (v_c - v_u)
            )

        # ---------------------------------------------------
        # JVP wrapper
        # ---------------------------------------------------

        def u_fn_jvp(
            z_arg,
            r_arg,
            t_arg,
            w_arg,
        ):
            return u_fn(
                z_arg,
                r_arg,
                t_arg,
                w_arg,
                c_dropped,
            )

        # ---------------------------------------------------
        # tangent computation
        # ---------------------------------------------------

        with torch.no_grad():

            v_c_inst = u_fn_jvp(
                z_t,
                t,
                t,
                w,
            )

            # tangent clipping
            tangent_norm = v_c_inst.norm(
                dim=-1,
                keepdim=True,
            ).clamp(min=1e-8)

            clip_scale = torch.clamp(
                self.tangent_clip / tangent_norm,
                max=1.0,
            )

            v_c_inst = v_c_inst * clip_scale

        # ---------------------------------------------------
        # JVP
        # ---------------------------------------------------

        primals = (
            z_t,
            r,
            t,
            w,
        )

        tangents = (
            v_c_inst,
            torch.zeros_like(r),
            torch.ones_like(t),
            torch.zeros_like(w),
        )

        u, dudt = torch.func.jvp(
            u_fn_jvp,
            primals,
            tangents,
        )

        # derivative stabilization
        dudt = torch.nan_to_num(
            dudt,
            nan=0.0,
            posinf=self.dudt_clip,
            neginf=-self.dudt_clip,
        )

        dudt = torch.clamp(
            dudt,
            -self.dudt_clip,
            self.dudt_clip,
        )

        # ---------------------------------------------------
        # compound velocity
        # ---------------------------------------------------

        V = u + (t - r) * dudt.detach()

        # ---------------------------------------------------
        # stable MSE
        # ---------------------------------------------------

        loss = (V - v_g.detach()).pow(2).mean()

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
