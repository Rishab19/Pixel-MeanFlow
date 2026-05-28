import math
import torch
import torch.nn as nn

# Implementation 1
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


class MeanFlowGuidanceMLP(nn.Module):

    def __init__(
        self,
        data_dim=2,
        num_classes=10,
        hidden=256,
        depth=3,
        emb_dim=64,
    ):
        super().__init__()

        self.data_dim = data_dim
        self.num_classes = num_classes

        self.r_emb = SinusoidalEmbedding(emb_dim)
        self.t_emb = SinusoidalEmbedding(emb_dim)
        self.w_emb = SinusoidalEmbedding(emb_dim)

        self.class_emb = nn.Embedding(
            num_classes + 1,
            emb_dim
        )

        in_dim = data_dim + 4 * emb_dim

        layers = [
            nn.Linear(in_dim, hidden),
            nn.SiLU()
        ]

        for _ in range(depth - 1):
            layers += [
                nn.Linear(hidden, hidden),
                nn.SiLU()
            ]

        layers += [
            nn.Linear(hidden, data_dim)
        ]

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z,
        r,
        t,
        w,
        c,
    ):

        B = z.shape[0]

        if c is None:
            c = torch.full(
                (B,),
                self.num_classes,
                dtype=torch.long,
                device=z.device
            )

        r_e = self.r_emb(r.view(B))
        t_e = self.t_emb(t.view(B))
        w_e = self.w_emb(w.view(B))

        c_e = self.class_emb(c)

        inp = torch.cat([
            z,
            r_e,
            t_e,
            w_e,
            c_e
        ], dim=-1)

        return self.net(inp)    
# Alternate implementation

# import math
# import torch
# import torch.nn as nn

# class SinusoidalEmbedding(nn.Module):
#     def __init__(self, dim: int = 64):
#         super().__init__()
#         assert dim % 2 == 0
#         self.dim = dim

#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         t = t.view(-1, 1)
#         half = self.dim // 2
#         freqs = torch.exp(
#             -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
#         )
#         args = t * freqs.unsqueeze(0)
#         return torch.cat([args.sin(), args.cos()], dim=-1)


# class MeanFlowGuidanceMLP(nn.Module):
#     """
#     x-prediction MLP for 2-D data supporting Classifier-Free Guidance (Algorithm 2).
#     Inputs : z [B, 2], t [B, 1], h [B, 1], w [B, 1], c [B]
#     Output : x_hat [B, 2]
#     """
#     def __init__(
#         self, 
#         num_classes: int,
#         data_dim: int = 2, 
#         hidden: int = 512, 
#         depth: int = 5, 
#         emb_dim: int = 128
#     ):
#         super().__init__()
#         # Continuous variable embeddings
#         self.t_emb = SinusoidalEmbedding(emb_dim)
#         self.h_emb = SinusoidalEmbedding(emb_dim)
#         self.w_emb = SinusoidalEmbedding(emb_dim)  # Embed CFG scale w similarly to time

#         # Categorical variable embedding
#         # We add +1 to num_classes to account for the unconditional/null token
#         self.c_emb = nn.Embedding(num_classes + 1, emb_dim)

#         # Total feature injection dimension = data + 4 * emb_dim
#         in_dim = data_dim + (4 * emb_dim)
        
#         layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
#         for _ in range(depth - 1):
#             layers += [nn.Linear(hidden, hidden), nn.SiLU()]
#         layers += [nn.Linear(hidden, data_dim)]
#         self.net = nn.Sequential(*layers)

#     def forward(
#         self, 
#         z: torch.Tensor, 
#         t: torch.Tensor, 
#         h: torch.Tensor, 
#         w: torch.Tensor, 
#         c: torch.Tensor
#     ) -> torch.Tensor:
#         B = z.shape[0]
        
#         # Extract embeddings and ensure consistent shape mapping [B, emb_dim]
#         t_e = self.t_emb(t.view(B))
#         h_e = self.h_emb(h.view(B))
#         w_e = self.w_emb(w.view(B))
#         c_e = self.c_emb(c.view(B))
        
#         # Concat inputs horizontally along the feature dimension
#         features = torch.cat([z, t_e, h_e, w_e, c_e], dim=-1)
        
#         return self.net(features)
