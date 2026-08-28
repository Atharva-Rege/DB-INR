import math

import numpy as np
import torch
import torch.nn as nn

PHI = (1.0 + 5.0 ** 0.5) / 2.0


class DualBandSineLayer(nn.Module):
    """Dual-band sinusoidal residual layer.

    y = g * sin(w * r_lo * W_lo x) + (1 - g) * sin(w * r_hi * W_hi x)
    h = (y + R(x)) / sqrt(2)

    The released configuration fixes g = 1/2, omits the normalising denominator and
    places the bands symmetrically about the base frequency.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0,
                 low_freq_ratio=1.0 / math.sqrt(PHI), high_freq_ratio=None,
                 share_linear=False, norm_gate=False, residual=True, scale_residual=True,
                 gate_from="x", fixed_gate=0.5, learnable_omega=False):
        super().__init__()
        self.is_first, self.in_features, self.out_features = is_first, in_features, out_features
        self.share_linear, self.norm_gate = share_linear, norm_gate
        self.residual, self.scale_residual = residual, scale_residual
        self.gate_from, self.fixed_gate = gate_from, fixed_gate

        if high_freq_ratio is None:
            high_freq_ratio = math.sqrt(PHI)
        self.r_lo, self.r_hi = float(low_freq_ratio), float(high_freq_ratio)

        if learnable_omega:
            self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
        else:
            self.register_buffer("omega_0", torch.tensor(float(omega_0)))

        self.lin_lo = nn.Linear(in_features, out_features, bias=bias)
        self.lin_hi = self.lin_lo if share_linear else nn.Linear(in_features, out_features, bias=bias)

        if fixed_gate is None:
            gin = in_features if gate_from == "x" else out_features
            self.gate = nn.Linear(gin, out_features, bias=True)
        else:
            self.gate = None

        if residual:
            self.res_proj = (nn.Identity() if in_features == out_features
                             else nn.Linear(in_features, out_features, bias=False))
        else:
            self.res_proj = None
        self.init_weights()

    def init_weights(self):
        om = float(self.omega_0.detach().item() if isinstance(self.omega_0, nn.Parameter)
                   else self.omega_0.item())
        with torch.no_grad():
            for lin, r in ((self.lin_lo, self.r_lo), (self.lin_hi, self.r_hi)):
                # a shared map has no per-branch r to initialise for, so it falls back
                # to SIREN's convention at the base frequency
                r_init = 1.0 if self.share_linear else r
                if self.is_first:
                    b = 1.0 / self.in_features / r_init
                else:
                    b = np.sqrt(6.0 / self.in_features) / (om * r_init)
                lin.weight.uniform_(-b, b)
                if lin.bias is not None:
                    lin.bias.zero_()
                if self.share_linear:
                    break
            if self.gate is not None:
                self.gate.weight.zero_()
                self.gate.bias.zero_()
            if isinstance(self.res_proj, nn.Linear):
                # preserves Var = 1/2 through the projection
                b = math.sqrt(3.0 / self.in_features)
                self.res_proj.weight.uniform_(-b, b)

    def forward(self, x):
        om = torch.clamp(self.omega_0, min=1e-6)
        z_lo = self.lin_lo(x)
        z_hi = z_lo if self.share_linear else self.lin_hi(x)
        y_lo = torch.sin((om * self.r_lo) * z_lo)
        y_hi = torch.sin((om * self.r_hi) * z_hi)

        if self.fixed_gate is not None:
            g = torch.full_like(y_lo, float(self.fixed_gate))
        else:
            g = torch.sigmoid(self.gate(x if self.gate_from == "x" else z_lo))

        y = g * y_lo + (1.0 - g) * y_hi
        if self.norm_gate:
            y = y / torch.sqrt(g * g + (1.0 - g) * (1.0 - g) + 1e-8)
        if self.residual:
            y = y + self.res_proj(x)
            if self.scale_residual:
                y = y * (1.0 / math.sqrt(2.0))
        return y
