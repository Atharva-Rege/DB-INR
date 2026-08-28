import numpy as np
import torch
import torch.nn as nn

from .layers import DualBandSineLayer


class INR(nn.Module):
    """L-layer coordinate network built from DB-INR layers."""

    def __init__(self, in_features=2, hidden_features=377, hidden_layers=3, out_features=1,
                 omega_0=20.0, **layer_kw):
        super().__init__()
        layers = [DualBandSineLayer(in_features, hidden_features, is_first=True,
                                    omega_0=omega_0, **layer_kw)]
        for _ in range(hidden_layers):
            layers.append(DualBandSineLayer(hidden_features, hidden_features, is_first=False,
                                            omega_0=omega_0, **layer_kw))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            c = np.sqrt(6.0 / hidden_features) / max(omega_0, 1e-12)
            final.weight.uniform_(-c, c)
        layers.append(final)
        self.layers = nn.ModuleList(layers)

    def forward(self, coords):
        x = coords
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_w_features(self, coords):
        x, feats = coords, []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return x, feats


class SegmentationHead(nn.Module):
    """Per-pixel read-out from the penultimate coordinate features."""

    def __init__(self, in_features, hidden=(512,), num_classes=25):
        super().__init__()
        dims = [in_features] + list(hidden)
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        blocks.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class SegINR(nn.Module):
    def __init__(self, inr_cfg, head_hidden=(512,), num_classes=25):
        super().__init__()
        self.inr = INR(**inr_cfg)
        self.segmentation_head = SegmentationHead(inr_cfg["hidden_features"],
                                                  hidden=head_hidden,
                                                  num_classes=num_classes)

    def forward(self, coords):
        rec, feats = self.inr.forward_w_features(coords)
        return rec, self.segmentation_head(feats[-2])


def count_params(m):
    return sum(p.numel() for p in m.parameters())
