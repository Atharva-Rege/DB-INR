import argparse
import json

import numpy as np
import torch

from dbinr.data import get_coords
from dbinr.engine import fit_inr_single
from dbinr.metrics import evaluate_slice
from dbinr.models import SegINR
from train import build_dataset


def load(ckpt, dev):
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    a = d["args"]
    # the bands are stored explicitly, since not every reported configuration places them
    # symmetrically about omega: the volumetric models use r_lo=0.5, r_hi=0.5*rho
    if a.get("r_lo") is not None and a.get("r_hi") is not None:
        r_lo, r_hi = a["r_lo"], a["r_hi"]
    else:
        rho = a["rho"]
        r_lo, r_hi = 1.0 / np.sqrt(rho), np.sqrt(rho)
    dim = 3 if a["dataset"] == "oasis3d" else 2
    cfg = dict(in_features=dim, hidden_features=a["hidden"], hidden_layers=a["layers"],
               out_features=1, omega_0=a["omega"],
               low_freq_ratio=float(r_lo), high_freq_ratio=float(r_hi),
               fixed_gate=a["fixed_gate"], norm_gate=a["norm_gate"],
               residual=not a["no_residual"], scale_residual=not a["no_scale_residual"],
               share_linear=a["share_linear"])
    model = SegINR(cfg, head_hidden=(a["head_hidden"],),
                   num_classes=a["classes"] + 1).to(dev).float()
    model.segmentation_head.load_state_dict(d["head"])
    model.eval()
    return model, {k: v.to(dev) for k, v in d["init_inr"].items()}, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, init, a = load(args.ckpt, dev)
    dim = 3 if a["dataset"] == "oasis3d" else 2
    coords = get_coords((a["res"],) * dim, dev)
    ds = build_dataset(a["dataset"], args.split, a["classes"], a["res"],
                       a["intensity_norm"], n=args.n or None)
    C = a["classes"] + 1

    scores = []
    for i in range(len(ds)):
        it = ds[i]
        img = it["img"].to(dev).float()
        out = fit_inr_single(model.inr, init, coords, img, a["Tf"], a["inner_lr"])[a["Tf"]]
        with torch.no_grad():
            pred = model.segmentation_head(out["features"]).argmax(-1)
        scores.append(evaluate_slice(pred, it["seg_int"].to(dev).long(), C))
    keys = scores[0].keys()
    summary = {k: float(np.nanmean([s[k] for s in scores])) for k in keys}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
