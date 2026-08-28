import argparse
import json
import os
import time

import numpy as np
import torch

from dbinr.data import OASIS2D, ACDC2D, MSB2D, OASIS3D, get_coords
from dbinr.engine import fit_inr_batch, meta_train, train_head
from dbinr.losses import CompositeLoss, FocalLoss, ReconLoss
from dbinr.meta import INRMetaLearner
from dbinr.metrics import evaluate_slice
from dbinr.models import SegINR, SegmentationHead, count_params

PHI = (1.0 + 5.0 ** 0.5) / 2.0


def build_dataset(name, mode, classes, res, norm, n=None):
    if name.startswith("msb:"):
        return MSB2D(name.split(":", 1)[1], mode, resolution=res, n_samples=n)
    if name == "oasis3d":
        return OASIS3D(mode, num_classes=classes, resolution=res,
                       intensity_norm=norm, n_samples=n)
    if name == "acdc":
        return ACDC2D(os.environ["ACDC_SPLITS"], mode, num_classes=classes,
                      resolution=res, n_samples=n, root=os.environ.get("ACDC_ROOT", "."))
    return OASIS2D(os.environ["OASIS_SPLITS"], mode, classes, res, norm,
                   n_samples=n, root=os.environ.get("OASIS_ROOT", "."))


def to_device(ds, dev, coords_n):
    items = []
    for i in range(len(ds)):
        it = ds[i]
        items.append({"img": it["img"].to(dev).float(),
                      "seg_int": it["seg_int"].to(dev).long()})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="oasis", help="oasis | oasis3d | acdc | msb:<Name>")
    ap.add_argument("--classes", type=int, default=24, choices=[1, 3, 4, 24])
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--intensity-norm", default="per_image", choices=["none", "per_image"])
    ap.add_argument("--hidden", type=int, default=377)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--omega", type=float, default=20.0)
    ap.add_argument("--rho", type=float, default=PHI, help="band ratio r_hi / r_lo")
    ap.add_argument("--fixed-gate", type=float, default=0.5)
    ap.add_argument("--norm-gate", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-scale-residual", action="store_true")
    ap.add_argument("--share-linear", action="store_true")
    ap.add_argument("--Ti", type=int, default=2)
    ap.add_argument("--Tf", type=int, default=100)
    ap.add_argument("--outer-epochs", type=int, default=10)
    ap.add_argument("--head-epochs", type=int, default=1500)
    ap.add_argument("--inner-lr", type=float, default=1e-4)
    ap.add_argument("--outer-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=3)
    ap.add_argument("--head-bs", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dbinr")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    C = args.classes + 1
    d = 3 if args.dataset == "oasis3d" else 2
    shape = (args.res,) * d
    coords = get_coords(shape, dev)

    r_lo, r_hi = 1.0 / np.sqrt(args.rho), np.sqrt(args.rho)
    layer_kw = dict(low_freq_ratio=float(r_lo), high_freq_ratio=float(r_hi),
                    fixed_gate=args.fixed_gate, norm_gate=args.norm_gate,
                    residual=not args.no_residual,
                    scale_residual=not args.no_scale_residual,
                    share_linear=args.share_linear)
    inr_cfg = dict(in_features=d, hidden_features=args.hidden, hidden_layers=args.layers,
                   out_features=1, omega_0=args.omega, **layer_kw)
    model = SegINR(inr_cfg, head_hidden=(args.head_hidden,), num_classes=C).to(dev).float()
    print(f"[DB-INR] params={count_params(model):,}  rho={args.rho:.4f}  "
          f"r_lo={r_lo:.4f} r_hi={r_hi:.4f} omega={args.omega}")

    tr = build_dataset(args.dataset, "train", args.classes, args.res, args.intensity_norm)
    va = build_dataset(args.dataset, "val", args.classes, args.res, args.intensity_norm)
    te = build_dataset(args.dataset, "test", args.classes, args.res, args.intensity_norm)
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}")
    tr_items = to_device(tr, dev, coords.shape[0])
    va_items = to_device(va, dev, coords.shape[0])

    loss_fn = CompositeLoss(rec=ReconLoss(), seg=FocalLoss(gamma=args.gamma))
    meta = INRMetaLearner(model, inner_steps=args.Ti, inner_lr=args.inner_lr,
                          outer_lr=args.outer_lr, loss_fn=loss_fn)

    def val_fn(m):
        p = m.get()
        feats, segs = [], []
        for i in range(0, len(va_items), args.bs):
            chunk = va_items[i:i + args.bs]
            tgt = torch.stack([c["img"] for c in chunk])
            o = fit_inr_batch(m.model.inr, {k[4:]: v for k, v in p.items() if k.startswith("inr.")},
                              coords, tgt, args.Tf)
            feats.append(o[args.Tf]["features"])
            segs.append(torch.stack([c["seg_int"] for c in chunk]))
        f, s = torch.cat(feats), torch.cat(segs)
        with torch.no_grad():
            pred = m.model.segmentation_head(f).argmax(-1)
        dice = float(np.mean([evaluate_slice(pred[i], s[i], C)["dice"] for i in range(len(s))]))
        return dice, {"val_dice": dice}

    t0 = time.time()
    best, hist = meta_train(meta, tr_items, coords, args.outer_epochs,
                            log_every=args.val_every, val_fn=val_fn, seed=args.seed)
    print(f"  [stage1] best val dice {best['score']:.4f} @ step {best['step']}  "
          f"({time.time() - t0:.0f}s)")
    meta.load(best["params"])

    init = {k[4:]: v for k, v in meta.get().items() if k.startswith("inr.")}
    def cache(items):
        F_, S_ = [], []
        for i in range(0, len(items), args.bs):
            chunk = items[i:i + args.bs]
            tgt = torch.stack([c["img"] for c in chunk])
            o = fit_inr_batch(model.inr, init, coords, tgt, args.Tf)
            F_.append(o[args.Tf]["features"].cpu())
            S_.append(torch.stack([c["seg_int"] for c in chunk]).cpu())
        return torch.cat(F_), torch.cat(S_)

    f_tr, s_tr = cache(tr_items)
    f_va, s_va = cache(va_items)
    head = SegmentationHead(args.hidden, hidden=(args.head_hidden,), num_classes=C).to(dev)
    train_head(head, f_tr, s_tr, f_va, s_va, epochs=args.head_epochs, lr=args.head_lr,
               gamma=args.gamma, batch=args.head_bs, verbose=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"init_inr": init, "head": head.state_dict(), "args": vars(args)},
               args.out + ".pt")
    json.dump({"args": vars(args), "val_dice": best["score"], "history": hist},
              open(args.out + ".json", "w"), indent=1)
    print("  saved", args.out + ".pt")


if __name__ == "__main__":
    main()
