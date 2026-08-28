"""Segmentation and reconstruction metrics.

Dice is reported two ways because the literature mixes them:
  - `dice_metaseg`  : torchmetrics generalized_dice_score(weight_type='simple',
                      include_background=True).
  - `dice_per_class`: the standard mean over classes PRESENT in the ground truth,
                      excluding background. This is the MICCAI-conventional number
                      and is strictly harder.
Boundary quality: HD95, ASSD and Normalised Surface Dice (NSD).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from torchmetrics.functional.segmentation import generalized_dice_score


# ------------------------------- overlap ----------------------------------- #
def dice_metaseg(pred_oh, gt_oh, num_classes):
    """pred_oh/gt_oh: (*spatial, C) one-hot tensors. Matches the released evaluation.
    Spatial rank is arbitrary, so the same call works for 2D slices and 3D volumes."""
    d = pred_oh.dim()                       # spatial dims + channel
    perm = (0, d) + tuple(range(1, d))      # (1, C, *spatial)
    p = pred_oh[None].permute(*perm)
    g = gt_oh[None].permute(*perm)
    return float(generalized_dice_score(p.int(), g.int(), num_classes=num_classes,
                                        include_background=True, weight_type="simple").mean())


def dice_per_class(pred_lbl, gt_lbl, num_classes, include_background=False, eps=1e-7):
    """Mean Dice over classes present in the ground truth. Returns (mean, per-class dict)."""
    out, lo = {}, 0 if include_background else 1
    for c in range(lo, num_classes):
        g = gt_lbl == c
        if not g.any():
            continue
        p = pred_lbl == c
        out[c] = float((2 * (p & g).sum()) / (p.sum() + g.sum() + eps))
    return (float(np.mean(list(out.values()))) if out else float("nan")), out


def iou_per_class(pred_lbl, gt_lbl, num_classes, include_background=False, eps=1e-7):
    out, lo = {}, 0 if include_background else 1
    for c in range(lo, num_classes):
        g = gt_lbl == c
        if not g.any():
            continue
        p = pred_lbl == c
        out[c] = float((p & g).sum() / ((p | g).sum() + eps))
    return (float(np.mean(list(out.values()))) if out else float("nan")), out


# ------------------------------- boundary ---------------------------------- #
def _surface(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, bool)
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def _surface_distances(p, g, spacing=1.0):
    """Symmetric surface distances (in `spacing` units) between two binary masks."""
    sp, sg = _surface(p), _surface(g)
    if sp.sum() == 0 or sg.sum() == 0:
        return None, None
    dt_g = distance_transform_edt(~sg, sampling=spacing)
    dt_p = distance_transform_edt(~sp, sampling=spacing)
    return dt_g[sp], dt_p[sg]


def hd95(pred_lbl, gt_lbl, num_classes, spacing=1.0, include_background=False):
    """95th-percentile Hausdorff distance, averaged over classes present in BOTH."""
    vals, lo = [], 0 if include_background else 1
    for c in range(lo, num_classes):
        g, p = gt_lbl == c, pred_lbl == c
        if not g.any():
            continue
        if not p.any():
            vals.append(float(np.linalg.norm(gt_lbl.shape)) * spacing)   # worst case
            continue
        d1, d2 = _surface_distances(p, g, spacing)
        if d1 is None:
            continue
        vals.append(float(np.percentile(np.concatenate([d1, d2]), 95)))
    return float(np.mean(vals)) if vals else float("nan")


def assd(pred_lbl, gt_lbl, num_classes, spacing=1.0, include_background=False):
    vals, lo = [], 0 if include_background else 1
    for c in range(lo, num_classes):
        g, p = gt_lbl == c, pred_lbl == c
        if not g.any() or not p.any():
            continue
        d1, d2 = _surface_distances(p, g, spacing)
        if d1 is None:
            continue
        vals.append(float((d1.sum() + d2.sum()) / (len(d1) + len(d2))))
    return float(np.mean(vals)) if vals else float("nan")


def nsd(pred_lbl, gt_lbl, num_classes, tau=1.0, spacing=1.0, include_background=False):
    """Normalised Surface Dice at tolerance tau the boundary metric recommended
    for medical segmentation (Nikolov et al.). Higher is better, in [0,1]."""
    vals, lo = [], 0 if include_background else 1
    for c in range(lo, num_classes):
        g, p = gt_lbl == c, pred_lbl == c
        if not g.any():
            continue
        if not p.any():
            vals.append(0.0)
            continue
        d1, d2 = _surface_distances(p, g, spacing)
        if d1 is None:
            continue
        vals.append(float(((d1 <= tau).sum() + (d2 <= tau).sum()) / (len(d1) + len(d2))))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------- reconstruction ------------------------------- #
def psnr(pred, gt, data_range=None):
    """PSNR with the data range taken from the ground truth unless given."""
    pred = np.asarray(pred, np.float64).ravel()
    gt = np.asarray(gt, np.float64).ravel()
    dr = float(gt.max() - gt.min()) if data_range is None else float(data_range)
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(dr ** 2 / mse))


def evaluate_slice(logits, gt_onehot, gt_int, num_classes, hw, recon=None, img=None, spacing=1.0):
    """One-stop evaluation of a single 2D slice."""
    pred = logits.softmax(-1).argmax(-1).reshape(hw).cpu().numpy().astype(np.int64)
    gt = gt_int.reshape(hw).cpu().numpy().astype(np.int64)
    pred_oh = torch.from_numpy(np.eye(num_classes, dtype=np.int64)[pred])
    gt_oh = gt_onehot.detach().cpu().reshape(*hw, num_classes).long()
    res = {
        "dice_metaseg": dice_metaseg(pred_oh, gt_oh, num_classes),
        "dice": dice_per_class(pred, gt, num_classes)[0],
        "dice_fg_bg": dice_per_class(pred, gt, num_classes, include_background=True)[0],
        "iou": iou_per_class(pred, gt, num_classes)[0],
        "hd95": hd95(pred, gt, num_classes, spacing),
        "assd": assd(pred, gt, num_classes, spacing),
        "nsd1": nsd(pred, gt, num_classes, 1.0, spacing),
        "nsd2": nsd(pred, gt, num_classes, 2.0, spacing),
    }
    if recon is not None and img is not None:
        r = recon.reshape(-1).detach().cpu().numpy()
        i = img.reshape(-1).detach().cpu().numpy()
        res["psnr"] = psnr(r, i)
        res["psnr_dr1"] = psnr(r, i, data_range=1.0)
    return res
