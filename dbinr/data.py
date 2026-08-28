import json
import os

import cv2
import nibabel as nib
import numpy as np
import torch


def get_coords(shape, device="cpu", dtype=torch.float32):
    """Normalised coordinate grid in [-1,1]^d."""
    axes = [torch.linspace(-1, 1, s, dtype=dtype) for s in shape]
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grid], dim=-1).to(device)


def pad_resize(image, resolution, mode="nearest"):
    """Square-pad then resize."""
    h, w = image.shape[:2]
    if h > w:
        l = (h - w) // 2
        image = np.pad(image, ((0, 0), (l, h - w - l)), mode="constant")
    elif w > h:
        t = (w - h) // 2
        image = np.pad(image, ((t, w - h - t), (0, 0)), mode="constant")
    h, w = image.shape[:2]
    if (h, w) != tuple(resolution):
        image = cv2.resize(image, tuple(resolution),
                           interpolation=cv2.INTER_NEAREST if mode == "nearest" else cv2.INTER_LINEAR)
    return image


def onehot(labels, num_classes):
    oh = np.zeros((*labels.shape, num_classes), dtype=np.float32)
    for c in range(num_classes):
        oh[..., c] = (labels == c)
    return oh


class OASIS2D(torch.utils.data.Dataset):
    """One aligned coronal slice per subject."""

    def __init__(self, json_file, mode, num_classes=4, resolution=192,
                 intensity_norm="per_image", augment=False, n_samples=None, root=None):
        assert num_classes in (4, 24)
        with open(json_file) as f:
            self.data = json.load(f)[mode]
        if n_samples:
            self.data = self.data[:n_samples]
        self.num_classes = num_classes
        self.n_out = num_classes + 1
        self.res = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
        self.intensity_norm = intensity_norm
        self.augment = augment
        self.root = root

    def __len__(self):
        return len(self.data)

    def _load(self, p):
        if self.root:
            p = f"{self.root}/{p}"
        a = nib.load(p).get_fdata()
        return a.squeeze(-1) if a.shape[-1] == 1 else a

    def __getitem__(self, i):
        d = self.data[i]
        img = np.clip(self._load(d["img"]), 0, None)
        seg = self._load(d[f"seg{self.num_classes}"])

        img = pad_resize(img, self.res, "bilinear")
        seg = pad_resize(seg, self.res, "nearest")

        if self.intensity_norm == "per_image":
            m = img.max()
            if m > 0:
                img = img / m
        if self.augment:
            if np.random.rand() < 0.5:
                img, seg = np.flip(img, 0), np.flip(seg, 0)
            k = np.random.randint(0, 4)
            img, seg = np.rot90(img, k), np.rot90(seg, k)

        seg_int = np.ascontiguousarray(seg).astype(np.int64)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)).float().reshape(-1, 1),
            "seg_onehot": torch.from_numpy(onehot(seg_int, self.n_out)).float().reshape(-1, self.n_out),
            "seg_int": torch.from_numpy(seg_int).long().reshape(-1),
            "subject": d["img"].split("/")[-2],
            "index": i,
        }


# --------------------------------------------------------------------------- #
#  Geometric perturbation, used by the spatial-alignment probe.
#  Rotation/translation are applied to the IMAGE AND MASK TOGETHER, so a model
#  that segments from image content is unaffected while a model that has memorised
#  a coordinate atlas degrades.
# --------------------------------------------------------------------------- #


class ACDC2D(torch.utils.data.Dataset):
    """Pre-built 2D ACDC slices. Same interface as OASIS2D."""

    def __init__(self, json_file, mode, num_classes=3, resolution=192,
                 intensity_norm="per_image", augment=False, n_samples=None, root=None,
                 max_angle=0.0, max_shift=0, seed=0, elastic=0.0, aniso=0.0):
        with open(json_file) as f:
            self.data = json.load(f)[mode]
        if n_samples:
            self.data = self.data[:n_samples]
        self.num_classes, self.n_out = num_classes, num_classes + 1
        self.res = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
        self.root = root
        self.max_angle, self.max_shift, self.seed = max_angle, max_shift, seed
        self.elastic, self.aniso = elastic, aniso

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        z = np.load(f"{self.root}/{d['npz']}" if self.root else d["npz"])
        img, seg = z["img"].astype(np.float32), z["seg"].astype(np.int64)
        if self.max_angle or self.max_shift or self.elastic or self.aniso:
            rng = np.random if self.seed is None else np.random.RandomState(self.seed * 100003 + i)
            ang = rng.uniform(-self.max_angle, self.max_angle) if self.max_angle else 0.0
            tx = rng.randint(-self.max_shift, self.max_shift + 1) if self.max_shift else 0
            ty = rng.randint(-self.max_shift, self.max_shift + 1) if self.max_shift else 0
            img, seg = rigid_perturb(img, seg, ang, tx, ty)
            if self.aniso:
                img, seg = affine_perturb(img, seg, 1.0 + rng.uniform(-self.aniso, self.aniso),
                                          rng.uniform(-self.aniso, self.aniso))
            if self.elastic:
                img, seg = elastic_perturb(img, seg, self.elastic, 12.0, rng)
        else:
            ang = tx = ty = 0
        seg = np.ascontiguousarray(seg).astype(np.int64)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)).float().reshape(-1, 1),
            "seg_onehot": torch.from_numpy(onehot(seg, self.n_out)).float().reshape(-1, self.n_out),
            "seg_int": torch.from_numpy(seg).long().reshape(-1),
            "subject": d["subject"], "index": i,
            "angle": float(ang), "tx": int(tx), "ty": int(ty),
        }


# --------------------------------------------------------------------------- #
#  MedSegBench
#
#  The datasets are split into two groups BEFORE any model is run, by the
#  structure they contain, because the gated layer is a high-frequency band
#  selector and should help on one group and not the other:
#     THIN  vessels and other elongated structures, dominated by high spatial
#              frequency and long thin boundaries.
#     BLOB  lesions and organs, compact regions dominated by low frequency.
#  Both groups are reported, so the split is a prediction rather than a selection.
# --------------------------------------------------------------------------- #
MSB_THIN = ["Drive", "ChaseDB1", "Dca1", "Chuac"]
MSB_BLOB = ["Busi", "Isic2016", "Kvasir", "Covid19Radio"]


class MSB2D(torch.utils.data.Dataset):
    """One MedSegBench dataset as binary 2D segmentation, grayscale, in [0,1]."""

    def __init__(self, name, mode, num_classes=1, resolution=128, root=None,
                 n_samples=None, max_angle=0.0, max_shift=0, seed=0, **kw):
        import medsegbench as msb
        cls = getattr(msb, f"{name}MSBench")
        split = {"train": "train", "val": "val", "test": "test"}[mode]
        self.ds = cls(split=split, download=True, size=int(resolution))
        self.n = len(self.ds) if n_samples is None else min(n_samples, len(self.ds))
        self.res = (resolution, resolution)
        self.num_classes, self.n_out = 1, 2
        self.max_angle, self.max_shift, self.seed = max_angle, max_shift, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, y = self.ds[i]
        img = np.asarray(x, dtype=np.float32)
        if img.ndim == 3:
            img = img[..., 1] if img.shape[-1] == 3 else img.mean(-1)   # green channel for fundus
        img = img / max(img.max(), 1e-6)
        seg = (np.asarray(y) > 127).astype(np.int64)
        if self.max_angle or self.max_shift:
            rng = np.random if self.seed is None else np.random.RandomState(self.seed * 100003 + i)
            ang = rng.uniform(-self.max_angle, self.max_angle) if self.max_angle else 0.0
            tx = rng.randint(-self.max_shift, self.max_shift + 1) if self.max_shift else 0
            ty = rng.randint(-self.max_shift, self.max_shift + 1) if self.max_shift else 0
            img, seg = rigid_perturb(img, seg, ang, tx, ty)
        else:
            ang = tx = ty = 0
        seg = np.ascontiguousarray(seg).astype(np.int64)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)).float().reshape(-1, 1),
            "seg_onehot": torch.from_numpy(onehot(seg, 2)).float().reshape(-1, 2),
            "seg_int": torch.from_numpy(seg).long().reshape(-1),
            "subject": f"{i}", "index": i,
            "angle": float(ang), "tx": int(tx), "ty": int(ty),
        }


# --------------------------------------------------------------------------- #
#  OASIS as 3D volumes. The coordinate network is dimension-agnostic, so the only
#  thing that changes is the loader and the shape passed to get_coords.
# --------------------------------------------------------------------------- #


class OASIS3D(torch.utils.data.Dataset):
    """neurite-OASIS volumes, affinely aligned, resampled to a cube of side `resolution`."""

    ROOT = os.environ.get("OASIS3D_ROOT", "data/oasis3d")

    def __init__(self, mode, num_classes=4, resolution=64, intensity_norm="per_image",
                 n_samples=None, root=None, seed=0, **kw):
        import glob as _g
        root = root or self.ROOT
        subs = sorted(d for d in _g.glob(f"{root}/OASIS_OAS1_*/")
                      if os.path.exists(d + "aligned_norm.nii.gz")
                      and os.path.exists(d + f"aligned_seg{num_classes}.nii.gz"))
        # deterministic subject-level split, disjoint by construction
        n = len(subs)
        cuts = {"train": slice(0, int(0.60 * n)),
                "val":   slice(int(0.60 * n), int(0.75 * n)),
                "test":  slice(int(0.75 * n), n)}
        self.subs = subs[cuts[mode]]
        if n_samples:
            self.subs = self.subs[:n_samples]
        self.res = int(resolution)
        self.num_classes = num_classes
        self.n_out = num_classes + 1
        self.intensity_norm = intensity_norm

    def __len__(self):
        return len(self.subs)

    def _resize(self, v, mode):
        t = torch.from_numpy(np.ascontiguousarray(v)).float()[None, None]
        t = F.interpolate(t, size=(self.res,) * 3, mode=mode,
                          align_corners=False if mode == "trilinear" else None)
        return t[0, 0].numpy()

    def __getitem__(self, i):
        import nibabel as nib
        d = self.subs[i]
        img = nib.load(d + "aligned_norm.nii.gz").get_fdata()
        seg = nib.load(d + f"aligned_seg{self.num_classes}.nii.gz").get_fdata()
        img = self._resize(np.clip(img, 0, None), "trilinear")
        seg = self._resize(seg, "nearest")
        if self.intensity_norm == "per_image":
            m = img.max()
            if m > 0:
                img = img / m
        seg_int = np.rint(seg).astype(np.int64)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)).float().reshape(-1, 1),
            "seg_onehot": torch.from_numpy(onehot(seg_int, self.n_out)).float().reshape(-1, self.n_out),
            "seg_int": torch.from_numpy(seg_int).long().reshape(-1),
            "subject": d.rstrip("/").split("/")[-1],
            "index": i,
        }
