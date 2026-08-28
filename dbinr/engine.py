from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def _stack(params, B):
    return {k: v.detach().clone().unsqueeze(0).expand(B, *v.shape).contiguous()
            for k, v in params.items()}


@torch.no_grad()
def _adam(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, keys=None):
    for k in (p if keys is None else keys):
        m[k].mul_(b1).add_(g[k], alpha=1 - b1)
        v[k].mul_(b2).addcmul_(g[k], g[k], value=1 - b2)
        mh = m[k] / (1 - b1 ** t)
        vh = (v[k] / (1 - b2 ** t)).clamp_(min=1e-12)
        p[k].addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)


def fit_inr_batch(inr, init_params, coords, targets, steps, lr=1e-4,
                  checkpoints=(), feature_layer=-2, return_recon=True):
    """Fit inr from init_params to B images at once with a functional Adam loop.

    coords : (N,d) shared grid or (B,N,d);  targets: (B,N,1)
    returns {step: {'features','recon','mse'}}
    """
    B = targets.shape[0]
    if coords.dim() == 2:
        coords = coords.unsqueeze(0).expand(B, *coords.shape)
    p = _stack(init_params, B)
    for v in p.values():
        v.requires_grad_(False)
    m = {k: torch.zeros_like(v) for k, v in p.items()}
    v_ = {k: torch.zeros_like(x) for k, x in p.items()}

    def loss_one(par, c, y):
        return ((functional_call(inr, par, (c,)) - y) ** 2).mean()

    gfn = vmap(grad(loss_one))
    ckpts = sorted(set(int(c) for c in checkpoints) | {int(steps)})
    out = {}
    for t in range(1, steps + 1):
        g = gfn(p, coords, targets)
        _adam(p, g, m, v_, t, lr)
        if t in ckpts:
            with torch.no_grad():
                rec, feats = vmap(lambda par, c: _functional_features(inr, par, c))(p, coords)
            out[t] = {"features": feats, "mse": ((rec - targets) ** 2).mean(dim=(1, 2))}
            if return_recon:
                out[t]["recon"] = rec
    out["params"] = p
    return out


def _functional_features(inr, params, coords, feature_layer=-2):
    x, feats = coords, []
    for i, layer in enumerate(inr.layers):
        sub = {k.split(".", 2)[2]: v for k, v in params.items()
               if k.startswith(f"layers.{i}.")}
        x = functional_call(layer, sub, (x,))
        feats.append(x)
    return x, feats[feature_layer]


def fit_inr_single(inr, init_params, coords, target, steps, lr=1e-4,
                   checkpoints=(), feature_layer=-2):
    import copy
    net = copy.deepcopy(inr)
    net.load_state_dict({k: v.detach().clone() for k, v in init_params.items()},
                        strict=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ckpts = sorted(set(int(c) for c in checkpoints) | {int(steps)})
    out = {}
    for t in range(1, steps + 1):
        opt.zero_grad()
        loss = ((net(coords) - target) ** 2).mean()
        loss.backward()
        opt.step()
        if t in ckpts:
            with torch.no_grad():
                rec, feats = net.forward_w_features(coords)
            out[t] = {"features": feats[feature_layer], "recon": rec,
                      "mse": ((rec - target) ** 2).mean(),
                      # the fitted network itself, so callers can re-query it at
                      # coordinates other than the ones it was fitted on
                      "net": net}
    return out


def meta_train(meta, train_items, coords, epochs, log_every=50, val_fn=None, seed=0,
               resample_fn=None):
    """train_items: list of dicts with 'img' (N,1) and 'seg_int' (N,) already on GPU.

    `resample_fn(epoch) -> new train_items` is called at the START of every outer epoch.
    Augmentation runs through this hook so each image gets a FRESH random transform each
    epoch; pre-materialising a perturbed split once would instead give a permanently
    misaligned training set, which is a different (and much harsher) experiment.
    """
    g = torch.Generator().manual_seed(seed)
    hist, best = [], {"score": -1e9, "params": None}
    step = 0
    for ep in range(epochs):
        if resample_fn is not None:
            train_items = resample_fn(ep)
        order = torch.randperm(len(train_items), generator=g).tolist()
        for j in order:
            it = train_items[j]
            pack = {"gt": it["img"][None], "seg_int": it["seg_int"][None]}
            loss, info = meta.outer_step(coords[None], pack)
            step += 1
            if val_fn and step % log_every == 0:
                score, extra = val_fn(meta)
                hist.append({"step": step, "loss": loss, **info, **extra})
                if score > best["score"]:
                    best = {"score": score, "params": meta.get(), "step": step}
    if best["params"] is None:
        best = {"score": float("nan"), "params": meta.get(), "step": step}
    return best, hist


def train_head(head, feats_tr, seg_tr, feats_va, seg_va, epochs=4000, lr=1e-4,
               gamma=2.0, eval_every=50, batch=None, verbose=False):
    """feats_*: (M,N,h) cached penultimate features; seg_*: (M,N) integer labels."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    def focal(logits, tgt):
        logp = F.log_softmax(logits, -1)
        lt = torch.gather(logp, -1, tgt.unsqueeze(-1)).squeeze(-1)
        return (-((1 - lt.exp()) ** gamma) * lt).mean()

    M = feats_tr.shape[0]
    bs = batch or M
    dev = next(head.parameters()).device
    def _get(t, idx):
        return t[idx].to(dev, non_blocking=True).float() if t.device.type == "cpu" else t[idx].float()
    best = {"loss": 1e9, "state": None, "epoch": -1}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(M, generator=gen)
        tot = 0.0
        for s0 in range(0, M, bs):
            idx = perm[s0:s0 + bs]
            opt.zero_grad()
            l = focal(head(_get(feats_tr, idx)), seg_tr[idx].to(dev))
            l.backward(); opt.step(); tot += float(l.detach()) * len(idx)
        loss = torch.tensor(tot / M)
        if ep % eval_every == 0:
            head.eval()
            with torch.no_grad():
                vl = float(np.mean([
                    float(focal(head(_get(feats_va, slice(i, i + 16))), seg_va[i:i + 16].to(dev)))
                    for i in range(0, feats_va.shape[0], 16)]))
            sched.step(vl)
            if vl < best["loss"]:
                best = {"loss": vl, "state": {k: v.clone() for k, v in head.state_dict().items()},
                        "epoch": ep}
            if verbose and ep % (eval_every * 10) == 0:
                print(f"    head ep {ep:5d} train {float(loss):.5f} val {vl:.5f} best {best['loss']:.5f}")
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    return best
