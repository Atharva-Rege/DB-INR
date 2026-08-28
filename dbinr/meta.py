"""MAML meta-learner with a differentiable Adam inner loop.

The inner loop runs
`inner_steps` differentiable Adam updates on ALL parameters (INR + head) from the
current meta-initialisation, and the outer loss is evaluated at the adapted
parameters with create_graph=True (second-order MAML).
"""
from collections import OrderedDict
from copy import deepcopy

import torch


class INRMetaLearner:
    def __init__(self, model, inner_steps=2, inner_lr=1e-4, outer_lr=1e-4,
                 loss_fn=None, first_order=False):
        self.model = model
        self.inner_steps = inner_steps
        self.inner_lr, self.outer_lr = inner_lr, outer_lr
        self.loss_fn = loss_fn
        self.first_order = first_order
        self.params = OrderedDict(
            (k, v.clone().detach().requires_grad_()) for k, v in model.named_parameters())
        self.opt = torch.optim.Adam(self.params.values(), lr=outer_lr)

    # ---------------------------------------------------------------- utils --
    def get(self, prefix=None, copy=True):
        return OrderedDict((k, v.clone().detach() if copy else v)
                           for k, v in self.params.items()
                           if prefix is None or k.startswith(prefix))

    def load(self, params):
        for k, v in params.items():
            if k in self.params:
                self.params[k].data.copy_(v.data if hasattr(v, "data") else v)

    @staticmethod
    def _adam_step(params, grads, step, m, v, lr, b1=0.9, b2=0.999, eps=1e-8):
        new = OrderedDict()
        for (k, p), g in zip(params.items(), grads):
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g ** 2
            mh = m[k] / (1 - b1 ** step)
            vh = torch.clamp(v[k] / (1 - b2 ** step), min=1e-12)
            new[k] = p - lr * mh / (vh.sqrt() + eps)
        return new, m, v

    # ------------------------------------------------------------ meta loop --
    def inner_loop(self, coords, pack):
        params = OrderedDict((k, v.clone()) for k, v in self.params.items())
        m = {k: torch.zeros_like(v) for k, v in params.items()}
        v = {k: torch.zeros_like(x) for k, x in params.items()}
        for i in range(self.inner_steps):
            out = torch.func.functional_call(self.model, params, coords)
            pack["output"] = out
            loss, _ = self.loss_fn(pack)
            if torch.isnan(loss):
                break
            grads = torch.autograd.grad(loss, params.values(),
                                        create_graph=not self.first_order)
            params, m, v = self._adam_step(params, grads, i + 1, m, v, self.inner_lr)
        return params

    def outer_step(self, coords, pack):
        self.opt.zero_grad()
        adapted = self.inner_loop(coords, pack)
        pack["output"] = torch.func.functional_call(self.model, adapted, coords)
        loss, info = self.loss_fn(pack)
        loss.backward()
        self.opt.step()
        return float(loss.detach()), info
