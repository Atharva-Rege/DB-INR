# DB-INR: Dual-Band Implicit Neural Representations for Medical Image Segmentation

Reference implementation of the DB-INR layer and the meta-learned implicit segmentation
pipeline it is trained in.

## The layer

Each layer carries a low- and a high-frequency sinusoidal branch with its own projection,
combines them with a fixed equal mixture, and adds a residual scaled by `1/sqrt(2)`:

```
y = 0.5 * sin(w * r_lo * W_lo x) + 0.5 * sin(w * r_hi * W_hi x)
h = (y + R(x)) / sqrt(2)
```

with `r_lo = 1/sqrt(rho)`, `r_hi = sqrt(rho)`, `rho = phi` by default, `W_lo` and `W_hi`
drawn from `U(+-sqrt(6/n)/(w*r))` at each branch's own effective frequency, and `R` the
identity where the width is unchanged and otherwise `U(+-sqrt(3/n))`.

`dbinr/layers.py` also exposes the variants used in the ablation: `fixed_gate=None` restores
a learned gate, `norm_gate=True` adds the variance-preserving denominator, `share_linear=True`
gives both branches one projection, and `scale_residual=False` removes the `1/sqrt(2)`.

## Install

```bash
pip install -r requirements.txt
```

## Data

Set the split files and roots before training:

```bash
export OASIS_SPLITS=/path/to/oasis_splits.json
export OASIS_ROOT=/path/to/oasis
export ACDC_SPLITS=/path/to/acdc_splits.json
export ACDC_ROOT=/path/to/acdc
export OASIS3D_ROOT=/path/to/oasis3d
```

MedSegBench datasets download on first use through the `medsegbench` package.

## Train

OASIS-2D with 25 labels at the 1.0M budget, which is the configuration reported in the paper:

```bash
python train.py --dataset oasis --classes 24 --res 192 \
    --hidden 377 --layers 3 --head-hidden 512 --omega 20 \
    --Tf 100 --outer-epochs 10 --head-epochs 1500 \
    --bs 3 --head-bs 8 --seed 0 --out runs/dbinr_oasis
```

Cardiac cine MRI:

```bash
python train.py --dataset acdc --classes 3 --omega 30 --out runs/dbinr_acdc
```

Volumetric:

```bash
python train.py --dataset oasis3d --classes 4 --res 64 --omega 30 \
    --bs 1 --head-epochs 800 --out runs/dbinr_oasis3d
```

MedSegBench:

```bash
python train.py --dataset msb:Dca1 --classes 1 --res 128 --omega 30 --out runs/dbinr_dca1
```

Training runs in two stages. Stage one meta-learns the coordinate network and head with
MAML under a reconstruction and a focal term. Stage two refits the network to every training
scan, caches the penultimate features and trains the read-out on them with early stopping.

## Evaluate

```bash
python evaluate.py --ckpt runs/dbinr_oasis.pt --split test
```

At test time the network is fitted to the intensities of an unseen scan for `Tf` steps with
the reconstruction term alone, and labels are read from the fitted features in one forward
pass with the head frozen. No annotation of the test scan is used.

## Pretrained weights

Checkpoints for the three benchmarks of Tab. 1:

| File | Dataset | Labels | Res | `omega` | Bands | Download |
|---|---|---|---|---|---|---|
| `dbinr_oasis2d.pt` | OASIS-2D, brain MRI | 25 | 192 | 20 | 0.786, 1.272 | [link](https://drive.google.com/file/d/1wwZjdaieBG2Bl7H-29l-uz_9Zm1Po1MN/view?usp=sharing) |
| `dbinr_acdc.pt` | ACDC, cardiac cine MRI | 3 | 192 | 30 | 0.786, 1.272 | [link](https://drive.google.com/file/d/1rdFk542_-Utr8Uo1nQnnrDvaEndNi9K1/view?usp=sharing) |
| `dbinr_oasis3d.pt` | OASIS-3D, brain volumes | 4 | 64 | 30 | 0.500, 0.809 | [link](https://drive.google.com/file/d/1rcAEzCC1ENXMnITrWKt9UpcjtMlRmzMV/view?usp=sharing) |

Pass the `.pt` file to `evaluate.py`:

```bash
python evaluate.py --ckpt dbinr_oasis2d.pt --split test
```