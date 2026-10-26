<h1 align="center">
<strong>DB-INR: Dual-Band Implicit Neural Representations for Medical Image Segmentation</strong>
</h1>

<div align="center">

<a href="https://git.io/typing-svg">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=147B82&center=true&width=620&lines=Two+sinusoidal+bands+per+layer.;A+fixed+equal+mixture+and+a+scaled+residual.;Segment+an+unseen+scan+from+its+intensities+alone."
alt="Typing SVG"
style="margin-bottom:-10px; display:block;" />
</a>

[![ISBI 2027](https://img.shields.io/badge/IEEE_ISBI_2027-Submission-8C1B13?style=for-the-badge)](#-citation)
[![Weights](https://img.shields.io/badge/Weights-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](#-pretrained-weights)
[![Task](https://img.shields.io/badge/Task-Implicit_Segmentation-37CCB6?style=for-the-badge)](#-method)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#️-installation)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FAtharva-Rege%2FDB-INR&label=Views&countColor=%23147b82&style=for-the-badge)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FAtharva-Rege%2FDB-INR)

<h3>🧠 <a href="#-method">Method</a> &nbsp;|&nbsp; 🏆 <a href="#-results">Results</a> &nbsp;|&nbsp; 🧩 <a href="#-pretrained-weights">Weights</a> &nbsp;|&nbsp; 🏋️ <a href="#️-training">Train</a> &nbsp;|&nbsp; ⚡ <a href="#-evaluation">Evaluate</a></h3>

<img src="https://raw.githubusercontent.com/genmilab/VGS-Decoding/main/docs/assets/genmilab-logo.png" alt="GenMI Lab" height="60"/>

</div>

## 🔥 News
- **[Oct 2026]** 🚀 Code and checkpoints for the three benchmarks of Table 1 are released.
- 📄 The paper is submitted to **IEEE ISBI 2027**.

## Overview
Reference implementation of the **DB-INR** layer and the meta-learned implicit segmentation pipeline it is trained in.

A coordinate network maps a spatial location to an intensity, and a small per-pixel head reads class labels from its penultimate features. **DB-INR** replaces the single sinusoid of each layer with a **low- and a high-frequency band**, each with its own projection and initialisation, mixed with a fixed equal weight and added to a residual scaled by 1/√2. At test time the network is fitted to the intensities of an unseen scan, and the labels are read off in one forward pass. **No annotation of the test scan is used.**

<p align="center">
<img src="./docs/assets/fig1_architecture.png" alt="DB-INR pipeline and layer" width="100%"/>
<br/>
<em><b>Fig. 1. Overview of the pipeline and of the DB-INR layer.</b> <b>(I)</b> A coordinate network f<sub>θ</sub> and a per-pixel head g<sub>φ</sub> are meta-learned with MAML under a reconstruction and a classification term. <b>(II)</b> At test time f<sub>θ</sub> is fitted to an unseen scan with the reconstruction term alone and labels are read from its penultimate features in one pass, with φ frozen, so no annotation of the test scan is used. The block inside f<sub>θ*</sub> is the DB-INR layer: a projection per band at its own effective frequency, the mixing gate pinned to the constant g = ½ rather than learned, and a residual scaled by 1/√2.</em>
</p>

## 📖 Contents
- [🧠 Method](#-method)
- [🏆 Results](#-results)
- [🔍 Qualitative Results](#-qualitative-results)
- [⛏️ Installation](#️-installation)
- [🗃️ Data](#️-data)
- [🧩 Pretrained Weights](#-pretrained-weights)
- [🏋️ Training](#️-training)
- [⚡ Evaluation](#-evaluation)
- [🗂️ Repository Structure](#️-repository-structure)
- [⚠️ Limitations](#️-limitations)
- [📝 Citation](#-citation)
- [📚 Acknowledgments](#-acknowledgments)
- [📨 Contact](#-contact)

## 🧠 Method

### The layer

Each layer carries a low- and a high-frequency sinusoidal branch with its own projection, combines them with a fixed equal mixture, and adds a residual scaled by `1/sqrt(2)`:

```
y = 0.5 * sin(w * r_lo * W_lo x) + 0.5 * sin(w * r_hi * W_hi x)
h = (y + R(x)) / sqrt(2)
```

with `r_lo = 1/sqrt(rho)`, `r_hi = sqrt(rho)`, `rho = phi` (the golden ratio) by default, `W_lo` and `W_hi` drawn from `U(±sqrt(6/n)/(w*r))` at each branch's own effective frequency, and `R` the identity where the width is unchanged and otherwise `U(±sqrt(3/n))`.

The supplementary derives four design conditions, each tested in the ablations:

| Condition | Design choice | Ablation flag (`dbinr/layers.py` / `train.py`) |
|---|---|---|
| P1 | the mixture is fixed, not learned | `fixed_gate=None` / `--fixed-gate` restores a learned gate; `norm_gate=True` / `--norm-gate` adds the variance-preserving denominator |
| P2 | each branch has its own projection and initialisation | `share_linear=True` / `--share-linear` gives both branches one projection |
| P3 | the residual is scaled by 1/√2 | `scale_residual=False` / `--no-scale-residual`; `--no-residual` removes it |
| P4 | the two bands sit near the base frequency | `--rho` sets the band ratio r_hi / r_lo |

### The pipeline

1. **Stage one (meta-learning).** The coordinate network and the head are meta-learned jointly with MAML, each scan a task. The inner loop adapts the network to one scan for `Ti` Adam steps with the reconstruction term alone; the outer loop updates network and head under a reconstruction and a focal term.
2. **Stage two (read-out).** The network is refitted to every training scan, the penultimate features are cached, and the read-out (one 512-wide hidden layer) is trained on them with early stopping.
3. **Test time.** The network is fitted to the intensities of an unseen scan for `Tf` steps with the reconstruction term alone, and labels are read from the fitted features in one forward pass with the head frozen.

> [!NOTE]
> Every backbone in the paper (SIREN, Gauss, WIRE, FINER, HOSC and DB-INR) is trained and evaluated under exactly this protocol and a matched parameter budget. Only the coordinate activation and its validated base frequency differ.

## 🏆 Results

Mean Dice, NSD and HD95 over the full test sets are in Table 1 of the paper. The supplementary repeats every comparison case by case (same trained models, two-sided Wilcoxon signed-rank test over test cases):

<div align="center">

| Baseline | OASIS-2D (100 subjects)<br/>ahead · median Δ · *p* | OASIS-3D (40 volumes)<br/>ahead · median Δ · *p* | ACDC (50 slices)<br/>ahead · median Δ · *p* |
|---|:---:|:---:|:---:|
| SIREN | 96/100 · +0.0173 · 5.8e-18 | 40/40 · +0.0051 · 1.8e-12 | 30/50 · +0.0092 · 2.5e-01 |
| Gauss | 100/100 · +0.0378 · 3.9e-18 | 40/40 · +0.0213 · 1.8e-12 | 39/50 · +0.0251 · 1.0e-04 |
| WIRE | 98/100 · +0.0321 · 4.5e-18 | 40/40 · +0.0295 · 1.8e-12 | 43/50 · +0.0381 · 6.0e-07 |
| FINER | 100/100 · +0.0770 · 3.9e-18 | 40/40 · +0.0733 · 1.8e-12 | 48/50 · +0.3550 · 8.9e-15 |
| HOSC | 100/100 · +0.0582 · 3.9e-18 | 40/40 · +0.0398 · 1.8e-12 | 42/50 · +0.0317 · 2.1e-08 |

</div>

- On both brain tasks the gain is spread across cases rather than concentrated in a few.
- On ACDC, DB-INR separates from four backbones but not from SIREN (30/50, *p* = 0.25): its margin over SIREN sits inside the noise.

<p align="center">
<img src="./docs/assets/fig_paired.png" alt="Per-case Dice against SIREN" width="100%"/>
<br/>
<em><b>Per-case Dice against SIREN</b>, the strongest baseline on each benchmark, with DB-INR on the vertical axis and SIREN on the horizontal, one point per test case, identity line in grey; points above the identity line are cases in which DB-INR achieves higher Dice.</em>
</p>

## 🔍 Qualitative Results

> [!NOTE]
> The cases below were selected as examples with large DB-INR gains over the strongest baseline. They show where the layer helps, not typical test-set performance.

<p align="center">
<img src="./docs/assets/fig_grid_main.png" alt="Every backbone on the same OASIS-2D and ACDC scans" width="100%"/>
<br/>
<em><b>Every backbone on the same scans.</b> Five OASIS-2D slices with 25 labels above, five ACDC slices with three labels below, all at the 1.0 M budget. Columns are the input, the five baselines, DB-INR and the reference.</em>
</p>

On OASIS-2D every backbone recovers the same gross anatomy and they differ at the boundaries of the thin cortical labels. On ACDC the baselines return speckle over the myocardium and blood pool where DB-INR returns connected regions.

<p align="center">
<img src="./docs/assets/fig_collage_3d.png" alt="Volumetric result on OASIS-3D" width="85%"/>
<br/>
<em><b>Volumetric result on OASIS-3D</b> at 64 × 64 × 64. Three orthogonal views of one test volume, segmented by the same layer and protocol with a three-dimensional coordinate; per-case Dice inset.</em>
</p>

<details>
<summary><strong>MedSegBench: ISIC-2016, BBBC010, COVID-19 Radiography</strong></summary>

<p align="center">
<img src="./docs/assets/fig_grid_msb.png" alt="Every backbone on three MedSegBench benchmarks" width="100%"/>
<br/>
<em><b>Every backbone on three MedSegBench benchmarks.</b> Three test images each from ISIC-2016 dermoscopy, BBBC010 microscopy and COVID-19 Radiography chest X-ray, at 128² in grayscale with at most 160 training images.</em>
</p>

</details>

## ⛏️ Installation

```bash
git clone https://github.com/Atharva-Rege/DB-INR.git
cd DB-INR
pip install -r requirements.txt
```

Requirements: PyTorch ≥ 2.2, NumPy, OpenCV, NiBabel, SciPy and `medsegbench`.

## 🗃️ Data

Set the split files and roots before training:

```bash
export OASIS_SPLITS=/path/to/oasis_splits.json
export OASIS_ROOT=/path/to/oasis
export ACDC_SPLITS=/path/to/acdc_splits.json
export ACDC_ROOT=/path/to/acdc
export OASIS3D_ROOT=/path/to/oasis3d
```

MedSegBench datasets download on first use through the `medsegbench` package (`--dataset msb:<Name>`).

## 🧩 Pretrained Weights

Checkpoints for the three benchmarks of Table 1:

| File | Dataset | Labels | Res | `omega` | Bands | Download |
|---|---|---|---|---|---|---|
| `dbinr_oasis2d.pt` | OASIS-2D, brain MRI | 25 | 192 | 20 | 0.786, 1.272 | [link](https://drive.google.com/file/d/1wwZjdaieBG2Bl7H-29l-uz_9Zm1Po1MN/view?usp=sharing) |
| `dbinr_acdc.pt` | ACDC, cardiac cine MRI | 3 | 192 | 30 | 0.786, 1.272 | [link](https://drive.google.com/file/d/1rdFk542_-Utr8Uo1nQnnrDvaEndNi9K1/view?usp=sharing) |
| `dbinr_oasis3d.pt` | OASIS-3D, brain volumes | 4 | 64 | 30 | 0.500, 0.809 | [link](https://drive.google.com/file/d/1rcAEzCC1ENXMnITrWKt9UpcjtMlRmzMV/view?usp=sharing) |

Pass the `.pt` file to `evaluate.py`:

```bash
python evaluate.py --ckpt dbinr_oasis2d.pt --split test
```

## 🏋️ Training

OASIS-2D with 25 labels at the 1.0 M budget, the configuration reported in the paper:

```bash
python train.py --dataset oasis --classes 24 --res 192 \
    --hidden 377 --layers 3 --head-hidden 512 --omega 20 \
    --Tf 100 --outer-epochs 10 --head-epochs 1500 \
    --bs 3 --head-bs 8 --seed 0 --out runs/dbinr_oasis
```

<details open>
<summary><strong>Other benchmarks</strong></summary>

```bash
# Cardiac cine MRI (ACDC)
python train.py --dataset acdc --classes 3 --omega 30 --out runs/dbinr_acdc

# Volumetric (OASIS-3D)
python train.py --dataset oasis3d --classes 4 --res 64 --omega 30 \
    --bs 1 --head-epochs 800 --out runs/dbinr_oasis3d

# MedSegBench (e.g. DCA1)
python train.py --dataset msb:Dca1 --classes 1 --res 128 --omega 30 --out runs/dbinr_dca1
```

</details>

| Argument | Default | Meaning |
|---|---|---|
| `--hidden`, `--layers` | 377, 3 | coordinate network width and depth |
| `--head-hidden` | 512 | read-out hidden width |
| `--omega` | 20 | base frequency ω |
| `--rho` | φ | band ratio r_hi / r_lo |
| `--Ti`, `--Tf` | 2, 100 | inner-loop steps (meta-training), fitting steps (refit and test) |
| `--outer-epochs`, `--head-epochs` | 10, 1500 | stage one and stage two epochs |
| `--inner-lr`, `--outer-lr`, `--head-lr` | 1e-4 | learning rates |
| `--gamma` | 2.0 | focal-loss γ |

Each run writes `<out>.pt` (meta-learned initialisation, head and arguments) and `<out>.json` (validation Dice and history).

## ⚡ Evaluation

```bash
python evaluate.py --ckpt runs/dbinr_oasis.pt --split test
```

The network is fitted to the intensities of each test scan for `Tf` steps with the reconstruction term alone, and the frozen head reads the labels. `--n` limits the number of test cases.

## 🗂️ Repository Structure

```text
DB-INR/
├── dbinr/
│   ├── layers.py     # DualBandSineLayer and its ablation variants
│   ├── models.py     # INR, segmentation head, SegINR
│   ├── meta.py       # MAML meta-learner (inner and outer loop)
│   ├── engine.py     # batched fitting, meta-training, head training
│   ├── losses.py     # reconstruction, focal and composite losses
│   ├── data.py       # OASIS-2D, ACDC, OASIS-3D and MedSegBench loaders
│   └── metrics.py    # Dice, IoU, HD95, ASSD, NSD, PSNR
├── train.py          # two-stage training
├── evaluate.py       # test-time fitting and evaluation
├── docs/assets/      # README figures (architecture, results, qualitative)
└── requirements.txt
```

## ⚠️ Limitations

- On ACDC the margin over SIREN is not significant (30 of 50 slices, *p* = 0.25).
- Qualitative cases were selected for large gains and do not represent the test-set distribution.
- Test-time segmentation needs a per-scan fit of `Tf` steps, so inference is slower than a single feed-forward pass.

## 📝 Citation

The citation will be added once the review process concludes.

## 📚 Acknowledgments

The baselines are SIREN, Gauss, WIRE, FINER and HOSC. Meta-learning follows MAML (Finn et al., 2017). We thank the providers of OASIS, ACDC and MedSegBench.

## 📨 Contact
For questions, please open an [issue](https://github.com/Atharva-Rege/DB-INR/issues).

> [!IMPORTANT]
> DB-INR is intended for research only. It is not validated for clinical use and must not inform any diagnostic or treatment decision.
