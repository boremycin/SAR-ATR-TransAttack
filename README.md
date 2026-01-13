# SAR-ATR-TransAttack


This repository provides the official implementation for the paper:

**"SRAW-Attack: Space-Reweighted Adversarial Warping Attack for SAR Target Recognition"**

The codebase is designed to support reproducible research on *adversarial robustness and vulnerability analysis of deep learning–based SAR Automatic Target Recognition (SAR-ATR) systems*, with a particular focus on **spatial/warping-based adversarial attacks** that depart from conventional additive perturbation paradigms.

---

## 1. Overview

Deep neural networks (DNNs) have become the dominant approach for SAR Automatic Target Recognition (ATR). However, due to the intrinsic sparsity, strong structural priors, and speckle-dominated background of SAR imagery, their robustness characteristics differ fundamentally from those observed in optical imagery.

This repository investigates **warping-based adversarial attacks** for SAR-ATR, including spatial warping and geometry-aware perturbations, and provides:

- Implementations of several *baseline adversarial attacks* adapted to SAR imagery
- A novel **warping-based adversarial attack framework** (TransAttack)
- Evaluation protocols tailored for SAR-ATR models
- Experimental code supporting the results reported in the paper

The overall goal is to **analyze, rather than merely degrade, SAR-ATR robustness**, and to better understand how geometric distortions and pixel re-arrangements can serve as effective adversarial mechanisms in SAR.

---

## 2. Key Contributions Reflected in This Code

- **Beyond additive perturbations**  
  Unlike FGSM/PGD-style methods that rely on pixel-wise intensity changes, this work formulates adversarial generation as a *spatial transformation problem*, which is particularly suitable for SAR images with strong structural cues.

- **SAR-specific adversarial insights**  
  The implementation explicitly considers SAR characteristics such as sparse scattering centers, background speckle, and geometry sensitivity, making it fundamentally different from off-the-shelf vision attack code.

- **Unified evaluation pipeline**  
  The repository supports consistent comparison between classical additive attacks and transformation-based attacks under the same SAR-ATR setting.

---

## 3. Repository Structure

```
SAR-ATR-TransAttack/
├── attacks/            # Adversarial attack implementations
│   ├── fgsm.py
│   ├── pgd.py
│   ├── cw.py
│   ├── ...
│   └── sraw.py  # Transformation-based attack (core contribution)
│
├── models/             # SAR-ATR network architectures
│   ├── resnet.py
│   ├── vgg.py
│   └── ...
│
├── datasets/           # Dataset loading and preprocessing
│   └── mstar.py
│
├── utils/              # Helper functions (metrics, visualization, etc.)
│
├── configs/            # Experiment configuration files
│
├── train.py            # Model training script
├── test.py             # Standard evaluation
├── attack_eval.py      # Adversarial evaluation entry point
│
└── README.md
```

> **Note**: The exact structure may evolve as experiments are extended. The current layout reflects the version used for the paper's reported results.

---

## 4. Supported Attacks

The following adversarial methods are implemented or supported:

- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent)
- **MI-FGSM**
- **CW Attack**
- **SRAW Attack** *(proposed)*

The proposed SRAW optimizes adversarial effectiveness by learning **spatial deformation fields** rather than direct pixel perturbations, enabling subtle but semantically meaningful attacks in SAR imagery.

---

## 5. Datasets

Experiments are conducted on **SAR-ATR benchmark datasets** (e.g., MSTAR-style datasets).

Due to dataset licensing restrictions, **raw SAR data are not included** in this repository. Please prepare the dataset separately and organize it according to the expected directory structure defined in `datasets/`.

---

## 6. Usage

### 6.1 Environment Setup

```bash
conda create -n sar-attack python=3.8
conda activate sar-attack
pip install -r requirements.txt
```

### 6.2 Training a SAR-ATR Model

```bash
python train.py --config configs/train.yaml
```

### 6.3 Adversarial Evaluation

```bash
python attack_evaluation.py \
    --attack transattack \
    --model resnet \
    --config configs/attack.yaml
```

---

## 7. Evaluation Metrics

The code supports commonly used metrics for adversarial evaluation in SAR-ATR, including:

- Classification accuracy under attack
- PSNR / SSIM
- Perceptual similarity metrics
- Attack success rate (ASR)

These metrics are reported in the paper to quantify both *attack effectiveness* and *visual plausibility*.

---

## 8. Reproducibility

- All experiments are conducted with fixed random seeds.
- Configuration files used for the paper are provided in `configs/`.
- Model checkpoints can be regenerated following the provided scripts.

If you encounter discrepancies, please verify dataset preprocessing and normalization settings, which are critical for SAR data.

---

## 9. Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{SRAW,
  title={SRAW-Attack: Space-Reweighted Adversarial Warping Attack for SAR Target Recognition},
  author={Yiming Zhang and others},
  journal={Arxiv},
  year={2026}
}
```

---

## 10. License

This project is released for **academic research purposes only**.

Please check the license file for details and ensure proper citation when using this code in your work.

---

## 11. Contact

For questions, discussions, or potential collaborations, feel free to open an issue or contact the authors.

---

**This repository aims to facilitate transparent, reproducible, and SAR-aware adversarial research.**

