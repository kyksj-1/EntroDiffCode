# EntroDiff (MVP Codebase)

Welcome to the minimal viable product (MVP) codebase for **EntroDiff: Entropy-aware diffusion for hyperbolic PDEs** (NeurIPS 2026 Target).

This codebase is specifically designed to run robustly on multiple environments (PC, Dedicated Servers, Colab, Kaggle). We emphasize separation of configurations from code, adhering strongly to deterministic multi-environment setups.

## 1. Quick Start and Environment Configuration

As highlighted in the project documentation: **Never commit absolute paths**. 

1. Copy the configuration template:
   ```bash
   cp PROJECT/black/configs/env_config.yaml.example PROJECT/black/configs/env_config.yaml
   ```
2. Open configs/env_config.yaml and edit the data_dir and output_dir corresponding to your current hardware.
3. You can also override device requests here: device: "cuda". For Windows local tests with GPU constraints, leave num_workers: 0 to avoid buggy multiprocessing overhead.

## 2. Directory Structure

```text
PROJECT/black/
├── configs/
│   ├── env_config.yaml         # Your local env configuration (Do not commit)
│   └── ...                     # Other exp configurations 
├── src/
│   ├── data/                   # Data generation and loaders (Burgers 1D Solver)
│   ├── models/                 # Neural Architectures (1D U-Net Backbone)
│   ├── diffusion/              # Diffusion Core (Viscosity schedules, Godunov sampler)
│   └── utils/                  # Environment parsers
├── scripts/
│   ├── generate_data.py        # Generates Ground Truth Godunov data
│   ├── train_mvp.py            # End-to-end training script
│   └── eval_viz.py             # (WIP) Evaluation and visualizations
└── README.md
```

## 3. Workflow (MVP)

The MVP validates EntroDiff theory on a 1D Inviscid Burgers Equation comparing purely vanilla DSM with Kruzhkov Entropy and physical schedules.

### Step 1: Generate Data
Run the internal rigorous finite-volume Godunov solver to yield Ground Truth tracking shock development:
```bash
python PROJECT/black/scripts/generate_data.py
```
*Outputs are saved to your data_dir defined in env_config.yaml.*

### Step 2: Train MVP Model
The MVP setup currently runs 1D U-Net using our custom score configurations and Godunov flux constraints.
```bash
python PROJECT/black/scripts/train_mvp.py
```
*Modify hyperparameters like epochs = 10 or lr = 2e-4 inside train_mvp.py manually during the MVP rapid iteration phase. Checkpoints will be flushed explicitly to your output_dir.*

## 4. Theory & Implementation Divergence Notes

Important architectural notes and mappings to the 03_method.tex methodology equations:
- **src/diffusion/schedules.py**: Accurate implementation of Viscosity-matched tracking (sigma^2(tau) = 2*nu*tau).
- **src/models/score_param.py**: The BVAwareScore contains proxy forward steps in the MVP. Fully rigorous graph-created autograd derivations will be plugged in via explicit references left throughout the comments.
- **src/diffusion/samplers.py**: We implemented the Heun integration step. PDE direction strictly uses proxy Godunov residual flow rather than its computationally intensive backwards node tracking (\nabla L_PDE) to maintain fast local throughput natively, mirroring physical displacement efficiently.
