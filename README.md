# Creating environment with uv

This is specific to IsaacSim 5.0.0 with IsaacLab 2.3.0 because this uses specific package versions. Other versions are not tested. 

## Installation

The installation process largely follows [the official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html), but with minor modifications to maximize the potential with uv. 

### 1. Install Isaac Sim

Follow the guide to everything before the `Installing Isaac Lab` section. 

### 2. Install IsaacLab

Use the IsaacLab with modified environment in this repo and create the Isaac Sim Symbolic Link. 

### 3. Setting up a Python Environment (DIFFERENT!)

The remainder of this guide assumes that your IsaacLab will be under the root of your project folder like
```bash
-- Delta-Action-Model-for-Contact-Rich-Tasks
 |  -- main.py
 |  -- IsaacLab
 |  -- configs/
 |  -- scripts/
```

**NOTE** The way IsaacLab's official way of installing with uv is suboptimal:  
1. Does not take advantage of uv's ultra-fast dependency parsing.
2. Does not use uv's strict version locking and guarantee system. 

which are the two major selling points of `uv` in the first place. Essentially, all it uses uv for is creating a venv, and rolls back to pip for everything else.  

Therefore, we modify the env creation and package installation process in Step 4. 

### 4. Installation

First, install the system dependencies: 
```bash
# these dependencies are needed by robomimic which is not available on Windows
sudo apt install cmake build-essential

# additional dependencies for pygame (required by rl-games)
sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libjpeg-dev libpng-dev libportmidi-dev
```

Then, replacing the official guide to run `isaaclab.sh --uv my_env`, actually run:
```bash
uv init
uv sync     # to create the .venv
```
in your project root, so you have control of your environment in your project, not within the path of IsaacLab. 

To add environment variables, the official script injects commands into `.venv/bin/activate` (or the conda equivalent), but this is not very convenient, since `uv run` and VSCode debugger bypass that. 

The alternative recommendation is to use `direnv` as a shell extension, which auto-loads and unloads env variables as you enter and exit directories. Create a `.envrc` file and register some environment variables required by IsaacLab/IsaacSim: 

```bash
# Isaac Lab setup
export ISAACLAB_PATH="${PWD}/IsaacLab"
alias isaaclab="${PWD}/IsaacLab/isaaclab.sh"
export RESOURCE_NAME="IsaacSim"

# Source Isaac Sim environment if available (binary install)
if [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]; then
    . "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
fi

# Add project root to PYTHONPATH for local imports
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

Then, replacing the official guide to run `isaaclab.sh --install`, modify your `pyproject.toml` so that ALL required packages (RL, testing, dev tools) are in the main dependency list (no extras needed):

```toml
[project]
name = "delta-action-model"
version = "0.1.0"
description = "Learning a delta action model that makes up for dynamic difference between sim and real for contact-rich tasks"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  # PyTorch (CUDA 12.8)
  "torch==2.7.0",
  "torchvision==0.22.0",

  # Core dependencies
  "numpy<2",
  "onnx>=1.18.0",
  "prettytable==3.3.0",
  "toml",

  # Devices
  "hidapi==0.14.0.post2",

  # Reinforcement learning
  "gymnasium==1.2.1",
  "isaaclab-rl[all]",
  "isaaclab-mimic[all]",

  # Procedural generation
  "trimesh",
  "pyglet<2",

  # Image processing / modeling
  "transformers",
  "einops",
  "pillow==11.2.1",
  "warp-lang",

  # Livestream / web
  "starlette==0.45.3",

  # Misc
  "pygame>=2.3.0",

  # Testing
  "pytest",
  "pytest-mock",
  "junitparser",
  "flatdict==4.0.1",
  "flaky",

  # Dev tools
  "pre-commit",

  # Isaac Lab editable packages
  "isaaclab",
  "isaaclab-assets",
  "isaaclab-tasks",
  "isaaclab-rl",
  "isaaclab-mimic",
]

[tool.uv.sources]
isaaclab = { path = "IsaacLab/source/isaaclab", editable = true }
isaaclab-assets = { path = "IsaacLab/source/isaaclab_assets", editable = true }
isaaclab-tasks = { path = "IsaacLab/source/isaaclab_tasks", editable = true }
isaaclab-rl = { path = "IsaacLab/source/isaaclab_rl", editable = true }
isaaclab-mimic = { path = "IsaacLab/source/isaaclab_mimic", editable = true }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv]
index-strategy = "unsafe-best-match"
```

Then, run `uv sync` to install all required dependencies.

### 5. Verifying the Installation

Since we are using the python from the uv venv, we can test the IsaacLab installation with:
```bash
source .venv/bin/activate

# Empty example
python3 IsaacLab/scripts/tutorials/00_sim/create_empty.py

# Train a robot example
python3 IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

**Note:** We are not using `uv run` because this command bypasses the `activate` command which sets some aliases and env variables that are required for IsaacSim to be detected. 

### 6. Running the Delta Action Model

Once the environment is set up, you can run the training scripts for the delta action model:

```bash
source .venv/bin/activate

# Run your training script (adjust based on your actual scripts)
python3 scripts/train_delta_model.py --config configs/your_config.yaml
```

## Project Structure

```
Delta-Action-Model-for-Contact-Rich-Tasks/
├── IsaacLab/           # Modified Issac Lab for delta action training and policy fine-tuning
├── configs/            # Configuration files
├── scripts/            # Training and evaluation scripts
├── models/             # Model definitions
├── utils/              # Utility functions
├── pyproject.toml      # Project dependencies
├── . envrc              # Environment variables (direnv)
└── README.md           # This file
```

## Adding Packages to Your Project

Just follow the standard `uv` practice and add packages and package index to `pyproject.toml`, then run `uv sync` to realize changes. No extras flag is needed since everything is in the base dependency list.

## Troubleshooting

- **PyGame installation fails**: Make sure you have all the SDL2 system dependencies installed (see Step 4).
- **IsaacSim not detected**: Ensure you've sourced the activate script (`source .venv/bin/activate`) and that the `.envrc` file is properly configured.
- **CUDA issues**: Verify that you have CUDA 12.8 installed and that your GPU drivers are up to date.

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{your_paper,
  title={Learning a Delta Action Model for Contact-Rich Tasks},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2025}
}
```

