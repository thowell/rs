# Random Search
A simple [JAX](https://github.com/google/jax)-based implementation of [random search](https://arxiv.org/abs/1803.07055) for [locomotion tasks](https://github.com/openai/gym/tree/master/gym/envs/mujoco) using [MuJoCo XLA (MJX)](https://mujoco.readthedocs.io/en/stable/mjx.html).

## Installation
Clone the repository:
```sh
git clone https://github.com/thowell/rs
```

Optionally, create a conda environment:
```sh
conda create -n rs python=3.10
conda activate rs
```

pip install:
```sh
pip install -e .
```

## Environments
- [Ant](rs/envs/ant.py)
  - based on [ant_v5](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v5.py)
  - modified solver settings
  - only contact between feet and floor
  - no rewards or observations dependent on contact forces
- [Cheetah](rs/envs/cheetah.py)
  - based on [half_cheetah_v5](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/half_cheetah_v5.py)
  - modified solver settings
- [Humanoid](rs/envs/humanoid.py)
  - based on [humanoid_v5](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v5.py)
  - modified solver settings
  - only contact between feet and floor
  - no rewards or observations dependent on contact forces
- [Walker](rs/envs/walker.py)
  - based on [walker2d_v5](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v5.py)
  - modified solver settings
  - only contact between feet and floor


## Run
Experiments performed with [Nvidia RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) and [Ubuntu 22.04.4 LTS](https://releases.ubuntu.com/jammy/).

**Note**: run multiple times to find good policies.

First, change to `rs/` directory:
```sh
cd rs
```

### Ant
Search:
```sh
python train.py --env ant --search
```

Visualize policy checkpoint:
```sh
python train.py --env ant --mode visualize --load pretrained/ant
```

### Cheetah
Search:
```sh
python train.py --env cheetah --search
```

Visualize policy checkpoint:
```sh
python train.py --env cheetah --load pretrained/cheetah --visualize
```

### Humanoid
Search:
```sh
python train.py --env humanoid --search
```

Visualize policy checkpoint:
```sh
python train.py --env humanoid --load pretrained/humanoid --visualize
```

### Walker
Search:
```sh
python train.py --env walker --search
```

Visualize policy checkpoint:
```sh
python train.py --env walker --load pretrained/walker --visualize
```

### Command line arguments
- `--env`: `ant`, `cheetah`, `humanoid`, `walker`
- `--search`: run random search to improve policy
- `--checkpoint`: filename in `checkpoint/` to save policy
- `--load`: provide string in `checkpoint/` 
directory to load policy from checkpoint
- `--seed`: int for random number generation
- `--visualize`: visualize policy 


## Mapping notation from the paper to code
$\alpha$: `update_step`

$\nu$: `random_step`

$N$: `nsample`

$b$: `ntop`

## Notes
