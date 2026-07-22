# Approximate C-Uniform Sampling: An Information-Theoretic and Bayesian Inference Perspective
Accompanying code for the paper

## Prerequisites
Clone this repo and create inside a virtual env. We use `uv` package manager:
```shell
uv venv
source .venv/bin/activate
```
In in the same directory level, clone also two more repositories:
```
git clone https://github.com/TimeEscaper/pyminisim.git
git clone https://github.com/planning-team/oo-ctrl-py.git
```
The resulting directory structure should have form:
```
.
├── <this repo>
├── pyminisim
├── oo-ctrl-py
```
Then, install main dependencies in venv:
```shell
uv sync
```

## Running the code

Sampling models training procedure is presented in notebook [train_mi.ipynb](notebooks/train_mi.ipynb) for Mutual Information-based method. The dataset sampling and training for HSVGD-based method is located in [scripts/svgd](scripts/svgd) directory. Sampled datasets used to train models are located in [vc_uniform_samples](artifacts/datasets/vc_uniform_samples), but you can generate them using [sample_dataset.py](scripts/svgd/sample_dataset.py) script. For all models, pre-trained weights available in [deploy_checkpoints](deploy_checkpoints) directory.

Evaluation in PyMiniSim is available in [pms_benchmark.py](scripts/eval/pms_benchmark.py) script. Plots visualization is available in [make_sampling_plots.py](scripts/eval/make_sampling_plots.py) script.
