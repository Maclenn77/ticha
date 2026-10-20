# Environment for the dataset pipeline

How to reproduce the environment used to run
[`notebooks/ticha_pipeline.ipynb`](notebooks/ticha_pipeline.ipynb)
(download → summary → line pairing → review → splits → augmentation).

The notebook imports the code from `src/` in this repository, so the `ticha`
package does **not** need to be installed: clone the repo, create the
environment, run the notebook.

## Requirements

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html) (tested with conda 25.5)
- **Google Chrome**: only for the first step (the manuscript list is scraped
  with Selenium; Selenium downloads a matching chromedriver by itself)
- ~5 GB of disk for the environment (PyTorch), plus space for the page images
- Internet access on the first run: Ticha (documents and images), Zenodo (the
  McCATMuS HTR model, fetched once by kraken)

## Setup

```bash
git clone https://github.com/maclenn77/ticha.git
cd ticha
conda env create -f environment.yml
conda activate ticha
python -m ipykernel install --user --name ticha --display-name "Python (ticha)"
```

Check it:

```bash
python -m pytest -q
```

Then start Jupyter and open the notebook with the **Python (ticha)** kernel
(the notebook already asks for it):

```bash
jupyter lab notebooks/ticha_pipeline.ipynb
```

The first code cell checks that `ticha` is imported from this repo's `src/`
and stops otherwise.

## Two environment files

| File | Use |
|---|---|
| [`environment.yml`](environment.yml) | The requirements (minimum versions; kraken pinned). Works on macOS, Linux and Windows. Use this one normally. |
| [`environment.osx-arm64.lock.yml`](environment.osx-arm64.lock.yml) | Exact versions of every package in the environment the pipeline was tested with, on macOS / Apple Silicon. Use it to reproduce results exactly on that platform: `conda env create -f environment.osx-arm64.lock.yml` |

Main versions in the tested environment:

| Package | Version |
|---|---|
| Python | 3.12.15 |
| kraken | 7.1.1 |
| PyTorch / torchvision | 2.14.0 / 0.29.1 |
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| Pillow | 12.3.0 |
| selenium | 4.51.0 |
| JupyterLab | 4.6.4 |

### Why these versions

- **kraken 7.1.1** (`>=6,<8` in `pyproject.toml`): the code fetches the HTR
  model with `htrmopo`, which comes with kraken 6+. kraken 7 still provides
  `blla.segment()` and `TorchVGSLModel.load_model` but marks them deprecated
  ("will be removed with kraken 8"), so kraken 8 will need code changes. The
  deprecation warnings printed in step 4 are expected.
- **Python 3.12**: supported by kraken 6 (`<3.13`) and 7 (`<3.14`).
- **PyTorch comes from pip** (as a kraken dependency), which has the Apple
  Silicon (MPS) and CUDA builds.

## Speeding up step 4 (GPU and parallel workers)

Kraken segmentation and recognition of every page is the slowest step. It
runs on CPU by default (`DEVICE = "cpu"`, `WORKERS = 1` in the notebook's
configuration cell).

- **Parallel workers** (`WORKERS`): documents are processed in separate
  processes, each with its own copy of the models (~2 GB of RAM each at
  peak, measured segmenting a 2500x3500 page). Measured on an M2 MacBook Air
  (8 cores, 8 GB), segmenting a page: 7.4 s with 1 process, 4.7 s with
  `WORKERS = 2`, 3.9 s with `WORKERS = 3`. With 8 GB of RAM use 2 (3 at
  most); fanless laptops also slow down when they heat up on long runs.
- Apple Silicon GPU (`DEVICE = "mps"`): works, but was not faster than CPU
  for segmentation (most of the time goes to CPU post-processing). Prefer
  workers.
- NVIDIA: `DEVICE = "cuda:0"`. On Linux, pip's default PyTorch wheel already
  includes CUDA. If it doesn't match your driver, install a matching build
  after creating the environment, following <https://pytorch.org/get-started/locally/>.

Check what PyTorch sees:

```bash
python -c "import torch; print('mps', torch.backends.mps.is_available(), 'cuda', torch.cuda.is_available())"
```

## Data location

Everything the notebook downloads or produces goes to `data/` in the repo
(ignored by git). To keep it elsewhere, e.g. on an external disk:

```bash
TICHA_DATA=/Volumes/disk/ticha-data jupyter lab notebooks/ticha_pipeline.ipynb
```

## Updating the environment

After changing `environment.yml`:

```bash
conda env update -n ticha -f environment.yml --prune
python -m pytest -q
conda env export -n ticha --no-builds | grep -v '^prefix:' > environment.osx-arm64.lock.yml   # on macOS arm64
```

To remove it: `conda env remove -n ticha` and `jupyter kernelspec remove ticha`.

## TrOCR fine-tuning on a GPU VM

[`notebooks/ticha_trocr_finetuning.ipynb`](notebooks/ticha_trocr_finetuning.ipynb)
fine-tunes `microsoft/trocr-base-handwritten` on the
[`maclenn77/ticha-htr-lines`](https://huggingface.co/datasets/maclenn77/ticha-htr-lines)
dataset from the Hub, so it doesn't need `data/` or the conda environment. It
runs in Google Colab (installs its packages itself, saves to Drive) or on a
CUDA machine with [`requirements-trocr.txt`](requirements-trocr.txt).

Tested on a GCP VM: `g2-standard-8` with one NVIDIA L4 (23 GB), Deep Learning
VM image `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (Python 3.10, PyTorch
2.9.1+cu129). The notebook uses batch size 8 and bf16 on the L4 (2 x 4
gradient accumulation and fp16 on a 16 GB T4); one training epoch takes about
4 minutes.

### Create the VM

L4s are often out of stock (`ZONE_RESOURCE_POOL_EXHAUSTED`); try another zone
or add `--provisioning-model SPOT`:

```bash
gcloud compute instances create ticha-l4 --zone us-central1-a \
  --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE --boot-disk-size 150GB \
  --image-family pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project deeplearning-platform-release \
  --metadata install-nvidia-driver=True
```

### Set it up

On the VM (`gcloud compute ssh ticha-l4 --zone us-central1-a`). The venv
reuses the image's CUDA PyTorch through `--system-site-packages`:

```bash
git clone https://github.com/maclenn77/ticha.git && cd ticha
python3 -m venv --system-site-packages ~/venvs/ticha
~/venvs/ticha/bin/pip install -r requirements-trocr.txt
~/venvs/ticha/bin/python -m ipykernel install --user --name ticha-trocr --display-name "Python (ticha-trocr)"
```

### Open Jupyter through an SSH tunnel

Don't open a firewall port for Jupyter. Run it on the VM's localhost only and
forward the port over SSH:

```bash
# on the VM (stops when you log out; use tmux or a systemd user service to keep it running)
~/venvs/ticha/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser

# on your computer: keep this running, then open http://localhost:8888
gcloud compute ssh ticha-l4 --zone us-central1-a -- -N -L 8888:localhost:8888
```

Outputs go to `~/ticha-runs/` on the VM (set `TICHA_OUTPUT_DIR` to change it).
Copy results back with `gcloud compute scp --recurse ticha-l4:ticha-runs/... .`
and stop the VM when you're done, since it is billed while running:
`gcloud compute instances stop ticha-l4 --zone us-central1-a`.
