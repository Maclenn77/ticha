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
