# Ticha

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23640278.svg)](https://doi.org/10.5281/zenodo.23640278)

Tools for the Colonial Zapotec manuscripts of the
[Ticha Digital Text Explorer](https://ticha.haverford.edu/): download the
manuscripts and their transcriptions, build a line-level handwritten text
recognition (HTR) dataset from them, and read new lines with fine-tuned TrOCR
models.

| Stage | Command / module | Output |
|---|---|---|
| Scrape | `ticha manuscripts`, `texts`, `images`, `printed` | CSVs of metadata and transcriptions, page images |
| Segment and pair | `ticha lines` (`ticha.core.line_pairing`) | PAGE XML + line crops paired with their text |
| Review | `ticha check`, `ticha review` | contact sheets; reviewed PAGE XML |
| Augment | `ticha augment` (`ticha.core.augmentation`) | degraded and varied copies of the lines |
| Build the dataset | `scripts/build_dataset.py`, `scripts/export_hf_dataset.py` | train / validation / test splits, [`maclenn77/ticha-htr-lines`](https://huggingface.co/datasets/maclenn77/ticha-htr-lines) |
| Recognize | `ticha.core.htr` | text of a line image, with [`maclenn77/ticha-trocr-base`](https://huggingface.co/maclenn77/ticha-trocr-base) |

## Installation

The package is not on PyPI; install it from the repository (Python 3.10+):

```bash
git clone https://github.com/maclenn77/ticha.git
cd ticha
pip install -e .            # scrapers
pip install -e '.[lines]'   # + segmentation and pairing (kraken)
pip install -e '.[htr]'     # + TrOCR recognition (torch, transformers)
pip install -e '.[dev]'     # + tests and linters
```

The notebooks use the conda environment described in
[`ENVIRONMENT.md`](ENVIRONMENT.md).

Before 1.0 the command was called `ticha-scraper`. It still works, with a
deprecation warning, and will be removed in 2.0.

## Usage

### Manuscript list (`manuscripts` command)

```bash
ticha manuscripts -o manuscripts.csv
ticha manuscripts -o manuscripts.csv -r 3.0 -v   # 3 s between requests, verbose
ticha manuscripts -o manuscripts.csv --no-headless   # show the browser
```

```python
from ticha import TichaScraper

scraper = TichaScraper(rate_limit=2.0)
df = scraper.scrape_manuscripts()
df.to_csv("manuscripts.csv", index=False)
```

### Downloading page images

The PDF links on Ticha return `500 Internal Server Error`, so pages are
rebuilt from the Deep Zoom Images (DZI) shown in the OpenSeadragon viewer
(`/images/dzi/<ticha_id>_<page>.dzi`): the scraper reads each descriptor,
downloads the tiles of one pyramid level
(`/images/dzi/<ticha_id>_<page>_files/{level}/{col}_{row}.jpg`) and stitches
them into a single image. Both page layouts are supported:

- `/en/texts/<id>/` (the `document_link` values from the manuscripts CSV):
  pages come from the `data-pages` range used by `call_the_dragon.js`.
- `/en/text/<id>/<n>/`: one image per page; every page is visited using the
  `pageMap` embedded in the page.

The original PDF is tried first (one attempt) unless `--skip-pdf` is given.

```bash
# One document
ticha images -u https://ticha.haverford.edu/en/texts/Oc111/ -d images

# All documents from the manuscripts CSV, max 3000px per side, plus a PDF per document
ticha images -i manuscripts.csv -d images --max-dimension 3000 --pdf
```

```python
from ticha import TichaImageScraper

scraper = TichaImageScraper(rate_limit=2.0)
result = scraper.download_document_images(url, "images", image_format="png")
print(result["image_paths"])
```

Images are written to `<output-dir>/<document>/page_001.png`, ...

### Text of the printed works (`printed` command)

Downloads only the transcriptions (no images or PDFs) of the printed works
in Ticha's texts menu: Agüero's *Misceláneo espiritual*, Córdova's *Arte*,
and Levanto's *Arte* and *Cathecismo*. This is extra Zapotec text, for
example to train a language model or the embeddings of a decoder. The
testaments featured on the site are already handled by the `texts` command.

```bash
ticha printed --list                                   # works available
ticha printed -o printed.csv --corpus-dir printed_corpus
ticha printed -o printed.csv -w aguero-miscelaneo      # one work
```

`printed.csv` has one row per page: `transcription` (original spelling),
`transcription_regular` (Ticha's regularized version, when there is one) and
`zapotec_marked` (the words Ticha glossed, which in the *Artes* are the
Zapotec forms inside the Spanish text). Glosses and catchwords are removed.
Pages already in the CSV are skipped, so you can resume an interrupted run.
`--corpus-dir` also writes one `.txt` per work.

### Line-level dataset for HTR (`lines` command)

Segments the downloaded page images into lines with
[kraken](https://kraken.re) and pairs each line with its transcription.

```bash
pip install 'ticha[lines]'   # installs kraken (and PyTorch)

ticha texts  -i manuscripts.csv -o texts.csv
ticha images -i manuscripts.csv -d images -o images.csv
ticha lines  -t texts.csv -i images.csv -d dataset --export-lines
```

`-w N` processes N documents in parallel, each in its own process with its
own copy of the models (~2 GB of RAM per worker). On an 8-core Apple Silicon
laptop with 8 GB, `-w 2` segments about 1.6x faster than one process (`-w 3`,
1.9x, is the most that fits in memory).

Steps:

1. **Cleaning**: documents without a transcription are skipped, and by
   default only `Completed and Checked` and `Completed but not Checked` are
   kept (`-s STATUS` to choose, `-s all` for every status).
2. **Segmentation**: kraken finds the baselines. Tiny or squeezed lines
   (folio numbers, words inserted between lines) are kept as *fragments*
   without text; lines are read top to bottom and left to right (signature
   columns row by row, two-page spreads left page first).
3. **Recognition**: a generic HTR model
   ([McCATMuS](https://doi.org/10.5281/zenodo.13788177), downloaded on first
   use; `--model` to change it) reads each line roughly. The reading is only
   used to find which transcription line belongs to which image line.
4. **Alignment**: the transcription is treated as a stream of words and cut
   into the spans read on each image line, so transcriptions whose line
   breaks do not follow the manuscript (re-flowed text, two transcription
   lines on one manuscript line, or the reverse) are paired too; the
   transcription's own breaks are kept when they fit. A word the scribe broke
   across two lines (reading ends in `-`/`=`) is split between them. Image
   lines may stay without text (stamped-paper headers, untranscribed
   signatures) and words may stay without a line. Pages are aligned on their
   own when the transcription's folio markers (`(8r)`, `[40r]`) match the
   image folios, otherwise the whole document at once.
5. **Confidence**: a pair is confident when its text fits the reading better
   than the neighbouring lines' texts (an off-by-one alignment would fit a
   neighbour as well), the reading has a plausible length for the text, and
   either its CER is at most `--cer-threshold` (0.5) or, up to `--max-cer`
   (0.75), it beats the neighbours by `--min-margin` (0.15): the generic
   model misreads Zapotec, so right pairs often have a high CER.
   `TextEquiv/@conf` is `1 - CER`, kept at 0.5 or above for confident pairs
   and below 0.5 for doubtful ones.
6. **Sorting**: pages go to `auto/` when at most `--review-tolerance` (20%)
   of their pairs are doubtful and of their transcription found no line;
   doubtful lines are written there **without text**, so they are left out
   of training (their best guess stays in `lines.csv`). `--review-tolerance 0`
   restores the strict rule. **Nothing is discarded**: every other page goes
   to `review/`, pre-filled with the best alignment, with the transcription
   assigned to it in a `.txt`. Pages to which no transcription was assigned
   (blank backs, covers) go to `untranscribed/`. The reason is in
   `manifest.csv`.

#### Applying a review (`review` command)

Pages in `review/` can be reviewed line by line against the images (with
`check`) and the decisions written to a text file, one line per decision:
`doc page line keep|drop|fix [text]`. `review` applies the file and writes
the reviewed PAGE XML, which the dataset build uses for those pages. The
review of the published dataset is in `review/decisions.txt`:

```bash
ticha review -d data/paired --decisions review/decisions.txt -o data/reviewed
```

#### Reproducing the published dataset

[`notebooks/ticha_data_preparation_pipeline.ipynb`](notebooks/ticha_data_preparation_pipeline.ipynb) runs every step. The pieces that make the
result reproducible:

- **Segmentation:** kraken's default baseline model, fixed by `kraken==7.1.1`
  in `environment.yml`.
- **Pairing:** McCATMuS fine-tuned on Ticha lines (`FINETUNED_SOURCE =
  "download"` takes the published model; `"train"` fine-tunes it again on a
  first pass with McCATMuS) and the confidence settings `CER_THRESHOLD = 0.5`,
  `MAX_CER = 0.75`, `MIN_MARGIN = 0.15`, `REVIEW_TOLERANCE = 0.2` (the
  defaults of the `lines` command).
- **Review:** `review/decisions.txt`, applied in step 6.
- **Split and augmentation:** fixed seed (`SEED = 42`).

#### Building the train / validation / test dataset

`scripts/build_dataset.py` (steps 6-10 of `notebooks/ticha_data_preparation_pipeline.ipynb`)
takes the `auto/` pages and the pages corrected by hand in `data/reviewed/`,
splits the documents 80/10/10 (by lines, within each language) and cuts the
lines. Augmented copies are kept in a separate folder:

```
data/splits/{train,validation,test}/lines/...   original lines (+ train.txt, validation.txt, test.txt)
data/augmented/train/lines/...                   degrade + variety copies of train lines (train.txt)
data/augmented/test_degraded/lines/...           degrade-only copies of test lines (test_degraded.txt)
```

```bash
python scripts/build_dataset.py --copies 4 --degrade-ratio 0.5 --test-degraded-copies 1
cat data/splits/train.txt data/augmented/train.txt > train_all.txt
ketos train -f path -t train_all.txt -e data/splits/validation.txt
ketos test -f path -m model.safetensors -e data/splits/test.txt              # real scans
ketos test -f path -m model.safetensors -e data/augmented/test_degraded.txt  # low quality
```

#### Checking the pairing by eye

`check` draws random line crops with the text paired with them, the HTR
reading and the CER (green: confident, red: doubtful, gray: no text), to
see whether the pairs are right before training on them:

```bash
ticha check -d dataset --only doubtful -n 30            # 30 doubtful lines
ticha check -d dataset --only confident -s auto -n 45   # what goes to training
ticha check -d dataset --doc SDE740dT -n 0 -o check/SDE740dT.png  # a whole document
```

Sheets are written as `check/check_001.png`, `check/check_002.png`, ...
(15 lines each, `--per-sheet` to change).

`--no-recognition` skips steps 3-4 and pairs lines in order (a page is
accepted only when the line counts agree).

Output:

```
dataset/
├── manifest.csv            # one row per page: status, reason, line counts
├── lines.csv               # one row per line: text, HTR reading, CER, confident
├── auto/                   # PAGE XML + image, ready for training
│   ├── Al714_page_001.png
│   └── Al714_page_001.xml
├── review/                 # PAGE XML + image + .txt for human correction
├── untranscribed/          # pages without transcription (segmentation only)
├── lines/                  # with --export-lines: line_001.png + line_001.gt.txt
│                           #   (every confident pair, also from review pages)
├── segmentation/           # kraken caches: re-runs skip segmentation
└── recognition/            #   and recognition
```

The PAGE XML (2019-07-15 schema) files can be imported into
[eScriptorium](https://escriptorium.inria.fr) together with their images to
review/correct them, and used to train a model with
`ketos train -f page dataset/auto/*.xml`. The confidence of each pair is in
`TextEquiv/@conf`.

### Data augmentation (`augment` command)

Augments the confidently paired line images written by
`lines --export-lines`. Each augmented image keeps the transcription it was
paired with; nothing is segmented or aligned again.

```bash
ticha lines   -t texts.csv -i images.csv -d dataset --export-lines
ticha augment -d dataset -o augmented -n 4 --seed 1
```

Two kinds of copies (`--degrade-ratio`, default half and half):

- **degrade**: imitates Ticha's untranscribed manuscripts, so a model trained
  on the transcribed ones transfers to them. Measured on a sample of
  untranscribed pages (mostly the Santo Domingo Etla notarial series), those
  scans are about 0.6x the resolution (~1450 px wide), almost grey with a
  cool cast and faded brown ink, and mostly washed out (paper ~220) or
  sometimes underexposed (dark grey, low contrast; `--underexposed-ratio`),
  with uneven lighting and stains. With `--target DIR` the tone, colour and
  resolution of real low-quality images in `DIR` are imitated instead of
  the built-in ranges.
- **variety**: brightness, contrast, gamma, warm/cool paper, ink weight,
  blur, noise, JPEG compression, small rotation, slant, scale and elastic
  distortion (`--no-geometry` to disable the geometric ones).

Output: `augmented/lines/<page>/line_NNN_augKK.png` + `.gt.txt` (same format
as `lines/`) and `augment.csv` with the parameters of every copy. The white
area outside each line's polygon stays white, as in the original crops.

Lines of `review/` pages are used too when they were paired confidently
(`--min-conf`, default 0.5); `--auto-only` restricts to `auto/` pages.
`--pages` also augments whole pages as PAGE XML + image (reusing the paired
lines and transforming their coordinates), e.g. to train segmentation.

Split train/validation/test **by document before augmenting**, so copies of
the same line never end up on both sides.

### Handwritten text recognition (`ticha.core.htr`)

Reads line images with TrOCR base fine-tuned on the Ticha lines. The model
proposes its 10 best readings with beam search and a second model, trained to
write each line right to left, reranks them. Both are in
[`maclenn77/ticha-trocr-base`](https://huggingface.co/maclenn77/ticha-trocr-base).

```bash
pip install -e '.[htr]'
```

```python
from ticha.core.htr import LineReader, clean_text

reader = LineReader.from_pretrained()          # rerank=False: left-to-right model only
reader.transcribe("line.png")                  # one image -> str
reader.transcribe(["a.png", "b.png"])          # several -> list of str

clean_text("p[es]os de Dom(ing)o")             # Ticha transcription -> diplomatic text
```

`clean_text` is the rule the models were trained and evaluated with: it
removes expansions and glosses and returns `None` for lines with damage or
editorial notes.

### Notebooks

- [`notebooks/ticha_data_preparation_pipeline.ipynb`](notebooks/ticha_data_preparation_pipeline.ipynb):
  every step from downloading to the final train/validation/test dataset with
  augmentation, using the code in `src/` directly.
- [`notebooks/ticha_trocr_usage_example.ipynb`](notebooks/ticha_trocr_usage_example.ipynb):
  reading manuscript lines with the published models; no training needed.
- [`notebooks/train/`](notebooks/train): TrOCR training: fine-tuning, the
  ablation of label smoothing and the learning-rate schedule, and the
  right-to-left reranker. The notebooks share their settings and code through
  four modules in the same folder, so every model is trained and scored on the
  same data in the same way. Training hyperparameters stay in each notebook.
  - [`trocr_setup.py`](notebooks/train/trocr_setup.py): runs folder, pretrained
    model, dataset version, seed, decoder start token.
  - [`trocr_data.py`](notebooks/train/trocr_data.py): loading, cleaning
    (`ticha.core.htr.clean_text`) and the checks (no document in two splits,
    no train document repeating evaluation lines).
  - [`trocr_model.py`](notebooks/train/trocr_model.py): batch collators, model
    setup for training and decoding, validation metrics.
  - [`trocr_eval.py`](notebooks/train/trocr_eval.py): decoding a split, CER and
    WER by language, paired bootstrap.
- [`notebooks/train/trained/`](notebooks/train/trained): the training notebooks
  exactly as they ran to produce the models in the report, with their outputs.
  They predate the shared modules, so their code is self-contained; a note at
  the top of each says what the run used.
- [`notebooks/reports/`](notebooks/reports): evaluation of the models and the
  metrics in the report.

#### Models in the report

The training notebooks save each model under `$TICHA_RUNS_DIR` (default
`~/ticha-runs`), where the later notebooks read it. Run them in this order,
because each one reads the models of the previous ones:

| Order | Notebook | Models in the report | Run folder | Run record |
|---|---|---|---|---|
| 1 | [`train/ticha_trocr_finetuning.ipynb`](notebooks/train/ticha_trocr_finetuning.ipynb) | **Smoothing, cosine** (run D) | `trocr-base-handwritten-finetuned-smooth-cosine` | [outputs](notebooks/train/trained/ticha_trocr_finetuning.with-outputs.ipynb) |
| 2 | [`train/ticha_trocr_ablation_smoothing_schedule.ipynb`](notebooks/train/ticha_trocr_ablation_smoothing_schedule.ipynb) | **No smoothing, linear** (run A) | `ablation-smoothing-schedule/a-no-smoothing-linear` | [outputs](notebooks/train/trained/ticha_trocr_ablation_smoothing_schedule.with-outputs.ipynb) |
| | | **Smoothing, linear** (run B) | `ablation-smoothing-schedule/b-smoothing-linear` | |
| | | **Smoothing, linear + augmentation** (run E, the best fine-tuned model) | `ablation-smoothing-schedule/e-smoothing-linear-augmentation` | |
| 3 | [`train/ticha_trocr_beams_r2l_rerank.ipynb`](notebooks/train/ticha_trocr_beams_r2l_rerank.ipynb) | **Right-to-left reranker** | `trocr-base-handwritten-finetuned-smooth-cosine-aug-r2l-th687` | [outputs](notebooks/train/trained/ticha_trocr_beams_r2l_rerank.with-outputs.ipynb) |
| | | **Run E with 5, 8 or 10 beams and with reranking**; the best system is run E + 10 beams + right-to-left rerank (new decodings of run E, not new models) | `beams-r2l-rerank-linear-aug-th687` | |
| 4 | [`reports/ticha_trocr_evaluation.ipynb`](notebooks/reports/ticha_trocr_evaluation.ipynb) | **Pretrained (no fine-tuning)**: `microsoft/trocr-base-handwritten`, scored zero-shot | `evaluation/` | |
| 5 | [`reports/ticha_report_metrics.ipynb`](notebooks/reports/ticha_report_metrics.ipynb) | None: collects the numbers, tables and figures of the report | `data/evaluation/report_metrics/` in the repository | |

The fine-tuning notebook also trains two runs that the ablation uses as
reference points but the report's table leaves out: the 5-epoch first run
(`trocr-base-handwritten-finetuned`) and smoothing + cosine + augmentation
(`trocr-base-handwritten-finetuned-smooth-cosine-aug`). The ablation also trains
run C (no smoothing, cosine schedule). The models that read new lines,
[`maclenn77/ticha-trocr-base`](https://huggingface.co/maclenn77/ticha-trocr-base),
are published, so [`ticha_trocr_usage_example.ipynb`](notebooks/ticha_trocr_usage_example.ipynb)
needs no training.

The dataset pipeline, the usage example and the report notebooks use the conda environment
described in [`ENVIRONMENT.md`](ENVIRONMENT.md). The training notebooks run on
a GPU machine with [`requirements-trocr.txt`](requirements-trocr.txt), also
described there.

## Data Structure

The scraper extracts the following columns based on the Ticha website table structure:

- `document_name`: Name of the manuscript
- `document_link`: Link to the full document
- `file_type`: Type of file (PDF, Image, etc.)
- `ticha_id`: Unique Ticha identifier
- `year`: Year of the document
- `town`: Town where document was created
- `archive`: Archive where document is stored
- `doc_type`: Type of document (Testament, Receipt, etc.)
- `language`: Language(s) of the document
- `trptn_status`: Transcription status
- `legibility`: Legibility assessment

### Text content (`texts` command)

`ticha texts -i manuscripts.csv -o texts.csv` adds these columns to
each manuscript row (empty when the site has no such text):

- `transcription`: transcribed text, line breaks preserved
- `interlinear`: interlinear analysis (glosses aligned with spaces)
- `modern_spanish`: modern Spanish normalization
- `spanish_translation`, `english_translation`: translations (featured texts)
- metadata fields from the document page, e.g. `title`, `scribe`,
  `document_type`, `call_number`, `date_digitized`, `town_modern_official`,
  `primary_parties`, `witnesses`, `permission_file`

Featured texts (`/en/text/<id>/<n>/`) are paginated: every page is visited,
the texts are joined in order, metadata comes from their `overview` page, and
`page_count` is added.

## Requirements

- Python 3.10+
- Linux or macOS. CI tests Python 3.10 and 3.12 on Linux and builds the
  conda environment on Linux and macOS; **Windows is not tested**.
- Chrome (for the Selenium-based `manuscripts` and `texts` commands)
- `lines` extra: kraken and PyTorch
- `htr` extra: PyTorch and transformers

## Ethical Usage

This package is designed for academic research purposes. Please:

- Use respectful rate limiting (default 2 seconds between requests)
- Cite the Ticha project in your research
- Follow academic fair use guidelines
- Check robots.txt and terms of service

## Citation

If you use this software, the dataset or the models, please cite them
([`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" gives APA and
BibTeX):

> Pérez-Tejada, J. P. (2026). *ticha: Colonial Zapotec manuscripts from Ticha, a
> line-level HTR dataset and TrOCR models*. Zenodo.
> https://doi.org/10.5281/zenodo.23640278

This DOI always resolves to the latest release; each release also has its own
DOI on Zenodo.

When using data downloaded with this tool, please also cite:

> Ticha: a digital text explorer for Colonial Zapotec. Brook Danielle Lillehaugen, George Aaron Broadwell, et al. https://ticha.haverford.edu/

## License

MIT License - see LICENSE file for details.
