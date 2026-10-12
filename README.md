# Ticha

A Python package for scraping Colonial Zapotec manuscript data from the Ticha Digital Text Explorer.

## Installation

```bash
pip install ticha-scraper
```

For development:
```bash
git clone https://github.com/maclenn77/ticha.git
cd ticha
pip install -e .[dev]
```

## Usage

### Python API

```python
from ticha_scraper import TichaScraper
import pandas as pd

# Create scraper instance
scraper = TichaScraper(rate_limit=2.0)

# Scrape all manuscript data
df = scraper.scrape_manuscripts()

# Work with the DataFrame
print(f"Found {len(df)} manuscripts")
print(df.head())

# Save to CSV
df.to_csv('manuscripts.csv', index=False)
```

### Command Line Interface

```bash
# Basic usage
ticha-scraper -o manuscripts.csv

# With custom rate limiting
ticha-scraper -o manuscripts.csv -r 3.0

# Verbose output
ticha-scraper -o manuscripts.csv -v

# Run with visible browser (non-headless)
ticha-scraper -o manuscripts.csv --no-headless
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
ticha-scraper images -u https://ticha.haverford.edu/en/texts/Oc111/ -d images

# All documents from the manuscripts CSV, max 3000px per side, plus a PDF per document
ticha-scraper images -i manuscripts.csv -d images --max-dimension 3000 --pdf
```

```python
from ticha import TichaImageScraper

scraper = TichaImageScraper(rate_limit=2.0)
result = scraper.download_document_images(url, "images", image_format="png")
print(result["image_paths"])
```

Images are written to `<output-dir>/<document>/page_001.png`, ...

### Line-level dataset for HTR (`lines` command)

Segments the downloaded page images into lines with
[kraken](https://kraken.re) and pairs each line with its transcription.

```bash
pip install 'ticha[lines]'   # installs kraken (and PyTorch)

ticha-scraper texts  -i manuscripts.csv -o texts.csv
ticha-scraper images -i manuscripts.csv -d images -o images.csv
ticha-scraper lines  -t texts.csv -i images.csv -d dataset --export-lines
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
ticha-scraper review -d data/paired --decisions review/decisions.txt -o data/reviewed
```

#### Reproducing the published dataset

`notebooks/ticha_pipeline.ipynb` runs every step. The pieces that make the
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

`scripts/build_dataset.py` (steps 6-10 of `notebooks/ticha_pipeline.ipynb`)
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
ticha-scraper check -d dataset --only doubtful -n 30            # 30 doubtful lines
ticha-scraper check -d dataset --only confident -s auto -n 45   # what goes to training
ticha-scraper check -d dataset --doc SDE740dT -n 0 -o check/SDE740dT.png  # a whole document
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
ticha-scraper lines   -t texts.csv -i images.csv -d dataset --export-lines
ticha-scraper augment -d dataset -o augmented -n 4 --seed 1
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

### Full pipeline notebook

[`notebooks/ticha_pipeline.ipynb`](notebooks/ticha_pipeline.ipynb) runs every
step, from downloading to the final train/validation/test dataset with
augmentation and metrics, using the code in `src/` directly. Set up its conda
environment as described in [`ENVIRONMENT.md`](ENVIRONMENT.md).

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

`ticha-scraper texts -i manuscripts.csv -o texts.csv` adds these columns to
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

- Python 3.9+
- Chrome browser (for Selenium WebDriver)
- pandas
- selenium

## Ethical Usage

This scraper is designed for academic research purposes. Please:

- Use respectful rate limiting (default 2 seconds between requests)
- Cite the Ticha project in your research
- Follow academic fair use guidelines
- Check robots.txt and terms of service

## Citation

When using data scraped with this tool, please cite:

> Ticha: a digital text explorer for Colonial Zapotec. Brook Danielle Lillehaugen, George Aaron Broadwell, et al. https://ticha.haverford.edu/

## License

MIT License - see LICENSE file for details.
