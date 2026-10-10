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
4. **Alignment**: image lines and transcription lines are aligned in order;
   image lines may stay without text (stamped-paper headers, untranscribed
   signatures) and transcription lines may stay without an image line. Each
   pair gets a confidence `1 - CER` between reading and transcription
   (`--cer-threshold`, default 0.5). Pages are aligned on their own when the
   transcription's folio markers (`(8r)`, `[40r]`) match the image folios,
   otherwise the whole document at once.
5. **Sorting**: pages where every transcription line found a confident line
   go to `auto/`. **Nothing is discarded**: every other page goes to
   `review/`, pre-filled with the best alignment, with the transcription
   lines assigned to it in a `.txt`. The reason is in `manifest.csv`.

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
