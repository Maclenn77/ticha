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
