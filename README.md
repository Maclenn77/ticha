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
