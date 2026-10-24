# Changelog

## 1.0.0

The package is no longer only a scraper: it now builds a line-level HTR
dataset from the Ticha manuscripts and reads new lines with fine-tuned TrOCR
models. The version reflects this new scope. The 0.2.0 Python API is
unchanged.

### Changed

- The command is now `ticha`. `ticha-scraper` still works but prints a
  deprecation warning, and will be removed in 2.0.
- Python 3.10 or newer is required (3.9 is end of life).
- `TichaTextScraper` was rewritten against the live site: it reads featured
  (paginated) texts, translations, interlinear analysis and more document
  metadata.
- New package description, keywords and classifiers.

### Added

- `ticha images`: rebuilds page images from Ticha's Deep Zoom tiles
  (`TichaImageScraper`), with optional PDFs.
- `ticha printed`: transcriptions of the printed works (`TichaPrintedScraper`).
- `ticha lines`: kraken segmentation, alignment of image lines with their
  transcription and confidence sorting into PAGE XML (`ticha.core.line_pairing`,
  `lines` extra).
- `ticha check` and `ticha review`: contact sheets to check the pairing by eye,
  and a decisions file to apply a line-by-line review.
- `ticha augment`: degraded and varied copies of the lines
  (`ticha.core.augmentation`).
- `ticha.core.htr`: `LineReader` for the fine-tuned TrOCR models with
  right-to-left reranking, and `clean_text`, the transcription rule they were
  trained with (`htr` extra).
- `scripts/build_dataset.py` and `scripts/export_hf_dataset.py`: the
  train / validation / test splits and their export to the Hugging Face Hub.
- Notebooks for the data pipeline, TrOCR training, evaluation and report
  metrics.

### Fixed

- README: install instructions (the package is not on PyPI), the Python import
  path, CLI examples without a subcommand and links to a notebook that did not
  exist.

## 0.2.0

- Manuscript list scraper (`TichaScraper`) and first text scraper.
