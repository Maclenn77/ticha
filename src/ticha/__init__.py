"""Ticha Colonial Zapotec manuscript scraper."""

__version__ = "0.2.0"

from .core.scraper import TichaScraper
from .core.text_scraper import TichaTextScraper
from .core.image_scraper import TichaImageScraper

__all__ = ["TichaScraper", "TichaTextScraper", "TichaImageScraper"]
