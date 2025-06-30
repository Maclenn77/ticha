"""Ticha Colonial Zapotec manuscript scraper."""

__version__ = "0.1.0"

from .core.scraper import TichaScraper
from .core.text_scraper import TichaTextScraper

__all__ = ["TichaScraper", "TichaTextScraper"]
