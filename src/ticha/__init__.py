"""Colonial Zapotec manuscripts from Ticha: scraping, HTR dataset and recognition."""

__version__ = "1.0.0"

__all__ = ["TichaScraper", "TichaTextScraper", "TichaImageScraper"]

# The scrapers are imported on first use, so modules that don't need Selenium
# (e.g. ticha.core.htr) can be imported without it.
_SCRAPERS = {
    "TichaScraper": ".core.scraper",
    "TichaTextScraper": ".core.text_scraper",
    "TichaImageScraper": ".core.image_scraper",
}


def __getattr__(name):
    if name in _SCRAPERS:
        from importlib import import_module

        return getattr(import_module(_SCRAPERS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
