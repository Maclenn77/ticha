"""Text and metadata scraper for individual Ticha documents.

Ticha has two document layouts:

- Handwritten texts (``/<lang>/texts/<id>/``, the ``document_link`` values of
  the manuscripts table): one page with ``metadata``, ``transcription``,
  ``interlinear`` and ``modern_spanish`` tabs.
- Featured texts (``/<lang>/text/<id>/<n>/``): metadata lives on
  ``/<lang>/text/<id>/overview/`` and every page ``<n>`` has its own
  ``transcription-tab``, ``analysis-tab``, ``normalized-spanish-tab``,
  ``spanish-translation-tab`` and ``english-translation-tab``. Pages are
  joined in order.
"""

import logging
import random
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from .image_scraper import USER_AGENT, TichaImageScraper

logger = logging.getLogger(__name__)

# Output column -> div id, per layout
HANDWRITTEN_FIELDS = {
    "transcription": "transcription",
    "interlinear": "interlinear",
    "modern_spanish": "modern_spanish",
}
FEATURED_FIELDS = {
    "transcription": "transcription-tab",
    "interlinear": "analysis-tab",
    "modern_spanish": "normalized-spanish-tab",
    "spanish_translation": "spanish-translation-tab",
    "english_translation": "english-translation-tab",
}
TEXT_FIELDS = list(FEATURED_FIELDS)

# "This document does not currently have an interlinear analysis available."
PLACEHOLDER_RE = re.compile(r"^this document does not currently have", re.I)
FEATURED_URL_RE = re.compile(r"^(.*/text/[^/]+/)(?:overview|\d+)/?$")
SOURCE_WHITESPACE_RE = re.compile(r"[ \t\r\n\f\v]+")
# Microsoft Word "conditional comments" pasted into some transcriptions
# (``[if gte mso 9]><xml>...<![endif]``, ``[if !supportLineBreakNewLine]``)
OFFICE_MARKUP_RE = re.compile(r"<?!?\[if [^\]]*\].*?\[endif\]>?|<!--.*?-->", re.S)


def strip_office_markup(text: Optional[str]) -> Optional[str]:
    """Remove Word conditional-comment blocks left in a transcription."""
    if not isinstance(text, str):
        return text
    cleaned = OFFICE_MARKUP_RE.sub("", text)
    cleaned = re.sub(r"<br\s*/?>|\b(?:Start|End)Fragment\b", "", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    if cleaned == text:
        return text
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None


def element_to_text(element: Optional[Tag]) -> Optional[str]:
    """
    Convert an HTML element to plain text, keeping its line structure.

    Markup with ``<br>``/``<p>`` is rendered like a browser would (source
    newlines are whitespace, ``<br>`` is a line break, ``<p>`` a paragraph);
    plain text is kept line by line. Non-breaking spaces, used to align the
    interlinear glosses, are preserved as spaces. Ticha's "not currently
    available" placeholders become None.
    """
    if element is None:
        return None
    if element.find(["br", "p", "div"]):
        for string in element.find_all(string=True):
            string.replace_with(SOURCE_WHITESPACE_RE.sub(" ", string))
        for br in element.find_all("br"):
            # unwrap, not replace: html.parser in bs4 < 4.15 can nest the rest
            # of the text inside a <br/> that follows a bare <br>
            br.insert_before("\n")
            br.unwrap()
        for block in element.find_all(["p", "div"]):
            block.insert_before("\n\n")
            block.append("\n\n")
    text = element.get_text()

    lines = [
        re.sub(r"[ \t\r\f\v]+", " ", line).strip().replace("\xa0", " ").rstrip()
        for line in text.split("\n")
    ]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    text = strip_office_markup(text)
    if not text or PLACEHOLDER_RE.match(text):
        return None
    return text


class TichaTextScraper:
    """Scraper for individual Ticha document texts and metadata."""

    def __init__(
        self,
        rate_limit: float = 2.0,
        headless: bool = True,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the text scraper.

        Args:
            rate_limit: Delay between requests in seconds
            headless: Kept for backwards compatibility (no browser is used)
            session: Optional requests session (useful for testing)
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.headless = headless
        self.timeout = timeout
        self.base_url = "https://ticha.haverford.edu"
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    def _get_page(self, document_url: str) -> Optional[bytes]:
        """Load a document page."""
        response = self.session.get(document_url, timeout=self.timeout)

        if response.status_code != 200:
            logger.error(
                f"Failed to access document URL: {document_url} "
                f"(status code: {response.status_code})"
            )
            return None

        return response.content

    def _intelligent_delay(self):
        """Add intelligent delay with randomization."""
        jitter = random.uniform(0.8, 1.2)
        time.sleep(self.rate_limit * jitter)

    def _normalize_metadata_key(self, key: str) -> str:
        """
        Normalize metadata keys to lowercase with underscores.

        Args:
            key: Original metadata key

        Returns:
            Normalized key (lowercase, underscores instead of spaces)
        """
        # Remove extra whitespace and convert to lowercase
        normalized = key.strip().lower()
        # Replace spaces and other separators with underscores
        normalized = re.sub(r'[\s\-\.]+', '_', normalized)
        # Remove parentheses and other special characters
        normalized = re.sub(r'[^\w_]', '', normalized)
        # Clean up multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')

        return normalized

    def _extract_metadata(self, soup: BeautifulSoup, page_url: str = "") -> Dict[str, str]:
        """
        Extract metadata from div with id="metadata".

        Each field is ``<p><span class="meta-title">Key</span> value</p>``;
        ``Key: value`` paragraphs are accepted too.

        Returns:
            Dictionary with normalized metadata keys and values
        """
        metadata: Dict[str, str] = {}
        metadata_div = soup.find('div', id='metadata')
        if metadata_div is None:
            logger.warning("No metadata div found")
            return metadata

        for p_tag in metadata_div.find_all('p'):
            title = p_tag.find('span', class_='meta-title')
            if title:
                key = title.get_text(strip=True)
                title.extract()
                value = p_tag.get_text(" ", strip=True)
            elif ':' in p_tag.get_text():
                key, value = p_tag.get_text().split(':', 1)
            else:
                continue

            value = re.sub(r'\s+', ' ', value).strip()
            # Links shown as "[PDF]" are more useful as their URL
            link = p_tag.find('a', href=True)
            if link and re.fullmatch(r'\[.*\]', value):
                value = urljoin(page_url, link['href'])

            key = self._normalize_metadata_key(key)
            if key and value:
                metadata[key] = value
                logger.debug(f"Extracted metadata: {key} = {value}")

        logger.info(f"Extracted {len(metadata)} metadata fields")
        return metadata

    def _extract_text_content(self, soup: BeautifulSoup, div_id: str) -> Optional[str]:
        """
        Extract the text of a div, omitting HTML tags.

        Args:
            div_id: ID of the div to extract text from

        Returns:
            Clean text content or None if missing/empty/placeholder
        """
        text = element_to_text(soup.find('div', id=div_id))
        if text:
            logger.debug(f"Extracted {len(text)} characters from {div_id}")
        else:
            logger.debug(f"No text content found in {div_id}")
        return text

    def _scrape_featured(self, document_url: str, base: str) -> Dict[str, Any]:
        """Scrape a featured text: overview metadata plus every page's tabs."""
        overview_url = f"{base}overview/"
        overview = self._get_page(overview_url)
        metadata = (
            self._extract_metadata(BeautifulSoup(overview, 'html.parser'), overview_url)
            if overview else {}
        )

        first_url = f"{base}0/"
        first_html = self._get_page(first_url)
        if first_html is None:
            raise ValueError(f"Could not load {first_url}")
        page_urls = TichaImageScraper(rate_limit=0).find_page_urls(first_html, first_url)

        texts: Dict[str, List[str]] = {field: [] for field in FEATURED_FIELDS}
        for page_url in page_urls:
            html = first_html if page_url == first_url else self._get_page(page_url)
            if html is None:
                continue
            soup = BeautifulSoup(html, 'html.parser')
            for field, div_id in FEATURED_FIELDS.items():
                text = self._extract_text_content(soup, div_id)
                if text:
                    texts[field].append(text)
            if page_url != page_urls[-1]:
                self._intelligent_delay()

        return {
            'url': document_url,
            'page_count': len(page_urls),
            **{field: '\n\n'.join(parts) or None for field, parts in texts.items()},
            **metadata,
        }

    def scrape_document(self, document_url: str) -> Dict[str, Any]:
        """
        Scrape a single document's metadata and text content.

        Args:
            document_url: URL of the document to scrape

        Returns:
            Dictionary containing metadata and text content
        """
        try:
            # Ensure we have a full URL
            if not document_url.startswith('http'):
                document_url = urljoin(self.base_url, document_url)

            logger.info(f"Scraping document: {document_url}")

            featured = FEATURED_URL_RE.match(document_url)
            if featured:
                document_data = self._scrape_featured(document_url, featured.group(1))
            else:
                content = self._get_page(document_url)
                if content is None:
                    raise ValueError(f"Could not load {document_url}")
                soup = BeautifulSoup(content, 'html.parser')
                metadata = self._extract_metadata(soup, document_url)
                document_data = {
                    'url': document_url,
                    **{field: None for field in TEXT_FIELDS},
                    **{
                        field: self._extract_text_content(soup, div_id)
                        for field, div_id in HANDWRITTEN_FIELDS.items()
                    },
                    **metadata,  # Unpack metadata into the main dictionary
                }

            # Add rate limiting
            self._intelligent_delay()

            logger.info(f"Successfully scraped document: {document_url}")
            return document_data

        except Exception as e:
            logger.error(f"Error scraping document {document_url}: {e}")
            return {'url': document_url, 'error': str(e)}

    def scrape_documents_from_dataframe(self, df: pd.DataFrame,
                                      link_column: str = 'document_link',
                                      max_documents: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape multiple documents from a DataFrame containing document links.

        Args:
            df: DataFrame containing document links
            link_column: Name of the column containing document links
            max_documents: Maximum number of documents to scrape (None for all)

        Returns:
            DataFrame with scraped text and metadata
        """

        scraped_data = []

        # Filter out rows without links
        valid_df = df[df[link_column].notna() & (df[link_column] != '')].copy()

        if max_documents:
            valid_df = valid_df.head(max_documents)

        total_docs = len(valid_df)
        logger.info(f"Starting to scrape {total_docs} documents")

        for idx, (_, row) in enumerate(valid_df.iterrows(), 1):
            document_url = row[link_column]

            logger.info(f"Processing document {idx}/{total_docs}: {document_url}")

            # Scrape the document
            doc_data = self.scrape_document(document_url)

            # Add original row data to preserve manuscript info
            combined_data = {**row.to_dict(), **doc_data}
            scraped_data.append(combined_data)

            # Progress logging
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{total_docs} documents processed")

        logger.info(f"Completed scraping {len(scraped_data)} documents")

        return pd.DataFrame(scraped_data)

    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.info("Text scraper session closed")
