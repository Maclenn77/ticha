"""Text-only scraper for Ticha's printed works (artes, catechisms, sermons).

The printed works are listed on ``/en/texts/`` with links like
``/en/texts/<slug>/0/original/``. Each page's text is served as JSON at
``/texts/<slug>/<n>/json`` (``transcription`` and ``transcription_regular``,
both HTML). Agüero's *Misceláneo* leaves the JSON empty and puts the text
in the ``tab1-slug`` div of the page itself, so the page is read instead.

Only text is downloaded: no images, no PDFs. Words analysed by Ticha are
wrapped in popovers (``mark.trigger`` + a hidden gloss table); the gloss is
dropped and the marked words are also kept apart, since in the artes they
are the Zapotec forms inside Spanish prose.
"""

import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .image_scraper import USER_AGENT
from .text_scraper import element_to_text

logger = logging.getLogger(__name__)

BASE_URL = "https://ticha.haverford.edu"
INDEX_URL = f"{BASE_URL}/en/texts/"
WORK_LINK_RE = re.compile(r"/texts/([\w-]+)/0/original/?$")
VIEWER_INIT_RE = re.compile(r"documentViewerInit\(\s*'([^']+)'\s*,\s*\d+\s*,\s*(\d+)")
PAGE_COLUMNS = [
    "slug",
    "work",
    "page",
    "page_label",
    "url",
    "transcription",
    "transcription_regular",
    "zapotec_marked",
]


def html_to_text(html: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Plain text of a printed page plus the words Ticha marked for analysis.

    Gloss tables, catchwords (the first word of the next page, printed at
    the foot of each page) and the crowdsourcing form shown on untranscribed
    pages are removed; headings become their own paragraph.
    """
    if not html or not html.strip():
        return {"text": None, "marked": None}
    soup = BeautifulSoup(html, "html.parser")
    for hidden in soup.select(
        "span.content.hide, div.catch, #submitTranscription, " "form, script, style"
    ):
        hidden.decompose()
    marked = [m.get_text(" ", strip=True) for m in soup.select("mark.trigger")]
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading.name = "div"
    # Agüero's pages are several sibling <div>s, not one wrapper
    return {
        "text": element_to_text(soup),
        "marked": "\n".join(re.sub(r"\s+", " ", m) for m in marked if m) or None,
    }


class TichaPrintedScraper:
    """Download the transcriptions of Ticha's printed works, page by page."""

    def __init__(
        self,
        rate_limit: float = 2.0,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
    ):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    def _get(self, url: str) -> Optional[requests.Response]:
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            logger.error(
                f"Failed to access {url} (status code: {response.status_code})"
            )
            return None
        return response

    def _delay(self):
        time.sleep(self.rate_limit * random.uniform(0.8, 1.2))

    def list_works(self) -> pd.DataFrame:
        """Printed works linked from the Ticha texts menu: slug, title, viewer URL."""
        response = self._get(INDEX_URL)
        if response is None:
            raise ValueError(f"Could not load {INDEX_URL}")
        soup = BeautifulSoup(response.content, "html.parser")
        works: Dict[str, Dict[str, str]] = {}
        for link in soup.find_all("a", href=True):
            match = WORK_LINK_RE.search(link["href"])
            if not match or match.group(1) in works:
                continue
            # The menu entry is "<li class=dropdown><a>Title</a><ul>...Text link"
            dropdown = link.find_parent("li", class_="dropdown")
            toggle = dropdown.find("a", class_="dropdown-toggle") if dropdown else None
            title = toggle.get_text(" ", strip=True) if toggle else match.group(1)
            works[match.group(1)] = {
                "slug": match.group(1),
                "title": title,
                "url": urljoin(BASE_URL, link["href"]),
            }
        return pd.DataFrame(list(works.values()), columns=["slug", "title", "url"])

    def _page_url(self, slug: str, page: int) -> str:
        return f"{BASE_URL}/en/texts/{slug}/{page}/original/"

    def last_page(self, slug: str) -> int:
        """Number of the work's last page, read from the viewer's init call."""
        response = self._get(self._page_url(slug, 0))
        if response is None:
            raise ValueError(f"Could not load the first page of {slug}")
        match = VIEWER_INIT_RE.search(response.text)
        if not match:
            raise ValueError(f"No page count found for {slug}")
        return int(match.group(2))

    def scrape_page(self, slug: str, page: int, work: str = "") -> Dict[str, Any]:
        """Text of one page (JSON endpoint, or the page HTML when that is empty)."""
        row: Dict[str, Any] = {
            "slug": slug,
            "work": work or slug,
            "page": page,
            "page_label": None,
            "url": self._page_url(slug, page),
        }
        data: Dict[str, Any] = {}
        response = self._get(f"{BASE_URL}/texts/{slug}/{page}/json")
        if response is not None:
            try:
                data = response.json()
            except ValueError:
                logger.warning(f"Invalid JSON for {slug} page {page}")
        row["page_label"] = data.get("n")

        original = html_to_text(data.get("transcription"))
        regular = html_to_text(data.get("transcription_regular"))
        if original["text"] is None:
            self._delay()
            page_response = self._get(row["url"])
            if page_response is not None:
                tab = BeautifulSoup(page_response.content, "html.parser").find(
                    "div", id="tab1-slug"
                )
                if tab is not None:
                    original = html_to_text(str(tab))

        row["transcription"] = original["text"]
        row["transcription_regular"] = regular["text"]
        row["zapotec_marked"] = original["marked"] or regular["marked"]
        return row

    def iter_pages(
        self,
        slug: str,
        work: str = "",
        skip: Optional[set] = None,
        max_pages: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield every page of a work, skipping page numbers in ``skip``."""
        last = self.last_page(slug)
        pages = [p for p in range(last + 1) if p not in (skip or set())]
        if max_pages is not None:
            pages = pages[:max_pages]
        logger.info(f"{slug}: {last + 1} pages, {len(pages)} to download")
        for page in pages:
            self._delay()
            try:
                yield self.scrape_page(slug, page, work)
            except requests.RequestException as e:
                logger.error(f"Error on {slug} page {page}: {e}")

    def scrape(
        self,
        output: Path,
        works: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
        save_every: int = 20,
    ) -> pd.DataFrame:
        """
        Download the selected works (all by default) into one CSV, one row per page.

        Pages already in ``output`` are skipped, so an interrupted run resumes.
        """
        output = Path(output)
        done = (
            pd.read_csv(output)
            if output.exists()
            else pd.DataFrame(columns=PAGE_COLUMNS)
        )
        rows = done.to_dict("records")
        catalogue = self.list_works()
        if works:
            unknown = set(works) - set(catalogue["slug"])
            if unknown:
                raise ValueError(
                    f"Unknown works: {sorted(unknown)}; "
                    f"available: {catalogue['slug'].tolist()}"
                )
            catalogue = catalogue[catalogue["slug"].isin(works)]

        def save():
            frame = pd.DataFrame(rows, columns=PAGE_COLUMNS).sort_values(
                ["slug", "page"]
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)
            return frame

        for work in catalogue.itertuples():
            skip = set(done.loc[done["slug"] == work.slug, "page"].astype(int))
            for i, row in enumerate(
                self.iter_pages(work.slug, work.title, skip, max_pages), 1
            ):
                rows.append(row)
                if i % save_every == 0:
                    save()
                    logger.info(f"{work.slug}: {i} pages downloaded")
            save()
        return save()

    def close(self):
        self.session.close()


def write_corpus(
    pages: pd.DataFrame, output_dir: Path, column: str = "transcription"
) -> List[Path]:
    """Join each work's pages, in order, into ``<output_dir>/<slug>.txt``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for slug, group in pages.sort_values("page").groupby("slug"):
        texts = group[column].dropna().astype(str)
        path = output_dir / f"{slug}.txt"
        path.write_text("\n\n".join(texts) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
