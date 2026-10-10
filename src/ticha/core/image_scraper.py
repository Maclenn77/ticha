"""Image scraper for Ticha documents.

Ticha shows manuscript pages with OpenSeadragon using Deep Zoom Images (DZI):
an XML descriptor (``/images/dzi/<ticha_id>_<page>.dzi``) plus a pyramid of
tiles stored at ``<ticha_id>_<page>_files/{level}/{col}_{row}.jpg``. The PDF
links on the site return 500, so this module rebuilds each page by downloading
the tiles of one pyramid level and stitching them into a single image
(PNG/JPEG/TIFF), and can optionally bundle the pages into a local PDF.
"""

import json
import logging
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

logger = logging.getLogger(__name__)

# Manuscript scans are large; let Pillow handle them without warnings.
Image.MAX_IMAGE_PIXELS = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# A quoted JS string ending in .dzi, optionally built by concatenating string
# literals, e.g. ``"/images/dzi/" + "Ma733_1r.dzi"``.
DZI_EXPR_RE = re.compile(
    r"""((?:["'][^"'\n]*["']\s*\+\s*)*["'][^"'\n]*\.dzi(?:\?[^"'\n]*)?["'])""",
    re.I,
)
JS_STRING_RE = re.compile(r"""["']([^"'\n]*)["']""")
# ``const pageMap = {"01r": "0", "01v": "1"}`` on /<lang>/text/<id>/<n>/ pages
PAGE_MAP_RE = re.compile(r"pageMap\s*=\s*(\{[^}]*\})")
TICHA_ID_RE = re.compile(r"/texts?/([^/]+?)(?:\.pdf)?(?:/|$)")
DZI_PREFIX = "/images/dzi/"
# Inline tile sources: tileSources: { Image: { ... } }
INLINE_DZI_RE = re.compile(r"""\{\s*["']?Image["']?\s*:\s*\{""")


@dataclass
class DZIInfo:
    """Parsed Deep Zoom descriptor."""

    url: str
    width: int
    height: int
    tile_size: int
    overlap: int
    format: str
    tiles_url: str

    @property
    def max_level(self) -> int:
        """Level holding the full-resolution image."""
        return math.ceil(math.log2(max(self.width, self.height, 1)))

    def level_size(self, level: int) -> tuple:
        """Width and height of the image at a pyramid level."""
        scale = 2 ** (self.max_level - level)
        return (
            max(1, math.ceil(self.width / scale)),
            max(1, math.ceil(self.height / scale)),
        )

    def level_for_max_dimension(self, max_dimension: Optional[int]) -> int:
        """Highest level whose largest side does not exceed max_dimension."""
        if not max_dimension:
            return self.max_level
        for level in range(self.max_level, -1, -1):
            if max(self.level_size(level)) <= max_dimension:
                return level
        return 0

    def tile_url(self, level: int, col: int, row: int) -> str:
        return f"{self.tiles_url}{level}/{col}_{row}.{self.format}"


def _tiles_url_for(dzi_url: str) -> str:
    """``.../page.dzi`` -> ``.../page_files/``."""
    base = dzi_url.split("?", 1)[0]
    base = re.sub(r"\.(dzi|xml)$", "", base, flags=re.I)
    return f"{base}_files/"


def parse_dzi(xml_content: Any, dzi_url: str) -> DZIInfo:
    """Parse a DZI XML descriptor."""
    root = ET.fromstring(xml_content)
    # Strip the namespace ({http://schemas.microsoft.com/deepzoom/2008}Image)
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    size = root.find("Size")
    if root.tag != "Image" or size is None:
        raise ValueError(f"Not a DZI descriptor: {dzi_url}")
    return DZIInfo(
        url=dzi_url,
        width=int(size.get("Width")),
        height=int(size.get("Height")),
        tile_size=int(root.get("TileSize")),
        overlap=int(root.get("Overlap", 0)),
        format=root.get("Format", "jpg"),
        tiles_url=_tiles_url_for(dzi_url),
    )


def parse_inline_dzi(obj: Dict[str, Any], page_url: str) -> DZIInfo:
    """Parse an inline OpenSeadragon tile source ``{Image: {Url, Size, ...}}``."""
    image = obj["Image"]
    size = image["Size"]
    tiles_url = urljoin(page_url, image["Url"])
    if not tiles_url.endswith("/"):
        tiles_url += "/"
    return DZIInfo(
        url=tiles_url,
        width=int(size["Width"]),
        height=int(size["Height"]),
        tile_size=int(image["TileSize"]),
        overlap=int(image.get("Overlap", 0)),
        format=image.get("Format", "jpg"),
        tiles_url=tiles_url,
    )


def _extract_braced(text: str, start: int) -> Optional[str]:
    """Return the balanced ``{...}`` block that starts at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _js_object_to_json(block: str) -> str:
    """Loosely convert a JS object literal to JSON (quote keys, fix quotes)."""
    block = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', block)
    block = block.replace("'", '"')
    return re.sub(r",\s*([}\]])", r"\1", block)


def _page_range_to_list(pages: str) -> List[str]:
    """
    Expand a Ticha ``data-pages`` value: ranges and comma-separated parts.

    ``"788-789, 791"`` -> ``["788", "789", "791"]``. (Ticha's own viewer
    ignores everything after the comma, but those images exist.)
    """
    result: List[str] = []
    for part in pages.split(","):
        if part.strip():
            result.extend(_expand_page_range(part))
    return result


def _expand_page_range(pages: str) -> List[str]:
    """
    Expand one ``data-pages`` range exactly like ``call_the_dragon.js``.

    ``"09r-10v"`` -> ``["9r", "9v", "10r", "10v"]``; ``"1-3"`` -> ``["1", "2", "3"]``.
    A single page is kept verbatim. As in the JS, ranges go through
    ``parseInt``, which drops leading zeros (the files on the server match).
    """
    split = pages.strip().split("-")
    if len(split) == 1:
        return split
    first, last = split[0], split[1]
    lead = re.compile(r"^\s*(\d+)")

    def to_int(value: str) -> int:
        match = lead.match(value)
        if not match:
            raise ValueError(f"Invalid page range: {pages}")
        return int(match.group(1))

    if first.endswith(("r", "v")):
        start = to_int(first) * 2 + (1 if first.endswith("v") else 0)
        end = to_int(last) * 2 + (1 if last.endswith("v") else 0)
        return [f"{n // 2}{'v' if n % 2 else 'r'}" for n in range(start, end + 1)]
    return [str(n) for n in range(to_int(first), to_int(last) + 1)]


def _zero_stripped(dzi_url: str) -> Optional[str]:
    """``.../Oc111_09r.dzi`` -> ``.../Oc111_9r.dzi`` (None if unchanged)."""
    alt = re.sub(r"_0+(\d+[rv]?\.dzi)$", r"_\1", dzi_url)
    return alt if alt != dzi_url else None


def ticha_id_from_url(url: str) -> Optional[str]:
    """Extract the Ticha id from ``/en/texts/Oc111/`` or ``/en/text/Ma733/0/``."""
    match = TICHA_ID_RE.search(url)
    return match.group(1) if match else None


def folio_from_source(source: Any, ticha_id: str = "") -> str:
    """Folio of a DZI source: ``/images/dzi/Ma733T_1v.dzi`` -> ``1v``."""
    url = source.url if isinstance(source, DZIInfo) else str(source)
    name = re.sub(r"(_files/?|\.dzi)(\?.*)?$", "", url.rstrip("/")).rsplit("/", 1)[-1]
    if ticha_id and name.startswith(f"{ticha_id}_"):
        return name[len(ticha_id) + 1 :]
    return name.rsplit("_", 1)[-1]


class TichaImageScraper:
    """Download page images of Ticha documents from their DZI tile pyramids."""

    def __init__(
        self,
        rate_limit: float = 2.0,
        tile_delay: float = 0.05,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
        retries: int = 3,
    ):
        """
        Initialize the image scraper.

        Args:
            rate_limit: Delay between documents in seconds
            tile_delay: Delay between tile requests in seconds
            session: Optional requests session (useful for testing)
            timeout: Request timeout in seconds
            retries: Attempts per request before giving up
        """
        self.rate_limit = rate_limit
        self.tile_delay = tile_delay
        self.timeout = timeout
        self.retries = retries
        self.base_url = "https://ticha.haverford.edu"
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    def _intelligent_delay(self):
        """Add intelligent delay with randomization."""
        jitter = random.uniform(0.8, 1.2)
        time.sleep(self.rate_limit * jitter)

    def _get(self, url: str, retries: Optional[int] = None) -> requests.Response:
        """GET with retries on network errors and 5xx responses."""
        last_error: Optional[Exception] = None
        retries = self.retries if retries is None else retries
        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = requests.HTTPError(
                    f"{response.status_code} for {url}", response=response
                )
            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                last_error = e
            if attempt < retries:
                time.sleep(2**attempt * 0.5)
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def find_image_sources(self, html: Any, page_url: str) -> List[Any]:
        """
        Find the DZI sources referenced in a document page.

        Returns a list in page order of absolute ``.dzi`` URLs (str) or
        already-parsed inline sources (DZIInfo).
        """
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        sources: List[Any] = []
        seen = set()

        def add_url(raw: str):
            url = urljoin(page_url, raw.strip())
            if url not in seen:
                seen.add(url)
                sources.append(url)

        # Handwritten texts: <script id="call_the_dragon" data-ticha-id="Oc111"
        # data-pages="09r-09v">, expanded by call_the_dragon.js into
        # /images/dzi/Oc111_9r.dzi, /images/dzi/Oc111_9v.dzi
        for tag in soup.find_all(attrs={"data-ticha-id": True, "data-pages": True}):
            try:
                pages = _page_range_to_list(tag["data-pages"])
            except ValueError as e:
                logger.warning(str(e))
                continue
            for page in pages:
                add_url(f"{DZI_PREFIX}{tag['data-ticha-id']}_{page}.dzi")

        # Attributes (href, src, data-*) that point to a .dzi file
        for tag in soup.find_all(True):
            for value in tag.attrs.values():
                values = value if isinstance(value, list) else [value]
                for v in values:
                    if isinstance(v, str) and re.search(r"\.dzi(\?|$)", v, re.I):
                        add_url(v)

        # Quoted .dzi URLs inside scripts (OpenSeadragon tileSources)
        for script in soup.find_all("script"):
            text = script.string or script.get_text() or ""
            for match in DZI_EXPR_RE.finditer(text):
                url = "".join(JS_STRING_RE.findall(match.group(1)))
                add_url(url.replace("\\/", "/"))
            for match in INLINE_DZI_RE.finditer(text):
                block = _extract_braced(text, match.start())
                if not block:
                    continue
                try:
                    info = parse_inline_dzi(
                        json.loads(_js_object_to_json(block)), page_url
                    )
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Could not parse inline tile source: {e}")
                    continue
                if info.tiles_url not in seen:
                    seen.add(info.tiles_url)
                    sources.append(info)

        logger.info(f"Found {len(sources)} DZI sources in {page_url}")
        return sources

    def find_page_urls(self, html: Any, page_url: str) -> List[str]:
        """
        List every page URL of a paginated ``/<lang>/text/<id>/<n>/`` document.

        These pages show one image each and embed ``pageMap`` (folio -> index).
        Returns ``[page_url]`` for documents that are not paginated this way.
        """
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        match = PAGE_MAP_RE.search(html)
        if not match or not re.search(r"/\d+/?$", page_url):
            return [page_url]
        try:
            page_map = json.loads(_js_object_to_json(match.group(1)))
        except ValueError:
            return [page_url]
        indexes = sorted({int(v) for v in page_map.values() if str(v).isdigit()})
        if not indexes:
            return [page_url]
        return [re.sub(r"/\d+/?$", f"/{i}/", page_url) for i in indexes]

    def find_pdf_links(self, html: Any, page_url: str) -> List[str]:
        """Find links to PDF files (``*.pdf`` or ``download="*.pdf"``)."""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            if "permissions" in a["href"]:
                continue
            if re.search(r"\.pdf(\?|$)", a["href"], re.I) or re.search(
                r"\.pdf$", a.get("download") or "", re.I
            ):
                url = urljoin(page_url, a["href"])
                if url not in links:
                    links.append(url)
        return links

    def load_dzi(self, source: Any) -> DZIInfo:
        """Return a DZIInfo, fetching and parsing the descriptor if needed."""
        if isinstance(source, DZIInfo):
            return source
        try:
            return parse_dzi(self._get(source).content, source)
        except requests.HTTPError as e:
            alt = _zero_stripped(source)
            status = getattr(e.response, "status_code", None)
            if alt and status == 404:
                logger.debug(f"{source} not found, trying {alt}")
                return parse_dzi(self._get(alt).content, alt)
            raise

    # ------------------------------------------------------------------ #
    # Download + stitching
    # ------------------------------------------------------------------ #

    def stitch_dzi(self, info: DZIInfo, level: Optional[int] = None) -> Image.Image:
        """
        Download all tiles of one pyramid level and stitch them together.

        Args:
            info: Parsed DZI descriptor
            level: Pyramid level (default: full resolution)
        """
        level = info.max_level if level is None else level
        width, height = info.level_size(level)
        cols = math.ceil(width / info.tile_size)
        rows = math.ceil(height / info.tile_size)
        logger.info(
            f"Stitching {cols}x{rows} tiles at level {level} ({width}x{height}px)"
        )

        canvas = Image.new("RGB", (width, height), "white")
        for row in range(rows):
            for col in range(cols):
                url = info.tile_url(level, col, row)
                tile = Image.open(BytesIO(self._get(url).content))
                tile.load()
                if tile.mode != "RGB":
                    tile = tile.convert("RGB")
                # Tiles beyond the first row/column carry `overlap` extra
                # pixels on their left/top edge.
                x = col * info.tile_size - (info.overlap if col else 0)
                y = row * info.tile_size - (info.overlap if row else 0)
                canvas.paste(tile, (x, y))
                if self.tile_delay:
                    time.sleep(self.tile_delay)
        return canvas

    def download_dzi(
        self,
        source: Any,
        output_path: Any,
        level: Optional[int] = None,
        max_dimension: Optional[int] = None,
    ) -> Path:
        """
        Download a DZI image and save it as a single file.

        The output format comes from the file extension (.png, .jpg, .tif).
        """
        info = self.load_dzi(source)
        if level is None:
            level = info.level_for_max_dimension(max_dimension)
        image = self.stitch_dzi(info, level)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"quality": 95} if output_path.suffix.lower() in (
            ".jpg",
            ".jpeg",
        ) else {}
        image.save(output_path, **save_kwargs)
        logger.info(f"Saved {output_path}")
        return output_path

    def _download_pdf(self, pdf_url: str, output_path: Path) -> Optional[Path]:
        """Try to download an original PDF; return None if the server fails."""
        try:
            # The site answers 500 for most PDFs; don't insist.
            response = self._get(pdf_url, retries=1)
        except requests.RequestException as e:
            logger.warning(f"PDF unavailable ({e}); falling back to DZI images")
            return None
        if not response.content.startswith(b"%PDF"):
            logger.warning(f"{pdf_url} did not return a PDF; falling back to DZI")
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path

    @staticmethod
    def images_to_pdf(image_paths: List[Path], output_path: Any) -> Path:
        """Combine page images into a single PDF."""
        output_path = Path(output_path)
        pages = [Image.open(p).convert("RGB") for p in image_paths]
        pages[0].save(output_path, save_all=True, append_images=pages[1:])
        return output_path

    def download_document_images(
        self,
        document_url: str,
        output_dir: Any,
        image_format: str = "png",
        max_dimension: Optional[int] = None,
        make_pdf: bool = False,
        try_pdf: bool = True,
    ) -> Dict[str, Any]:
        """
        Download every page image of a document.

        Args:
            document_url: URL of the document page
            output_dir: Directory where images are written
            image_format: png, jpg or tif
            max_dimension: Limit the largest side of each image (None = full res)
            make_pdf: Also bundle the downloaded pages into a PDF
            try_pdf: Try the original PDF link first

        Returns:
            Dictionary with url, image_paths, pdf_path and error (if any)
        """
        if not document_url.startswith("http"):
            document_url = urljoin(self.base_url, document_url)
        result: Dict[str, Any] = {
            "url": document_url,
            "image_paths": [],
            "image_folios": [],
            "pdf_path": None,
            "error": None,
        }
        try:
            html = self._get(document_url).content
            output_dir = Path(output_dir)
            stem = re.sub(
                r"[^\w\-]+",
                "_",
                ticha_id_from_url(document_url)
                or document_url.rstrip("/").split("/")[-1],
            ) or "document"

            if try_pdf:
                for pdf_url in self.find_pdf_links(html, document_url):
                    pdf = self._download_pdf(pdf_url, output_dir / f"{stem}.pdf")
                    if pdf:
                        result["pdf_path"] = str(pdf)
                        break

            sources = []
            for page_url in self.find_page_urls(html, document_url):
                page_html = (
                    html if page_url == document_url else self._get(page_url).content
                )
                for source in self.find_image_sources(page_html, page_url):
                    if source not in sources:
                        sources.append(source)
            if not sources and not result["pdf_path"]:
                result["error"] = "No DZI images or PDF found"
                return result

            for page, source in enumerate(sources, 1):
                path = output_dir / stem / f"page_{page:03d}.{image_format}"
                try:
                    self.download_dzi(source, path, max_dimension=max_dimension)
                    result["image_paths"].append(str(path))
                    result["image_folios"].append(folio_from_source(source, stem))
                except Exception as e:
                    logger.error(f"Failed page {page} of {document_url}: {e}")

            if make_pdf and result["image_paths"] and not result["pdf_path"]:
                pdf = self.images_to_pdf(
                    [Path(p) for p in result["image_paths"]],
                    output_dir / f"{stem}.pdf",
                )
                result["pdf_path"] = str(pdf)
        except Exception as e:
            logger.error(f"Error downloading images for {document_url}: {e}")
            result["error"] = str(e)
        finally:
            self._intelligent_delay()
        return result

    def download_images_from_dataframe(
        self,
        df: pd.DataFrame,
        output_dir: Any,
        link_column: str = "document_link",
        max_documents: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Download images for every document in a DataFrame of links."""
        valid_df = df[df[link_column].notna() & (df[link_column] != "")]
        if max_documents:
            valid_df = valid_df.head(max_documents)

        rows = []
        total = len(valid_df)
        for idx, (_, row) in enumerate(valid_df.iterrows(), 1):
            logger.info(f"Processing document {idx}/{total}: {row[link_column]}")
            result = self.download_document_images(
                row[link_column], output_dir, **kwargs
            )
            result["image_count"] = len(result["image_paths"])
            result["image_paths"] = ";".join(result["image_paths"])
            result["image_folios"] = ";".join(result["image_folios"])
            rows.append({**row.to_dict(), **result})
        return pd.DataFrame(rows)
