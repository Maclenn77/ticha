"""Pair segmented manuscript lines with their Ticha transcriptions.

Input: the CSVs written by ``ticha-scraper texts`` (transcriptions) and
``ticha-scraper images`` (page images). Pipeline per document:

1. Clean: keep documents with a transcription (and, optionally, a given
   transcription status).
2. Segment each page image into baselines with kraken (cached as JSON).
   Tiny or squeezed lines (folio numbers, interlinear insertions) are marked
   as fragments; lines are read top to bottom, left to right (signature
   columns row by row, two-page spreads left page first).
3. Recognize each line with a generic HTR model (McCATMuS by default). The
   rough reading is only used to find which transcription line goes where.
4. Align segmented lines and transcription lines with dynamic programming:
   lines may stay without text (stamped-paper headers, untranscribed
   signatures) and transcription lines may stay without a line. Each pair
   gets a confidence from the character error rate (CER) between reading and
   transcription. Pages are aligned on their own when the transcription's
   folio markers match the image folios, otherwise the whole document is
   aligned at once. Without a recognizer, lines are paired in order and a
   page is accepted only when the line counts agree.

Output (``output_dir``):

- ``auto/``: pages where every transcription line was paired confidently,
  as PAGE XML (2019 schema; ``TextEquiv/@conf`` holds the confidence) next
  to a copy of the image. Readable by eScriptorium, Transkribus and
  ``ketos train -f page``.
- ``review/``: every other page, pre-filled with the best alignment, plus a
  ``.txt`` with the transcription lines assigned to it. Nothing is
  discarded: import both into eScriptorium to correct them.
- ``lines/`` (optional): crops + ``.gt.txt`` of confidently paired lines.
- ``manifest.csv`` (one row per page) and ``lines.csv`` (one row per line).
"""

import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

DEFAULT_STATUSES = ("Completed and Checked", "Completed but not Checked")
# McCATMuS: generic HTR model for 16th-21st c. manuscripts (es, fr, la, ...)
DEFAULT_MODEL = "10.5281/zenodo.13788177"

# Folio / page markers at the start of a line: (8r) [40r] (02v) (446) [777]
FOLIO_MARKER_RE = re.compile(r"^\s*[\(\[]\s*(\d{1,4}\s*[rv]?)\s*[\)\]]\s*", re.I)
# Runs of spaces Ticha uses to lay out columns (signatures, witness lists)
COLUMN_GAP_RE = re.compile(r" {4,}")

# A segmenter takes an image path and returns
# {"lines": [{"baseline": [[x, y], ...], "boundary": [[x, y], ...],
#             "regions": [region_id]}],
#  "regions": [{"id": region_id, "boundary": [[x, y], ...]}]}
Segmenter = Callable[[Path], Dict[str, Any]]
# A recognizer takes an image path and lines, and returns one string per line
Recognizer = Callable[[Path, Sequence["Line"]], List[str]]


# ---------------------------------------------------------------------- #
# Transcription handling
# ---------------------------------------------------------------------- #


def split_transcription_pages(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Split a transcription on folio markers.

    Returns ``[(folio, page_text), ...]``. Without markers the whole text is
    one page with folio None. Text before the first marker is kept as its
    own page only if it is not empty.
    """
    pages: List[Tuple[Optional[str], List[str]]] = [(None, [])]
    for line in text.splitlines():
        match = FOLIO_MARKER_RE.match(line)
        if match:
            pages.append((match.group(1).replace(" ", ""), []))
            rest = line[match.end() :]
            if rest.strip():
                pages[-1][1].append(rest)
        else:
            pages[-1][1].append(line)
    result = [(folio, "\n".join(lines).strip()) for folio, lines in pages]
    if not result[0][1] and len(result) > 1:
        result = result[1:]
    return result


def transcription_lines(page_text: str, split_columns: bool = True) -> List[str]:
    """Non-empty lines of a page; column layouts split into separate lines."""
    lines = []
    for line in page_text.splitlines():
        parts = COLUMN_GAP_RE.split(line) if split_columns else [line]
        lines.extend(p.strip() for p in parts if p.strip())
    return lines


def normalize_folio(folio: Optional[str]) -> Optional[str]:
    """``02r`` -> ``2r``, ``(446)`` -> ``446``."""
    if folio is None:
        return None
    folio = re.sub(r"[\s\(\)\[\]]", "", str(folio)).lower()
    return re.sub(r"^0+(?=\d)", "", folio)


def normalize_for_matching(text: str) -> str:
    """
    Reduce a line to what a generic HTR model and Ticha agree on.

    Drops editorial additions (``q[ue]``, ``(...)``, ``(rao?)``), accents,
    case, punctuation and spaces (word separation in the manuscripts is
    irregular), and merges letters used interchangeably (u/v, i/j/y).
    """
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))
    return text.translate(str.maketrans("vjy", "uii"))


def cer(reading: str, reference: str) -> float:
    """Character error rate of a reading against a reference, after normalizing."""
    a, b = normalize_for_matching(reading), normalize_for_matching(reference)
    if not b:
        return 0.0 if not a else 1.0
    try:
        from rapidfuzz.distance import Levenshtein

        distance = Levenshtein.distance(a, b)
    except ImportError:  # pragma: no cover - small fallback
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        distance = prev[-1]
    return distance / len(b)


# ---------------------------------------------------------------------- #
# Line geometry
# ---------------------------------------------------------------------- #


@dataclass
class Line:
    baseline: List[List[int]]
    boundary: List[List[int]]
    region: Optional[str] = None
    text: Optional[str] = None
    fragment: bool = False
    reading: Optional[str] = None
    cer: Optional[float] = None
    confident: bool = False

    @property
    def x_range(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.baseline]
        return min(xs), max(xs)

    @property
    def width(self) -> float:
        x0, x1 = self.x_range
        return x1 - x0

    @property
    def y(self) -> float:
        return statistics.fmean(p[1] for p in self.baseline)

    @property
    def x_center(self) -> float:
        x0, x1 = self.x_range
        return (x0 + x1) / 2


def _mark_fragments(
    lines: List[Line], fragment_ratio: float, tiny_ratio: float
) -> None:
    widths = sorted(line.width for line in lines)
    reference = widths[int(0.9 * (len(widths) - 1))]  # a typical full line
    full = sorted(
        (ln for ln in lines if ln.width >= fragment_ratio * reference),
        key=lambda ln: ln.y,
    )
    gaps = [b.y - a.y for a, b in zip(full, full[1:]) if b.y - a.y > 1]
    spacing = statistics.median(gaps) if gaps else float("inf")
    for line in lines:
        if line.width < tiny_ratio * reference:
            line.fragment = True
        elif line.width < fragment_ratio * reference:
            nearest = min((abs(line.y - f.y) for f in full), default=float("inf"))
            line.fragment = nearest < 0.6 * spacing
        else:
            line.fragment = False


def _band_order(lines: List[Line]) -> List[Line]:
    """Top to bottom; lines side by side at the same height left to right."""
    main = sorted((ln for ln in lines if not ln.fragment), key=lambda ln: ln.y)
    gaps = [b.y - a.y for a, b in zip(main, main[1:]) if b.y - a.y > 1]
    tolerance = 0.5 * statistics.median(gaps) if gaps else 0

    bands: List[List[Line]] = []
    for line in main:
        band = bands[-1] if bands else None
        if (
            band
            and abs(line.y - statistics.fmean(b.y for b in band)) < tolerance
            and all(
                line.x_range[0] > b.x_range[1] or line.x_range[1] < b.x_range[0]
                for b in band
            )
        ):
            band.append(line)
        else:
            bands.append([line])

    ordered: List[Line] = []
    for band in bands:
        ordered.extend(sorted(band, key=lambda ln: ln.x_range[0]))

    for frag in sorted((ln for ln in lines if ln.fragment), key=lambda ln: ln.y):
        index = next((i for i, ln in enumerate(ordered) if ln.y > frag.y), len(ordered))
        ordered.insert(index, frag)
    return ordered


def order_lines(
    lines: List[Line],
    fragment_ratio: float = 0.35,
    tiny_ratio: float = 0.1,
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Line]:
    """
    Mark fragments and return lines in reading order.

    A line is a fragment (kept without text) when it is tiny (narrower than
    ``tiny_ratio`` x a full line: folio numbers, stray marks) or when it is
    short (``fragment_ratio``) and squeezed against a full line, like the
    words Ticha's transcriptions fold into the line below or above.
    Short lines on their own row (titles, signatures) are kept.

    Main lines are grouped into horizontal bands (lines at the same height
    that do not overlap horizontally, e.g. two signature columns) and read
    band by band, left to right. On landscape images (two-page spreads) the
    left page is read before the right one.
    """
    if not lines:
        return []
    if image_size and image_size[0] > 1.2 * image_size[1]:
        middle = image_size[0] / 2
        left = [ln for ln in lines if ln.x_center < middle]
        right = [ln for ln in lines if ln.x_center >= middle]
        ordered: List[Line] = []
        for side in (left, right):
            if side:
                _mark_fragments(side, fragment_ratio, tiny_ratio)
                ordered.extend(_band_order(side))
        return ordered
    _mark_fragments(lines, fragment_ratio, tiny_ratio)
    return _band_order(lines)


# ---------------------------------------------------------------------- #
# Alignment
# ---------------------------------------------------------------------- #


def align(
    readings: Sequence[str],
    texts: Sequence[str],
    skip_line_cost: float = 0.5,
    skip_text_cost: float = 0.5,
) -> List[Tuple[Optional[int], Optional[int], Optional[float]]]:
    """
    Monotonic alignment of line readings with transcription lines.

    Returns the path as ``(line_index | None, text_index | None, cer)``:
    a pair, a line without text, or a text without line. Pairing costs the
    CER of the pair; leaving a line or a text unpaired costs the skip costs.
    """
    n, m = len(readings), len(texts)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    cache: Dict[Tuple[int, int], float] = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and cost[i][j] + skip_line_cost < cost[i + 1][j]:
                cost[i + 1][j] = cost[i][j] + skip_line_cost
                move[i + 1][j] = "line"
            if j < m and cost[i][j] + skip_text_cost < cost[i][j + 1]:
                cost[i][j + 1] = cost[i][j] + skip_text_cost
                move[i][j + 1] = "text"
            if i < n and j < m:
                c = cache.setdefault((i, j), cer(readings[i], texts[j]))
                if cost[i][j] + c < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1] = cost[i][j] + c
                    move[i + 1][j + 1] = "pair"

    path: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
    i, j = n, m
    while i or j:
        step = move[i][j]
        if step == "pair":
            path.append((i - 1, j - 1, cache[(i - 1, j - 1)]))
            i, j = i - 1, j - 1
        elif step == "line":
            path.append((i - 1, None, None))
            i -= 1
        else:
            path.append((None, j - 1, None))
            j -= 1
    return path[::-1]


# ---------------------------------------------------------------------- #
# Segmentation and recognition with kraken
# ---------------------------------------------------------------------- #


def _require_kraken():
    try:
        import kraken  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "kraken is required for line segmentation: pip install 'ticha[lines]'"
        ) from e


class KrakenSegmenter:
    """Baseline segmentation with kraken's default model, cached as JSON."""

    def __init__(self, cache_dir: Optional[Path] = None, device: str = "cpu"):
        _require_kraken()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device

    def __call__(self, image_path: Path) -> Dict[str, Any]:
        cache = None
        if self.cache_dir:
            cache = self.cache_dir / f"{image_path.parent.name}_{image_path.stem}.json"
            if cache.exists():
                return json.loads(cache.read_text())

        from kraken import blla

        logger.info(f"Segmenting {image_path}")
        with Image.open(image_path) as im:
            seg = blla.segment(im, device=self.device)
        result = {
            "lines": [
                {
                    "baseline": [list(map(int, p)) for p in line.baseline],
                    "boundary": [list(map(int, p)) for p in line.boundary],
                    "regions": list(line.regions or []),
                }
                for line in seg.lines
            ],
            "regions": [
                {
                    "id": region.id,
                    "boundary": [list(map(int, p)) for p in region.boundary],
                }
                for regions in (seg.regions or {}).values()
                for region in regions
            ],
        }
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result))
        return result


class KrakenRecognizer:
    """Line recognition with a kraken model (path or DOI), cached as JSON."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ):
        _require_kraken()
        from kraken.lib import models

        path = Path(model)
        if not path.exists():
            from htrmopo import get_model

            logger.info(f"Fetching recognition model {model}")
            model_dir = Path(get_model(model))
            files = sorted(model_dir.glob("*.mlmodel")) + sorted(
                model_dir.glob("*.safetensors")
            )
            if not files:
                raise FileNotFoundError(f"No model file in {model_dir}")
            path = files[0]
        self.name = path.stem
        self.model = models.load_any(str(path), device=device)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __call__(self, image_path: Path, lines: Sequence[Line]) -> List[str]:
        key = hashlib.sha1(
            json.dumps([ln.baseline for ln in lines]).encode()
        ).hexdigest()[:12]
        cache = None
        if self.cache_dir:
            cache = (
                self.cache_dir
                / f"{image_path.parent.name}_{image_path.stem}.{self.name}.{key}.json"
            )
            if cache.exists():
                return json.loads(cache.read_text())

        from kraken import rpred
        from kraken.containers import BaselineLine, Segmentation

        seg = Segmentation(
            type="baselines",
            imagename=str(image_path),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=[
                BaselineLine(id=f"l{i}", baseline=ln.baseline, boundary=ln.boundary)
                for i, ln in enumerate(lines)
            ],
            regions={},
            line_orders=[],
        )
        logger.info(f"Recognizing {len(lines)} lines of {image_path}")
        with Image.open(image_path) as im:
            readings = [
                record.prediction for record in rpred.rpred(self.model, im, seg)
            ]
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(readings, ensure_ascii=False))
        return readings


# ---------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------- #


def _points(points: Iterable[Iterable[int]]) -> str:
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def _bbox_polygon(points: List[List[int]]) -> List[List[int]]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [
        [min(xs), min(ys)],
        [max(xs), min(ys)],
        [max(xs), max(ys)],
        [min(xs), max(ys)],
    ]


def write_page_xml(
    lines: List[Line],
    regions: List[Dict[str, Any]],
    image_name: str,
    image_size: Tuple[int, int],
    xml_path: Path,
    comments: str = "",
) -> None:
    """Write a PAGE XML (2019-07-15) file with lines in the given order."""
    ET.register_namespace("", PAGE_NS)

    def q(tag: str) -> str:
        return f"{{{PAGE_NS}}}{tag}"

    root = ET.Element(q("PcGts"))
    meta = ET.SubElement(root, q("Metadata"))
    ET.SubElement(meta, q("Creator")).text = "ticha line_pairing"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    ET.SubElement(meta, q("Created")).text = now
    ET.SubElement(meta, q("LastChange")).text = now
    if comments:
        ET.SubElement(meta, q("Comments")).text = comments
    page = ET.SubElement(
        root,
        q("Page"),
        imageFilename=image_name,
        imageWidth=str(image_size[0]),
        imageHeight=str(image_size[1]),
    )

    # Keep kraken's regions; lines without one go to a synthetic region.
    region_bounds = {r["id"]: r["boundary"] for r in regions}
    grouped: Dict[str, List[Tuple[int, Line]]] = {}
    for index, line in enumerate(lines):
        key = line.region if line.region in region_bounds else "_default"
        grouped.setdefault(key, []).append((index, line))
    if "_default" in grouped:
        points = [p for _, line in grouped["_default"] for p in line.boundary]
        region_bounds["_default"] = _bbox_polygon(points)

    region_order = sorted(grouped, key=lambda key: grouped[key][0][0])
    reading = ET.SubElement(page, q("ReadingOrder"))
    ordered_group = ET.SubElement(reading, q("OrderedGroup"), id="ro_1")
    for r_index in range(len(region_order)):
        ET.SubElement(
            ordered_group,
            q("RegionRefIndexed"),
            index=str(r_index),
            regionRef=f"r{r_index + 1}",
        )

    for r_index, key in enumerate(region_order):
        region_el = ET.SubElement(
            page,
            q("TextRegion"),
            id=f"r{r_index + 1}",
            custom=f"readingOrder {{index:{r_index};}}",
        )
        ET.SubElement(region_el, q("Coords"), points=_points(region_bounds[key]))
        for l_index, (index, line) in enumerate(grouped[key]):
            custom = f"readingOrder {{index:{l_index};}}"
            if line.fragment:
                custom += " structure {type:fragment;}"
            line_el = ET.SubElement(
                region_el, q("TextLine"), id=f"l{index + 1}", custom=custom
            )
            ET.SubElement(line_el, q("Coords"), points=_points(line.boundary))
            ET.SubElement(line_el, q("Baseline"), points=_points(line.baseline))
            if line.text:
                attrs = {}
                if line.cer is not None:
                    attrs["conf"] = f"{max(0.0, 1 - line.cer):.3f}"
                equiv = ET.SubElement(line_el, q("TextEquiv"), **attrs)
                ET.SubElement(equiv, q("Unicode")).text = line.text

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)


def crop_line(image: Image.Image, boundary: List[List[int]]) -> Image.Image:
    """Crop a line polygon, painting everything outside it white."""
    box = _bbox_polygon(boundary)
    x0, y0 = box[0]
    x1, y1 = box[2]
    region = image.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGB")
    mask = Image.new("L", region.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in boundary], fill=255)
    background = Image.new("RGB", region.size, "white")
    return Image.composite(region, background, mask)


def _place_image(source: Path, target: Path) -> None:
    """Hard-link (or copy) the page image next to its PAGE XML."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


# ---------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------- #


@dataclass
class PageImage:
    number: int
    path: Path
    folio: Optional[str] = None


@dataclass
class DocTask:
    doc_id: str
    pages: List[PageImage]
    transcription: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    image: PageImage
    lines: List[Line] = field(default_factory=list)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)
    texts: List[str] = field(default_factory=list)  # transcription lines assigned
    unpaired_texts: int = 0
    reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def main(self) -> List[Line]:
        return [ln for ln in self.lines if not ln.fragment]


class LinePairer:
    """Build a line-level HTR dataset from Ticha images and transcriptions."""

    def __init__(
        self,
        segmenter: Optional[Segmenter] = None,
        recognizer: Optional[Recognizer] = None,
        statuses: Optional[Iterable[str]] = DEFAULT_STATUSES,
        fragment_ratio: float = 0.35,
        split_columns: bool = True,
        cer_threshold: float = 0.5,
        export_lines: bool = False,
    ):
        """
        Args:
            segmenter: Callable returning lines/regions for an image
                (``run`` creates a KrakenSegmenter if None)
            recognizer: Callable reading lines, used for alignment. Without
                one, lines are paired in order (pages need equal line counts)
            statuses: Keep only these ``trptn_status`` values (None keeps all)
            fragment_ratio: Short lines squeezed against a full line are
                fragments (insertions) and get no text; see ``order_lines``
            split_columns: Split transcription lines on runs of 4+ spaces
            cer_threshold: Highest CER for a pair to count as confident
            export_lines: Also write crops + .gt.txt of confident lines
        """
        self.segmenter = segmenter
        self.recognizer = recognizer
        self.statuses = set(statuses) if statuses else None
        self.fragment_ratio = fragment_ratio
        self.split_columns = split_columns
        self.cer_threshold = cer_threshold
        self.export_lines = export_lines

    # -- input -------------------------------------------------------- #

    @staticmethod
    def _resolve(path: str, base: Path) -> Path:
        candidate = Path(path)
        if candidate.exists() or candidate.is_absolute():
            return candidate
        return base / candidate

    def load(self, texts_csv: Any, images_csv: Any) -> Tuple[List[DocTask], List[Dict]]:
        """Join texts and images; return document tasks and skipped documents."""
        texts = pd.read_csv(texts_csv, keep_default_na=False, na_values=[""])
        images = pd.read_csv(images_csv, keep_default_na=False, na_values=[""])
        images_base = Path(images_csv).parent
        image_columns = ["url", "image_paths"] + (
            ["image_folios"] if "image_folios" in images.columns else []
        )
        merged = texts.merge(images[image_columns], on="url", how="left")

        tasks: List[DocTask] = []
        skipped: List[Dict] = []
        for _, row in merged.iterrows():
            doc_id = row.get("ticha_id")
            if not isinstance(doc_id, str) or not doc_id:
                from .image_scraper import ticha_id_from_url

                doc_id = ticha_id_from_url(str(row["url"])) or str(row["url"])
            status = row.get("trptn_status")
            transcription = row.get("transcription")
            image_paths = row.get("image_paths")

            reason = None
            if not isinstance(transcription, str) or not transcription.strip():
                reason = "no transcription"
            elif (
                self.statuses
                and isinstance(status, str)
                and status not in self.statuses
            ):
                reason = f"status: {status}"
            elif not isinstance(image_paths, str) or not image_paths.strip():
                reason = "no images"
            if reason:
                skipped.append(
                    {"doc_id": doc_id, "status": "skipped", "reason": reason}
                )
                continue

            paths = [p for p in image_paths.split(";") if p]
            folios = str(row.get("image_folios") or "").split(";")
            pages = []
            for number, path in enumerate(paths, 1):
                resolved = self._resolve(path, images_base)
                folio = folios[number - 1] if len(folios) == len(paths) else None
                pages.append(PageImage(number, resolved, folio or None))
            tasks.append(
                DocTask(
                    doc_id=doc_id,
                    pages=pages,
                    transcription=transcription,
                    metadata={"url": row["url"], "trptn_status": status},
                )
            )
        return tasks, skipped

    # -- processing --------------------------------------------------- #

    def _segment(self, page: PageImage) -> PageResult:
        result = PageResult(image=page)
        try:
            segmentation = self.segmenter(page.path)
            with Image.open(page.path) as im:
                result.size = im.size
            lines = [
                Line(
                    baseline=ln["baseline"],
                    boundary=ln["boundary"],
                    region=(ln.get("regions") or [None])[0],
                )
                for ln in segmentation["lines"]
            ]
            result.lines = order_lines(
                lines, self.fragment_ratio, image_size=result.size
            )
            result.regions = segmentation.get("regions", [])
            if self.recognizer and result.main:
                readings = self.recognizer(page.path, result.main)
                for line, reading in zip(result.main, readings):
                    line.reading = reading
        except Exception as e:
            logger.error(f"Failed {page.path}: {e}")
            result.error = str(e)
        return result

    def _units(
        self, task: DocTask, results: List[PageResult]
    ) -> Tuple[List[Tuple[List[PageResult], List[str]]], List[str]]:
        """
        Group pages with the transcription lines they must be aligned with.

        Per page when every image folio has a transcription page with the
        same folio marker; otherwise the whole document as one unit.
        Returns (units, document-level notes).
        """
        pages = split_transcription_pages(task.transcription)
        by_folio = {normalize_folio(f): text for f, text in pages if f}
        image_folios = [normalize_folio(r.image.folio) for r in results]

        def lines_of(text: str) -> List[str]:
            return transcription_lines(text, self.split_columns)

        if by_folio and all(f in by_folio for f in image_folios):
            units = [
                ([r], lines_of(by_folio[f])) for r, f in zip(results, image_folios)
            ]
            missing = [f for f in by_folio if f not in image_folios]
            notes = (
                [f"transcribed folio(s) without image: {', '.join(missing)}"]
                if missing
                else []
            )
            return units, notes
        all_lines = [ln for _, text in pages for ln in lines_of(text)]
        return [(results, all_lines)], []

    def _pair_unit(self, pages: List[PageResult], texts: List[str]) -> None:
        sequence = [(page, line) for page in pages for line in page.main]
        if not sequence:
            if pages:
                pages[0].texts.extend(texts)
                pages[0].unpaired_texts += len(texts)
            return

        if self.recognizer:
            path = align([line.reading or "" for _, line in sequence], texts)
        elif len(pages) == 1 and len(sequence) == len(texts):
            path = [(i, i, None) for i in range(len(texts))]
        else:
            # No recognizer and counts differ: pre-fill in order for review.
            k = min(len(sequence), len(texts))
            path = [(i, i, None) for i in range(k)]
            path += [(i, None, None) for i in range(k, len(sequence))]
            path += [(None, j, None) for j in range(k, len(texts))]
            for page in pages:
                segmented = sum(len(p.main) for p in pages)
                page.reasons.append(
                    f"line count mismatch: {segmented} segmented, "
                    f"{len(texts)} transcribed"
                )

        current: Optional[PageResult] = None
        pending: List[str] = []  # unpaired texts before the next pairing
        for line_index, text_index, error in path:
            if line_index is not None:
                page, line = sequence[line_index]
                if text_index is not None:
                    line.text = texts[text_index]
                    line.cer = error
                    line.confident = error is None or error <= self.cer_threshold
                    # Unpaired texts between two pairs belong to both pages
                    for target in {id(p): p for p in (current, page) if p}.values():
                        target.unpaired_texts += len(pending)
                    page.texts.extend(pending)
                    page.texts.append(texts[text_index])
                    pending = []
                    current = page
                elif current is None:
                    current = page
            else:
                pending.append(texts[text_index])
        if pending:
            last = current or pages[-1]
            last.texts.extend(pending)
            last.unpaired_texts += len(pending)

    def process_document(self, task: DocTask) -> Tuple[List[PageResult], List[str]]:
        """Segment, recognize and align one document."""
        results = [self._segment(page) for page in task.pages]
        ok = [r for r in results if not r.error]
        units, notes = self._units(task, ok)
        for pages, texts in units:
            self._pair_unit(pages, texts)
        return results, notes

    def run(
        self,
        texts_csv: Any,
        images_csv: Any,
        output_dir: Any,
        max_documents: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the full pipeline; returns the manifest (one row per page)."""
        output_dir = Path(output_dir)
        if self.segmenter is None:
            self.segmenter = KrakenSegmenter(cache_dir=output_dir / "segmentation")

        tasks, skipped = self.load(texts_csv, images_csv)
        if max_documents:
            tasks = tasks[:max_documents]

        rows: List[Dict[str, Any]] = list(skipped)
        line_rows: List[Dict[str, Any]] = []
        for index, task in enumerate(tasks, 1):
            logger.info(f"[{index}/{len(tasks)}] {task.doc_id}")
            results, notes = self.process_document(task)
            for result in results:
                row, lines = self._write_page(task, result, notes, output_dir)
                rows.append(row)
                line_rows.extend(lines)

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = pd.DataFrame(rows)
        manifest.to_csv(output_dir / "manifest.csv", index=False)
        pd.DataFrame(line_rows).to_csv(output_dir / "lines.csv", index=False)
        return manifest

    def _write_page(
        self,
        task: DocTask,
        result: PageResult,
        notes: List[str],
        output_dir: Path,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        page = result.image
        stem = f"{task.doc_id}_page_{page.number:03d}"
        row: Dict[str, Any] = {
            "doc_id": task.doc_id,
            "page": page.number,
            "folio": page.folio,
            "image": str(page.path),
            **task.metadata,
        }
        if result.error:
            row.update(status="error", reason=result.error)
            return row, []

        main = result.main
        paired = [ln for ln in main if ln.text]
        doubtful = [ln for ln in paired if not ln.confident]
        reasons = list(result.reasons)
        if not main:
            reasons.append("no lines segmented")
        if not paired and main:
            reasons.append("no transcription paired")
        if result.unpaired_texts:
            reasons.append(
                f"{result.unpaired_texts} transcription line(s) without a line"
            )
        if doubtful:
            reasons.append(f"{len(doubtful)} low-confidence pair(s)")
        reasons.extend(notes)
        status = "review" if reasons and not (notes and reasons == notes) else "auto"

        target = output_dir / status
        image_name = f"{stem}{page.path.suffix}"
        _place_image(page.path, target / image_name)
        write_page_xml(
            result.lines,
            result.regions,
            image_name,
            result.size,
            target / f"{stem}.xml",
            comments=f"{task.doc_id} {task.metadata.get('url', '')} "
            f"folio={page.folio or ''} status={status}"
            + (f" ({'; '.join(reasons)})" if reasons else ""),
        )
        if status == "review" and result.texts:
            (target / f"{stem}.txt").write_text(
                "\n".join(result.texts) + "\n", encoding="utf-8"
            )

        line_rows = []
        crops_dir = output_dir / "lines" / stem
        image = Image.open(page.path) if self.export_lines else None
        try:
            for number, line in enumerate(main, 1):
                crop = None
                if image is not None and line.text and line.confident:
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    crop = crops_dir / f"line_{number:03d}.png"
                    crop_line(image, line.boundary).save(crop)
                    crop.with_suffix("").with_suffix(".gt.txt").write_text(
                        line.text + "\n", encoding="utf-8"
                    )
                line_rows.append(
                    {
                        "doc_id": task.doc_id,
                        "page": page.number,
                        "page_status": status,
                        "line": number,
                        "text": line.text,
                        "reading": line.reading,
                        "cer": None if line.cer is None else round(line.cer, 3),
                        "confident": bool(line.text) and line.confident,
                        "crop": str(crop) if crop else None,
                    }
                )
        finally:
            if image is not None:
                image.close()

        row.update(
            status=status,
            reason="; ".join(reasons) or None,
            segmented_lines=len(main),
            fragments=len(result.lines) - len(main),
            paired_lines=len(paired),
            confident_lines=len(paired) - len(doubtful),
            unpaired_transcription_lines=result.unpaired_texts,
            xml=str(target / f"{stem}.xml"),
        )
        return row, line_rows
