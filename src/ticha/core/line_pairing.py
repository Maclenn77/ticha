"""Pair segmented manuscript lines with their Ticha transcriptions.

Input: the CSVs written by ``ticha-scraper texts`` (transcriptions) and
``ticha-scraper images`` (page images). Pipeline per document:

1. Clean: keep documents with a transcription (and, optionally, a given
   transcription status).
2. Split the transcription into pages using Ticha's folio markers
   (``(8r)``, ``[40r]``, ``(446)``) and match them to the page images.
3. Segment each page image into baselines with kraken (cached as JSON).
4. Drop tiny fragments (folio numbers, interlinear insertions), order lines
   top-to-bottom / left-to-right, and pair them with the transcription lines.

Output (``output_dir``):

- ``auto/``: pages whose segmented and transcribed line counts agree,
  as PAGE XML (2019 schema) next to a copy of the image. Readable by
  eScriptorium, Transkribus and ``ketos train -f page``.
- ``review/``: pages that need a human. The PAGE XML is pre-filled in order
  (lines may be shifted) and a ``.txt`` file holds the page transcription.
  Import both into eScriptorium to correct them.
- ``lines/`` (optional): line crops + ``.gt.txt`` for the ``auto`` pages.
- ``manifest.csv``: one row per page with status and reason.
"""

import json
import logging
import os
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

DEFAULT_STATUSES = ("Completed and Checked", "Completed but not Checked")

# Folio / page markers at the start of a line: (8r) [40r] (02v) (446) [777]
FOLIO_MARKER_RE = re.compile(r"^\s*[\(\[]\s*(\d{1,4}\s*[rv]?)\s*[\)\]]\s*", re.I)
# Runs of spaces Ticha uses to lay out columns (signatures, witness lists)
COLUMN_GAP_RE = re.compile(r" {4,}")

# A segmenter takes an image path and returns
# {"lines": [{"baseline": [[x, y], ...], "boundary": [[x, y], ...],
#             "regions": [region_id]}],
#  "regions": [{"id": region_id, "boundary": [[x, y], ...]}]}
Segmenter = Callable[[Path], Dict[str, Any]]


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


def order_lines(
    lines: List[Line], fragment_ratio: float = 0.35, tiny_ratio: float = 0.1
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
    band by band, left to right. Fragments are placed by height.
    """
    if not lines:
        return []
    widths = sorted(line.width for line in lines)
    reference = widths[int(0.9 * (len(widths) - 1))]  # a typical full line

    full = sorted(
        (ln for ln in lines if ln.width >= fragment_ratio * reference),
        key=lambda ln: ln.y,
    )
    full_gaps = [b.y - a.y for a, b in zip(full, full[1:]) if b.y - a.y > 1]
    spacing = statistics.median(full_gaps) if full_gaps else float("inf")
    for line in lines:
        if line.width < tiny_ratio * reference:
            line.fragment = True
        elif line.width < fragment_ratio * reference:
            nearest = min((abs(line.y - f.y) for f in full), default=float("inf"))
            line.fragment = nearest < 0.6 * spacing
        else:
            line.fragment = False

    main = sorted((line for line in lines if not line.fragment), key=lambda ln: ln.y)
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

    # Insert fragments by height
    for frag in sorted((line for line in lines if line.fragment), key=lambda ln: ln.y):
        index = next((i for i, ln in enumerate(ordered) if ln.y > frag.y), len(ordered))
        ordered.insert(index, frag)
    return ordered


# ---------------------------------------------------------------------- #
# Segmentation
# ---------------------------------------------------------------------- #


class KrakenSegmenter:
    """Baseline segmentation with kraken's default model, cached as JSON."""

    def __init__(self, cache_dir: Optional[Path] = None, device: str = "cpu"):
        try:
            from kraken import blla  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "kraken is required for line segmentation: pip install 'ticha[lines]'"
            ) from e
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device

    def __call__(self, image_path: Path) -> Dict[str, Any]:
        cache = None
        if self.cache_dir:
            stem = f"{image_path.parent.name}_{image_path.stem}"
            cache = self.cache_dir / f"{stem}.json"
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
    q = lambda tag: f"{{{PAGE_NS}}}{tag}"  # noqa: E731

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
                equiv = ET.SubElement(line_el, q("TextEquiv"))
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
class PageTask:
    doc_id: str
    page_number: int
    image_path: Path
    folio: Optional[str]
    text: Optional[str]
    doc_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LinePairer:
    """Build a line-level HTR dataset from Ticha images and transcriptions."""

    def __init__(
        self,
        segmenter: Optional[Segmenter] = None,
        statuses: Optional[Iterable[str]] = DEFAULT_STATUSES,
        fragment_ratio: float = 0.35,
        split_columns: bool = True,
        export_lines: bool = False,
    ):
        """
        Args:
            segmenter: Callable returning lines/regions for an image
                (default: KrakenSegmenter, created on first use)
            statuses: Keep only these ``trptn_status`` values (None keeps all)
            fragment_ratio: Lines narrower than this fraction of a full line
                and squeezed against one are fragments (insertions) and get
                no text; see ``order_lines``
            split_columns: Split transcription lines on runs of 4+ spaces
            export_lines: Also write line crops + .gt.txt for ``auto`` pages
        """
        self.segmenter = segmenter
        self.statuses = set(statuses) if statuses else None
        self.fragment_ratio = fragment_ratio
        self.split_columns = split_columns
        self.export_lines = export_lines

    # -- input -------------------------------------------------------- #

    @staticmethod
    def _resolve(path: str, base: Path) -> Path:
        candidate = Path(path)
        if candidate.exists() or candidate.is_absolute():
            return candidate
        return base / candidate

    def load(
        self, texts_csv: Any, images_csv: Any
    ) -> Tuple[List[PageTask], List[Dict]]:
        """Join texts and images; return page tasks and skipped documents."""
        texts = pd.read_csv(texts_csv)
        images = pd.read_csv(images_csv)
        images_base = Path(images_csv).parent
        merged = texts.merge(
            images[["url", "image_paths"]], on="url", how="left", suffixes=("", "_img")
        )

        tasks: List[PageTask] = []
        skipped: List[Dict] = []
        for _, row in merged.iterrows():
            doc_id = row.get("ticha_id")
            if not isinstance(doc_id, str):
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

            paths = [self._resolve(p, images_base) for p in image_paths.split(";") if p]
            pages = split_transcription_pages(transcription)
            if len(pages) == len(paths):
                doc_reason = None
                page_texts = pages
            else:
                doc_reason = (
                    f"page split mismatch: {len(pages)} transcription page(s) "
                    f"for {len(paths)} image(s)"
                )
                # Can't tell which text goes with which image: give every
                # page the full transcription for the reviewer.
                page_texts = [(None, transcription)] * len(paths)

            for number, (path, (folio, text)) in enumerate(zip(paths, page_texts), 1):
                tasks.append(
                    PageTask(
                        doc_id=doc_id,
                        page_number=number,
                        image_path=path,
                        folio=folio,
                        text=text,
                        doc_reason=doc_reason,
                        metadata={"url": row["url"], "trptn_status": status},
                    )
                )
        return tasks, skipped

    # -- processing --------------------------------------------------- #

    def pair_page(
        self, task: PageTask, segmentation: Dict[str, Any]
    ) -> Tuple[List[Line], str, Optional[str], int]:
        """Pair one page. Returns (lines, status, reason, transcribed line count)."""
        lines = [
            Line(
                baseline=ln["baseline"],
                boundary=ln["boundary"],
                region=(ln.get("regions") or [None])[0],
            )
            for ln in segmentation["lines"]
        ]
        ordered = order_lines(lines, self.fragment_ratio)
        main = [ln for ln in ordered if not ln.fragment]
        text_lines = (
            transcription_lines(task.text, self.split_columns)
            if task.text and not task.doc_reason
            else []
        )

        if task.doc_reason:
            return ordered, "review", task.doc_reason, len(text_lines)
        for line, text in zip(main, text_lines):
            line.text = text
        if not main:
            return ordered, "review", "no lines segmented", len(text_lines)
        if len(main) != len(text_lines):
            reason = (
                f"line count mismatch: {len(main)} segmented, "
                f"{len(text_lines)} transcribed"
            )
            return ordered, "review", reason, len(text_lines)
        return ordered, "auto", None, len(text_lines)

    def run(
        self,
        texts_csv: Any,
        images_csv: Any,
        output_dir: Any,
        max_documents: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the full pipeline and return the manifest."""
        output_dir = Path(output_dir)
        if self.segmenter is None:
            self.segmenter = KrakenSegmenter(cache_dir=output_dir / "segmentation")

        tasks, skipped = self.load(texts_csv, images_csv)
        if max_documents:
            keep = list(dict.fromkeys(t.doc_id for t in tasks))[:max_documents]
            tasks = [t for t in tasks if t.doc_id in keep]

        rows: List[Dict[str, Any]] = list(skipped)
        for index, task in enumerate(tasks, 1):
            logger.info(f"[{index}/{len(tasks)}] {task.doc_id} page {task.page_number}")
            rows.append(self._process(task, output_dir))

        manifest = pd.DataFrame(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_dir / "manifest.csv", index=False)
        return manifest

    def _process(self, task: PageTask, output_dir: Path) -> Dict[str, Any]:
        stem = f"{task.doc_id}_page_{task.page_number:03d}"
        row: Dict[str, Any] = {
            "doc_id": task.doc_id,
            "page": task.page_number,
            "folio": task.folio,
            "image": str(task.image_path),
            **task.metadata,
        }
        try:
            segmentation = self.segmenter(task.image_path)
            lines, status, reason, n_text = self.pair_page(task, segmentation)
        except Exception as e:
            logger.error(f"Failed {stem}: {e}")
            row.update(status="error", reason=str(e))
            return row

        target = output_dir / status
        image_name = f"{stem}{task.image_path.suffix}"
        _place_image(task.image_path, target / image_name)
        with Image.open(task.image_path) as im:
            size = im.size
            write_page_xml(
                lines,
                segmentation.get("regions", []),
                image_name,
                size,
                target / f"{stem}.xml",
                comments=f"{task.doc_id} {task.metadata.get('url', '')} "
                f"folio={task.folio or ''} status={status} {reason or ''}".strip(),
            )
            if status == "review" and task.text:
                (target / f"{stem}.txt").write_text(task.text + "\n", encoding="utf-8")
            if status == "auto" and self.export_lines:
                self._export_lines(im, lines, output_dir / "lines" / stem)

        main = [ln for ln in lines if not ln.fragment]
        row.update(
            status=status,
            reason=reason,
            segmented_lines=len(main),
            fragments=len(lines) - len(main),
            transcribed_lines=n_text,
            xml=str(target / f"{stem}.xml"),
        )
        return row

    @staticmethod
    def _export_lines(image: Image.Image, lines: List[Line], directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        n = 0
        for line in lines:
            if line.fragment or not line.text:
                continue
            n += 1
            crop_line(image, line.boundary).save(directory / f"line_{n:03d}.png")
            (directory / f"line_{n:03d}.gt.txt").write_text(
                line.text + "\n", encoding="utf-8"
            )
