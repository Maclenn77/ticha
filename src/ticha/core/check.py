"""Contact sheets to check line pairing by eye.

Reads the output of ``ticha lines`` (``manifest.csv``, ``lines.csv``
and the PAGE XML files) and draws a sample of line crops, each with the
transcription paired with it, the HTR reading and its CER. Confident pairs
are labelled in green, doubtful ones in red and lines without text in gray.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .line_pairing import crop_line

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)
SELECTIONS = ("confident", "doubtful", "unpaired", "paired", "all")
COLORS = {"confident": "#1a7f37", "doubtful": "#cf222e", "unpaired": "#6e7781"}


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _kind(row: pd.Series) -> str:
    if not isinstance(row["text"], str) or not row["text"]:
        return "unpaired"
    return "confident" if bool(row["confident"]) else "doubtful"


def main_line_boundaries(xml_path: Path) -> List[List[List[int]]]:
    """Polygons of the non-fragment lines of a PAGE XML, in reading order."""
    lines = []
    for el in ET.parse(xml_path).getroot().iter():
        if not el.tag.endswith("}TextLine") or "fragment" in el.get("custom", ""):
            continue
        coords = next((c for c in el if c.tag.endswith("}Coords")), None)
        number = re.sub(r"\D", "", el.get("id", "")) or "0"
        points = [
            [int(float(v)) for v in p.split(",")]
            for p in (coords.get("points") if coords is not None else "").split()
        ]
        lines.append((int(number), points))
    return [points for _, points in sorted(lines)]


def _find_xml(dataset: Path, xml: str) -> Path:
    """The manifest's XML path, or the same file in another status folder."""
    path = Path(xml)
    if path.exists():
        return path
    for folder in ("auto", "review", "untranscribed"):
        candidate = dataset / folder / path.name
        if candidate.exists():
            return candidate
    return path


def select_lines(
    dataset: Path,
    only: str = "all",
    statuses: Optional[Iterable[str]] = None,
    docs: Optional[Iterable[str]] = None,
    sample: Optional[int] = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Rows of ``lines.csv`` to draw, with the page's XML and image paths."""
    if only not in SELECTIONS:
        raise ValueError(f"only must be one of {SELECTIONS}")
    lines = pd.read_csv(dataset / "lines.csv")
    manifest = pd.read_csv(dataset / "manifest.csv")
    manifest = manifest[manifest["status"].isin(["auto", "review", "untranscribed"])]
    lines = lines.merge(
        manifest[["doc_id", "page", "xml", "image"]], on=["doc_id", "page"]
    )
    lines["xml"] = lines["xml"].map(lambda x: str(_find_xml(dataset, x)))
    lines["kind"] = lines.apply(_kind, axis=1)
    if only == "paired":
        lines = lines[lines["kind"] != "unpaired"]
    elif only != "all":
        lines = lines[lines["kind"] == only]
    if statuses:
        lines = lines[lines["page_status"].isin(list(statuses))]
    if docs:
        lines = lines[lines["doc_id"].isin(list(docs))]
    if sample and len(lines) > sample:
        lines = lines.sample(sample, random_state=seed)
    return lines.sort_values(["doc_id", "page", "line"]).reset_index(drop=True)


def _row_image(
    page: Image.Image, boundary: List[List[int]], row: Dict[str, Any], width: int, font
) -> Image.Image:
    crop = crop_line(page, boundary)
    scale = min(1.0, width / crop.width)
    crop = crop.resize(
        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    )
    size = font.size if hasattr(font, "size") else 20
    canvas = Image.new("RGB", (width, crop.height + 3 * size + 16), "white")
    canvas.paste(crop, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = crop.height + 4
    kind = row["kind"]
    text = row["text"] if kind != "unpaired" else "(no text)"
    draw.text((6, y), f"T: {text}", fill=COLORS[kind], font=font)
    cer = "" if pd.isna(row["cer"]) else f"  cer={row['cer']:.2f}"
    reading = "" if pd.isna(row["reading"]) else row["reading"]
    draw.text((6, y + size + 4), f"R: {reading}", fill="#57606a", font=font)
    draw.text(
        (6, y + 2 * size + 8),
        f"{row['doc_id']} page {row['page']} line {row['line']}  "
        f"[{row['page_status']}] {kind}{cer}",
        fill="#57606a",
        font=font,
    )
    return canvas


def draw_sheets(
    rows: pd.DataFrame,
    output: Path,
    per_sheet: int = 15,
    width: int = 1600,
    font_size: int = 22,
) -> List[Path]:
    """Draw ``rows`` into ``output_001.png``, ``output_002.png``, ..."""
    font = _font(font_size)
    images: List[Image.Image] = []
    boundaries: Dict[str, Sequence] = {}
    for (xml, image_path), group in rows.groupby(["xml", "image"], sort=False):
        if xml not in boundaries:
            boundaries[xml] = main_line_boundaries(Path(xml))
        with Image.open(image_path) as page:
            for _, row in group.iterrows():
                index = int(row["line"]) - 1
                if index >= len(boundaries[xml]) or len(boundaries[xml][index]) < 3:
                    continue
                images.append(
                    _row_image(page, boundaries[xml][index], row, width, font)
                )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(images), per_sheet):
        chunk = images[start : start + per_sheet]
        sheet = Image.new(
            "RGB", (width, sum(im.height + 12 for im in chunk)), "#d0d7de"
        )
        y = 0
        for im in chunk:
            sheet.paste(im, (0, y))
            y += im.height + 12
        path = output.with_name(f"{output.stem}_{start // per_sheet + 1:03d}.png")
        sheet.save(path)
        paths.append(path)
    return paths


def check_pairing(
    dataset: Any,
    output: Any,
    only: str = "all",
    statuses: Optional[Iterable[str]] = None,
    docs: Optional[Iterable[str]] = None,
    sample: Optional[int] = 30,
    seed: int = 0,
    per_sheet: int = 15,
) -> List[Path]:
    """Select lines of a ``lines`` output and draw them; returns the sheets."""
    rows = select_lines(Path(dataset), only, statuses, docs, sample, seed)
    return draw_sheets(rows, Path(output), per_sheet=per_sheet)
