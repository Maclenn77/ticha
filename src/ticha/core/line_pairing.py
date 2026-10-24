"""Pair segmented manuscript lines with their Ticha transcriptions.

Input: the CSVs written by ``ticha texts`` (transcriptions) and
``ticha images`` (page images). Pipeline per document:

1. Clean: keep documents with a transcription (and, optionally, a given
   transcription status).
2. Segment each page image into baselines with kraken (cached as JSON).
   Tiny or squeezed lines (folio numbers, interlinear insertions) are marked
   as fragments; lines are read top to bottom, left to right (signature
   columns row by row, two-page spreads left page first).
3. Recognize each line with a generic HTR model (McCATMuS by default). The
   rough reading is only used to find which transcription line goes where.
4. Align the line readings with the transcription as a stream of words
   (``align_words``): each line gets the span of words it reads, so
   transcriptions whose line breaks do not follow the manuscript are cut
   again; lines may stay without text and words without a line. A pair is
   confident when its text fits better than the neighbouring texts and its
   CER is low enough (see ``LinePairer``). Pages are aligned on their own
   when the transcription's folio markers match the image folios, otherwise
   the whole document is aligned at once. Without a recognizer, lines are
   paired in order and a page is accepted only when the line counts agree.

Output (``output_dir``):

- ``auto/``: pages paired confidently up to ``review_tolerance`` (doubtful
  lines are written without text), as PAGE XML (2019 schema;
  ``TextEquiv/@conf`` holds the confidence) next to a copy of the image.
  Readable by eScriptorium, Transkribus and ``ketos train -f page``.
- ``review/``: every other page, pre-filled with the best alignment, plus a
  ``.txt`` with the transcription lines assigned to it. Nothing is
  discarded: import both into eScriptorium to correct them.
- ``untranscribed/``: pages to which no transcription was assigned.
- ``lines/`` (optional): crops + ``.gt.txt`` of confidently paired lines.
- ``manifest.csv`` (one row per page) and ``lines.csv`` (one row per line).
"""

import bisect
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import statistics
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw

from .text_scraper import strip_office_markup

logger = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

DEFAULT_STATUSES = ("Completed and Checked", "Completed but not Checked")
# Output folders of a page
PAGE_STATUSES = ("auto", "review", "untranscribed")
# McCATMuS: generic HTR model for 16th-21st c. manuscripts (es, fr, la, ...)
DEFAULT_MODEL = "10.5281/zenodo.13788177"

# Folio / page markers at the start of a line: (8r) [40r] (02v) (446) [777]
FOLIO_MARKER_RE = re.compile(r"^\s*[\(\[]\s*(\d{1,4}\s*[rv]?)\s*[\)\]]\s*", re.I)
# Runs of spaces Ticha uses to lay out columns (signatures, witness lists)
COLUMN_GAP_RE = re.compile(r" {4,}")
# Marks of a word broken at the end of a line
HYPHENS = {"-", "=", "¬", "⸗"}

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


# Line references some transcriptions start lines with: "63r-5 Joseph de ..."
LINE_REFERENCE_RE = re.compile(r"^\s*\d{1,4}[rv]-\d{1,3}\s+")


def transcription_lines(page_text: str, split_columns: bool = True) -> List[str]:
    """
    Non-empty lines of a page; column layouts split into separate lines and
    line references (``63r-5``) removed.
    """
    lines = []
    for line in page_text.splitlines():
        line = LINE_REFERENCE_RE.sub("", line)
        parts = COLUMN_GAP_RE.split(line) if split_columns else [line]
        lines.extend(p.strip() for p in parts if p.strip())
    return lines


def normalize_folio(folio: Optional[str]) -> Optional[str]:
    """``02r`` -> ``2r``, ``(446)`` -> ``446``."""
    if folio is None:
        return None
    folio = re.sub(r"[\s\(\)\[\]]", "", str(folio)).lower()
    return re.sub(r"^0+(?=\d)", "", folio)


_EDITORIAL_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def _normalize_kept(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))
    return text.translate(str.maketrans("vjy", "uii"))


def normalize_for_matching(text: str) -> str:
    """
    Reduce a line to what a generic HTR model and Ticha agree on.

    Drops editorial additions (``q[ue]``, ``(...)``, ``(rao?)``), accents,
    case, punctuation and spaces (word separation in the manuscripts is
    irregular), and merges letters used interchangeably (u/v, i/j/y).
    """
    return _normalize_kept(_EDITORIAL_RE.sub("", text))


def _distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:  # pragma: no cover - small fallback
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def cer(reading: str, reference: str) -> float:
    """Character error rate of a reading against a reference, after normalizing."""
    a, b = normalize_for_matching(reading), normalize_for_matching(reference)
    if not b:
        return 0.0 if not a else 1.0
    return _distance(a, b) / len(b)


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
    margin: Optional[float] = None  # CER against neighbouring texts minus own CER
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
    gaps = [b.y - a.y for a, b in pairwise(full) if b.y - a.y > 1]
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
    gaps = [b.y - a.y for a, b in pairwise(main) if b.y - a.y > 1]
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


@dataclass
class Word:
    """A transcription word: ``texts[line][start:end]`` and its normalized form."""

    line: int
    start: int
    end: int
    norm: str
    line_start: bool = False
    line_end: bool = False


def split_words(texts: Sequence[str]) -> List[Word]:
    """
    Split transcription lines into words, remembering the original breaks.

    Tokens that normalize to nothing (``(?)``, ``=``) are glued to the
    previous word of the line (or the next one at the start of a line).
    """
    words: List[Word] = []
    for index, text in enumerate(texts):
        dropped = [m.span() for m in _EDITORIAL_RE.finditer(text)]
        line_words: List[Word] = []
        for match in re.finditer(r"\S+", text):
            kept = "".join(
                c
                for k, c in enumerate(match.group(), match.start())
                if not any(a <= k < b for a, b in dropped)
            )
            norm = _normalize_kept(kept)
            if line_words and (not norm or not line_words[-1].norm):
                line_words[-1].end = match.end()
                line_words[-1].norm += norm
                continue
            line_words.append(Word(index, match.start(), match.end(), norm))
        if line_words:
            line_words[0].line_start = True
            line_words[-1].line_end = True
        words.extend(line_words)
    return words


def words_text(texts: Sequence[str], words: Sequence[Word], a: int, b: int) -> str:
    """Original text of ``words[a:b]``; pieces of different lines joined by a space."""
    parts: List[List[int]] = []
    for word in words[a:b]:
        if parts and parts[-1][0] == word.line:
            parts[-1][2] = word.end
        else:
            parts.append([word.line, word.start, word.end])
    return " ".join(texts[line][start:end] for line, start, end in parts)


def _word_band(
    readings: Sequence[str], words: Sequence[Word], width: int
) -> Tuple[List[int], List[int]]:
    """Window of word positions per line, around a global character alignment."""
    from rapidfuzz.distance import Levenshtein

    joined_r = "".join(readings)
    joined_t = "".join(w.norm for w in words)
    # r2t[p]: position in the text aligned with position p of the readings
    r2t = [0] * (len(joined_r) + 1)
    for tag, r0, r1, t0, t1 in Levenshtein.opcodes(joined_r, joined_t):
        for k in range(r0, r1 + 1):
            r2t[k] = t0 + (min(k - r0, t1 - t0) if tag != "delete" else 0)
    word_starts, pos = [], 0
    for word in words:
        word_starts.append(pos)
        pos += len(word.norm)

    lo, hi, pos = [], [], 0
    for i in range(len(readings) + 1):
        anchor = bisect.bisect_right(word_starts, r2t[pos]) - 1 if words else 0
        lo.append(max(0, anchor - width))
        hi.append(min(len(words), anchor + width))
        if i < len(readings):
            pos += len(readings[i])
    hi[-1] = len(words)
    for i in range(1, len(lo)):  # windows must move forward
        lo[i] = max(lo[i], lo[i - 1])
    for i in range(len(hi) - 2, -1, -1):
        hi[i] = min(hi[i], hi[i + 1])
    lo[0] = 0
    return lo, hi


AlignStep = Tuple[str, Optional[int], Optional[int], Optional[int], Optional[int]]


def align_words(
    readings: Sequence[str],
    words: Sequence[Word],
    skip_line_cost: float = 0.6,
    skip_word_cost: float = 0.6,
    break_penalty: float = 3.0,
    band: int = 60,
    max_states: int = 40000,
) -> List[AlignStep]:
    """
    Cut a stream of transcription words into the spans read on each line.

    Unlike ``align``, a line can take part of a transcription line or several
    of them, so transcriptions whose line breaks do not follow the manuscript
    can be paired too. Costs are edit distances on normalized text: pairing
    a line with ``words[a:b]`` costs their distance (plus ``break_penalty``
    for each end that does not fall on an original line break, so the
    transcription's own breaks win when they fit); leaving a line or words
    unpaired costs ``skip_*_cost`` per character.

    Returns steps ``("pair", line, a, b, distance)``, ``("line", line, None,
    None, None)`` (line without text) and ``("words", None, a, b, None)``
    (words without line), in order.
    """
    reads = [normalize_for_matching(r or "") for r in readings]
    n, m = len(reads), len(words)
    if (n + 1) * (m + 1) > max_states and n and m:
        lo, hi = _word_band(reads, words, band)
    else:
        lo, hi = [0] * (n + 1), [m] * (n + 1)

    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[Tuple[int, int, int]]]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    cost[0][0] = 0.0
    for i in range(n + 1):
        row, row_back = cost[i], back[i]
        for w in range(lo[i], hi[i] + 1):
            here = row[w]
            if here == inf:
                continue
            if w < hi[i]:  # leave word w without a line
                c = here + skip_word_cost * len(words[w].norm)
                if c < row[w + 1]:
                    row[w + 1] = c
                    row_back[w + 1] = (i, w, -1)
            if i == n:
                continue
            reading = reads[i]
            nxt, nxt_back = cost[i + 1], back[i + 1]
            if lo[i + 1] <= w <= hi[i + 1]:  # leave line i without text
                c = here + skip_line_cost * len(reading)
                if c < nxt[w]:
                    nxt[w] = c
                    nxt_back[w] = (i, w, -2)
            if w == m:
                continue
            limit = 2 * len(reading) + 20
            start_penalty = 0.0 if words[w].line_start else break_penalty
            span = ""
            for v in range(w + 1, min(m, hi[i + 1]) + 1):
                span += words[v - 1].norm
                if len(span) > limit and v > w + 1:
                    break
                if v < lo[i + 1]:
                    continue
                c = here + _distance(reading, span) + start_penalty
                if not words[v - 1].line_end:
                    c += break_penalty
                if c < nxt[v]:
                    nxt[v] = c
                    nxt_back[v] = (i, w, v)

    if cost[n][m] == inf:  # the window missed the path: align without it
        return align_words(
            readings,
            words,
            skip_line_cost,
            skip_word_cost,
            break_penalty,
            band,
            max_states=(n + 1) * (m + 1),
        )

    steps: List[AlignStep] = []
    i, w = n, m
    while i or w:
        pi, pw, kind = back[i][w]
        if kind == -1:
            if steps and steps[-1][0] == "words" and steps[-1][2] == w:
                steps[-1] = ("words", None, pw, steps[-1][3], None)
            else:
                steps.append(("words", None, pw, w, None))
        elif kind == -2:
            steps.append(("line", pi, None, None, None))
        else:
            span = "".join(word.norm for word in words[pw:w])
            steps.append(("pair", pi, pw, w, _distance(reads[pi], span)))
        i, w = pi, pw
    return steps[::-1]


class _Cut:
    """
    The text paired with a line: ``words[start:end]``, plus the end of a word
    broken on the previous line (``head``) and/or the start of one broken on
    the next line (``tail``).
    """

    def __init__(
        self, texts: Sequence[str], words: Sequence[Word], a: int, b: int, d: int
    ):
        self.texts, self.words = texts, words
        self.start, self.end = a, b
        self.head = self.tail = ("", "")  # (text, normalized)
        self.distance = d

    @property
    def text(self) -> str:
        parts = [
            self.head[0],
            words_text(self.texts, self.words, self.start, self.end),
            self.tail[0],
        ]
        return " ".join(p for p in parts if p)

    @property
    def norm(self) -> str:
        middle = "".join(w.norm for w in self.words[self.start : self.end])
        return self.head[1] + middle + self.tail[1]


def _split_broken_word(
    raw_reading: str, read1: str, read2: str, first: _Cut, second: _Cut
) -> bool:
    """
    Split a word the scribe broke across two lines; True if split.

    Used where the cut falls inside a transcription line (the transcription
    did not follow the manuscript's breaks). The last word of the first line
    or the first word of the second one is split where the readings fit
    best: by at least 2 edits when the first line's reading ends with a
    hyphen (the first part then gets it), by at least 3 when the scribe broke
    the word without one.
    """
    hyphen = raw_reading.rstrip()[-1:]
    if hyphen not in HYPHENS:
        hyphen = ""
    words, texts = first.words, first.texts
    best = (first.distance + second.distance - (2 if hyphen else 3), None)
    candidates = []
    if first.end - 1 >= first.start:
        candidates.append(first.end - 1)  # last word of the first line
    if second.start < second.end:
        candidates.append(second.start)  # first word of the second line
    for w in candidates:
        word = words[w]
        original = texts[word.line][word.start : word.end]
        core = original.rstrip(".,;:")  # "Verdad," splits as "Ver" + "dad,"
        if not core.isalpha() or len(word.norm) != len(core):
            continue  # editorial marks or letters that normalize away
        before = first.head[1] + "".join(x.norm for x in words[first.start : w])
        after = "".join(x.norm for x in words[w + 1 : second.end])
        for j in range(1, len(word.norm)):
            d1 = _distance(read1, before + word.norm[:j])
            d2 = _distance(read2, word.norm[j:] + after)
            if d1 + d2 < best[0]:
                best = (d1 + d2, (w, j, d1, d2))
    if best[1] is None:
        return False
    w, j, d1, d2 = best[1]
    word = words[w]
    original = texts[word.line][word.start : word.end]
    first.end = w
    first.tail = (original[:j] + hyphen, word.norm[:j])
    second.start = w + 1
    second.head = (original[j:], word.norm[j:])
    first.distance, second.distance = d1, d2
    return True


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
        self.path = path
        self.device = device
        self.model = self._load()
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _load(self) -> Any:
        """kraken >= 7 task API (reads .mlmodel and .safetensors), else load_any."""
        try:
            from kraken.tasks import RecognitionTaskModel
        except ImportError:
            from kraken.lib import models

            return models.load_any(str(self.path), device=self.device)
        return RecognitionTaskModel.load_model(self.path)

    def __getstate__(self) -> Dict[str, Any]:
        # Worker processes reload the model from its file instead of pickling it.
        state = dict(self.__dict__)
        state.pop("model", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.model = self._load()

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
            if hasattr(self.model, "predict"):
                from kraken.configs import RecognitionInferenceConfig

                # Explicit accelerator: "auto" picks Apple's MPS, which hangs
                accelerator = "cpu" if self.device == "cpu" else "auto"
                config = RecognitionInferenceConfig(
                    accelerator=accelerator, device="auto", num_line_workers=0
                )
                records = self.model.predict(im, seg, config)
            else:
                records = rpred.rpred(self.model, im, seg)
            readings = [record.prediction for record in records]
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(readings, ensure_ascii=False))
        return readings


# ---------------------------------------------------------------------- #
# Worker processes (LinePairer.run with workers > 1)
# ---------------------------------------------------------------------- #

_worker_pairer: Optional["LinePairer"] = None


def _init_worker(pairer: "LinePairer", threads: int) -> None:
    global _worker_pairer
    _worker_pairer = pairer
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def _process_in_worker(task: "DocTask") -> Tuple[List["PageResult"], List[str]]:
    return _worker_pairer.process_document(task)


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
                    # 1 - CER, kept >= 0.5 for confident pairs and below for
                    # doubtful ones, whatever their CER
                    conf = max(0.0, 1 - line.cer)
                    conf = max(conf, 0.5) if line.confident else min(conf, 0.49)
                    attrs["conf"] = f"{conf:.3f}"
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
    unpaired_chars: int = 0
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
        max_cer: float = 0.75,
        min_margin: float = 0.15,
        review_tolerance: float = 0.2,
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
            cer_threshold: A pair with a CER up to this is confident (if it
                matches its own text better than its neighbours' texts)
            max_cer: A pair with a CER up to this is still confident when
                its text fits clearly better than the neighbouring texts
                (by ``min_margin``): the generic model misreads Zapotec, so
                right pairs often have a high CER
            min_margin: See ``max_cer``
            review_tolerance: Share of a page's lines that may stay doubtful
                (or of its transcription that may stay without a line) for
                the page to go to ``auto/``; doubtful lines are written there
                without text, so they are left out of training
            export_lines: Also write crops + .gt.txt of confident lines
        """
        self.segmenter = segmenter
        self.recognizer = recognizer
        self.statuses = set(statuses) if statuses else None
        self.fragment_ratio = fragment_ratio
        self.split_columns = split_columns
        self.cer_threshold = cer_threshold
        self.max_cer = max_cer
        self.min_margin = min_margin
        self.review_tolerance = review_tolerance
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
            transcription = strip_office_markup(row.get("transcription"))
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
                for line, reading in zip(result.main, readings, strict=True):
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
                ([r], lines_of(by_folio[f]))
                for r, f in zip(results, image_folios, strict=True)
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

        # path: (line index | None, text | None, chars of unpaired text)
        path: List[Tuple[Optional[int], Optional[str], int]]
        if self.recognizer:
            path = self._align_with_readings(sequence, texts)
        elif len(pages) == 1 and len(sequence) == len(texts):
            path = [(i, texts[i], 0) for i in range(len(texts))]
            for _, line in sequence:
                line.confident = True
        else:
            # No recognizer and counts differ: pre-fill in order for review.
            k = min(len(sequence), len(texts))
            path = [(i, texts[i], 0) for i in range(k)]
            path += [(i, None, 0) for i in range(k, len(sequence))]
            path += [
                (None, texts[j], len(normalize_for_matching(texts[j])))
                for j in range(k, len(texts))
            ]
            segmented = len(sequence)
            for page in pages:
                page.reasons.append(
                    f"line count mismatch: {segmented} segmented, "
                    f"{len(texts)} transcribed"
                )
            for i in range(k):
                sequence[i][1].confident = True

        current: Optional[PageResult] = None
        pending: List[Tuple[str, int]] = []  # unpaired texts before the next pairing
        for line_index, text, chars in path:
            if line_index is not None:
                page, line = sequence[line_index]
                if text is not None:
                    line.text = text
                    # Unpaired texts between two pairs belong to both pages
                    for target in {id(p): p for p in (current, page) if p}.values():
                        target.unpaired_texts += len(pending)
                        target.unpaired_chars += sum(c for _, c in pending)
                    page.texts.extend(t for t, _ in pending)
                    page.texts.append(text)
                    pending = []
                    current = page
                elif current is None:
                    current = page
            else:
                pending.append((text, chars))
        if pending:
            last = current or pages[-1]
            last.texts.extend(t for t, _ in pending)
            last.unpaired_texts += len(pending)
            last.unpaired_chars += sum(c for _, c in pending)

    def _align_with_readings(
        self, sequence: List[Tuple[PageResult, Line]], texts: List[str]
    ) -> List[Tuple[Optional[int], Optional[str], int]]:
        """
        Align line readings with the transcription words; set CER, margin
        and confidence of each paired line.

        A pair is confident when its text fits the reading better than the
        texts of the neighbouring pairs (an off-by-one alignment fits a
        neighbour as well) and either its CER is at most ``cer_threshold``
        or, up to ``max_cer``, it beats the neighbours by ``min_margin``.
        Pairs whose boundary with the next pair falls inside a transcription
        line and could move by one word at no cost are not confident.
        """
        words = split_words(texts)
        reads = [normalize_for_matching(line.reading or "") for _, line in sequence]
        steps = align_words([line.reading or "" for _, line in sequence], words)

        def span(a: int, b: int) -> str:
            return "".join(w.norm for w in words[a:b])

        def error(reading: str, reference: str) -> float:
            if not reference:
                return 0.0 if not reading else 1.0
            return _distance(reading, reference) / len(reference)

        pairs = [s for s in steps if s[0] == "pair"]
        ambiguous = set()  # boundaries (index of the first pair) that could move
        for i, ((_, k1, a1, b1, d1), (_, k2, a2, b2, d2)) in enumerate(pairwise(pairs)):
            if b1 != a2 or (words[b1 - 1].line_end and words[a2].line_start):
                continue  # not contiguous, or on an original line break
            now = d1 + d2
            moves = []
            if b1 - 1 > a1:  # last word of the first line to the second one
                moves.append(
                    _distance(reads[k1], span(a1, b1 - 1))
                    + _distance(reads[k2], span(b1 - 1, b2))
                )
            if a2 + 1 < b2:  # first word of the second line to the first one
                moves.append(
                    _distance(reads[k1], span(a1, a2 + 1))
                    + _distance(reads[k2], span(a2 + 1, b2))
                )
            if moves and min(moves) <= now:
                ambiguous.add(i)

        cut = {k: _Cut(texts, words, a, b, d) for _, k, a, b, d in pairs}
        for i, ((_, k1, _, b1, _), (_, k2, a2, _, _)) in enumerate(pairwise(pairs)):
            if (
                b1 == a2
                and not words[a2].line_start
                and _split_broken_word(
                    sequence[k1][1].reading or "",
                    reads[k1],
                    reads[k2],
                    cut[k1],
                    cut[k2],
                )
            ):
                ambiguous.discard(i)  # the split settled where the line breaks
        ambiguous_lines = {k for i in ambiguous for k in (pairs[i][1], pairs[i + 1][1])}

        for index, (_, k, _, _, _) in enumerate(pairs):
            line = sequence[k][1]
            own, d = cut[k].norm, cut[k].distance
            line.cer = d / len(own) if own else (0.0 if not reads[k] else 1.0)
            neighbours = [
                pairs[j][1]
                for j in (index - 2, index - 1, index + 1, index + 2)
                if 0 <= j < len(pairs)
            ]
            if neighbours:
                line.margin = (
                    min(error(reads[k], cut[n].norm) for n in neighbours) - line.cer
                )
            # The model misreads letters but rarely drops or invents many:
            # a reading much shorter or longer than its text is a bad cut.
            ratio = len(reads[k]) / max(1, len(own))
            beats = line.margin is None or line.margin > 0
            line.confident = k not in ambiguous_lines and (
                (line.cer <= self.cer_threshold and beats and 0.7 <= ratio <= 1.4)
                or (
                    line.cer <= self.max_cer
                    and line.margin is not None
                    and line.margin >= self.min_margin
                    and 0.8 <= ratio <= 1.25
                )
            )

        path: List[Tuple[Optional[int], Optional[str], int]] = []
        for kind, k, a, b, _ in steps:
            if kind == "pair":
                path.append((k, cut[k].text, 0))
            elif kind == "line":
                path.append((k, None, 0))
            else:
                path.append((None, words_text(texts, words, a, b), len(span(a, b))))
        return path

    def process_document(self, task: DocTask) -> Tuple[List[PageResult], List[str]]:
        """Segment, recognize and align one document."""
        results = [self._segment(page) for page in task.pages]
        ok = [r for r in results if not r.error]
        units, notes = self._units(task, ok)
        for pages, texts in units:
            self._pair_unit(pages, texts)
        if self.recognizer and units and all(len(pages) == 1 for pages, _ in units):
            notes = notes + self._retry_other_folios(task, units)
        return results, notes

    @staticmethod
    def _confident_count(page: PageResult) -> int:
        return sum(1 for ln in page.main if ln.text and ln.confident)

    @staticmethod
    def _snapshot(page: PageResult) -> Tuple[Any, ...]:
        return (
            [(ln.text, ln.cer, ln.margin, ln.confident) for ln in page.main],
            list(page.texts),
            page.unpaired_texts,
            page.unpaired_chars,
        )

    @staticmethod
    def _restore(page: PageResult, snapshot: Tuple[Any, ...]) -> None:
        lines, texts, unpaired_texts, unpaired_chars = snapshot
        for line, (text, error, margin, confident) in zip(
            page.main, lines, strict=True
        ):
            line.text, line.cer, line.margin, line.confident = (
                text,
                error,
                margin,
                confident,
            )
        page.texts, page.unpaired_texts, page.unpaired_chars = (
            list(texts),
            unpaired_texts,
            unpaired_chars,
        )

    def _retry_other_folios(
        self, task: DocTask, units: List[Tuple[List[PageResult], List[str]]]
    ) -> List[str]:
        """
        Pair badly paired pages with the other folios' transcriptions.

        Image folios and transcription folio markers sometimes disagree (a
        verso labelled as the recto, pages in another order). A page where
        fewer than half of its pairs are confident is aligned again with each
        other folio's text, and keeps the one that pairs at least twice as
        many lines confidently (5 or more, and half of its pairs).
        """
        folios = [
            (folio, transcription_lines(text, self.split_columns))
            for folio, text in split_transcription_pages(task.transcription)
            if folio
        ]
        if len(folios) < 2:
            return []
        notes = []
        for pages, texts in units:
            page = pages[0]
            paired = [ln for ln in page.main if ln.text]
            confident = self._confident_count(page)
            if len(paired) < 3 or confident >= 0.5 * len(paired):
                continue
            best_count, best, best_folio = confident, self._snapshot(page), None
            for folio, candidate in folios:
                if candidate == texts:
                    continue
                self._restore(
                    page, ([(None, None, None, False)] * len(page.main), [], 0, 0)
                )
                self._pair_unit([page], candidate)
                count = self._confident_count(page)
                paired_now = sum(1 for ln in page.main if ln.text)
                if count >= max(2 * best_count, 5) and count >= 0.5 * paired_now:
                    best_count, best, best_folio = count, self._snapshot(page), folio
            self._restore(page, best)
            if best_folio:
                notes.append(
                    f"page {page.image.number} paired with folio {best_folio} "
                    f"(image folio {page.image.folio})"
                )
        return notes

    def run(
        self,
        texts_csv: Any,
        images_csv: Any,
        output_dir: Any,
        max_documents: Optional[int] = None,
        workers: int = 1,
        threads_per_worker: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline; returns the manifest (one row per page).

        Args:
            workers: Documents processed in parallel, each in its own process
                with its own copy of the models (mind the RAM: ~2 GB per
                worker with kraken on a 2500x3500 page). 1 processes them in this process.
            threads_per_worker: Torch threads per worker (default: CPU
                cores / workers)
        """
        output_dir = Path(output_dir)
        if self.segmenter is None:
            self.segmenter = KrakenSegmenter(cache_dir=output_dir / "segmentation")

        tasks, skipped = self.load(texts_csv, images_csv)
        if max_documents:
            tasks = tasks[:max_documents]

        rows: List[Dict[str, Any]] = list(skipped)
        line_rows: List[Dict[str, Any]] = []
        for index, (task, (results, notes)) in enumerate(
            zip(
                tasks,
                self._process_all(tasks, workers, threads_per_worker),
                strict=True,
            ),
            1,
        ):
            logger.info(f"[{index}/{len(tasks)}] {task.doc_id}")
            for result in results:
                row, lines = self._write_page(task, result, notes, output_dir)
                rows.append(row)
                line_rows.extend(lines)

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = pd.DataFrame(rows)
        manifest.to_csv(output_dir / "manifest.csv", index=False)
        pd.DataFrame(line_rows).to_csv(output_dir / "lines.csv", index=False)
        return manifest

    def _process_all(
        self,
        tasks: List[DocTask],
        workers: int,
        threads_per_worker: Optional[int],
    ) -> Iterable[Tuple[List[PageResult], List[str]]]:
        """Process documents in order, in worker processes when workers > 1."""
        if workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                yield self.process_document(task)
            return
        threads = threads_per_worker or max(1, (os.cpu_count() or 1) // workers)
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self, threads),
        ) as pool:
            yield from pool.map(_process_in_worker, tasks)

    def _page_status(
        self, result: PageResult, notes: List[str]
    ) -> Tuple[str, List[str]]:
        """
        ``auto`` or ``review``, and the reasons (also given for auto pages).

        ``untranscribed`` when no transcription text was assigned to the
        page. A page goes to review when nothing could be paired, or when more
        than ``review_tolerance`` of its pairs are doubtful or of its
        transcription (in characters) found no line.
        """
        main = result.main
        paired = [ln for ln in main if ln.text]
        doubtful = [ln for ln in paired if not ln.confident]
        transcribed = result.unpaired_chars + sum(
            len(normalize_for_matching(ln.text)) for ln in paired
        )
        reasons = list(result.reasons)
        if not paired and not result.unpaired_chars and not reasons:
            # Nothing of the transcription belongs here (blank page, cover,
            # untranscribed page): nothing to train on or to review.
            return "untranscribed", ["no transcription for this page"] + notes
        blocking = bool(reasons)
        if not main:
            reasons.append("no lines segmented")
            blocking = True
        if not paired and main:
            reasons.append("no transcription paired")
            blocking = True
        if result.unpaired_texts:
            reasons.append(
                f"{result.unpaired_texts} transcription part(s) without a line "
                f"({result.unpaired_chars} chars)"
            )
            blocking |= result.unpaired_chars > self.review_tolerance * transcribed
        if doubtful:
            reasons.append(f"{len(doubtful)} low-confidence pair(s)")
            blocking |= len(doubtful) > self.review_tolerance * len(paired)
        reasons.extend(notes)
        return ("review" if blocking else "auto"), reasons

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
        doubtful_ids = {id(ln) for ln in doubtful}
        status, reasons = self._page_status(result, notes)

        target = output_dir / status
        # A page re-run with other settings may change folder: drop old copies
        for other in PAGE_STATUSES:
            if other != status:
                for old in (output_dir / other).glob(f"{stem}.*"):
                    old.unlink()
        image_name = f"{stem}{page.path.suffix}"
        _place_image(page.path, target / image_name)
        # Doubtful lines of auto pages are written without text: they are
        # left out of training instead of sending the page to review.
        xml_lines = (
            [
                replace(ln, text=None) if id(ln) in doubtful_ids else ln
                for ln in result.lines
            ]
            if status == "auto"
            else result.lines
        )
        write_page_xml(
            xml_lines,
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
