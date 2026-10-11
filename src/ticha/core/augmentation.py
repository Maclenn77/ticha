"""Data augmentation for the Ticha line dataset.

Two families of transformations:

- ``degrade``: reproduces the look of Ticha's untranscribed manuscripts so a
  model trained on the transcribed ones transfers to them. Measured on a
  sample of untranscribed pages (mostly the Santo Domingo Etla notarial
  series) against transcribed ones, those images are:

  * about 0.6x the linear resolution (~1450 px wide instead of ~2450),
  * almost grey (saturation ~5 instead of ~20) with a slightly cool cast,
  * either washed out (paper ~220, faded ink) or, less often, underexposed
    (dark grey paper ~130-170, low ink/paper contrast),
  * unevenly lit (bound-volume gutters), sometimes stained.

  Parameters come from built-in ranges measured on that sample, or from
  real target images (``ImageProfile.from_image``), in which case each
  augmented image matches the tone, colour and resolution of one target.

- ``variety``: generic handwriting augmentations to enlarge the dataset:
  brightness/contrast/gamma, warm or cool paper, ink weight, blur, noise,
  JPEG compression, small rotation/slant/scale and, for line images,
  elastic distortion.

Line crops (``.png`` + ``.gt.txt`` from ``lines --export-lines``) are
augmented directly, so every augmented image keeps the transcription it
was paired with: nothing is segmented or aligned again. Whole pages can
optionally be augmented too (PAGE XML + image, e.g. to train
segmentation): the already paired lines are reused, geometric changes are
applied to their coordinates, and only confident lines keep their text.
"""

import json
import logging
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from .line_pairing import PAGE_NS

logger = logging.getLogger(__name__)

Matrix = Tuple[float, float, float, float, float, float]  # x' = a x + b y + c ...
IDENTITY: Matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------- #
# Image statistics
# ---------------------------------------------------------------------- #


def _luminance(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _saturation(a: np.ndarray) -> float:
    mx, mn = a.max(axis=-1), a.min(axis=-1)
    return float(np.mean(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)) * 255)


def background_mask(a: np.ndarray) -> np.ndarray:
    """Pure-white pixels: the area ``crop_line`` paints outside a line's
    polygon. Real paper never reaches it (its 99th percentile is ~230)."""
    return np.all(a >= 250, axis=-1)


@dataclass
class ImageProfile:
    """Tone and colour statistics of an image (central area)."""

    ink: float  # 5th percentile of luminance
    paper: float  # 90th percentile of luminance
    saturation: float  # mean HSV saturation, 0-255
    cast: float  # mean red - mean blue
    width: int = 0

    @classmethod
    def from_array(
        cls, a: np.ndarray, width: int = 0, crop: bool = True
    ) -> "ImageProfile":
        if crop and min(a.shape[:2]) > 16:
            h, w = a.shape[:2]
            a = a[h // 8 : h - h // 8, w // 8 : w - w // 8]
        width = width or a.shape[1]
        background = background_mask(a)
        if background.any() and (~background).sum() > 100:
            a = a[~background][None, ...]  # keep only the manuscript pixels
        y = _luminance(a)
        ink, paper = np.percentile(y, [5, 90])
        return cls(
            ink=float(ink),
            paper=float(paper),
            saturation=_saturation(a),
            cast=float(a[..., 0].mean() - a[..., 2].mean()),
            width=width,
        )

    @classmethod
    def from_image(cls, path: Any) -> "ImageProfile":
        with Image.open(path) as im:
            im = im.convert("RGB")
            width = im.width
            im.thumbnail((1200, 1200))
            return cls.from_array(np.asarray(im).astype(np.float32), width=width)


# ---------------------------------------------------------------------- #
# Photometric operations (float32 RGB arrays, 0-255)
# ---------------------------------------------------------------------- #


def tone_map(
    a: np.ndarray, src: Tuple[float, float], dst: Tuple[float, float]
) -> np.ndarray:
    """Linear tone curve moving ink level src[0] -> dst[0] and paper level
    src[1] -> dst[1], continued (and clipped) beyond them."""
    s_ink, s_paper = src
    d_ink, d_paper = dst
    if s_paper - s_ink < 5:
        return a
    d_ink = min(d_ink, d_paper - 10)
    slope = (d_paper - d_ink) / (s_paper - s_ink)
    return np.clip(d_ink + (a - s_ink) * slope, 0, 255).astype(np.float32)


def set_saturation(a: np.ndarray, factor: float) -> np.ndarray:
    gray = _luminance(a)[..., None]
    return gray + factor * (a - gray)


def shift_cast(a: np.ndarray, delta: float) -> np.ndarray:
    """Add ``delta`` to (red - blue): positive is warmer, negative cooler."""
    return a + np.array([delta / 2, 0.0, -delta / 2], dtype=np.float32)


def gamma(a: np.ndarray, g: float) -> np.ndarray:
    return 255.0 * (np.clip(a, 0, 255) / 255.0) ** g


def contrast(a: np.ndarray, factor: float, brightness: float = 0.0) -> np.ndarray:
    mean = a.mean()
    return (a - mean) * factor + mean + brightness


def illumination(
    a: np.ndarray, rng: random.Random, strength: float, gutter: float = 0.0
) -> np.ndarray:
    """Multiply by a smooth lighting field: gradient + vignette + gutter shadow."""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w - 1, 1)
    yy /= max(h - 1, 1)
    angle = rng.uniform(0, 2 * math.pi)
    grad = (xx - 0.5) * math.cos(angle) + (yy - 0.5) * math.sin(angle)
    vignette = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    field = 1 - strength * (grad + 0.5) * 0.6 - strength * vignette * 0.8
    if gutter:
        side = xx if rng.random() < 0.5 else 1 - xx
        field -= gutter * np.exp(-side / 0.06)
    return a * np.clip(field, 0.2, 1.2)[..., None]


def stains(a: np.ndarray, rng: random.Random, amount: float) -> np.ndarray:
    """Darken a few soft, brownish blotches."""
    h, w = a.shape[:2]
    small = (
        np.random.default_rng(rng.randrange(2**32)).random((8, 8)).astype(np.float32)
    )
    field = (
        np.asarray(
            Image.fromarray((small * 255).astype(np.uint8)).resize(
                (w, h), Image.BICUBIC
            )
        ).astype(np.float32)
        / 255
    )
    mask = np.clip((field - 0.7) / 0.3, 0, 1) * amount
    tint = np.array([0.85, 0.75, 0.6], dtype=np.float32)
    return a * (1 - mask[..., None] * (1 - tint))


def brown_ink(a: np.ndarray, strength: float) -> np.ndarray:
    """Give dark strokes the faded iron-gall brown seen on real scans, while
    the paper stays neutral or cool."""
    y = _luminance(a)
    ink, paper = np.percentile(y, [3, 90])
    darkness = np.clip((paper - y) / max(paper - ink, 1), 0, 1)[..., None]
    brown = np.array([14.0, 2.0, -16.0], dtype=np.float32)
    return a + strength * darkness * brown


def gaussian_noise(a: np.ndarray, rng: random.Random, sigma: float) -> np.ndarray:
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, a.shape[:2])
    return a + noise[..., None].astype(np.float32)


def _to_image(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def _to_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB")).astype(np.float32)


def resolution(im: Image.Image, factor: float) -> Image.Image:
    """Lose detail like a lower-resolution capture (size is kept)."""
    if factor >= 0.999:
        return im
    small = im.resize(
        (max(1, round(im.width * factor)), max(1, round(im.height * factor))),
        Image.BOX,
    )
    return small.resize(im.size, Image.BICUBIC)


def jpeg(im: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    im.save(buffer, "JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def ink_weight(im: Image.Image, delta: int) -> Image.Image:
    """Thicker (delta > 0) or thinner (delta < 0) dark strokes."""
    if delta > 0:
        return im.filter(ImageFilter.MinFilter(2 * delta + 1))
    if delta < 0:
        return im.filter(ImageFilter.MaxFilter(-2 * delta + 1))
    return im


# ---------------------------------------------------------------------- #
# Geometry
# ---------------------------------------------------------------------- #


def affine_matrix(
    size: Tuple[int, int], rotate: float = 0.0, shear: float = 0.0, scale: float = 1.0
) -> Matrix:
    """Forward matrix: rotation (degrees), horizontal shear (slant) and scale
    around the image centre."""
    w, h = size
    cx, cy = w / 2, h / 2
    t = math.radians(rotate)
    cos, sin = math.cos(t), math.sin(t)
    # M = R * S(scale) * Shear; x' = a x + b y + c
    a = scale * cos
    b = scale * (cos * shear - sin)
    d = scale * sin
    e = scale * (sin * shear + cos)
    c = cx - a * cx - b * cy
    f = cy - d * cx - e * cy
    return (a, b, c, d, e, f)


def invert(m: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    det = a * e - b * d
    ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
    return (ia, ib, -(ia * c + ib * f), id_, ie, -(id_ * c + ie * f))


def apply_points(m: Matrix, points: Sequence[Sequence[float]]) -> List[List[int]]:
    a, b, c, d, e, f = m
    return [[round(a * x + b * y + c), round(d * x + e * y + f)] for x, y in points]


def warp(im: Image.Image, m: Matrix, fill: Tuple[int, int, int]) -> Image.Image:
    if m == IDENTITY:
        return im
    return im.transform(
        im.size, Image.AFFINE, invert(m), resample=Image.BICUBIC, fillcolor=fill
    )


def elastic(im: Image.Image, seed: int, alpha: float, grid: int = 6) -> Image.Image:
    """Smooth random displacement (pixels up to ~alpha), for line images.
    The same seed gives the same displacement (used to move masks along)."""
    w, h = im.size
    g = np.random.default_rng(seed)
    a = _to_array(im)
    fields = []
    for _ in range(2):
        coarse = g.uniform(-1, 1, (max(2, grid * h // max(w, 1) + 2), grid + 2))
        field = Image.fromarray(((coarse + 1) * 127.5).astype(np.uint8)).resize(
            (w, h), Image.BICUBIC
        )
        fields.append((np.asarray(field).astype(np.float32) / 127.5 - 1) * alpha)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.clip(xx + fields[0], 0, w - 1)
    sy = np.clip(yy + fields[1], 0, h - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (sx - x0)[..., None], (sy - y0)[..., None]
    out = (
        a[y0, x0] * (1 - wx) * (1 - wy)
        + a[y0, x1] * wx * (1 - wy)
        + a[y1, x0] * (1 - wx) * wy
        + a[y1, x1] * wx * wy
    )
    return _to_image(out)


# ---------------------------------------------------------------------- #
# Recipes
# ---------------------------------------------------------------------- #

# Ranges measured on untranscribed Ticha pages (see module docstring)
WASHED = dict(paper=(218, 234), ink=(110, 140), gap=(75, 115))
UNDEREXPOSED = dict(paper=(125, 175), ink=(70, 115), gap=(45, 70))


class Augmenter:
    """Sample and apply random augmentations."""

    def __init__(
        self,
        targets: Optional[List[ImageProfile]] = None,
        seed: Optional[int] = None,
        underexposed_ratio: float = 0.2,
    ):
        """
        Args:
            targets: Profiles of real low-quality images to imitate in
                ``degrade``; built-in measured ranges when None
            seed: Random seed for reproducible output
            underexposed_ratio: Share of ``degrade`` outputs that are dark
                (the rest are washed out) when no targets are given
        """
        self.targets = targets or []
        self.rng = random.Random(seed)
        self.underexposed_ratio = underexposed_ratio

    # -- parameter sampling ------------------------------------------- #

    def sample(
        self, preset: str, source: ImageProfile, geometry: bool, line: bool
    ) -> Dict[str, Any]:
        r = self.rng
        if preset == "degrade":
            if self.targets:
                t = r.choice(self.targets)
                paper = t.paper + r.uniform(-4, 4)
                ink = t.ink + r.uniform(-6, 6)
                sat = t.saturation / max(source.saturation, 1)
                cast = t.cast
                res = (
                    t.width / max(source.width, 1) if t.width and source.width else 0.5
                )
                mode = "target"
            else:
                mode = (
                    "underexposed" if r.random() < self.underexposed_ratio else "washed"
                )
                ranges = UNDEREXPOSED if mode == "underexposed" else WASHED
                paper = r.uniform(*ranges["paper"])
                # keep at least the ink/paper gap seen on real pages
                ink = min(r.uniform(*ranges["ink"]), paper - r.uniform(*ranges["gap"]))
                sat = r.uniform(0.08, 0.3)
                cast = r.uniform(-6, 3)
                # untranscribed scans are ~1350-1500 px wide for a whole page
                res = r.uniform(1350, 1500) / max(source.width, 1)
            if line:
                # a crop's width says nothing about the page resolution
                res = r.uniform(0.4, 0.65)
            params = dict(
                preset=preset,
                mode=mode,
                tone=[round(paper, 1), round(ink, 1)],
                saturation=round(min(max(sat, 0.05), 1.0), 3),
                cast_target=round(cast, 1),
                resolution=round(min(max(res, 0.3), 1.0), 3),
                illumination=round(r.uniform(0.0, 0.35), 3),
                gutter=(
                    round(r.uniform(0.15, 0.4), 3)
                    if not line and r.random() < 0.4
                    else 0.0
                ),
                stains=round(r.uniform(0.2, 0.5), 3) if r.random() < 0.3 else 0.0,
                ink_tint=round(r.uniform(0.4, 1.0), 3),
                noise=round(r.uniform(0.8, 2.5), 2),
                jpeg=r.randint(45, 85),
            )
            if geometry:
                params.update(rotate=round(r.uniform(-0.8, 0.8), 2))
            return params

        if preset == "variety":
            params = dict(
                preset=preset,
                brightness=round(r.uniform(-25, 25), 1),
                contrast=round(r.uniform(0.75, 1.25), 3),
                gamma=round(r.uniform(0.8, 1.25), 3),
                saturation=round(r.uniform(0.3, 1.3), 3),
                cast=round(r.uniform(-12, 12), 1),
                ink_weight=r.choice([-1, 0, 0, 1]) if line else 0,
                blur=round(r.uniform(0.3, 1.2), 2) if r.random() < 0.4 else 0.0,
                noise=round(r.uniform(0, 4), 2) if r.random() < 0.5 else 0.0,
                jpeg=r.randint(50, 95) if r.random() < 0.5 else 0,
            )
            if geometry:
                params.update(
                    rotate=round(
                        r.uniform(-2.5, 2.5) if line else r.uniform(-1.5, 1.5), 2
                    ),
                    shear=round(
                        r.uniform(-0.2, 0.2) if line else r.uniform(-0.05, 0.05), 3
                    ),
                    scale=round(r.uniform(0.92, 1.08), 3),
                    elastic=(
                        round(r.uniform(0.5, 2.0), 2)
                        if line and r.random() < 0.5
                        else 0.0
                    ),
                )
            return params
        raise ValueError(f"Unknown preset: {preset}")

    # -- application -------------------------------------------------- #

    def apply(
        self,
        im: Image.Image,
        params: Dict[str, Any],
        source: Optional[ImageProfile] = None,
        keep_background: bool = False,
    ) -> Tuple[Image.Image, Matrix]:
        """Apply sampled parameters. Returns (image, forward point matrix).

        With ``keep_background``, pure-white pixels (outside a line crop's
        polygon) stay white, moved by the same geometric transforms.
        """
        im = im.convert("RGB")
        source = source or ImageProfile.from_array(_to_array(im))
        r = random.Random(
            json.dumps(params, sort_keys=True)
        )  # deterministic per params
        background = None
        if keep_background:
            mask = background_mask(_to_array(im))
            if mask.any():
                background = Image.fromarray((mask * 255).astype(np.uint8)).convert(
                    "RGB"
                )
        if background is not None:
            fill = (255, 255, 255)
        else:
            fill = tuple(
                int(v) for v in np.percentile(_to_array(im).reshape(-1, 3), 90, axis=0)
            )

        m = affine_matrix(
            im.size,
            rotate=params.get("rotate", 0.0),
            shear=params.get("shear", 0.0),
            scale=params.get("scale", 1.0),
        )
        im = warp(im, m, fill)
        if background is not None:
            background = warp(background, m, (255, 255, 255))
        if params.get("elastic"):
            seed = r.randrange(2**32)
            im = elastic(im, seed, params["elastic"])
            if background is not None:
                background = elastic(background, seed, params["elastic"])
        if params.get("ink_weight"):
            im = ink_weight(im, params["ink_weight"])

        a = _to_array(im)
        if params["preset"] == "degrade":
            paper, ink = params["tone"]
            a = tone_map(a, (source.ink, source.paper), (ink, paper))
            a = set_saturation(a, params["saturation"])
            if params.get("ink_tint"):
                a = brown_ink(a, params["ink_tint"])
            # land on the target red-blue cast after desaturation and ink tint
            a = shift_cast(
                a, params["cast_target"] - float(a[..., 0].mean() - a[..., 2].mean())
            )
            if params["illumination"] or params["gutter"]:
                a = illumination(a, r, params["illumination"], params["gutter"])
            if params["stains"]:
                a = stains(a, r, params["stains"])
            im = resolution(_to_image(a), params["resolution"])
            a = gaussian_noise(_to_array(im), r, params["noise"])
            im = jpeg(_to_image(a), params["jpeg"])
        else:
            a = contrast(a, params["contrast"], params["brightness"])
            a = gamma(a, params["gamma"])
            a = set_saturation(a, params["saturation"])
            a = shift_cast(a, params["cast"])
            im = _to_image(a)
            if params["blur"]:
                im = im.filter(ImageFilter.GaussianBlur(params["blur"]))
            if params["noise"]:
                im = _to_image(gaussian_noise(_to_array(im), r, params["noise"]))
            if params["jpeg"]:
                im = jpeg(im, params["jpeg"])
        if background is not None:
            a = _to_array(im)
            a[np.asarray(background)[..., 0] > 127] = 255
            im = _to_image(a)
        return im, m


# ---------------------------------------------------------------------- #
# PAGE XML
# ---------------------------------------------------------------------- #


def _q(tag: str) -> str:
    return f"{{{PAGE_NS}}}{tag}"


def _parse_points(text: str) -> List[List[float]]:
    return [[float(v) for v in p.split(",")] for p in text.split()]


def transform_page_xml(
    xml_path: Path,
    out_path: Path,
    image_name: str,
    matrix: Matrix,
    min_conf: float,
    keep_unscored: bool,
) -> int:
    """Copy a PAGE XML with transformed coordinates; returns lines with text.

    Text is kept only on lines whose ``TextEquiv/@conf`` is at least
    ``min_conf`` (lines without a score are kept if ``keep_unscored``).
    """
    ET.register_namespace("", PAGE_NS)
    tree = ET.parse(xml_path)
    page = tree.getroot().find(_q("Page"))
    page.set("imageFilename", image_name)
    for el in page.iter():
        if el.tag in (_q("Coords"), _q("Baseline")) and el.get("points"):
            pts = apply_points(matrix, _parse_points(el.get("points")))
            el.set("points", " ".join(f"{x},{y}" for x, y in pts))
    kept = 0
    for line in page.iter(_q("TextLine")):
        equiv = line.find(_q("TextEquiv"))
        if equiv is None:
            continue
        conf = equiv.get("conf")
        ok = (float(conf) >= min_conf) if conf is not None else keep_unscored
        if ok:
            kept += 1
        else:
            line.remove(equiv)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return kept


# ---------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------- #


def load_targets(directory: Any) -> List[ImageProfile]:
    """Profiles of every image in a directory (searched recursively)."""
    paths = [
        p
        for p in sorted(Path(directory).rglob("*"))
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ]
    return [ImageProfile.from_image(p) for p in paths]


class DatasetAugmenter:
    """Augment the output of ``ticha-scraper lines``."""

    def __init__(
        self,
        augmenter: Augmenter,
        copies: int = 2,
        degrade_ratio: float = 0.5,
        min_conf: float = 0.5,
        include_review: bool = True,
        geometry: bool = True,
    ):
        """
        Args:
            augmenter: Samples and applies the transformations
            copies: Augmented copies per page / line image
            degrade_ratio: Share of copies using ``degrade`` (rest ``variety``)
            min_conf: Lowest pair confidence (1 - CER) kept as text
            include_review: Also use confident lines of review/ pages
            geometry: Allow rotation/slant/scale/elastic transforms
        """
        self.augmenter = augmenter
        self.copies = copies
        self.degrade_ratio = degrade_ratio
        self.min_conf = min_conf
        self.include_review = include_review
        self.geometry = geometry

    def _presets(self) -> List[str]:
        n_degrade = round(self.copies * self.degrade_ratio)
        presets = ["degrade"] * n_degrade + ["variety"] * (self.copies - n_degrade)
        self.augmenter.rng.shuffle(presets)
        return presets

    def augment_pages(self, dataset: Path, output: Path) -> List[Dict[str, Any]]:
        rows = []
        folders = ["auto"] + (["review"] if self.include_review else [])
        for folder in folders:
            for xml_path in sorted((dataset / folder).glob("*.xml")):
                page = ET.parse(xml_path).getroot().find(_q("Page"))
                image_path = xml_path.parent / page.get("imageFilename")
                if not image_path.exists():
                    logger.warning(f"Missing image for {xml_path}")
                    continue
                with Image.open(image_path) as im:
                    im = im.convert("RGB")
                    small = im.copy()
                    small.thumbnail((1200, 1200))
                    source = ImageProfile.from_array(_to_array(small), width=im.width)
                    for k, preset in enumerate(self._presets(), 1):
                        params = self.augmenter.sample(
                            preset, source, self.geometry, line=False
                        )
                        out, matrix = self.augmenter.apply(im, params, source)
                        stem = f"{xml_path.stem}_aug{k:02d}"
                        out_dir = output / "pages"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        image_name = f"{stem}.png"
                        kept = transform_page_xml(
                            xml_path,
                            out_dir / f"{stem}.xml",
                            image_name,
                            matrix,
                            self.min_conf,
                            keep_unscored=(folder == "auto"),
                        )
                        if not kept:
                            (out_dir / f"{stem}.xml").unlink()
                            continue
                        out.save(out_dir / image_name)
                        rows.append(
                            dict(
                                kind="page",
                                source=str(xml_path),
                                output=str(out_dir / f"{stem}.xml"),
                                lines=kept,
                                **{"params": json.dumps(params)},
                            )
                        )
                logger.info(f"Augmented {xml_path.name}")
        return rows

    def augment_lines(self, dataset: Path, output: Path) -> List[Dict[str, Any]]:
        """Augment line crops (``lines/<page>/line_NNN.png`` + ``.gt.txt``).

        ``--export-lines`` only writes confidently paired lines; without
        ``include_review`` only lines of ``auto/`` pages are used.
        """
        rows = []
        auto_pages = {p.stem for p in (dataset / "auto").glob("*.xml")}
        for gt in sorted((dataset / "lines").rglob("*.gt.txt")):
            image_path = gt.with_name(gt.name[: -len(".gt.txt")] + ".png")
            if not image_path.exists():
                continue
            if not self.include_review and image_path.parent.name not in auto_pages:
                continue
            text = gt.read_text(encoding="utf-8")
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                source = ImageProfile.from_array(_to_array(im), crop=False)
                out_dir = output / "lines" / image_path.parent.name
                out_dir.mkdir(parents=True, exist_ok=True)
                for k, preset in enumerate(self._presets(), 1):
                    params = self.augmenter.sample(
                        preset, source, self.geometry, line=True
                    )
                    out, _ = self.augmenter.apply(
                        im, params, source, keep_background=True
                    )
                    stem = f"{image_path.stem}_aug{k:02d}"
                    out.save(out_dir / f"{stem}.png")
                    (out_dir / f"{stem}.gt.txt").write_text(text, encoding="utf-8")
                    rows.append(
                        dict(
                            kind="line",
                            source=str(image_path),
                            output=str(out_dir / f"{stem}.png"),
                            lines=1,
                            params=json.dumps(params),
                        )
                    )
        return rows

    def run(
        self, dataset: Any, output: Any, pages: bool = False, lines: bool = True
    ) -> pd.DataFrame:
        dataset, output = Path(dataset), Path(output)
        rows: List[Dict[str, Any]] = []
        if pages:
            rows += self.augment_pages(dataset, output)
        if lines and (dataset / "lines").exists():
            rows += self.augment_lines(dataset, output)
        output.mkdir(parents=True, exist_ok=True)
        index = pd.DataFrame(
            rows, columns=["kind", "source", "output", "lines", "params"]
        )
        index.to_csv(output / "augment.csv", index=False)
        return index


__all__ = [
    "Augmenter",
    "DatasetAugmenter",
    "ImageProfile",
    "load_targets",
    "transform_page_xml",
    "affine_matrix",
    "apply_points",
    "invert",
]
