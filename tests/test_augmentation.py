"""Tests for dataset augmentation (synthetic images, no network)."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ticha.core.augmentation import (
    Augmenter,
    DatasetAugmenter,
    ImageProfile,
    affine_matrix,
    apply_points,
    tone_map,
    transform_page_xml,
    warp,
)
from ticha.core.line_pairing import PAGE_NS, Line, write_page_xml

NS = {"p": PAGE_NS}


def page_image(size=(1200, 1600), paper=(200, 190, 170), ink=(60, 45, 30)):
    """Warm paper with dark 'text' strokes."""
    im = Image.new("RGB", size, paper)
    d = ImageDraw.Draw(im)
    for y in range(200, size[1] - 200, 80):
        d.line([(150, y), (size[0] - 150, y)], fill=ink, width=6)
    return im


def test_tone_map_moves_ink_and_paper_levels():
    a = np.array([[[50.0, 50, 50], [100, 100, 100], [200, 200, 200]]], np.float32)
    out = tone_map(a, (100, 200), (120, 220))
    assert out[0, 1, 0] == pytest.approx(120)
    assert out[0, 2, 0] == pytest.approx(220)
    assert out[0, 0, 0] < 120  # monotone below the ink level


def test_profile_ignores_white_line_background():
    a = np.full((50, 200, 3), 255, np.float32)
    a[10:40, 10:190] = (180, 170, 150)
    a[20:30, 20:180] = (70, 60, 50)
    profile = ImageProfile.from_array(a, crop=False)
    assert profile.paper < 200  # the white mask is not the paper


def test_points_follow_the_image_transform():
    im = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(im).ellipse((95, 95, 105, 105), fill="black")  # dot at (100, 100)
    m = affine_matrix(im.size, rotate=8, shear=0.1, scale=0.9)
    out = np.asarray(warp(im, m, (255, 255, 255)).convert("L"))
    ys, xs = np.nonzero(out < 128)
    ((x, y),) = apply_points(m, [[100, 100]])
    assert abs(xs.mean() - x) < 2 and abs(ys.mean() - y) < 2


def write_page(path, image_name, confs):
    lines = []
    for i, conf in enumerate(confs):
        y = 300 + i * 100
        line = Line(
            baseline=[[150, y], [1050, y]],
            boundary=[[150, y - 40], [1050, y - 40], [1050, y + 10], [150, y + 10]],
            text=f"linea {i}",
            cer=None if conf is None else 1 - conf,
            confident=conf is not None and conf >= 0.5,
        )
        lines.append(line)
    write_page_xml(lines, [], image_name, (1200, 1600), path)


def texts_of(xml_path):
    page = ET.parse(xml_path).getroot().find("p:Page", NS)
    return [
        ln.findtext("p:TextEquiv/p:Unicode", namespaces=NS)
        for ln in page.iter(f"{{{PAGE_NS}}}TextLine")
    ]


def test_transform_page_xml_keeps_confident_text_and_moves_points(tmp_path):
    src = tmp_path / "p.xml"
    write_page(src, "p.png", [0.9, 0.3, None])
    out = tmp_path / "out.xml"
    m = affine_matrix((1200, 1600), rotate=0, scale=1.0)
    m = (m[0], m[1], m[2] + 10, m[3], m[4], m[5] + 5)  # shift by (10, 5)
    kept = transform_page_xml(src, out, "new.png", m, 0.5, keep_unscored=False)
    assert kept == 1
    assert texts_of(out) == ["linea 0", None, None]
    page = ET.parse(out).getroot().find("p:Page", NS)
    assert page.get("imageFilename") == "new.png"
    first = page.find(".//p:TextLine/p:Baseline", NS).get("points")
    assert first == "160,305 1060,305"


def test_degrade_reaches_measured_ranges():
    im = page_image()
    small = im.copy()
    source = ImageProfile.from_array(np.asarray(small).astype(np.float32), width=3000)
    aug = Augmenter(seed=1, underexposed_ratio=0)
    params = aug.sample("degrade", source, geometry=False, line=False)
    out, _ = aug.apply(im, params, source)
    profile = ImageProfile.from_array(np.asarray(out).astype(np.float32))
    assert params["mode"] == "washed"
    assert profile.saturation < source.saturation / 2
    assert profile.paper > source.paper  # washed out
    assert 0.4 < params["resolution"] < 0.55  # ~1400 px for a 3000 px page


def test_keep_background_leaves_mask_white():
    im = Image.new("RGB", (300, 80), "white")
    ImageDraw.Draw(im).polygon(
        [(10, 10), (290, 20), (290, 70), (10, 60)], fill=(190, 180, 160)
    )
    aug = Augmenter(seed=4)
    source = ImageProfile.from_array(np.asarray(im).astype(np.float32), crop=False)
    for preset in ("degrade", "variety"):
        params = aug.sample(preset, source, geometry=True, line=True)
        params.update(rotate=0, shear=0, scale=1, elastic=0)
        out, _ = aug.apply(im, params, source, keep_background=True)
        assert np.asarray(out)[2, 2].tolist() == [255, 255, 255]
        assert np.asarray(out)[40, 150].tolist() != [255, 255, 255]


def test_seed_makes_output_reproducible():
    im = page_image((300, 400))
    source = ImageProfile.from_array(np.asarray(im).astype(np.float32))
    outs = []
    for _ in range(2):
        aug = Augmenter(seed=9)
        params = aug.sample("variety", source, geometry=True, line=False)
        outs.append((params, np.asarray(aug.apply(im, params, source)[0])))
    assert outs[0][0] == outs[1][0]
    assert np.array_equal(outs[0][1], outs[1][1])


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    for folder, confs in (("auto", [0.9, 0.8]), ("review", [0.9, 0.2])):
        (root / folder).mkdir(parents=True)
        page_image().save(root / folder / f"{folder}_page_001.png")
        write_page(
            root / folder / f"{folder}_page_001.xml", f"{folder}_page_001.png", confs
        )
    lines = root / "lines" / "auto_page_001"
    lines.mkdir(parents=True)
    crop = Image.new("RGB", (400, 60), "white")
    ImageDraw.Draw(crop).rectangle((5, 5, 395, 55), fill=(190, 180, 160))
    crop.save(lines / "line_001.png")
    (lines / "line_001.gt.txt").write_text("linea 0\n", encoding="utf-8")
    return root


def test_dataset_augmenter_run(dataset, tmp_path):
    out = tmp_path / "aug"
    runner = DatasetAugmenter(Augmenter(seed=3), copies=2, degrade_ratio=0.5)
    index = runner.run(dataset, out, pages=True)

    pages = index[index.kind == "page"]
    assert len(pages) == 4  # 2 pages x 2 copies
    review_copy = out / "pages" / "review_page_001_aug01.xml"
    assert texts_of(review_copy) == ["linea 0", None]  # low-confidence text dropped
    assert (out / "pages" / "review_page_001_aug01.png").exists()

    presets = sorted(json.loads(p)["preset"] for p in pages.params)
    assert presets == ["degrade", "degrade", "variety", "variety"]

    lines = index[index.kind == "line"]
    assert len(lines) == 2
    gt = sorted((out / "lines" / "auto_page_001").glob("*.gt.txt"))
    assert [g.read_text() for g in gt] == ["linea 0\n", "linea 0\n"]
    assert (out / "augment.csv").exists()


def test_dataset_augmenter_auto_only(dataset, tmp_path):
    runner = DatasetAugmenter(Augmenter(seed=3), copies=1, include_review=False)
    index = runner.run(dataset, tmp_path / "aug", pages=True, lines=False)
    assert all("auto_page" in s for s in index.source)


def test_lines_are_the_default(dataset, tmp_path):
    index = DatasetAugmenter(Augmenter(seed=3), copies=1).run(dataset, tmp_path / "aug")
    assert set(index.kind) == {"line"}
