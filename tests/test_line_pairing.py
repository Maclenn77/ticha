"""Tests for line segmentation pairing (offline: fake segmenter/recognizer)."""

import xml.etree.ElementTree as ET

import pandas as pd
import pytest
from PIL import Image

from ticha.core.line_pairing import (
    PAGE_NS,
    Line,
    LinePairer,
    align,
    cer,
    normalize_for_matching,
    order_lines,
    split_transcription_pages,
    transcription_lines,
)

NS = {"p": PAGE_NS}


def make_line(x0, x1, y, region="r1"):
    return {
        "baseline": [[x0, y], [x1, y]],
        "boundary": [[x0, y - 30], [x1, y - 30], [x1, y + 10], [x0, y + 10]],
        "regions": [region],
    }


def page_segmentation():
    """A page like Tl675a: folio number, title, 3 body lines with an
    interlinear insertion, and a two-column signature block."""
    return {
        "lines": [
            make_line(2100, 2180, 100),  # folio number (tiny)
            make_line(300, 1100, 380),  # title (short, own row)
            make_line(250, 2450, 600),
            make_line(2200, 2420, 650),  # insertion squeezed above next line
            make_line(380, 2450, 700),
            make_line(380, 2450, 800),
            make_line(1600, 2250, 1100),  # right signature column, row 1
            make_line(360, 940, 1105),  # left signature column, row 1
            make_line(400, 1050, 1190),  # left, row 2
            make_line(1690, 2200, 1200),  # right, row 2
        ],
        "regions": [
            {"id": "r1", "boundary": [[0, 0], [2600, 0], [2600, 1300], [0, 1300]]}
        ],
    }


PAGE_TEXT = """escritora de uenta
Na pedro andres xoana
suani Ana chij Jueues a 14
cicanali rotia tobi cuelayo
Domingo rami                    lona Jhuan
res bernabe testigo            Garsia esc[ri]u[an] o"""

# Rough HTR readings for the 8 main lines of page_segmentation()
READINGS = [
    "escritora deuenta",
    "Na pedro andres xoana",
    "suani Ana chij jueues a 14",
    "cicanali rotia tobi cuelayo",
    "Domingo rami",
    "lona Jhuan",
    "res bernabe testigo",
    "Garsia escu o",
]


def fake_recognizer(readings_by_image):
    def recognize(path, lines):
        readings = readings_by_image[path.parent.name + "/" + path.name]
        return readings[: len(lines)] + [""] * (len(lines) - len(readings))

    return recognize


# -- text helpers -------------------------------------------------------- #


def test_split_transcription_pages():
    text = "(7r)\nline a\nline b\n(7v)\nline c\n[8r] line d\nline e"
    assert split_transcription_pages(text) == [
        ("7r", "line a\nline b"),
        ("7v", "line c"),
        ("8r", "line d\nline e"),
    ]
    assert split_transcription_pages("no markers\nhere") == [
        (None, "no markers\nhere")
    ]
    assert split_transcription_pages("(446)\na\n( 447 )\nb") == [
        ("446", "a"),
        ("447", "b"),
    ]


def test_transcription_lines_split_columns():
    assert transcription_lines(PAGE_TEXT)[-4:] == [
        "Domingo rami",
        "lona Jhuan",
        "res bernabe testigo",
        "Garsia esc[ri]u[an] o",
    ]
    assert len(transcription_lines(PAGE_TEXT, split_columns=False)) == 6


def test_normalize_and_cer():
    assert (
        normalize_for_matching("Sap[edr]o quijaa (...) Señor, Yva")
        == "sapoquiiaasenoriua"
    )
    assert cer("Sapo quiyaa", "Sap[edr]o quijaa") == 0
    assert cer("completely different", "nada que ver aqui") > 0.6


# -- geometry ------------------------------------------------------------ #


def test_order_lines_marks_fragments_and_reads_columns_by_row():
    lines = [
        Line(ln["baseline"], ln["boundary"]) for ln in page_segmentation()["lines"]
    ]
    ordered = order_lines(lines)
    fragments = [ln for ln in ordered if ln.fragment]
    assert [ln.baseline[0] for ln in fragments] == [[2100, 100], [2200, 650]]
    main_x_y = [tuple(ln.baseline[0]) for ln in ordered if not ln.fragment]
    assert main_x_y == [
        (300, 380), (250, 600), (380, 700), (380, 800),
        (360, 1105), (1600, 1100), (400, 1190), (1690, 1200),
    ]


def test_order_lines_reads_spread_left_page_first():
    left = [make_line(100, 900, y) for y in (200, 300, 400)]
    right = [make_line(1100, 1900, y) for y in (210, 310)]
    lines = [Line(ln["baseline"], ln["boundary"]) for ln in right + left]
    ordered = order_lines(lines, image_size=(2000, 1300))
    assert [tuple(ln.baseline[0]) for ln in ordered] == [
        (100, 200), (100, 300), (100, 400), (1100, 210), (1100, 310),
    ]


# -- alignment ----------------------------------------------------------- #


def test_align_skips_untranscribed_lines_and_missing_lines():
    readings = [
        "Valga para el Reynado",
        "SELLO QUARTO",
        "Don Domingo Correro",
        "ca de la Jurisdiccion",
        "seda y haya lugar",
    ]
    texts = [
        "Don Domingo Correro Carique",
        "ca de la Jurisdicción de este",
        "una linea que kraken no encontro",
        "seda y haya lugar Ante",
    ]
    path = align(readings, texts)
    pairs = [(i, j) for i, j, _ in path if i is not None and j is not None]
    assert pairs == [(2, 0), (3, 1), (4, 3)]
    assert (None, 2, None) in path
    assert (0, None, None) in path and (1, None, None) in path


# -- pipeline ------------------------------------------------------------ #


@pytest.fixture
def inputs(tmp_path):
    images = tmp_path / "images"
    for doc in ("Tl675a", "Two", "Miss"):
        (images / doc).mkdir(parents=True)
    for path in [
        "Tl675a/page_001.png",
        "Two/page_001.png",
        "Two/page_002.png",
        "Miss/page_001.png",
    ]:
        Image.new("RGB", (2600, 3000), "white").save(images / path)

    ids = ["Tl675a", "Two", "Miss", "Empty", "Prog"]
    texts = pd.DataFrame({
        "url": [f"u/{i}" for i in ids],
        "ticha_id": ids,
        "trptn_status": ["Completed and Checked"] * 4 + ["In Progress"],
        "transcription": [
            PAGE_TEXT,
            # no markers: aligned across both images
            PAGE_TEXT + "\nlinea de la segunda pagina",
            # folio 1v has no image
            f"(1r)\n{PAGE_TEXT}\n(1v)\nsin imagen",
            None,
            "x",
        ],
    })
    images_df = pd.DataFrame({
        "url": [f"u/{i}" for i in ids],
        "image_paths": [
            "images/Tl675a/page_001.png",
            "images/Two/page_001.png;images/Two/page_002.png",
            "images/Miss/page_001.png",
            "images/Tl675a/page_001.png",
            "images/Tl675a/page_001.png",
        ],
        "image_folios": ["8r", "1;2", "1r", "8r", "8r"],
    })
    texts.to_csv(tmp_path / "texts.csv", index=False)
    images_df.to_csv(tmp_path / "images.csv", index=False)
    return tmp_path


def segmenter(path):
    if path.parent.name == "Two" and path.name == "page_002.png":
        return {"lines": [make_line(300, 2400, 300)], "regions": []}
    return page_segmentation()


READINGS_BY_IMAGE = {
    "Tl675a/page_001.png": READINGS,
    "Two/page_001.png": READINGS,
    "Two/page_002.png": ["linea dela segunda pagina"],
    "Miss/page_001.png": READINGS,
}


def test_run_with_recognizer(inputs):
    out = inputs / "dataset"
    pairer = LinePairer(
        segmenter=segmenter,
        recognizer=fake_recognizer(READINGS_BY_IMAGE),
        export_lines=True,
    )
    manifest = pairer.run(inputs / "texts.csv", inputs / "images.csv", out)
    by_page = {(r.doc_id, r.page): r for r in manifest.itertuples()}

    assert by_page[("Tl675a", 1)].status == "auto"
    assert by_page[("Tl675a", 1)].confident_lines == 8
    # Unmarked transcription aligned across pages
    assert by_page[("Two", 1)].status == "auto"
    assert by_page[("Two", 2)].status == "auto"
    assert by_page[("Two", 2)].paired_lines == 1
    # A missing image is noted, but is no reason for review
    assert by_page[("Miss", 1)].status == "auto"
    assert "without image: 1v" in by_page[("Miss", 1)].reason
    skipped = manifest[manifest.status == "skipped"].set_index("doc_id")["reason"]
    assert skipped["Empty"] == "no transcription"
    assert skipped["Prog"] == "status: In Progress"

    xml = ET.parse(out / "auto" / "Tl675a_page_001.xml")
    page = xml.find("p:Page", NS)
    assert page.get("imageFilename") == "Tl675a_page_001.png"
    assert (out / "auto" / "Tl675a_page_001.png").exists()
    lines = page.findall(".//p:TextLine", NS)
    texts = [ln.findtext("p:TextEquiv/p:Unicode", namespaces=NS) for ln in lines]
    assert texts[:3] == [None, "escritora de uenta", "Na pedro andres xoana"]
    assert texts[-4:] == [
        "Domingo rami",
        "lona Jhuan",
        "res bernabe testigo",
        "Garsia esc[ri]u[an] o",
    ]
    assert "fragment" in lines[0].get("custom")
    assert lines[2].find("p:TextEquiv", NS).get("conf") == "1.000"
    assert lines[1].find("p:Baseline", NS).get("points") == "300,380 1100,380"

    gt = sorted((out / "lines" / "Tl675a_page_001").glob("*.gt.txt"))
    assert len(gt) == 8
    assert gt[0].read_text().strip() == "escritora de uenta"
    assert Image.open(gt[0].with_name("line_001.png")).size == (801, 41)

    lines_csv = pd.read_csv(out / "lines.csv")
    assert set(lines_csv.columns) >= {
        "doc_id", "page", "line", "text", "reading", "cer", "confident", "crop",
    }


def test_low_confidence_page_goes_to_review(inputs):
    bad = dict(READINGS_BY_IMAGE)
    bad["Tl675a/page_001.png"] = (
        READINGS[:3] + ["xxxxxxxx qqqq zzzzzz"] + READINGS[4:]
    )
    out = inputs / "dataset"
    manifest = LinePairer(segmenter=segmenter, recognizer=fake_recognizer(bad)).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    row = manifest[manifest.doc_id == "Tl675a"].iloc[0]
    assert row.status == "review"
    assert (out / "review" / "Tl675a_page_001.xml").exists()
    assert "cicanali rotia" in (out / "review" / "Tl675a_page_001.txt").read_text()


def test_run_without_recognizer_pairs_in_order(inputs):
    out = inputs / "dataset"
    manifest = LinePairer(segmenter=segmenter).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    by_page = {(r.doc_id, r.page): r for r in manifest.itertuples()}
    assert by_page[("Tl675a", 1)].status == "auto"
    # Without a recognizer, a multi-page alignment cannot be trusted
    assert by_page[("Two", 1)].status == "review"
    assert "line count mismatch" in by_page[("Two", 1)].reason


def test_statuses_none_keeps_all(inputs):
    pairer = LinePairer(segmenter=segmenter, statuses=None)
    tasks, skipped = pairer.load(inputs / "texts.csv", inputs / "images.csv")
    assert "Prog" in {t.doc_id for t in tasks}
    assert [s["doc_id"] for s in skipped] == ["Empty"]


def test_segmenter_error_is_recorded(inputs):
    def broken(path):
        raise RuntimeError("boom")

    manifest = LinePairer(segmenter=broken).run(
        inputs / "texts.csv", inputs / "images.csv", inputs / "out"
    )
    assert set(manifest[manifest.doc_id == "Tl675a"].status) == {"error"}
