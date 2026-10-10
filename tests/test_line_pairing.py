"""Tests for line segmentation pairing (offline, fake segmenter)."""

import xml.etree.ElementTree as ET

import pandas as pd
import pytest
from PIL import Image

from ticha.core.line_pairing import (
    PAGE_NS,
    Line,
    LinePairer,
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


def test_split_transcription_pages():
    text = "(7r)\nline a\nline b\n(7v)\nline c\n[8r] line d\nline e"
    assert split_transcription_pages(text) == [
        ("7r", "line a\nline b"),
        ("7v", "line c"),
        ("8r", "line d\nline e"),
    ]
    assert split_transcription_pages("no markers\nhere") == [(None, "no markers\nhere")]
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


def test_order_lines_marks_fragments_and_reads_columns_by_row():
    lines = [
        Line(ln["baseline"], ln["boundary"]) for ln in page_segmentation()["lines"]
    ]
    ordered = order_lines(lines)
    fragments = [ln for ln in ordered if ln.fragment]
    assert [ln.baseline[0] for ln in fragments] == [[2100, 100], [2200, 650]]
    main_x_y = [tuple(ln.baseline[0]) for ln in ordered if not ln.fragment]
    assert main_x_y == [
        (300, 380),
        (250, 600),
        (380, 700),
        (380, 800),
        (360, 1105),
        (1600, 1100),
        (400, 1190),
        (1690, 1200),
    ]


@pytest.fixture
def inputs(tmp_path):
    images = tmp_path / "images"
    (images / "Tl675a").mkdir(parents=True)
    (images / "Two").mkdir()
    for path in ["Tl675a/page_001.png", "Two/page_001.png", "Two/page_002.png"]:
        Image.new("RGB", (2600, 1300), "white").save(images / path)

    texts = pd.DataFrame(
        {
            "url": ["u/Tl675a", "u/Two", "u/Bad", "u/Empty", "u/Prog"],
            "ticha_id": ["Tl675a", "Two", "Bad", "Empty", "Prog"],
            "trptn_status": ["Completed and Checked"] * 4 + ["In Progress"],
            "transcription": [
                PAGE_TEXT,
                f"(1r)\n{PAGE_TEXT}\n(1v)\nonly one line",
                "a\nb",  # one transcription page for two images
                None,
                "x",
            ],
        }
    )
    images_df = pd.DataFrame(
        {
            "url": ["u/Tl675a", "u/Two", "u/Bad", "u/Empty", "u/Prog"],
            "image_paths": [
                "images/Tl675a/page_001.png",
                "images/Two/page_001.png;images/Two/page_002.png",
                "images/Two/page_001.png;images/Two/page_002.png",
                "images/Tl675a/page_001.png",
                "images/Tl675a/page_001.png",
            ],
        }
    )
    texts.to_csv(tmp_path / "texts.csv", index=False)
    images_df.to_csv(tmp_path / "images.csv", index=False)
    return tmp_path


def test_run_sorts_pages_into_auto_and_review(inputs):
    out = inputs / "dataset"
    pairer = LinePairer(segmenter=lambda path: page_segmentation(), export_lines=True)
    manifest = pairer.run(inputs / "texts.csv", inputs / "images.csv", out)

    by_page = {(r.doc_id, r.page): r for r in manifest.itertuples()}
    assert by_page[("Tl675a", 1)].status == "auto"
    assert by_page[("Two", 1)].status == "auto"
    assert by_page[("Two", 2)].status == "review"
    assert (
        "line count mismatch: 8 segmented, 1 transcribed" in by_page[("Two", 2)].reason
    )
    assert by_page[("Bad", 1)].status == "review"
    assert "page split mismatch" in by_page[("Bad", 1)].reason
    skipped = manifest[manifest.status == "skipped"].set_index("doc_id")["reason"]
    assert skipped["Empty"] == "no transcription"
    assert skipped["Prog"] == "status: In Progress"

    # PAGE XML with text on main lines, fragments without text
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
    assert lines[1].find("p:Baseline", NS).get("points") == "300,380 1100,380"

    # Review pages carry their transcription for the reviewer
    assert (out / "review" / "Two_page_002.txt").read_text().strip() == "only one line"
    assert (out / "review" / "Bad_page_001.txt").read_text().strip() == "a\nb"

    # Line crops + ground truth for auto pages only
    gt = sorted((out / "lines" / "Tl675a_page_001").glob("*.gt.txt"))
    assert len(gt) == 8
    assert gt[0].read_text().strip() == "escritora de uenta"
    assert Image.open(gt[0].with_name("line_001.png")).size == (801, 41)
    assert not (out / "lines" / "Two_page_002").exists()
    assert (out / "manifest.csv").exists()


def test_statuses_none_keeps_all(inputs):
    pairer = LinePairer(segmenter=lambda path: page_segmentation(), statuses=None)
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
