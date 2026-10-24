"""Tests for line segmentation pairing (offline: fake segmenter/recognizer)."""

import xml.etree.ElementTree as ET

import pandas as pd
import pytest
from PIL import Image

from ticha.core.line_pairing import (
    PAGE_NS,
    DocTask,
    Line,
    LinePairer,
    PageImage,
    align,
    align_words,
    cer,
    normalize_for_matching,
    order_lines,
    split_transcription_pages,
    split_words,
    transcription_lines,
    words_text,
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


def test_split_words_keeps_editorial_text_and_breaks():
    texts = ["q[ue] su (?) tengo", "(?) Juan"]
    words = split_words(texts)
    assert [w.norm for w in words] == ["q", "su", "tengo", "iuan"]
    assert words_text(texts, words, 1, 4) == "su (?) tengo (?) Juan"
    assert [w.line_start for w in words] == [True, False, False, True]
    assert [w.line_end for w in words] == [False, False, True, True]


def test_align_words_recuts_reflowed_transcription():
    # Transcription lines do not follow the manuscript lines
    texts = ["pagaran la pena de ocho pesos que se aga el", "gasto de qualquiera justicia"]
    readings = ["pagaran la pena de ocho", "pesos que se aga el gasto", "de qualquiera iusticia"]
    words = split_words(texts)
    steps = align_words(readings, words)
    cut = [words_text(texts, words, a, b) for kind, _, a, b, _ in steps if kind == "pair"]
    assert cut == [
        "pagaran la pena de ocho",
        "pesos que se aga el gasto",
        "de qualquiera justicia",
    ]


def test_align_words_merges_lines_and_skips_untranscribed():
    texts = ["Alcalde Juan Lopes", "Regidor Andres de la Cruz", "y lo firme"]
    readings = ["papel sellado quartillo", "Alcalde Juan Lopes Regidor Andres dela Cruz", "y lo firme"]
    words = split_words(texts)
    steps = align_words(readings, words)
    assert steps[0][:2] == ("line", 0)
    pairs = [(k, words_text(texts, words, a, b)) for kind, k, a, b, _ in steps if kind == "pair"]
    assert pairs == [
        (1, "Alcalde Juan Lopes Regidor Andres de la Cruz"),
        (2, "y lo firme"),
    ]


def test_hyphenated_word_is_split_between_lines(tmp_path):
    # The transcription keeps "Castillo" whole; the scribe broke it: "Cas-" / "tillo"
    texts = ["que paguen donde de via este difunto aora disen Castillo y su muger", "ya dicha"]
    readings = ["que paguen donde de via este difunto, aora disen Cas-", "tillo y su muger ya dicha"]
    image = tmp_path / "page_001.png"
    Image.new("RGB", (2600, 600), "white").save(image)

    def segment(path):
        return {"lines": [make_line(300, 2400, 300), make_line(300, 2400, 400)], "regions": []}

    pairer = LinePairer(segmenter=segment, recognizer=lambda path, lines: readings)
    page = pairer._segment(PageImage(1, image))
    pairer._pair_unit([page], texts)
    assert [ln.text for ln in page.main] == [
        "que paguen donde de via este difunto aora disen Cas-",
        "tillo y su muger ya dicha",
    ]
    assert all(ln.confident for ln in page.main)



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


class Recognizer:
    """Picklable fake recognizer, for worker processes."""

    def __init__(self, readings_by_image):
        self.readings_by_image = readings_by_image

    def __call__(self, path, lines):
        return fake_recognizer(self.readings_by_image)(path, lines)


def test_parallel_run_matches_sequential(inputs):
    def run(workers):
        out = inputs / f"dataset_{workers}"
        manifest = LinePairer(
            segmenter=segmenter,
            recognizer=Recognizer(READINGS_BY_IMAGE),
            export_lines=True,
        ).run(inputs / "texts.csv", inputs / "images.csv", out, workers=workers)
        return out, manifest

    seq_out, seq = run(1)
    par_out, par = run(2)
    path_cols = ["image", "xml"]
    pd.testing.assert_frame_equal(
        seq.drop(columns=path_cols),
        par.drop(columns=path_cols),
    )
    seq_lines = pd.read_csv(seq_out / "lines.csv").drop(columns="crop")
    par_lines = pd.read_csv(par_out / "lines.csv").drop(columns="crop")
    pd.testing.assert_frame_equal(seq_lines, par_lines)
    def files(out):
        return sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())

    assert files(seq_out) == files(par_out)


def test_low_confidence_page_goes_to_review(inputs):
    bad = dict(READINGS_BY_IMAGE)
    bad["Tl675a/page_001.png"] = (
        READINGS[:3] + ["xxxxxxxx qqqq zzzzzz"] + READINGS[4:]
    )
    out = inputs / "dataset"
    manifest = LinePairer(
        segmenter=segmenter, recognizer=fake_recognizer(bad), review_tolerance=0
    ).run(inputs / "texts.csv", inputs / "images.csv", out)
    row = manifest[manifest.doc_id == "Tl675a"].iloc[0]
    assert row.status == "review"
    assert (out / "review" / "Tl675a_page_001.xml").exists()
    assert "cicanali rotia" in (out / "review" / "Tl675a_page_001.txt").read_text()


def test_few_doubtful_lines_stay_auto_without_text(inputs):
    bad = dict(READINGS_BY_IMAGE)
    bad["Tl675a/page_001.png"] = (
        READINGS[:3] + ["xxxxxxxx qqqq zzzzzz"] + READINGS[4:]
    )
    out = inputs / "dataset"
    manifest = LinePairer(segmenter=segmenter, recognizer=fake_recognizer(bad)).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    row = manifest[manifest.doc_id == "Tl675a"].iloc[0]
    assert row.status == "auto"  # 1 doubtful line of 8 is within the tolerance
    assert row.confident_lines == 7
    xml = ET.parse(out / "auto" / "Tl675a_page_001.xml")
    texts = [
        ln.findtext("p:TextEquiv/p:Unicode", namespaces=NS)
        for ln in xml.iterfind(".//p:TextLine", NS)
    ]
    assert "cicanali rotia tobi cuelayo" not in texts
    assert "suani Ana chij Jueues a 14" in texts


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


def test_rerun_moves_page_out_of_old_folder(inputs):
    bad = dict(READINGS_BY_IMAGE)
    bad["Tl675a/page_001.png"] = READINGS[:3] + ["xxxxxxxx qqqq zzzzzz"] + READINGS[4:]
    out = inputs / "dataset"
    LinePairer(segmenter=segmenter, recognizer=fake_recognizer(bad), review_tolerance=0).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    assert (out / "review" / "Tl675a_page_001.xml").exists()
    LinePairer(segmenter=segmenter, recognizer=fake_recognizer(READINGS_BY_IMAGE)).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    assert (out / "auto" / "Tl675a_page_001.xml").exists()
    assert not list((out / "review").glob("Tl675a_page_001.*"))


def test_check_draws_sheets_of_selected_lines(inputs):
    from ticha.core.check import check_pairing, select_lines

    out = inputs / "dataset"
    LinePairer(segmenter=segmenter, recognizer=fake_recognizer(READINGS_BY_IMAGE)).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    rows = select_lines(out, only="confident", docs=["Tl675a"], sample=None)
    assert len(rows) == 8
    assert set(rows["kind"]) == {"confident"}

    sheets = check_pairing(out, inputs / "check" / "sheet.png", docs=["Tl675a"],
                           sample=None, per_sheet=5)
    assert [s.name for s in sheets] == ["sheet_001.png", "sheet_002.png"]
    assert Image.open(sheets[0]).width == 1600


def test_word_broken_without_hyphen_is_split(tmp_path):
    texts = ["y yo el escrivano doy fe de palabra de Verdad, como comparecio Juana"]
    readings = ["y yo el escrivano doy fe de palabra dever", "dad, como comparecio Juana"]
    image = tmp_path / "page_001.png"
    Image.new("RGB", (2600, 600), "white").save(image)

    def segment(path):
        return {"lines": [make_line(300, 2400, 300), make_line(300, 2400, 400)], "regions": []}

    pairer = LinePairer(segmenter=segment, recognizer=lambda path, lines: readings)
    page = pairer._segment(PageImage(1, image))
    pairer._pair_unit([page], texts)
    assert [ln.text for ln in page.main] == [
        "y yo el escrivano doy fe de palabra de Ver",
        "dad, como comparecio Juana",
    ]


def test_page_with_swapped_folio_is_paired_with_the_other_folio(tmp_path):
    recto = ["en el pueblo de santo domingo", "ante los alcaldes ordinarios",
             "parecio juana de silva viuda", "y vendio un pedaso de tierra",
             "que heredo de su marido difunto"]
    verso = ["y lo firme en esta ciudad", "de antequera a veinte de julio",
             "manuel diaz interprete nombrado", "ante mi juan de la cruz escribano",
             "testigos pedro y pablo mendes"]
    pages = [PageImage(1, tmp_path / "p1.png", "1r"), PageImage(2, tmp_path / "p2.png", "1v")]
    readings = {"p1.png": verso, "p2.png": recto}  # the images are swapped
    for page in pages:
        Image.new("RGB", (2600, 800), "white").save(page.path)

    def segment(path):
        return {"lines": [make_line(300, 2400, 150 * (i + 1)) for i in range(5)], "regions": []}

    pairer = LinePairer(segmenter=segment, recognizer=lambda path, lines: readings[path.name])
    task = DocTask("Doc", pages, "(1r)\n" + "\n".join(recto) + "\n(1v)\n" + "\n".join(verso))
    results, notes = pairer.process_document(task)
    assert [ln.text for ln in results[0].main] == verso
    assert [ln.text for ln in results[1].main] == recto
    assert all(ln.confident for r in results for ln in r.main)
    assert any("paired with folio 1v" in n for n in notes)


def test_transcription_lines_drop_line_references():
    assert transcription_lines("63r-1 loo ni cooti\n63r-5 Joseph de Paauedra\n8r de") == [
        "loo ni cooti", "Joseph de Paauedra", "8r de",
    ]


def test_apply_review_keeps_fixes_and_drops_lines(inputs, tmp_path):
    from ticha.core.review import apply_review, read_decisions

    bad = dict(READINGS_BY_IMAGE)
    bad["Tl675a/page_001.png"] = READINGS[:3] + ["xxxxxxxx qqqq zzzzzz"] + READINGS[4:]
    out = inputs / "dataset"
    LinePairer(segmenter=segmenter, recognizer=fake_recognizer(bad), review_tolerance=0).run(
        inputs / "texts.csv", inputs / "images.csv", out
    )
    decisions = tmp_path / "decisions.txt"
    decisions.write_text(
        "# doc page line action [text]\n"
        "Tl675a 1 4 fix cicanali rotia tobi cuelayo\n"
        "Tl675a 1 2 drop\n"
        "Gone 9 1 keep\n",
        encoding="utf-8",
    )
    assert read_decisions(decisions)[("Tl675a", 1, 4)].text == "cicanali rotia tobi cuelayo"
    result = apply_review(out, decisions, tmp_path / "reviewed")
    assert result[["kept", "fixed", "dropped"]].sum().tolist() == [6, 1, 1]
    assert result.attrs["unused"] == [("Gone", 9, 1)]
    xml = ET.parse(tmp_path / "reviewed" / "Tl675a_page_001.xml")
    texts = [
        ln.findtext("p:TextEquiv/p:Unicode", namespaces=NS)
        for ln in xml.iterfind(".//p:TextLine", NS)
        if "fragment" not in ln.get("custom", "")
    ]
    assert texts[1] is None  # dropped
    assert texts[3] == "cicanali rotia tobi cuelayo"  # fixed


def test_read_decisions_rejects_bad_lines(tmp_path):
    from ticha.core.review import read_decisions

    path = tmp_path / "d.txt"
    path.write_text("Al642 2 4 fix\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs the corrected text"):
        read_decisions(path)
