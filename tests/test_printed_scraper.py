"""Tests for the printed-works scraper (offline; markup mirrors the live site)."""

import json
from unittest.mock import Mock

import pandas as pd

from ticha.core.printed_scraper import TichaPrintedScraper, html_to_text, write_corpus

BASE = "https://ticha.haverford.edu"

INDEX_HTML = """
<ul class="mininav navbar-nav">
  <li><a href="/en/texts/handwritten/">Manuscripts</a></li>
  <li class="dropdown">
    <a class="dropdown-toggle" data-toggle="dropdown" href="#">Teitipac Testament</a>
    <ul class="dropdown-menu"><li><a href="/en/text/Te744/0/">Text</a></li></ul>
  </li>
  <li class="dropdown">
    <a class="dropdown-toggle" data-toggle="dropdown" href="#">Cordova, Arte <b class="caret"></b></a>
    <ul class="dropdown-menu">
      <li><a href="/en/arte/">Overview</a></li>
      <li><a href="/en/texts/cordova-arte/0/original/">Text</a></li>
    </ul>
  </li>
  <li class="dropdown">
    <a class="dropdown-toggle" data-toggle="dropdown" href="#">Aguero, Miscelaneo</a>
    <ul class="dropdown-menu"><li><a href="/en/texts/aguero-miscelaneo/0/original/">Text</a></li></ul>
  </li>
</ul>
<a href="/en/texts/cordova-arte/0/original/">Text</a>
"""

# /texts/<slug>/<n>/json: words with an analysis are popovers with a hidden gloss
CORDOVA_PAGE = """<div class="printed-text-page cordova-arte" data-n="5v"><div id="cordova-arte1">
<p>
   verbales. V.g. Dizen
   <span class="popover-markup inline"><mark class="trigger">nina,
   chija</mark><span class="content hide inline"><table><caption>nina chija</caption>
   <tr><td>nina</td><td>chi</td></tr><tr><td>REL</td><td>love</td></tr></table></span></span>,
   mi amado, <br>el de aquel. Y tam-<br>bien
</p>
<h4>DE LOS PRONOMBRES</h4>
<div class="catch">ver-</div>
</div></div>"""

AGUERO_HTML = """<html><body>
<script>documentViewerInit('aguero miscelaneo', 0, 2, {});</script>
<div class="tab-pane active" id="tab1-slug">
  <div>gabijlla ní raaqui, ni richoohuij Bezeeloo.</div>
  <div>Discipulo| Channa Maestrohe!</div>
</div>
<div id="tab2-slug"></div>
</body></html>"""


def response(text="", status=200):
    mock = Mock(status_code=status, text=text, content=text.encode())
    mock.json.side_effect = lambda: json.loads(text)
    return mock


def scraper_with(pages):
    session = Mock(headers={})
    session.get.side_effect = lambda url, timeout=None: pages.get(
        url, response(status=404)
    )
    return TichaPrintedScraper(rate_limit=0, session=session), session


def test_html_to_text_drops_glosses_and_catchwords():
    result = html_to_text(CORDOVA_PAGE)

    assert result["text"] == (
        "verbales. V.g. Dizen nina, chija, mi amado,\n"
        "el de aquel. Y tam-\nbien\n\nDE LOS PRONOMBRES"
    )
    assert "REL" not in result["text"] and "ver-" not in result["text"]
    assert result["marked"] == "nina, chija"


def test_html_to_text_keeps_sibling_divs():
    html = " <div>\n  Nijaxtenni toobi peaazi.\n </div>\n <div>\n  Discipulo| Channa!\n </div>"

    assert (
        html_to_text(html)["text"] == "Nijaxtenni toobi peaazi.\n\nDiscipulo| Channa!"
    )


def test_html_to_text_ignores_transcription_form():
    html = """<div id="tab1-slug"><div class="tab-pane" id="submitTranscription">
      <p>Instructions for Transcribing</p><script>function copyTranscription(){}</script>
      <form><div id="transcription_input"></div><textarea id="id_text"></textarea></form>
    </div></div>"""

    assert html_to_text(html)["text"] is None


def test_html_to_text_empty():
    assert html_to_text("") == {"text": None, "marked": None}


def test_list_works_finds_printed_texts_only():
    scraper, _ = scraper_with({f"{BASE}/en/texts/": response(INDEX_HTML)})

    works = scraper.list_works()

    assert works["slug"].tolist() == ["cordova-arte", "aguero-miscelaneo"]
    assert works["title"].tolist() == ["Cordova, Arte", "Aguero, Miscelaneo"]


def test_scrape_page_uses_json():
    payload = json.dumps(
        {"transcription": CORDOVA_PAGE, "transcription_regular": "", "n": "5v"}
    )
    scraper, _ = scraper_with({f"{BASE}/texts/cordova-arte/10/json": response(payload)})

    row = scraper.scrape_page("cordova-arte", 10, "Cordova, Arte")

    assert row["page_label"] == "5v"
    assert row["transcription"].startswith("verbales.")
    assert row["transcription_regular"] is None
    assert row["zapotec_marked"] == "nina, chija"


def test_scrape_page_falls_back_to_page_html():
    empty = json.dumps({"transcription": "", "transcription_regular": "", "n": "1"})
    scraper, _ = scraper_with(
        {
            f"{BASE}/texts/aguero-miscelaneo/1/json": response(empty),
            f"{BASE}/en/texts/aguero-miscelaneo/1/original/": response(AGUERO_HTML),
        }
    )

    row = scraper.scrape_page("aguero-miscelaneo", 1)

    assert row["transcription"] == (
        "gabijlla ní raaqui, ni richoohuij Bezeeloo.\n\n" "Discipulo| Channa Maestrohe!"
    )


def test_scrape_resumes_and_writes_corpus(tmp_path):
    empty = json.dumps({"transcription": "", "transcription_regular": "", "n": "x"})
    pages = {
        f"{BASE}/en/texts/": response(INDEX_HTML),
        f"{BASE}/en/texts/aguero-miscelaneo/0/original/": response(AGUERO_HTML),
    }
    for n in range(3):
        pages[f"{BASE}/texts/aguero-miscelaneo/{n}/json"] = response(empty)
        pages.setdefault(
            f"{BASE}/en/texts/aguero-miscelaneo/{n}/original/", response(AGUERO_HTML)
        )
    scraper, session = scraper_with(pages)
    output = tmp_path / "printed.csv"
    pd.DataFrame(
        [{"slug": "aguero-miscelaneo", "page": 0, "transcription": "ya"}]
    ).to_csv(output, index=False)

    result = scraper.scrape(output, works=["aguero-miscelaneo"])

    assert result["page"].tolist() == [0, 1, 2]
    requested = [call.args[0] for call in session.get.call_args_list]
    assert f"{BASE}/texts/aguero-miscelaneo/0/json" not in requested
    assert pd.read_csv(output)["page"].tolist() == [0, 1, 2]

    (path,) = write_corpus(result, tmp_path / "corpus")
    assert path.name == "aguero-miscelaneo.txt"
    assert path.read_text(encoding="utf-8").startswith("ya\n\ngabijlla")
