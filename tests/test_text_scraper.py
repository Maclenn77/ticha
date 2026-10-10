"""Tests for the Ticha text scraper (offline; markup mirrors the live site)."""

from unittest.mock import Mock

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from ticha.core.text_scraper import TichaTextScraper, element_to_text

BASE = "https://ticha.haverford.edu"

METADATA_HTML = """
<div role="tabpanel" class="tab-pane active" id="metadata">
  <br>
  <p>
  <span class="meta-title">Title</span>
    Translation of Bill of Sale from San Jerónimo Tlacochahuaya, 1675
  </p>
  <p>
  <span class="meta-title">Document Type</span>
    <a href="/en/texts/doc_type/Bill of Sale" target="_blank">Bill of Sale</a>
  </p>
  <p>
  <span class="meta-title">Town (Modern Official)</span>
    San Jerónimo Tlacochahuaya
  </p>
  <p>
  <span class="meta-title">Witnesses</span>

    Marcos Pérez, Ignacio Martín (alcalde)

  </p>
  <p>
  <span class="meta-title">Permission File</span>
    <a href="http://ticha.haverford.edu/images/permissions/AGEO.pdf">[PDF]</a>
  </p>
</div>
"""

# /en/texts/<id>/ layout: transcription as raw text + <br/>
HANDWRITTEN_HTML = METADATA_HTML + """
<div role="tabpanel" class="tab-pane" id="transcription">

  (8r) <br/>
  Yo Pedro Andres Principal y natural de este Pueb de Tlacochahuaya<br/>
  oy Jueves a catorze de Marzo<br/>

</div>
<div role="tabpanel" class="tab-pane" id="interlinear">
  <p>This document does not currently have an interlinear analysis available.</p>
</div>
<div role="tabpanel" class="tab-pane" id="modern_spanish">
  <p>This document does not currently have a modern Spanish normalization
  available.</p>
</div>
"""

# /en/text/<id>/<n>/ layout: one page per URL, pageMap lists the pages
FEATURED_PAGE_HTML = """
<div id="transcription-tab" class="tab-pane active">[{folio}]
Rinij naa Beni gichij
lo geche San Matheo</div>
<div id="normalized-spanish-tab" class="tab-pane"></div>
<div id="spanish-translation-tab" class="tab-pane"></div>
<div id="english-translation-tab" class="tab-pane">Page {folio} in English</div>
<div id="analysis-tab" class="tab-pane"><p>{folio}-1. Rinij naa</p>
<p>Ri-nij&nbsp;&nbsp;&nbsp;&nbsp;naa</p>
<p><em>Say I</em></p></div>
<script>
const pageMap = {{"01r": "0", "01v": "1"}} // get the pageMap from the model
</script>
"""


def fake_session(pages):
    session = Mock()
    session.headers = {}

    def get(url, timeout=None):
        response = Mock()
        response.status_code = 200 if url in pages else 404
        response.content = pages.get(url, "").encode()
        return response

    session.get.side_effect = get
    return session


@pytest.fixture
def text_scraper():
    return TichaTextScraper(rate_limit=0)


def test_init(text_scraper):
    assert text_scraper.rate_limit == 0
    assert text_scraper.base_url == BASE


def test_normalize_metadata_key(text_scraper):
    test_cases = [
        ("Title", "title"),
        ("Call Number", "call_number"),
        ("Date Digitized", "date_digitized"),
        ("Town (Modern Official)", "town_modern_official"),
        ("Primary Parties", "primary_parties"),
        ("Permission File", "permission_file"),
    ]
    for input_key, expected in test_cases:
        assert text_scraper._normalize_metadata_key(input_key) == expected


def test_extract_metadata(text_scraper):
    soup = BeautifulSoup(METADATA_HTML, "html.parser")
    assert text_scraper._extract_metadata(soup, f"{BASE}/en/texts/Tl675aT/") == {
        "title": "Translation of Bill of Sale from San Jerónimo Tlacochahuaya, 1675",
        "document_type": "Bill of Sale",
        "town_modern_official": "San Jerónimo Tlacochahuaya",
        "witnesses": "Marcos Pérez, Ignacio Martín (alcalde)",
        "permission_file": "http://ticha.haverford.edu/images/permissions/AGEO.pdf",
    }


def test_extract_metadata_colon_format(text_scraper):
    soup = BeautifulSoup(
        '<div id="metadata"><p>Title: Test Document</p><p>Archive: AGEO</p></div>',
        "html.parser",
    )
    assert text_scraper._extract_metadata(soup) == {
        "title": "Test Document",
        "archive": "AGEO",
    }


def test_extract_metadata_no_div(text_scraper):
    soup = BeautifulSoup("<div id='other'></div>", "html.parser")
    assert text_scraper._extract_metadata(soup) == {}


def test_text_with_br_lines():
    soup = BeautifulSoup(HANDWRITTEN_HTML, "html.parser")
    assert element_to_text(soup.find(id="transcription")) == (
        "(8r)\n"
        "Yo Pedro Andres Principal y natural de este Pueb de Tlacochahuaya\n"
        "oy Jueves a catorze de Marzo"
    )


def test_text_with_p_and_br():
    soup = BeautifulSoup(
        "<div id='t'><p>\n  +\n  <br/>\n  Niateni queacani\n  <br/>\n  ua nayo"
        "</p><p>segundo</p></div>",
        "html.parser",
    )
    assert element_to_text(soup.div) == "+\nNiateni queacani\nua nayo\n\nsegundo"


def test_text_plain_lines_and_nbsp_alignment():
    soup = BeautifulSoup(FEATURED_PAGE_HTML.format(folio="1r"), "html.parser")
    assert element_to_text(soup.find(id="transcription-tab")) == (
        "[1r]\nRinij naa Beni gichij\nlo geche San Matheo"
    )
    assert element_to_text(soup.find(id="analysis-tab")) == (
        "1r-1. Rinij naa\n\nRi-nij    naa\n\nSay I"
    )


def test_placeholders_and_empty_become_none(text_scraper):
    soup = BeautifulSoup(HANDWRITTEN_HTML, "html.parser")
    assert text_scraper._extract_text_content(soup, "interlinear") is None
    assert text_scraper._extract_text_content(soup, "modern_spanish") is None
    assert text_scraper._extract_text_content(soup, "missing") is None
    empty = BeautifulSoup('<div id="x" class="tab-pane"></div>', "html.parser")
    assert text_scraper._extract_text_content(empty, "x") is None


def test_scrape_handwritten_document():
    url = f"{BASE}/en/texts/Tl675aT/"
    scraper = TichaTextScraper(
        rate_limit=0, session=fake_session({url: HANDWRITTEN_HTML})
    )
    result = scraper.scrape_document("/en/texts/Tl675aT/")
    assert result["url"] == url
    assert result["transcription"].startswith("(8r)\nYo Pedro Andres")
    assert result["interlinear"] is None
    assert result["modern_spanish"] is None
    assert result["english_translation"] is None
    assert result["title"].startswith("Translation of Bill of Sale")


def test_scrape_featured_document_joins_pages():
    text = f"{BASE}/en/text/Ma733/"
    pages = {
        f"{text}overview/": METADATA_HTML,
        f"{text}0/": FEATURED_PAGE_HTML.format(folio="1r"),
        f"{text}1/": FEATURED_PAGE_HTML.format(folio="1v"),
    }
    scraper = TichaTextScraper(rate_limit=0, session=fake_session(pages))
    result = scraper.scrape_document(f"{text}0/")
    assert "error" not in result
    assert result["page_count"] == 2
    assert result["transcription"].startswith("[1r]\nRinij")
    assert "\n\n[1v]\nRinij" in result["transcription"]
    assert result["english_translation"] == "Page 1r in English\n\nPage 1v in English"
    assert result["interlinear"].startswith("1r-1. Rinij naa")
    assert result["modern_spanish"] is None
    assert result["title"].startswith("Translation of Bill of Sale")


def test_scrape_document_http_error():
    scraper = TichaTextScraper(rate_limit=0, session=fake_session({}))
    result = scraper.scrape_document("/en/texts/Missing/")
    assert "error" in result


def test_scrape_documents_from_dataframe():
    pages = {
        f"{BASE}/en/texts/A1/": HANDWRITTEN_HTML,
        f"{BASE}/en/texts/A2/": HANDWRITTEN_HTML,
    }
    scraper = TichaTextScraper(rate_limit=0, session=fake_session(pages))
    df = pd.DataFrame({
        "document_name": ["Doc 1", "Doc 2", "No link"],
        "document_link": ["/en/texts/A1/", "/en/texts/A2/", None],
    })
    result_df = scraper.scrape_documents_from_dataframe(df)
    assert len(result_df) == 2
    assert "document_name" in result_df.columns  # Original columns preserved
    assert "title" in result_df.columns  # New metadata added
    assert result_df["transcription"].notna().all()


def test_close():
    session = Mock()
    session.headers = {}
    TichaTextScraper(session=session).close()
    session.close.assert_called_once()
