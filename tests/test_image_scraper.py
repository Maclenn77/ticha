"""Tests for the Ticha DZI image scraper (offline, synthetic pyramid)."""

import math
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from PIL import Image

from ticha.core.image_scraper import (
    DZIInfo,
    TichaImageScraper,
    parse_dzi,
)

BASE = "https://ticha.haverford.edu"
DZI_URL = f"{BASE}/static/dzi/doc1/page1.dzi"


def make_source_image(width, height):
    """Image with a distinct color per pixel so misplaced tiles are detected."""
    img = Image.new("RGB", (width, height))
    img.putdata(
        [((x * 7) % 256, (y * 13) % 256, (x + y) % 256)
         for y in range(height) for x in range(width)]
    )
    return img


def build_pyramid(img, tile_size, overlap, fmt="png"):
    """Build a DZI pyramid the way deepzoom.py / vips do: {url: bytes}."""
    width, height = img.size
    max_level = math.ceil(math.log2(max(width, height)))
    files = {}
    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        lw, lh = math.ceil(width / scale), math.ceil(height / scale)
        level_img = img if scale == 1 else img.resize((lw, lh))
        for col in range(math.ceil(lw / tile_size)):
            for row in range(math.ceil(lh / tile_size)):
                x0 = max(col * tile_size - overlap, 0)
                y0 = max(row * tile_size - overlap, 0)
                x1 = min((col + 1) * tile_size + overlap, lw)
                y1 = min((row + 1) * tile_size + overlap, lh)
                buf = BytesIO()
                level_img.crop((x0, y0, x1, y1)).save(buf, "PNG")
                url = f"{BASE}/static/dzi/doc1/page1_files/{level}/{col}_{row}.{fmt}"
                files[url] = buf.getvalue()
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        f'Format="{fmt}" Overlap="{overlap}" TileSize="{tile_size}">'
        f'<Size Height="{height}" Width="{width}"/></Image>'
    ).encode()
    files[DZI_URL] = xml
    return files


def fake_session(files):
    session = Mock(spec=requests.Session)
    session.headers = {}

    def get(url, timeout=None):
        response = Mock()
        if url in files:
            response.status_code = 200
            response.content = files[url]
            response.raise_for_status = Mock()
        else:
            response.status_code = 404
            response.content = b""
            response.raise_for_status = Mock(
                side_effect=requests.HTTPError("404", response=response)
            )
        return response

    session.get.side_effect = get
    return session


@pytest.fixture
def source_image():
    return make_source_image(300, 190)


@pytest.fixture
def scraper_with_pyramid(source_image):
    files = build_pyramid(source_image, tile_size=64, overlap=2)
    scraper = TichaImageScraper(
        rate_limit=0, tile_delay=0, session=fake_session(files), retries=1
    )
    return scraper, files


def test_parse_dzi():
    xml = (
        b'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        b'Format="jpg" Overlap="1" TileSize="254"><Size Width="5000" Height="3000"/>'
        b"</Image>"
    )
    info = parse_dzi(xml, "https://x.org/a/page.dzi")
    assert (info.width, info.height) == (5000, 3000)
    assert info.tile_size == 254 and info.overlap == 1 and info.format == "jpg"
    assert info.tiles_url == "https://x.org/a/page_files/"
    assert info.max_level == 13
    assert info.level_size(13) == (5000, 3000)
    assert info.level_size(12) == (2500, 1500)
    assert info.tile_url(13, 2, 3) == "https://x.org/a/page_files/13/2_3.jpg"


def test_level_for_max_dimension():
    info = DZIInfo("u", 5000, 3000, 254, 1, "jpg", "u_files/")
    assert info.level_for_max_dimension(None) == 13
    assert info.level_for_max_dimension(2500) == 12
    assert info.level_for_max_dimension(2000) == 11


def test_stitch_reproduces_full_resolution(scraper_with_pyramid, source_image):
    scraper, _ = scraper_with_pyramid
    info = scraper.load_dzi(DZI_URL)
    stitched = scraper.stitch_dzi(info)
    assert stitched.size == source_image.size
    assert stitched.tobytes() == source_image.tobytes()


@pytest.mark.parametrize("tile_workers", [1, 4])
def test_stitch_is_identical_with_parallel_tiles(
    scraper_with_pyramid, source_image, tile_workers
):
    scraper, _ = scraper_with_pyramid
    scraper.tile_workers = tile_workers
    stitched = scraper.stitch_dzi(scraper.load_dzi(DZI_URL))
    assert stitched.tobytes() == source_image.tobytes()


def test_download_dzi_writes_file(scraper_with_pyramid, tmp_path):
    scraper, _ = scraper_with_pyramid
    out = scraper.download_dzi(DZI_URL, tmp_path / "p.png", max_dimension=160)
    assert Image.open(out).size == (150, 95)


def test_find_image_sources():
    html = """
    <html><body>
      <div class="viewer" data-dzi="/static/dzi/doc1/page1.dzi"></div>
      <script>
        var viewer = OpenSeadragon({
          id: "viewer",
          tileSources: ["/static/dzi/doc1/page1.dzi", '/static/dzi/doc1/page2.dzi']
        });
      </script>
      <script>
        OpenSeadragon({tileSources: {
          Image: {xmlns: "http://schemas.microsoft.com/deepzoom/2008",
                  Url: "/static/dzi/doc1/page3_files/", Format: "jpg",
                  Overlap: "1", TileSize: "254",
                  Size: {Width: "4000", Height: "3000"}}
        }});
      </script>
    </body></html>
    """
    scraper = TichaImageScraper(rate_limit=0)
    sources = scraper.find_image_sources(html, f"{BASE}/en/texts/doc1/")
    assert sources[:2] == [
        f"{BASE}/static/dzi/doc1/page1.dzi",
        f"{BASE}/static/dzi/doc1/page2.dzi",
    ]
    inline = sources[2]
    assert isinstance(inline, DZIInfo)
    assert inline.tiles_url == f"{BASE}/static/dzi/doc1/page3_files/"
    assert (inline.width, inline.height, inline.tile_size) == (4000, 3000, 254)


def test_download_document_falls_back_when_pdf_fails(
    scraper_with_pyramid, tmp_path
):
    scraper, files = scraper_with_pyramid
    page_url = f"{BASE}/en/texts/doc1/"
    files[page_url] = (
        b'<a href="/media/doc1.pdf">PDF</a>'
        b'<script>OpenSeadragon({tileSources: "/static/dzi/doc1/page1.dzi"})</script>'
    )
    # PDF endpoint answers 500 like the real site
    original_get = scraper.session.get.side_effect

    def get(url, timeout=None):
        if url.endswith(".pdf"):
            response = Mock(status_code=500, content=b"")
            return response
        return original_get(url, timeout)

    scraper.session.get.side_effect = get

    result = scraper.download_document_images(page_url, tmp_path, make_pdf=True)
    assert result["error"] is None
    assert len(result["image_paths"]) == 1
    assert result["pdf_path"].endswith("doc1.pdf")
    with open(result["pdf_path"], "rb") as f:
        assert f.read(4) == b"%PDF"


def test_download_images_from_dataframe(scraper_with_pyramid, tmp_path):
    scraper, files = scraper_with_pyramid
    page_url = f"{BASE}/en/texts/doc1/"
    files[page_url] = b'<div data-dzi="/static/dzi/doc1/page1.dzi"></div>'
    df = pd.DataFrame({"document_name": ["Doc 1", "No link"],
                       "document_link": [page_url, None]})
    result = scraper.download_images_from_dataframe(df, tmp_path, try_pdf=False)
    assert len(result) == 1
    assert result.iloc[0]["image_count"] == 1
    assert result.iloc[0]["document_name"] == "Doc 1"


def test_existing_pages_are_not_downloaded_again(scraper_with_pyramid, tmp_path):
    scraper, files = scraper_with_pyramid
    page_url = f"{BASE}/en/texts/doc1/"
    files[page_url] = b'<div data-dzi="/static/dzi/doc1/page1.dzi"></div>'
    first = scraper.download_document_images(page_url, tmp_path, try_pdf=False)
    saved = Path(first["image_paths"][0])
    assert not list(saved.parent.glob("*.part*"))
    requests_before = scraper.session.get.call_count
    assert any("_files/" in c.args[0] for c in scraper.session.get.call_args_list)

    again = scraper.download_document_images(page_url, tmp_path, try_pdf=False)
    tiles = [
        c.args[0] for c in scraper.session.get.call_args_list[requests_before:]
        if "_files/" in c.args[0]
    ]
    assert tiles == []
    assert again["image_paths"] == first["image_paths"]
    assert again["image_folios"] == first["image_folios"]


# Markup below mirrors the live Ticha pages (trimmed).

HANDWRITTEN_HTML = """
<li><a target="_blank" href="/en/texts/Oc111.pdf">PDF</a></li>
<a href="http://ticha.haverford.edu/images/permissions/Ticha_permissions_from_AGEO.pdf">[PDF]</a>
<div id="openseadragon1"></div>
<script id="call_the_dragon" type="text/javascript"
  src="/static/handwritten_texts/js/call_the_dragon.js"
  data-ticha-id="Oc111" data-pages="09r-10v"></script>
"""

PAGINATED_HTML = """
<a rel="noopener" download="testament, 1733.pdf" href="/text/Ma733/download/">
<button class="download">PDF</button></a>
<div id="openseadragon1"></div>
<script>
var url = "/images/dzi/" + "Ma733_1r.dzi";
var viewer = OpenSeadragon({id: "openseadragon1", tileSources: url});
</script>
<script>
const pageMap = {"01r": "0", "01v": "1"} // get the pageMap from the model
</script>
"""


@pytest.mark.parametrize(
    "pages, expected",
    [
        ("09r-10v", ["9r", "9v", "10r", "10v"]),
        ("1v-2r", ["1v", "2r"]),
        ("3-5", ["3", "4", "5"]),
        ("09r", ["09r"]),
        ("788-789, 791", ["788", "789", "791"]),
    ],
)
def test_page_range_matches_call_the_dragon(pages, expected):
    from ticha.core.image_scraper import _page_range_to_list

    assert _page_range_to_list(pages) == expected


def test_find_sources_handwritten_layout():
    scraper = TichaImageScraper(rate_limit=0)
    page_url = f"{BASE}/en/texts/Oc111/"
    assert scraper.find_image_sources(HANDWRITTEN_HTML, page_url) == [
        f"{BASE}/images/dzi/Oc111_{p}.dzi" for p in ("9r", "9v", "10r", "10v")
    ]
    assert scraper.find_pdf_links(HANDWRITTEN_HTML, page_url) == [
        f"{BASE}/en/texts/Oc111.pdf"
    ]
    assert scraper.find_page_urls(HANDWRITTEN_HTML, page_url) == [page_url]


def test_find_sources_paginated_layout():
    scraper = TichaImageScraper(rate_limit=0)
    page_url = f"{BASE}/en/text/Ma733/0/"
    assert scraper.find_image_sources(PAGINATED_HTML, page_url) == [
        f"{BASE}/images/dzi/Ma733_1r.dzi"
    ]
    assert scraper.find_pdf_links(PAGINATED_HTML, page_url) == [
        f"{BASE}/text/Ma733/download/"
    ]
    assert scraper.find_page_urls(PAGINATED_HTML, page_url) == [
        f"{BASE}/en/text/Ma733/0/",
        f"{BASE}/en/text/Ma733/1/",
    ]


def test_ticha_id_from_url():
    from ticha.core.image_scraper import ticha_id_from_url

    assert ticha_id_from_url(f"{BASE}/en/texts/Oc111/") == "Oc111"
    assert ticha_id_from_url(f"{BASE}/en/text/Ma733/0/") == "Ma733"


def test_load_dzi_retries_without_leading_zero(scraper_with_pyramid):
    scraper, files = scraper_with_pyramid
    padded = DZI_URL.replace("page1.dzi", "doc_09r.dzi")
    files[DZI_URL.replace("page1.dzi", "doc_9r.dzi")] = files[DZI_URL]
    info = scraper.load_dzi(padded)
    assert info.url.endswith("doc_9r.dzi")


def test_load_dzi_retries_with_leading_zero(scraper_with_pyramid):
    # data-pages="9r-9v" while the server stores Th687_09r.dzi
    scraper, files = scraper_with_pyramid
    files[DZI_URL.replace("page1.dzi", "doc_09r.dzi")] = files[DZI_URL]
    info = scraper.load_dzi(DZI_URL.replace("page1.dzi", "doc_9r.dzi"))
    assert info.url.endswith("doc_09r.dzi")


def test_load_dzi_raises_when_no_variant_exists(scraper_with_pyramid):
    scraper, _ = scraper_with_pyramid
    with pytest.raises(requests.HTTPError):
        scraper.load_dzi(DZI_URL.replace("page1.dzi", "doc_9r.dzi"))


@pytest.mark.parametrize(
    "folio, expected",
    [
        ("9r", ["09r", "009r"]),
        ("09r", ["9r", "009r"]),
        ("14", ["014"]),
        ("614", []),
    ],
)
def test_folio_variants(folio, expected):
    from ticha.core.image_scraper import _folio_variants

    url = f"{BASE}/images/dzi/Th687_{folio}.dzi"
    assert _folio_variants(url) == [f"{BASE}/images/dzi/Th687_{f}.dzi" for f in expected]


def test_failed_pages_are_reported(scraper_with_pyramid, tmp_path):
    scraper, files = scraper_with_pyramid
    page_url = f"{BASE}/en/texts/doc1/"
    files[page_url] = (
        b'<div data-dzi="/static/dzi/doc1/page1.dzi"></div>'
        b'<div data-dzi="/static/dzi/doc1/page2.dzi"></div>'
    )
    result = scraper.download_document_images(page_url, tmp_path, try_pdf=False)
    assert result["image_folios"] == ["page1"]
    assert result["missing_folios"] == ["page2"]
    assert result["error"] is None  # partial download is not an error


def test_document_without_any_page_is_an_error(scraper_with_pyramid, tmp_path):
    scraper, files = scraper_with_pyramid
    page_url = f"{BASE}/en/texts/doc1/"
    files[page_url] = b'<div data-dzi="/static/dzi/doc1/missing.dzi"></div>'
    df = pd.DataFrame({"document_link": [page_url]})
    row = scraper.download_images_from_dataframe(df, tmp_path, try_pdf=False).iloc[0]
    assert row["image_count"] == 0
    assert row["missing_folios"] == "missing"
    assert row["error"].startswith("All 1 pages failed")
