"""Tests for the Ticha DZI image scraper (offline, synthetic pyramid)."""

import math
from io import BytesIO
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
            response.raise_for_status = Mock(side_effect=requests.HTTPError("404"))
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
