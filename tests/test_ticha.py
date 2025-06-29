"""Tests for the Ticha scraper."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from ticha.core.scraper import TichaScraper


class TestTichaScraper:
    """Test cases for TichaScraper class."""
    
    @pytest.fixture
    def scraper(self):
        """Create a scraper instance for testing."""
        return TichaScraper(rate_limit=0.1)  # Fast for testing
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock WebDriver."""
        driver = Mock()
        driver.find_elements.return_value = []
        driver.find_element.return_value = Mock()
        return driver
    
    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.rate_limit == 0.1
        assert scraper.base_url == "https://ticha.haverford.edu/en/texts/handwritten/"
        assert scraper.driver is None
    
    def test_extract_text_from_cell_with_link(self, scraper):
        """Test text extraction from cell containing a link."""
        # Mock cell with link
        mock_cell = Mock()
        mock_link = Mock()
        mock_link.text = "Document Name"
        mock_cell.find_element.return_value = mock_link
        
        result = scraper._extract_text_from_cell(mock_cell)
        assert result == "Document Name"
    
    def test_extract_text_from_cell_without_link(self, scraper):
        """Test text extraction from cell without link."""
        # Mock cell without link
        mock_cell = Mock()
        mock_cell.text = "Plain Text"
        mock_cell.find_element.side_effect = NoSuchElementException()
        
        result = scraper._extract_text_from_cell(mock_cell)
        assert result == "Plain Text"
    
    def test_extract_link_from_cell(self, scraper):
        """Test link extraction from cell."""
        mock_cell = Mock()
        mock_link = Mock()
        mock_link.get_attribute.return_value = "https://example.com/doc"
        mock_cell.find_element.return_value = mock_link
        
        result = scraper._extract_link_from_cell(mock_cell)
        assert result == "https://example.com/doc"
    
    def test_extract_link_from_cell_no_link(self, scraper):
        """Test link extraction when no link exists."""
        mock_cell = Mock()
        mock_cell.find_element.side_effect = NoSuchElementException()
        
        result = scraper._extract_link_from_cell(mock_cell)
        assert result is None
    
    def test_has_next_page_enabled(self, scraper, mock_driver):
        """Test next page detection when enabled."""
        scraper.driver = mock_driver
        
        mock_button = Mock()
        mock_button.get_attribute.return_value = "paginate_button next"
        mock_driver.find_element.return_value = mock_button
        
        result = scraper._has_next_page()
        assert result is True
    
    def test_has_next_page_disabled(self, scraper, mock_driver):
        """Test next page detection when disabled."""
        scraper.driver = mock_driver
        
        mock_button = Mock()
        mock_button.get_attribute.return_value = "paginate_button next disabled"
        mock_driver.find_element.return_value = mock_button
        
        result = scraper._has_next_page()
        assert result is False
    
    def test_extract_table_data_structure(self, scraper, mock_driver):
        """Test table data extraction with correct structure."""
        scraper.driver = mock_driver
        
        # Mock table row
        mock_row = Mock()
        mock_cells = []
        for i in range(10):  # 10 columns as per structure
            cell = Mock()
            cell.text = f"Cell {i}"
            cell.find_element.side_effect = NoSuchElementException()  # No links
            mock_cells.append(cell)
        
        mock_row.find_elements.return_value = mock_cells
        mock_driver.find_elements.return_value = [mock_row]
        
        # Mock WebDriverWait
        with patch('ticha.core.scraper.WebDriverWait'):
            result = scraper._extract_table_data()
        
        assert len(result) == 1
        assert result[0]['document_name'] == "Cell 0"
        assert result[0]['ticha_id'] == "Cell 2"
        assert result[0]['legibility'] == "Cell 9"
    
    def test_close(self, scraper):
        """Test browser cleanup."""
        mock_driver = Mock()
        scraper.driver = mock_driver
        
        scraper.close()
        
        mock_driver.quit.assert_called_once()
        assert scraper.driver is None
