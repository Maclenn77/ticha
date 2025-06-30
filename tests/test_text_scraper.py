"""Tests for the Ticha text scraper."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from selenium.common.exceptions import NoSuchElementException

from ticha.core.text_scraper import TichaTextScraper


class TestTichaTextScraper:
    """Test cases for TichaTextScraper class."""
    
    @pytest.fixture
    def text_scraper(self):
        """Create a text scraper instance for testing."""
        return TichaTextScraper(rate_limit=0.1)
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock WebDriver."""
        driver = Mock()
        return driver
    
    def test_init(self, text_scraper):
        """Test text scraper initialization."""
        assert text_scraper.rate_limit == 0.1
        assert text_scraper.base_url == "https://ticha.haverford.edu"
        assert text_scraper.driver is None
    
    def test_normalize_metadata_key(self, text_scraper):
        """Test metadata key normalization."""
        test_cases = [
            ("Title", "title"),
            ("Call Number", "call_number"),
            ("Date Digitized", "date_digitized"),
            ("Town (Modern Official)", "town_modern_official"),
            ("Primary Parties", "primary_parties"),
            ("Permission File", "permission_file"),
        ]
        
        for input_key, expected in test_cases:
            result = text_scraper._normalize_metadata_key(input_key)
            assert result == expected
    
    def test_extract_metadata(self, text_scraper, mock_driver):
        """Test metadata extraction."""
        text_scraper.driver = mock_driver
        
        # Mock metadata div with p tags
        mock_metadata_div = Mock()
        mock_p_tags = []
        
        # Create mock p tags with metadata
        metadata_texts = [
            "Title: Test Document",
            "Archive: Test Archive",
            "Call Number: 123-ABC",
            "Date Digitized: 2023-01-01"
        ]
        
        for text in metadata_texts:
            mock_p = Mock()
            mock_p.text = text
            mock_p_tags.append(mock_p)
        
        mock_metadata_div.find_elements.return_value = mock_p_tags
        mock_driver.find_element.return_value = mock_metadata_div
        
        result = text_scraper._extract_metadata()
        
        expected = {
            'title': 'Test Document',
            'archive': 'Test Archive', 
            'call_number': '123-ABC',
            'date_digitized': '2023-01-01'
        }
        
        assert result == expected
    
    def test_extract_metadata_no_div(self, text_scraper, mock_driver):
        """Test metadata extraction when div doesn't exist."""
        text_scraper.driver = mock_driver
        mock_driver.find_element.side_effect = NoSuchElementException()
        
        result = text_scraper._extract_metadata()
        assert result == {}
    
    def test_extract_text_content(self, text_scraper, mock_driver):
        """Test text content extraction."""
        text_scraper.driver = mock_driver
        
        # Mock div with p tags
        mock_div = Mock()
        mock_p_tags = []
        
        # Create mock p tags with text content
        text_contents = [
            "First paragraph of text.",
            "Second paragraph of text.",
            "Third paragraph of text."
        ]
        
        for text in text_contents:
            mock_p = Mock()
            mock_p.text = text
            mock_p_tags.append(mock_p)
        
        mock_div.find_elements.return_value = mock_p_tags
        mock_driver.find_element.return_value = mock_div
        
        result = text_scraper._extract_text_content("transcription")
        
        expected = "First paragraph of text.\n\nSecond paragraph of text.\n\nThird paragraph of text."
        assert result == expected
    
    def test_extract_text_content_no_div(self, text_scraper, mock_driver):
        """Test text content extraction when div doesn't exist."""
        text_scraper.driver = mock_driver
        mock_driver.find_element.side_effect = NoSuchElementException()
        
        result = text_scraper._extract_text_content("nonexistent")
        assert result is None
    
    def test_scrape_document_structure(self, text_scraper, mock_driver):
        """Test document scraping structure."""
        text_scraper.driver = mock_driver
        
        # Mock successful page load
        with patch('ticha.core.text_scraper.WebDriverWait'):
            # Mock metadata extraction
            with patch.object(text_scraper, '_extract_metadata') as mock_metadata:
                mock_metadata.return_value = {'title': 'Test Doc', 'archive': 'Test Archive'}
                
                # Mock text content extraction
                with patch.object(text_scraper, '_extract_text_content') as mock_text:
                    mock_text.side_effect = lambda div_id: f"Content from {div_id}" if div_id == "transcription" else None
                    
                    result = text_scraper.scrape_document("https://example.com/doc")
                    
                    assert 'url' in result
                    assert 'transcription' in result
                    assert 'interlinear' in result
                    assert 'modern_spanish' in result
                    assert 'title' in result
                    assert 'archive' in result
                    assert result['title'] == 'Test Doc'
                    assert result['transcription'] == 'Content from transcription'
    
    def test_scrape_documents_from_dataframe(self, text_scraper):
        """Test scraping multiple documents from DataFrame."""
        # Create test DataFrame
        test_data = {
            'document_name': ['Doc 1', 'Doc 2', 'Doc 3'],
            'document_link': ['https://example.com/1', 'https://example.com/2', None],
            'ticha_id': ['ID1', 'ID2', 'ID3']
        }
        df = pd.DataFrame(test_data)
        
        # Mock the scrape_document method
        with patch.object(text_scraper, 'scrape_document') as mock_scrape:
            mock_scrape.side_effect = [
                {'url': 'https://example.com/1', 'title': 'Title 1', 'transcription': 'Text 1'},
                {'url': 'https://example.com/2', 'title': 'Title 2', 'transcription': 'Text 2'}
            ]
            
            with patch.object(text_scraper, '_setup_driver'):
                result_df = text_scraper.scrape_documents_from_dataframe(df)
        
        # Should only process 2 documents (excluding the one with None link)
        assert len(result_df) == 2
        assert 'document_name' in result_df.columns  # Original columns preserved
        assert 'title' in result_df.columns  # New metadata added
        assert 'transcription' in result_df.columns  # New content added
    
    def test_close(self, text_scraper):
        """Test browser cleanup."""
        mock_driver = Mock()
        text_scraper.driver = mock_driver
        
        text_scraper.close()
        
        mock_driver.quit.assert_called_once()
        assert text_scraper.driver is None
