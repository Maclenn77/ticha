"""Text and metadata scraper for individual Ticha documents."""

import time
import re
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class TichaTextScraper:
    """Scraper for individual Ticha document texts and metadata."""
    
    def __init__(self, rate_limit: float = 2.0, headless: bool = True):
        """
        Initialize the text scraper.
        
        Args:
            rate_limit: Delay between requests in seconds
            headless: Whether to run browser in headless mode
        """
        self.rate_limit = rate_limit
        self.headless = headless
        self.base_url = "https://ticha.haverford.edu"
        
    def _get_page(self, document_url: str):
        """Load a document page."""

        response = requests.get(document_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to access document URL: {document_url} (status code: {response.status_code})")
            return None
        
        return response.content
    
    def _intelligent_delay(self):
        """Add intelligent delay with randomization."""
        import random
        jitter = random.uniform(0.8, 1.2)
        time.sleep(self.rate_limit * jitter)
    
    def _normalize_metadata_key(self, key: str) -> str:
        """
        Normalize metadata keys to lowercase with underscores.
        
        Args:
            key: Original metadata key
            
        Returns:
            Normalized key (lowercase, underscores instead of spaces)
        """
        # Remove extra whitespace and convert to lowercase
        normalized = key.strip().lower()
        # Replace spaces and other separators with underscores
        normalized = re.sub(r'[\s\-\.]+', '_', normalized)
        # Remove parentheses and other special characters
        normalized = re.sub(r'[^\w_]', '', normalized)
        # Clean up multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract metadata from div with id="metadata".
        
        Returns:
            Dictionary with normalized metadata keys and values
        """
        metadata = {}


        
        try:
            # Find metadata div
            metadata_div = soup.find('div', id='metadata')
            
            # Find all <p> tags within metadata div
            p_tags = metadata_div.find_all('p') if metadata_div else []
            
            for p_tag in p_tags:
                text = p_tag.text.strip()
                if not text:
                    continue
                
                # Split on colon to separate key and value
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        key = self._normalize_metadata_key(parts[0])
                        value = parts[1].strip()
                        metadata[key] = value
                        logger.debug(f"Extracted metadata: {key} = {value}")
                else:
                    # Handle cases where there might not be a colon
                    logger.debug(f"Metadata without colon: {text}")
            
            logger.info(f"Extracted {len(metadata)} metadata fields")
            if metadata_div is None:
                logger.warning("No metadata div found")
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
                    
        
        return metadata
    
    def _extract_text_content(self, soup: BeautifulSoup, div_id: str) -> Optional[str]:
        """
        Extract text content from a div, parsing to omit HTML tags.
        
        Args:
            div_id: ID of the div to extract text from
            
        Returns:
            Clean text content or None if div not found
        """
        try:
            div_element = soup.find('div', id=div_id)
            
            # Find all <p> tags within the div
            p_tags = div_element.find_all('p') if div_element else []
            
            # Extract text from all paragraphs
            paragraphs = []
            for p_tag in p_tags:
                # Get text content, which automatically strips HTML tags
                text = p_tag.text.strip()
                if text:
                    paragraphs.append(text)
            
            if paragraphs:
                content = '\n\n'.join(paragraphs)
                logger.debug(f"Extracted {len(content)} characters from {div_id}")
                return content
            else:
                logger.debug(f"No text content found in {div_id}")
                return None
                
        except NoSuchElementException:
            logger.debug(f"Div with id '{div_id}' not found")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from {div_id}: {e}")
            return None
    
    def scrape_document(self, document_url: str) -> Dict[str, Any]:
        """
        Scrape a single document's metadata and text content.
        
        Args:
            document_url: URL of the document to scrape
            
        Returns:
            Dictionary containing metadata and text content
        """
        
        try:
            # Ensure we have a full URL
            if not document_url.startswith('http'):
                document_url = urljoin(self.base_url, document_url)
            
            logger.info(f"Scraping document: {document_url}")
            
            content = self._get_page(document_url)

            soup = BeautifulSoup(content, 'html.parser')
            

            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            # Extract text content from different divs
            transcription = self._extract_text_content(soup, "transcription")
            interlinear = self._extract_text_content(soup, "interLinear")
            modern_spanish = self._extract_text_content(soup, "modern_spanish")
            
            # Combine all data
            document_data = {
                'url': document_url,
                'transcription': transcription,
                'interlinear': interlinear,
                'modern_spanish': modern_spanish,
                **metadata  # Unpack metadata into the main dictionary
            }
            
            # Add rate limiting
            self._intelligent_delay()
            
            logger.info(f"Successfully scraped document with {len(metadata)} metadata fields")
            return document_data
            
        except Exception as e:
            logger.error(f"Error scraping document {document_url}: {e}")
            return {'url': document_url, 'error': str(e)}
    
    def scrape_documents_from_dataframe(self, df: pd.DataFrame, 
                                      link_column: str = 'document_link',
                                      max_documents: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape multiple documents from a DataFrame containing document links.
        
        Args:
            df: DataFrame containing document links
            link_column: Name of the column containing document links
            max_documents: Maximum number of documents to scrape (None for all)
            
        Returns:
            DataFrame with scraped text and metadata
        """
        
        scraped_data = []
        
        # Filter out rows without links
        valid_df = df[df[link_column].notna() & (df[link_column] != '')].copy()
        
        if max_documents:
            valid_df = valid_df.head(max_documents)
        
        total_docs = len(valid_df)
        logger.info(f"Starting to scrape {total_docs} documents")
        
        for idx, (_, row) in enumerate(valid_df.iterrows(), 1):
            document_url = row[link_column]
            
            logger.info(f"Processing document {idx}/{total_docs}: {document_url}")
            
            # Scrape the document
            doc_data = self.scrape_document(document_url)
            
            # Add original row data to preserve manuscript info
            combined_data = {**row.to_dict(), **doc_data}
            scraped_data.append(combined_data)
            
            # Progress logging
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{total_docs} documents processed")
        
        logger.info(f"Completed scraping {len(scraped_data)} documents")
        
        return pd.DataFrame(scraped_data)
    
    def close(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Text scraper browser driver closed")

