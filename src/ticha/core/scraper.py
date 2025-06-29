"""Main scraper class for Ticha website."""

import time
import random
from typing import List, Dict, Optional
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TichaScraper:
    """Scraper for Ticha Colonial Zapotec manuscript data."""
    
    def __init__(self, rate_limit: float = 2.0, headless: bool = True):
        """
        Initialize the scraper.
        
        Args:
            rate_limit: Delay between requests in seconds
            headless: Whether to run browser in headless mode
        """
        self.rate_limit = rate_limit
        self.headless = headless
        self.driver = None
        self.base_url = "https://ticha.haverford.edu/en/texts/handwritten/"
        
    def _setup_driver(self):
        """Set up Chrome WebDriver with appropriate options."""
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # Add user agent to appear more like a regular browser
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        return webdriver.Chrome(options=options)
    
    def _intelligent_delay(self):
        """Add intelligent delay with randomization."""
        jitter = random.uniform(0.8, 1.2)
        time.sleep(self.rate_limit * jitter)
    
    def _extract_text_from_cell(self, cell) -> str:
        """Extract text from table cell, handling links."""
        try:
            # First try to get text from link if exists
            link = cell.find_element(By.TAG_NAME, "a")
            return link.text.strip()
        except NoSuchElementException:
            # If no link, get direct text
            return cell.text.strip()
    
    def _extract_link_from_cell(self, cell) -> Optional[str]:
        """Extract href from link in table cell."""
        try:
            link = cell.find_element(By.TAG_NAME, "a")
            href = link.get_attribute("href")
            return href if href else None
        except NoSuchElementException:
            return None
    
    def _extract_table_data(self) -> List[Dict[str, str]]:
        """
        Extract data from current table page based on your specified structure:
        
        <table>
          <tbody>
            <tr role="row">
              <td> <a link_to_document> name 
              <td>  <a> file_type
             <td> ticha_id
             <td> <a> year
             <td> <a> town
            <td> <a> archive
            <td> <a> doc_type
            <td> <a> language
            <td> <a> trptn_status
           <td> <a> legibility
        """
        try:
            # Wait for table to be present
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            
            # Find all table rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr[role='row']")
            page_data = []
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    # Make sure we have enough cells (expecting 10 columns)
                    if len(cells) < 10:
                        logger.warning(f"Row has only {len(cells)} cells, skipping")
                        continue
                    
                    # Extract data according to your structure
                    row_data = {
                        'document_name': self._extract_text_from_cell(cells[0]),
                        'document_link': self._extract_link_from_cell(cells[0]),
                        'file_type': self._extract_text_from_cell(cells[1]),
                        'ticha_id': cells[2].text.strip(),  # No link expected here
                        'year': self._extract_text_from_cell(cells[3]),
                        'town': self._extract_text_from_cell(cells[4]),
                        'archive': self._extract_text_from_cell(cells[5]),
                        'doc_type': self._extract_text_from_cell(cells[6]),
                        'language': self._extract_text_from_cell(cells[7]),
                        'trptn_status': self._extract_text_from_cell(cells[8]),
                        'legibility': self._extract_text_from_cell(cells[9])
                    }
                    
                    # Only add row if we have essential data
                    if row_data['document_name'] and row_data['ticha_id']:
                        page_data.append(row_data)
                        
                except Exception as e:
                    logger.warning(f"Error extracting row data: {e}")
                    continue
            
            logger.info(f"Extracted {len(page_data)} records from current page")
            return page_data
            
        except TimeoutException:
            logger.error("Timeout waiting for table data")
            return []
        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return []
    
    def _has_next_page(self) -> bool:
        """
        Check if there's a next page available based on your pagination structure:
        <a class="paginate_button next disabled" aria-controls="myTable" data-dt-idx="7" tabindex="0" id="myTable_next">Next</a>
        """
        try:
            next_button = self.driver.find_element(By.ID, "myTable_next")
            classes = next_button.get_attribute("class")
            
            # If "disabled" is in the class, no more pages
            is_disabled = "disabled" in classes
            logger.info(f"Next button classes: {classes}, disabled: {is_disabled}")
            
            return not is_disabled
            
        except NoSuchElementException:
            logger.warning("Next button not found")
            return False
        except Exception as e:
            logger.error(f"Error checking for next page: {e}")
            return False
    
    def _go_to_next_page(self) -> bool:
        """Navigate to next page."""
        try:
            next_button = self.driver.find_element(By.ID, "myTable_next")
            
            # Check if button is enabled
            if "disabled" in next_button.get_attribute("class"):
                return False
            
            # Click the next button
            next_button.click()
            
            # Wait for page to load
            self._intelligent_delay()
            
            # Wait for table to update (look for loading to complete)
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            
            return True
            
        except Exception as e:
            logger.error(f"Error navigating to next page: {e}")
            return False
    
    def scrape_manuscripts(self) -> pd.DataFrame:
        """
        Scrape all manuscript data with pagination handling.
        
        Returns:
            pandas.DataFrame: DataFrame containing all scraped manuscript data
        """
        if self.driver is None:
            self.driver = self._setup_driver()
        
        try:
            logger.info(f"Starting scrape of {self.base_url}")
            self.driver.get(self.base_url)
            
            # Wait for initial page load
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
            
            all_data = []
            page_num = 1
            
            while True:
                logger.info(f"Scraping page {page_num}")
                
                # Extract data from current page
                page_data = self._extract_table_data()
                all_data.extend(page_data)
                
                logger.info(f"Page {page_num}: extracted {len(page_data)} records")
                logger.info(f"Total records so far: {len(all_data)}")
                
                # Check if there's a next page
                if not self._has_next_page():
                    logger.info("No more pages available")
                    break
                
                # Navigate to next page
                if not self._go_to_next_page():
                    logger.info("Could not navigate to next page")
                    break
                
                page_num += 1
                
                # Safety check to prevent infinite loops
                if page_num > 100:  # Adjust this limit as needed
                    logger.warning("Reached maximum page limit")
                    break
            
            logger.info(f"Scraping completed. Total records: {len(all_data)}")
            
            # Convert to DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                logger.warning("No data found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise
        finally:
            self.close()
    
    def close(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser driver closed")
