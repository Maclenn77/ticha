"""Command line interface for Ticha scraper."""

import argparse
import sys
from pathlib import Path
from .core.scraper import TichaScraper
from .utils.citation import generate_citation_info


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Scrape Ticha Colonial Zapotec manuscript data')
    parser.add_argument('-o', '--output', type=str, default='ticha_manuscripts.csv',
                        help='Output CSV file path (default: ticha_manuscripts.csv)')
    parser.add_argument('-r', '--rate-limit', type=float, default=2.0,
                        help='Rate limit in seconds between requests (default: 2.0)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Run browser in non-headless mode (visible)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Starting Ticha manuscript scraper...")
        
        scraper = TichaScraper(
            rate_limit=args.rate_limit,
            headless=not args.no_headless
        )
        
        # Scrape data
        df = scraper.scrape_manuscripts()
        
        if df.empty:
            print("No data was scraped. Please check the website and try again.")
            sys.exit(1)
        
        # Save to CSV
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        
        print(f"Successfully scraped {len(df)} records")
        print(f"Data saved to: {output_path.absolute()}")
        
        # Generate citation info
        citation = generate_citation_info("https://ticha.haverford.edu/en/texts/handwritten/")
        print(f"\nCitation info:")
        print(f"Source: {citation['source']}")
        print(f"URL: {citation['url']}")
        print(f"Access date: {citation['access_date']}")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()