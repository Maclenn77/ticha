"""Command line interface for Ticha scraper."""

import argparse
import sys
from pathlib import Path
from .core.scraper import TichaScraper
from .core.text_scraper import TichaTextScraper
from .utils.citation import generate_citation_info
import pandas as pd

def scrape_manuscripts(args):
    """Scrape manuscript list."""
    print("Starting Ticha manuscript scraper...")
    
    scraper = TichaScraper(
        rate_limit=args.rate_limit,
        headless=not args.no_headless
    )
    
    try:
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


def scrape_texts(args):
    """Scrape full text content and metadata."""
    print("Starting Ticha text content scraper...")
    
    # Load manuscript DataFrame
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        print("First run: ticha-scraper manuscripts -o manuscripts.csv")
        sys.exit(1)
    
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} manuscripts from {args.input}")
        
        # Check if document_link column exists
        if 'document_link' not in df.columns:
            print("Error: 'document_link' column not found in input file")
            sys.exit(1)
        
        # Filter out manuscripts without links
        valid_df = df[df['document_link'].notna() & (df['document_link'] != '')]
        print(f"Found {len(valid_df)} manuscripts with valid links")
        
        if len(valid_df) == 0:
            print("No valid document links found")
            sys.exit(1)
        
        # Initialize text scraper
        text_scraper = TichaTextScraper(
            rate_limit=args.rate_limit,
            headless=not args.no_headless
        )
        
        try:
            # Scrape text content
            texts_df = text_scraper.scrape_documents_from_dataframe(
                valid_df, 
                max_documents=args.max_docs
            )
            
            if texts_df.empty:
                print("No text data was scraped")
                sys.exit(1)
            
            # Save to CSV
            output_path = Path(args.output)
            texts_df.to_csv(output_path, index=False)
            
            print(f"Successfully scraped text content from {len(texts_df)} documents")
            print(f"Data saved to: {output_path.absolute()}")
            
            # Show some statistics
            transcription_count = texts_df['transcription'].notna().sum()
            interlinear_count = texts_df['interlinear'].notna().sum()
            spanish_count = texts_df['modern_spanish'].notna().sum()
            
            print(f"\nContent statistics:")
            print(f"- Documents with transcription: {transcription_count}")
            print(f"- Documents with interlinear: {interlinear_count}")
            print(f"- Documents with modern Spanish: {spanish_count}")
            
        finally:
            text_scraper.close()
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Scrape Ticha Colonial Zapotec manuscript data')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Manuscripts command (original functionality)
    manuscripts_parser = subparsers.add_parser('manuscripts', help='Scrape manuscript list')
    manuscripts_parser.add_argument('-o', '--output', type=str, default='ticha_manuscripts.csv',
                                  help='Output CSV file path (default: ticha_manuscripts.csv)')
    manuscripts_parser.add_argument('-r', '--rate-limit', type=float, default=2.0,
                                  help='Rate limit in seconds between requests (default: 2.0)')
    manuscripts_parser.add_argument('--no-headless', action='store_true',
                                  help='Run browser in non-headless mode (visible)')
    manuscripts_parser.add_argument('-v', '--verbose', action='store_true',
                                  help='Enable verbose logging')
    
    # Texts command (new functionality)
    texts_parser = subparsers.add_parser('texts', help='Scrape full text content and metadata')
    texts_parser.add_argument('-i', '--input', type=str, required=True,
                            help='Input CSV file with manuscript data (output from manuscripts command)')
    texts_parser.add_argument('-o', '--output', type=str, default='ticha_texts.csv',
                            help='Output CSV file path (default: ticha_texts.csv)')
    texts_parser.add_argument('-r', '--rate-limit', type=float, default=2.0,
                            help='Rate limit in seconds between requests (default: 2.0)')
    texts_parser.add_argument('--max-docs', type=int, default=None,
                            help='Maximum number of documents to scrape (default: all)')
    texts_parser.add_argument('--no-headless', action='store_true',
                            help='Run browser in non-headless mode (visible)')
    texts_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Handle verbose logging
    if hasattr(args, 'verbose') and args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Handle commands
    if args.command == 'manuscripts':
        scrape_manuscripts(args)
    elif args.command == 'texts':
        scrape_texts(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
