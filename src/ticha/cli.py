"""Command line interface for Ticha scraper."""

import argparse
import sys
from pathlib import Path
from .core.scraper import TichaScraper
from .core.text_scraper import TichaTextScraper
from .core.image_scraper import TichaImageScraper
from .core.augmentation import Augmenter, DatasetAugmenter, load_targets
from .core.line_pairing import (
    DEFAULT_MODEL,
    DEFAULT_STATUSES,
    KrakenRecognizer,
    KrakenSegmenter,
    LinePairer,
)
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
            print(f"\nContent statistics:")
            for column, label in [
                ('transcription', 'transcription'),
                ('interlinear', 'interlinear analysis'),
                ('modern_spanish', 'modern Spanish'),
                ('spanish_translation', 'Spanish translation'),
                ('english_translation', 'English translation'),
            ]:
                if column in texts_df.columns:
                    print(f"- Documents with {label}: {texts_df[column].notna().sum()}")
            
        finally:
            text_scraper.close()
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def scrape_images(args):
    """Download page images (DZI tiles stitched into single files)."""
    print("Starting Ticha image scraper...")

    if args.url:
        df = pd.DataFrame({'document_link': [args.url]})
    elif args.input and Path(args.input).exists():
        df = pd.read_csv(args.input)
        if 'document_link' not in df.columns:
            print("Error: 'document_link' column not found in input file")
            sys.exit(1)
    else:
        print("Error: provide --url or an existing --input CSV")
        print("First run: ticha-scraper manuscripts -o manuscripts.csv")
        sys.exit(1)

    image_scraper = TichaImageScraper(rate_limit=args.rate_limit)

    try:
        results_df = image_scraper.download_images_from_dataframe(
            df,
            args.output_dir,
            max_documents=args.max_docs,
            image_format=args.format,
            max_dimension=args.max_dimension,
            make_pdf=args.pdf,
            try_pdf=not args.skip_pdf,
        )

        output_path = Path(args.output)
        results_df.to_csv(output_path, index=False)

        total_images = int(results_df['image_count'].sum()) if not results_df.empty else 0
        print(f"Downloaded {total_images} images from {len(results_df)} documents")
        print(f"Images saved to: {Path(args.output_dir).absolute()}")
        print(f"Index saved to: {output_path.absolute()}")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def pair_lines(args):
    """Segment page images into lines and pair them with transcriptions."""
    print("Starting line segmentation and pairing...")

    for path in (args.texts, args.images):
        if not Path(path).exists():
            print(f"Error: Input file {path} not found")
            sys.exit(1)

    statuses = None if 'all' in (args.status or []) else (args.status or DEFAULT_STATUSES)
    output_dir = Path(args.output_dir)

    try:
        segmenter = KrakenSegmenter(cache_dir=output_dir / "segmentation", device=args.device)
        recognizer = None
        if not args.no_recognition:
            recognizer = KrakenRecognizer(
                args.model, cache_dir=output_dir / "recognition", device=args.device)
    except (ImportError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    pairer = LinePairer(
        segmenter=segmenter,
        recognizer=recognizer,
        statuses=statuses,
        split_columns=not args.no_split_columns,
        cer_threshold=args.cer_threshold,
        export_lines=args.export_lines,
    )

    try:
        manifest = pairer.run(args.texts, args.images, output_dir, max_documents=args.max_docs)
    except KeyboardInterrupt:
        print("\nInterrupted by user (finished pages are cached and kept)")
        sys.exit(1)

    counts = manifest['status'].value_counts().to_dict() if not manifest.empty else {}
    print(f"\nPages paired automatically: {counts.get('auto', 0)}  -> {output_dir / 'auto'}")
    print(f"Pages for human review:     {counts.get('review', 0)}  -> {output_dir / 'review'}")
    print(f"Documents skipped:          {counts.get('skipped', 0)}")
    if counts.get('error'):
        print(f"Pages with errors:          {counts['error']}")
    if not manifest.empty and 'confident_lines' in manifest:
        print(f"Confident line pairs:       {int(manifest['confident_lines'].sum())} "
              f"of {int(manifest['paired_lines'].sum())} paired")
    print(f"Manifest: {(output_dir / 'manifest.csv').absolute()}")


def augment_dataset(args):
    """Augment the line dataset (pages and line crops)."""
    print("Starting data augmentation...")
    dataset = Path(args.dataset)
    if not (dataset / "auto").exists():
        print(f"Error: {dataset} does not look like a 'lines' output (no auto/ folder)")
        sys.exit(1)

    targets = None
    if args.target:
        targets = load_targets(args.target)
        print(f"Imitating {len(targets)} target image(s) from {args.target}")
        if not targets:
            print("Error: no images found in --target")
            sys.exit(1)

    augmenter = DatasetAugmenter(
        Augmenter(targets=targets, seed=args.seed,
                  underexposed_ratio=args.underexposed_ratio),
        copies=args.copies,
        degrade_ratio=args.degrade_ratio,
        min_conf=args.min_conf,
        include_review=not args.auto_only,
        geometry=not args.no_geometry,
    )
    if not (dataset / "lines").exists() and not args.pages:
        print(f"Error: {dataset / 'lines'} not found; run the lines command with "
              "--export-lines (or pass --pages)")
        sys.exit(1)
    index = augmenter.run(dataset, args.output, pages=args.pages)

    pages = index[index['kind'] == 'page'] if not index.empty else index
    lines = index[index['kind'] == 'line'] if not index.empty else index
    print(f"Augmented line images: {len(lines)}  -> {Path(args.output) / 'lines'}")
    if args.pages:
        print(f"Augmented pages: {len(pages)} ({int(pages['lines'].sum()) if len(pages) else 0} "
              f"text lines)  -> {Path(args.output) / 'pages'}")
    print(f"Index: {(Path(args.output) / 'augment.csv').absolute()}")


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
    
    # Images command
    images_parser = subparsers.add_parser('images', help='Download document page images (DZI)')
    images_parser.add_argument('-i', '--input', type=str,
                             help='Input CSV file with manuscript data (output from manuscripts command)')
    images_parser.add_argument('-u', '--url', type=str,
                             help='Download a single document page URL instead of a CSV')
    images_parser.add_argument('-d', '--output-dir', type=str, default='ticha_images',
                             help='Directory for downloaded images (default: ticha_images)')
    images_parser.add_argument('-o', '--output', type=str, default='ticha_images.csv',
                             help='Output CSV index of downloaded images (default: ticha_images.csv)')
    images_parser.add_argument('-f', '--format', choices=['png', 'jpg', 'tif'], default='png',
                             help='Output image format (default: png)')
    images_parser.add_argument('--max-dimension', type=int, default=None,
                             help='Limit the largest side of each image in pixels (default: full resolution)')
    images_parser.add_argument('--pdf', action='store_true',
                             help='Also bundle the pages of each document into a PDF')
    images_parser.add_argument('--skip-pdf', action='store_true',
                             help='Do not try the original PDF links first')
    images_parser.add_argument('-r', '--rate-limit', type=float, default=2.0,
                             help='Rate limit in seconds between documents (default: 2.0)')
    images_parser.add_argument('--max-docs', type=int, default=None,
                             help='Maximum number of documents to process (default: all)')
    images_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Enable verbose logging')

    # Lines command
    lines_parser = subparsers.add_parser(
        'lines', help='Segment images into lines and pair them with transcriptions (kraken)')
    lines_parser.add_argument('-t', '--texts', type=str, required=True,
                            help='CSV from the texts command')
    lines_parser.add_argument('-i', '--images', type=str, required=True,
                            help='CSV from the images command')
    lines_parser.add_argument('-d', '--output-dir', type=str, default='ticha_lines',
                            help='Output directory (default: ticha_lines)')
    lines_parser.add_argument('-s', '--status', action='append',
                            help='Transcription status to keep; repeat for several, or "all" '
                                 '(default: "Completed and Checked" and "Completed but not Checked")')
    lines_parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                            help='kraken recognition model (path or DOI) used to align lines '
                                 f'with the transcription (default: McCATMuS, {DEFAULT_MODEL})')
    lines_parser.add_argument('--no-recognition', action='store_true',
                            help='Pair lines in order instead (pages need equal line counts)')
    lines_parser.add_argument('--cer-threshold', type=float, default=0.5,
                            help='Highest character error rate for a confident pair (default: 0.5)')
    lines_parser.add_argument('--export-lines', action='store_true',
                            help='Also write line images + .gt.txt of confidently paired lines')
    lines_parser.add_argument('--no-split-columns', action='store_true',
                            help='Do not split transcription lines on runs of 4+ spaces')
    lines_parser.add_argument('--device', type=str, default='cpu',
                            help='Device for kraken, e.g. cpu or cuda:0 (default: cpu)')
    lines_parser.add_argument('--max-docs', type=int, default=None,
                            help='Maximum number of documents to process (default: all)')
    lines_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable verbose logging')

    # Augment command
    augment_parser = subparsers.add_parser(
        'augment', help='Augment the line dataset (low-quality look + variety)')
    augment_parser.add_argument('-d', '--dataset', type=str, required=True,
                              help='Output directory of the lines command')
    augment_parser.add_argument('-o', '--output', type=str, default='ticha_augmented',
                              help='Output directory (default: ticha_augmented)')
    augment_parser.add_argument('-n', '--copies', type=int, default=2,
                              help='Augmented copies per page / line image (default: 2)')
    augment_parser.add_argument('--degrade-ratio', type=float, default=0.5,
                              help='Share of copies imitating untranscribed scans; '
                                   'the rest add generic variety (default: 0.5)')
    augment_parser.add_argument('--target', type=str, default=None,
                              help='Folder of real low-quality images to imitate '
                                   '(default: built-in ranges measured on Ticha)')
    augment_parser.add_argument('--underexposed-ratio', type=float, default=0.2,
                              help='Share of degraded copies that are dark instead of '
                                   'washed out, without --target (default: 0.2)')
    augment_parser.add_argument('--min-conf', type=float, default=0.5,
                              help='Lowest pair confidence kept as text (default: 0.5)')
    augment_parser.add_argument('--auto-only', action='store_true',
                              help='Use only auto/ pages (not confident lines of review/ pages)')
    augment_parser.add_argument('--no-geometry', action='store_true',
                              help='No rotation, slant, scaling or elastic distortion')
    augment_parser.add_argument('--pages', action='store_true',
                              help='Also augment whole pages as PAGE XML + image, reusing the '
                                   'paired lines (e.g. to train segmentation); line images '
                                   'are augmented by default')
    augment_parser.add_argument('--seed', type=int, default=None,
                              help='Random seed for reproducible output')
    augment_parser.add_argument('-v', '--verbose', action='store_true',
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
    elif args.command == 'images':
        scrape_images(args)
    elif args.command == 'lines':
        pair_lines(args)
    elif args.command == 'augment':
        augment_dataset(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
