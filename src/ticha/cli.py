"""Command line interface for Ticha scraper."""

import argparse
import sys
from pathlib import Path
from .core.scraper import TichaScraper
from .core.text_scraper import TichaTextScraper
from .core.image_scraper import TichaImageScraper
from .core.printed_scraper import TichaPrintedScraper, write_corpus
from .core.augmentation import Augmenter, DatasetAugmenter, load_targets
from .core.check import SELECTIONS, check_pairing
from .core.review import apply_review
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
        print("First run: ticha manuscripts -o manuscripts.csv")
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
        print("First run: ticha manuscripts -o manuscripts.csv")
        sys.exit(1)

    image_scraper = TichaImageScraper(rate_limit=args.rate_limit, tile_workers=args.tile_workers)

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


def scrape_printed(args):
    """Download the text of Ticha's printed works (no images or PDFs)."""
    scraper = TichaPrintedScraper(rate_limit=args.rate_limit)
    try:
        if args.list:
            works = scraper.list_works()
            for work in works.itertuples():
                print(f"{work.slug:24} {work.title}")
            return
        print("Starting Ticha printed-text scraper...")
        pages = scraper.scrape(Path(args.output), works=args.work, max_pages=args.max_pages)
    except KeyboardInterrupt:
        print("\nInterrupted by user (pages saved so far are kept; run again to resume)")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        scraper.close()

    print(f"Pages saved to: {Path(args.output).absolute()}")
    for slug, group in pages.groupby("slug"):
        chars = int(group["transcription"].fillna("").str.len().sum())
        print(f"- {slug}: {group['transcription'].notna().sum()} of {len(group)} pages "
              f"with text, {chars:,} characters")
    if args.corpus_dir:
        paths = write_corpus(pages, Path(args.corpus_dir), column=args.corpus_column)
        print(f"Plain-text corpus ({args.corpus_column}): {len(paths)} files in "
              f"{Path(args.corpus_dir).absolute()}")


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
        max_cer=args.max_cer,
        min_margin=args.min_margin,
        review_tolerance=args.review_tolerance,
        export_lines=args.export_lines,
    )

    try:
        manifest = pairer.run(args.texts, args.images, output_dir, max_documents=args.max_docs,
                              workers=args.workers)
    except KeyboardInterrupt:
        print("\nInterrupted by user (finished pages are cached and kept)")
        sys.exit(1)

    counts = manifest['status'].value_counts().to_dict() if not manifest.empty else {}
    print(f"\nPages paired automatically: {counts.get('auto', 0)}  -> {output_dir / 'auto'}")
    print(f"Pages for human review:     {counts.get('review', 0)}  -> {output_dir / 'review'}")
    if counts.get('untranscribed'):
        print(f"Pages without transcription: {counts['untranscribed']}  -> "
              f"{output_dir / 'untranscribed'}")
    print(f"Documents skipped:          {counts.get('skipped', 0)}")
    if counts.get('error'):
        print(f"Pages with errors:          {counts['error']}")
    if not manifest.empty and 'confident_lines' in manifest:
        print(f"Confident line pairs:       {int(manifest['confident_lines'].sum())} "
              f"of {int(manifest['paired_lines'].sum())} paired")
    print(f"Manifest: {(output_dir / 'manifest.csv').absolute()}")


def check_lines(args):
    """Draw line crops with their paired text, to check the pairing by eye."""
    dataset = Path(args.dataset)
    for name in ("manifest.csv", "lines.csv"):
        if not (dataset / name).exists():
            print(f"Error: {dataset / name} not found (run the lines command first)")
            sys.exit(1)
    sheets = check_pairing(
        dataset,
        args.output,
        only=args.only,
        statuses=args.status,
        docs=args.doc,
        sample=args.sample or None,
        seed=args.seed,
        per_sheet=args.per_sheet,
    )
    if not sheets:
        print("No lines match the selection")
        return
    print(f"Wrote {len(sheets)} sheet(s):")
    for sheet in sheets:
        print(f"  {sheet}")


def review_pages(args):
    """Apply a review decisions file to the review/ pages of a lines output."""
    dataset = Path(args.dataset)
    if not (dataset / "manifest.csv").exists():
        print(f"Error: {dataset / 'manifest.csv'} not found (run the lines command first)")
        sys.exit(1)
    result = apply_review(dataset, Path(args.decisions), Path(args.output), min_conf=args.min_conf)
    totals = result[["kept", "fixed", "dropped"]].sum()
    print(f"Reviewed pages: {len(result)} -> {args.output}")
    print(f"Lines kept: {int(totals['kept'])}, fixed: {int(totals['fixed'])}, "
          f"left without text: {int(totals['dropped'])}")
    if result.attrs["unused"]:
        print(f"Decisions for pages not in review (ignored): {len(result.attrs['unused'])}")


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
    parser = argparse.ArgumentParser(
        prog='ticha',
        description='Colonial Zapotec manuscripts from Ticha: scraping, line dataset and augmentation')
    
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
    images_parser.add_argument('--tile-workers', type=int, default=4,
                             help='Image tiles downloaded at the same time (default: 4)')
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

    # Printed command
    printed_parser = subparsers.add_parser(
        'printed', help='Download the text of the printed works (artes, catechism, Agüero)')
    printed_parser.add_argument('-o', '--output', type=str, default='ticha_printed.csv',
                                help='Output CSV, one row per page; existing pages are skipped '
                                     '(default: ticha_printed.csv)')
    printed_parser.add_argument('-w', '--work', action='append',
                                help='Only this work slug, e.g. aguero-miscelaneo (repeatable; '
                                     'default: all, see --list)')
    printed_parser.add_argument('--list', action='store_true',
                                help='List the printed works available and exit')
    printed_parser.add_argument('--corpus-dir', type=str, default=None,
                                help='Also write one plain-text file per work here')
    printed_parser.add_argument('--corpus-column', default='transcription',
                                choices=['transcription', 'transcription_regular', 'zapotec_marked'],
                                help='Text written to --corpus-dir (default: transcription)')
    printed_parser.add_argument('--max-pages', type=int, default=None,
                                help='Maximum pages per work (default: all)')
    printed_parser.add_argument('-r', '--rate-limit', type=float, default=2.0,
                                help='Rate limit in seconds between requests (default: 2.0)')
    printed_parser.add_argument('-v', '--verbose', action='store_true',
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
    lines_parser.add_argument('--max-cer', type=float, default=0.75,
                            help='Pairs up to this CER are still confident when their text fits '
                                 'clearly better than the neighbouring texts (default: 0.75)')
    lines_parser.add_argument('--min-margin', type=float, default=0.15,
                            help='How much better (in CER) than the neighbouring texts, see '
                                 '--max-cer (default: 0.15)')
    lines_parser.add_argument('--review-tolerance', type=float, default=0.2,
                            help='Share of doubtful lines (or of transcription without a line) '
                                 'a page may have and still go to auto/, with the doubtful lines '
                                 'left without text; 0 = every line must be confident '
                                 '(default: 0.2)')
    lines_parser.add_argument('--export-lines', action='store_true',
                            help='Also write line images + .gt.txt of confidently paired lines')
    lines_parser.add_argument('--no-split-columns', action='store_true',
                            help='Do not split transcription lines on runs of 4+ spaces')
    lines_parser.add_argument('--device', type=str, default='cpu',
                            help='Device for kraken, e.g. cpu or cuda:0 (default: cpu)')
    lines_parser.add_argument('-w', '--workers', type=int, default=1,
                            help='Documents processed in parallel, one process each '
                                 '(~2 GB RAM per worker; default: 1)')
    lines_parser.add_argument('--max-docs', type=int, default=None,
                            help='Maximum number of documents to process (default: all)')
    lines_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable verbose logging')

    # Augment command
    check_parser = subparsers.add_parser(
        'check', help='Draw sample line crops with their paired text, to check the pairing')
    check_parser.add_argument('-d', '--dataset', type=str, default='ticha_lines',
                              help='Output directory of the lines command (default: ticha_lines)')
    check_parser.add_argument('-o', '--output', type=str, default='check/check.png',
                              help='Sheet path; numbered _001, _002, ... (default: check/check.png)')
    check_parser.add_argument('--only', choices=SELECTIONS, default='all',
                              help='Lines to draw: confident, doubtful, unpaired (no text), '
                                   'paired (confident + doubtful) or all (default: all)')
    check_parser.add_argument('-s', '--status', action='append',
                              choices=['auto', 'review', 'untranscribed'],
                              help='Only lines of pages with this status (repeatable)')
    check_parser.add_argument('--doc', action='append',
                              help='Only lines of this document id, e.g. SDE740dT (repeatable)')
    check_parser.add_argument('-n', '--sample', type=int, default=30,
                              help='Random lines to draw; 0 = every selected line (default: 30)')
    check_parser.add_argument('--seed', type=int, default=0,
                              help='Random seed of the sample (default: 0)')
    check_parser.add_argument('--per-sheet', type=int, default=15,
                              help='Lines per image (default: 15)')

    review_parser = subparsers.add_parser(
        'review', help='Apply a line-by-line review (decisions file) to the review/ pages')
    review_parser.add_argument('-d', '--dataset', type=str, default='ticha_lines',
                               help='Output directory of the lines command (default: ticha_lines)')
    review_parser.add_argument('--decisions', type=str, default='review/decisions.txt',
                               help='Decisions file: "doc page line keep|drop|fix [text]" per line '
                                    '(default: review/decisions.txt)')
    review_parser.add_argument('-o', '--output', type=str, default='reviewed',
                               help='Folder for the reviewed PAGE XML (default: reviewed)')
    review_parser.add_argument('--min-conf', type=float, default=0.5,
                               help='Lines not in the decisions file keep their text if their '
                                    'confidence is at least this (default: 0.5)')

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
    elif args.command == 'printed':
        scrape_printed(args)
    elif args.command == 'lines':
        pair_lines(args)
    elif args.command == 'check':
        check_lines(args)
    elif args.command == 'review':
        review_pages(args)
    elif args.command == 'augment':
        augment_dataset(args)
    else:
        parser.print_help()
        sys.exit(1)


def deprecated_main():
    """Entry point of the old ``ticha-scraper`` command."""
    print("ticha-scraper is deprecated and will be removed in 2.0; use ticha.", file=sys.stderr)
    main()


if __name__ == "__main__":
    main()
