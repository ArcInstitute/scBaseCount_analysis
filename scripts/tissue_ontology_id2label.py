#!/usr/bin/env python3
"""
UBERON ID to Label Lookup using EBI OLS REST API
This version queries the API directly without downloading the entire ontology
"""

import requests
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import time
from bs4 import BeautifulSoup
import re
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

class UberonAPILookup:
    def __init__(self):
        """Initialize UBERON API lookup tool."""
        self.base_url = "https://www.ebi.ac.uk/ols4"
        self.cache = {}  # Simple in-memory cache
    
    def lookup_single(self, uberon_id: str) -> Optional[str]:
        """
        Look up a single UBERON ID using the OLS API.
        
        Args:
            uberon_id: UBERON ID (e.g., 'UBERON:0002106')
            
        Returns:
            Label string or None if not found
        """
        # Check cache first
        if uberon_id in self.cache:
            return self.cache[uberon_id]
        
        # Try different API endpoint format for OLS4
        # First try: search by obo_id
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/uberon/terms?obo_id={uberon_id}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse JSON response from API
                data = response.json()
                
                # Check if we got results in the embedded terms
                if '_embedded' in data and 'terms' in data['_embedded']:
                    terms = data['_embedded']['terms']
                    if terms:
                        # Get the first matching term
                        term = terms[0]
                        label = term.get('label', None)
                        
                        if label:
                            self.cache[uberon_id] = label
                            return label
                
                return None
            else:
                return None
        except Exception as e:
            print(f"Error looking up {uberon_id}: {e}", file=sys.stderr)
            return None
    
    def lookup_batch(self, uberon_ids: List[str], delay: float = 0.1, no_delay: bool = False, show_progress: bool = True) -> Dict[str, str]:
        """
        Look up multiple UBERON IDs with rate limiting.
        
        Args:
            uberon_ids: List of UBERON IDs
            delay: Delay between API calls in seconds
            no_delay: Skip delays between API calls
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping IDs to labels
        """
        results = {}
        total = len(uberon_ids)
        
        if not show_progress:
            print(f"Looking up {total} UBERON IDs...")
        
        if show_progress:
            console = Console()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Looking up UBERON IDs...", total=total)
                
                for i, uberon_id in enumerate(uberon_ids, 1):
                    progress.update(task, description=f"Processing {uberon_id}...")
                    
                    label = self.lookup_single(uberon_id)
                    results[uberon_id] = label if label else "unknown"
                    
                    progress.advance(task)
                    
                    # Rate limiting
                    if not no_delay and i < total:
                        time.sleep(delay)
                
                progress.update(task, description="Lookup complete!")
        else:
            for i, uberon_id in enumerate(uberon_ids, 1):
                label = self.lookup_single(uberon_id)
                results[uberon_id] = label if label else "unknown"
                
                # Progress indicator
                if i % 5 == 0:
                    print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
                
                # Rate limiting
                if not no_delay and i < total:
                    time.sleep(delay)
            
            print(f"Completed {total} lookups")
        
        return results


def read_ids_from_file(filepath: str) -> List[str]:
    """
    Read UBERON IDs from a file.
    
    Args:
        filepath: Path to file containing UBERON IDs (one per line)
        
    Returns:
        List of UBERON IDs
    """
    ids = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    ids.append(line)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)
    
    return ids


def output_results(results: Dict[str, str], output_format: str, output_file: Optional[str] = None):
    """
    Output results in the specified format.
    
    Args:
        results: Dictionary mapping UBERON IDs to labels
        output_format: Output format ('table', 'json', 'csv', 'tsv')
        output_file: Optional output file path
    """
    if output_format == 'table':
        console = Console()
        table = Table(title="UBERON ID Lookup Results")
        table.add_column("UBERON ID", style="cyan")
        table.add_column("Label", style="green")
        table.add_column("Status", style="yellow")
        
        for uberon_id, label in results.items():
            status = "Found" if label != "unknown" else "unknown"
            table.add_row(uberon_id, label, status)
        
        if output_file:
            with open(output_file, 'w') as f:
                console = Console(file=f, record=True)
                console.print(table)
        else:
            console.print(table)
            
    elif output_format == 'json':
        output_data = json.dumps(results, indent=2)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_data)
        else:
            print(output_data)
            
    elif output_format == 'csv':
        import csv
        if output_file:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['UBERON_ID', 'Label'])
                for uberon_id, label in results.items():
                    writer.writerow([uberon_id, label])
        else:
            print("UBERON_ID,Label")
            for uberon_id, label in results.items():
                print(f"{uberon_id},{label}")
                
    elif output_format == 'tsv':
        if output_file:
            with open(output_file, 'w') as f:
                f.write("UBERON_ID\tLabel\n")
                for uberon_id, label in results.items():
                    f.write(f"{uberon_id}\t{label}\n")
        else:
            print("UBERON_ID\tLabel")
            for uberon_id, label in results.items():
                print(f"{uberon_id}\t{label}")


def parse_args():
    """Parse command line arguments."""
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description="Look up UBERON tissue ontology IDs and return their labels using EBI OLS API",
        formatter_class=CustomFormatter,
        epilog="""
Examples:
  # Look up specific UBERON IDs
  python tissue-ontology-id-to-label.py UBERON:0000178 UBERON:0002371
  
  # Look up IDs from a file
  python tissue-ontology-id-to-label.py --input-file uberon_ids.txt
  
  # Output as JSON to a file
  python tissue-ontology-id-to-label.py --output-format json --output-file results.json UBERON:0000178
  
  # Dry run to see what would be looked up
  python tissue-ontology-id-to-label.py --dry-run UBERON:0000178 UBERON:0002371
  
  # Fast lookup with no delays (use cautiously)
  python tissue-ontology-id-to-label.py --no-delay UBERON:0000178 UBERON:0002371
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "uberon_ids",
        nargs="*",
        help="UBERON IDs to look up (e.g., UBERON:0000178)"
    )
    input_group.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="File containing UBERON IDs (one per line)"
    )
    
    # Output options
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["table", "json", "csv", "tsv"],
        default="table",
        help="Output format"
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        help="Output file path (if not specified, prints to stdout)"
    )
    
    # Rate limiting options
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between API calls"
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Skip delays between API calls (overrides --delay)"
    )
    
    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be looked up without making API calls"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Don't show progress bar"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get UBERON IDs from command line or file
    if args.input_file:
        uberon_ids = read_ids_from_file(args.input_file)
    else:
        uberon_ids = args.uberon_ids
    
    if not uberon_ids:
        print("Error: No UBERON IDs provided", file=sys.stderr)
        sys.exit(1)
    
    # Validate UBERON IDs format
    valid_ids = []
    for uberon_id in uberon_ids:
        if not uberon_id.startswith('UBERON:'):
            print(f"Warning: '{uberon_id}' doesn't appear to be a valid UBERON ID (should start with 'UBERON:')", file=sys.stderr)
        valid_ids.append(uberon_id)
    
    if args.dry_run:
        console = Console()
        console.print(f"[bold]DRY RUN:[/bold] Would look up {len(valid_ids)} UBERON IDs")
        console.print(f"[bold]Output format:[/bold] {args.output_format}")
        if args.output_file:
            console.print(f"[bold]Output file:[/bold] {args.output_file}")
        else:
            console.print("[bold]Output:[/bold] stdout")
        console.print(f"[bold]Delay:[/bold] {'None' if args.no_delay else f'{args.delay}s'}")
        console.print("\n[bold]IDs to look up:[/bold]")
        for i, uberon_id in enumerate(valid_ids, 1):
            console.print(f"  {i:2d}. {uberon_id}")
        return
    
    # Initialize API lookup
    lookup = UberonAPILookup()
    
    # Perform batch lookup
    try:
        results = lookup.lookup_batch(
            valid_ids,
            delay=args.delay,
            no_delay=args.no_delay,
            show_progress=not args.no_progress
        )
        
        # Output results
        output_results(results, args.output_format, args.output_file)
        
        # Print summary to stderr so it doesn't interfere with output
        found = sum(1 for label in results.values() if label != "unknown")
        unknown = len(results) - found
        
        if not args.no_progress:
            console = Console(file=sys.stderr)
            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f" - [green]Found: {found}[/green]")
            console.print(f" - [red]Unknown: {unknown}[/red]")
            console.print(f" - [blue]Total processed: {len(results)}[/blue]")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()