#!/usr/bin/env python3
"""
Script to fetch Entrez IDs for SRA experiment accessions using Bio.Entrez.
Reads a CSV file with SRA experiment accessions and adds corresponding Entrez IDs.
"""

import argparse
import csv
import sys
import time
from typing import List, Dict, Optional
import pandas as pd
from Bio import Entrez
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_entrez_id(accession: str, email: str, retries: int = 3, delay: float = 0.5) -> Optional[str]:
    """
    Get Entrez ID for a given SRA experiment accession.
    
    Args:
        accession: SRA experiment accession (e.g., SRX123456)
        email: Email address for Entrez queries (required by NCBI)
        retries: Number of retry attempts for failed queries
        delay: Delay between requests to respect rate limits
        
    Returns:
        Entrez ID as string, or None if not found/error occurred
    """
    Entrez.email = email
    
    for attempt in range(retries):
        try:
            # Search SRA database for the experiment accession
            handle = Entrez.esearch(db="sra", term=accession, rettype="uilist")
            search_results = Entrez.read(handle)
            handle.close()
            
            # Get the Entrez ID from search results
            if search_results["IdList"]:
                entrez_id = search_results["IdList"][0]
                logger.debug(f"Found Entrez ID {entrez_id} for {accession}")
                time.sleep(delay)  # Rate limiting
                return entrez_id
            else:
                logger.warning(f"No Entrez ID found for {accession}")
                time.sleep(delay)
                return None
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {accession}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"All attempts failed for {accession}")
                return None
    
    return None


def process_csv(input_file: str, output_file: str, accession_column: str, 
                email: str, batch_size: int = 10, max_records: int = None, 
                list_only: bool = False) -> None:
    """
    Process CSV file to add Entrez IDs for SRA experiment accessions.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        accession_column: Name of column containing SRA experiment accessions
        email: Email address for Entrez queries
        batch_size: Number of queries to process before saving progress
        max_records: Just process a max number of records (useful for testing)
        list_only: Output only a space-separated list of unique Entrez IDs to stdout
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")

        # Check if accession column exists
        if accession_column not in df.columns:
            raise ValueError(f"Column '{accession_column}' not found in input file")

        # Get unique accessions
        accessions = df[accession_column].dropna().unique().tolist()
        logger.info(f"Found {len(accessions)} unique accessions")

        # Limit records if specified
        if max_records:
            accessions = accessions[:max_records]
            logger.info(f"Processing first {len(accessions)} records due to --max-records")

        # Check if we already have some Entrez IDs (for resume functionality)
        temp_output = output_file + '.temp'
        processed_accessions = set()
        entrez_mapping = {}
        
        if os.path.exists(temp_output):
            logger.info(f"Found temporary file {temp_output}, resuming from previous run")
            temp_df = pd.read_csv(temp_output)
            if 'entrez_id' in temp_df.columns:
                for idx, row in temp_df.iterrows():
                    acc = row[accession_column]
                    entrez_id = row['entrez_id']
                    if pd.notna(entrez_id):
                        entrez_mapping[acc] = str(entrez_id)
                        processed_accessions.add(acc)
                logger.info(f"Resuming: {len(processed_accessions)} accessions already processed")

        # Process accessions in batches
        remaining_accessions = [acc for acc in accessions if acc not in processed_accessions]
        logger.info(f"Processing {len(remaining_accessions)} remaining accessions in batches of {batch_size}")
        
        total_processed = len(processed_accessions)
        
        for i in range(0, len(remaining_accessions), batch_size):
            batch = remaining_accessions[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: accessions {i+1}-{min(i+batch_size, len(remaining_accessions))} of {len(remaining_accessions)} remaining")
            
            # Process each accession in the batch
            for accession in batch:
                entrez_id = get_entrez_id(accession, email)
                entrez_mapping[accession] = str(entrez_id) if entrez_id else ""
                total_processed += 1
                
                if total_processed % 10 == 0:
                    logger.info(f"Progress: {total_processed}/{len(accessions)} accessions processed")
            
            # Save progress after each batch
            _save_progress(df, entrez_mapping, accession_column, temp_output)
            logger.info(f"Batch completed. Progress saved to {temp_output}")

        # Create final output with Entrez IDs
        logger.info("Creating final output file...")
        
        # Convert mapping to DataFrame
        entrez_list = [[acc, entrez_mapping.get(acc, "")] for acc in accessions]
        entrez_ids_df = pd.DataFrame(entrez_list, columns=[accession_column, "entrez_id"])
        
        # Merge with original dataframe
        df_final = df.merge(entrez_ids_df, on=accession_column, how='left')
        
        # Write final output
        if list_only:
            # write to stdout
            x = df_final["entrez_id"].dropna().unique()
            print(" ".join(x.astype(str).tolist()))
        else:
            df_final.to_csv(output_file, index=False)
        
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
            logger.info(f"Removed temporary file {temp_output}")

        # Final status
        successful_queries = sum(1 for v in entrez_mapping.values() if v and v != "")
        logger.info(f"Processing complete! {successful_queries}/{len(accessions)} accessions successfully mapped to Entrez IDs")
        logger.info(f"Final output written to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise



def _save_progress(df: pd.DataFrame, entrez_mapping: Dict[str, str], 
                   accession_column: str, temp_output: str) -> None:
    """Save current progress to temporary file."""
    try:
        # Create a copy of the dataframe with current Entrez ID mappings
        df_temp = df.copy()
        df_temp['entrez_id'] = df_temp[accession_column].map(entrez_mapping)
        df_temp.to_csv(temp_output, index=False)
    except Exception as e:
        logger.warning(f"Failed to save progress: {str(e)}")


def main():
    """Main function to handle command line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Fetch Entrez IDs for SRA experiment accessions from a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s experiments.csv -o experiments_entrez.csv -e your.email@domain.com
  %(prog)s data.csv -o output.csv -c experiment_id -e user@example.com
  %(prog)s input.csv -o output.csv -e user@example.com --batch-size 20 --verbose
  %(prog)s input.csv -e user@example.com --list-only --max-records 50
        """
    )
    
    parser.add_argument(
        'samples_table',
        help='Input CSV file containing SRA experiment accessions'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output CSV file with added Entrez IDs (default: input, but with "_entrez" suffix)'
    )
    
    parser.add_argument(
        '-c', '--column',
        default='accession',
        help='Name of column containing SRA experiment accessions (default: accession)'
    )
    
    parser.add_argument(
        '-e', '--email',
        required=True,
        help='Email address for NCBI Entrez queries (required by NCBI)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of queries to process before saving progress (default: 10)'
    )

    parser.add_argument(
        '--max-records',
        type=int,
        default=None,
        help='Just process a max number of records (useful for testing)'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Output only a space-separated list of unique Entrez IDs to stdout'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.batch_size < 1:
        parser.error("Batch size must be at least 1")
    
    if not args.output:
        args.output = os.path.splitext(args.samples_table)[0] + '_entrez.csv'

    # Process the CSV file
    try:
        process_csv(
            input_file=args.samples_table,
            output_file=args.output,
            accession_column=args.column,
            email=args.email,
            batch_size=args.batch_size,
            max_records=args.max_records,
            list_only=args.list_only
        )
        logger.info("Script completed successfully!")
        
    except FileNotFoundError:
        logger.error(f"Input file '{args.samples_table}' not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()