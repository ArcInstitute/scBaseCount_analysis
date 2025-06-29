#!/usr/bin/env python3
"""
Script to retrieve SRA project IDs for given experiment IDs using BigQuery.
"""

import argparse
import sys
from typing import List, Dict
from google.cloud import bigquery
import pandas as pd


def setup_bigquery_client():
    """Initialize BigQuery client."""
    try:
        client = bigquery.Client()
        return client
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        print("Make sure you have:")
        print("1. Google Cloud SDK installed")
        print("2. Authenticated with 'gcloud auth application-default login'")
        print("3. google-cloud-bigquery package installed")
        sys.exit(1)


def read_experiment_ids(input_file: str) -> List[str]:
    """Read experiment IDs from a file, one per line."""
    try:
        with open(input_file, 'r') as f:
            exp_ids = [line.strip() for line in f if line.strip()]
        return exp_ids
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{input_file}': {e}")
        sys.exit(1)


def query_sra_projects(client, experiment_ids: List[str]) -> Dict[str, str]:
    """
    Query SRA BigQuery to get project IDs for experiment IDs.
    
    Args:
        client: BigQuery client
        experiment_ids: List of SRA experiment IDs (e.g., ['SRX123456', 'SRX789012'])
    
    Returns:
        Dictionary mapping experiment ID to project ID
    """
    if not experiment_ids:
        return {}
    
    # Create the SQL query
    # The SRA metadata is in the nih-sra-datastore.sra_metadata.metadata table
    exp_ids_str = "', '".join(experiment_ids)
    query = f"""
    SELECT 
        m.sra_study,
        m.bioproject,
        m.experiment
    FROM `nih-sra-datastore.sra.metadata` as m
    WHERE m.experiment IN ('{exp_ids_str}')
    """
    
    try:
        print(f"Querying BigQuery for {len(experiment_ids)} experiment IDs...")
        query_job = client.query(query)
        results = query_job.result()
        
        # Convert to dictionary
        exp_to_project = {}
        for row in results:
            exp_to_project[row.experiment] = {
                'bioproject': row.bioproject,
                'sra_study': row.sra_study,
            }
        
        return exp_to_project
        
    except Exception as e:
        print(f"Error querying BigQuery: {e}")
        sys.exit(1)


def write_results(results: Dict[str, Dict], output_file: str = None, format_type: str = 'tsv'):
    """Write results to file or stdout."""
    if not results:
        print("No results to write.")
        return
    
    # Create DataFrame for easier formatting
    data = []
    for exp_id, info in results.items():
        data.append({
            'experiment_id': exp_id,
            'bioproject': info['bioproject'],
            'sra_study': info['sra_study']
        })
    
    df = pd.DataFrame(data)
    
    if output_file:
        if format_type == 'csv':
            df.to_csv(output_file, index=False)
            print(f"Results written to {output_file} (CSV format)")
        else:  # tsv
            df.to_csv(output_file, index=False, sep='\t')
            print(f"Results written to {output_file} (TSV format)")
    else:
        # Print to stdout
        if format_type == 'csv':
            print(df.to_csv(index=False))
        else:  # tsv
            print(df.to_csv(index=False, sep='\t'))


def main():
   
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description='Retrieve SRA project IDs for given experiment IDs using BigQuery',
        formatter_class=CustomFormatter,
        epilog="""
Examples:
  # Query single experiment ID
  python sra_exp_to_project.py -e SRX24267782
  
  # Query multiple experiment IDs
  python sra_exp_to_project.py -e SRX24267782 SRX15466901 ERX10981675
  
  # Read experiment IDs from file
  python sra_exp_to_project.py -i experiment_ids.txt
  
  # Save results to file
  python sra_exp_to_project.py -i experiment_ids.txt -o results.tsv
  
  # Output as CSV
  python sra_exp_to_project.py -e SRX123456 -f csv -o results.csv

Prerequisites:
  - Google Cloud SDK installed and authenticated
  - google-cloud-bigquery package: pip install google-cloud-bigquery pandas
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-e', '--experiments',
        nargs='+',
        help='SRA experiment IDs (space-separated)'
    )
    input_group.add_argument(
        '-i', '--input-file',
        help='File containing experiment IDs (one per line)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['tsv', 'csv'],
        default='tsv',
        help='Output format'
    )
    parser.add_argument(
        '--missing-only',
        action='store_true',
        help='Only show experiment IDs that were not found'
    )
    
    args = parser.parse_args()
    
    # Get experiment IDs
    if args.experiments:
        experiment_ids = args.experiments
    else:
        experiment_ids = read_experiment_ids(args.input_file)
    
    if not experiment_ids:
        print("No experiment IDs provided.")
        sys.exit(1)
    
    print(f"Processing {len(experiment_ids)} experiment IDs...")
    
    # Initialize BigQuery client
    client = setup_bigquery_client()
    
    # Query for project information
    results = query_sra_projects(client, experiment_ids)
    
    # Check for missing experiment IDs
    found_exp_ids = set(results.keys())
    input_exp_ids = set(experiment_ids)
    missing_exp_ids = input_exp_ids - found_exp_ids
    
    if missing_exp_ids:
        print(f"Warning: {len(missing_exp_ids)} experiment IDs not found in SRA:")
        for exp_id in sorted(missing_exp_ids):
            print(f"  {exp_id}")
        print()
    
    if args.missing_only:
        # Only output missing IDs
        if missing_exp_ids:
            for exp_id in sorted(missing_exp_ids):
                print(exp_id)
        else:
            print("All experiment IDs were found.")
    else:
        # Output results
        if results:
            print(f"Found {len(results)} experiment-to-project mappings:")
            write_results(results, args.output, args.format)
        else:
            print("No experiment IDs found in SRA database.")


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(override=True)
    main()
