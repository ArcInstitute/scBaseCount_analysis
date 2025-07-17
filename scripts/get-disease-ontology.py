#!/usr/bin/env python
"""
Script to obtain disease ontology for a set of SRX experiment accessions.
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_core.messages import HumanMessage
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from SRAgent.workflows.disease_ontology import create_disease_ontology_workflow
from SRAgent.tools.utils import set_entrez_access

# Create console instance
console = Console()


def parse_args():
    """Parse command line arguments."""
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description="Update missing tissue_ontology_term_id values in SRX metadata database",
        formatter_class=CustomFormatter,
        epilog="""
Examples:
  # Update missing tissue ontologies in test database (first 10 records)
  python update_tissue_ontologies.py --tenant test --limit 10
  
  # Update all missing tissue ontologies in production database
  python update_tissue_ontologies.py --tenant prod
  
  # Update with no delay between API calls (use cautiously)
  python update_tissue_ontologies.py --tenant test --no-delay
        """
    )
    parser.add_argument(
        "disease_csv",
        type=str,
        help="CSV file containing diseases ('disease' column)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="diease-ontology.csv",
        help="Output CSV file to write results to"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Save checkpoint every N records. The checkpoint file is {--output-csv}.checkpoint"
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart the workflow, regardless of checkpoint"
    )
    return parser.parse_args()

async def get_disease_ontology(disease: str) -> List[str]:
    """
    Get disease ontology for a given disease.
    """
    workflow = create_disease_ontology_workflow(model_name="o4-mini", reasoning_effort="medium", temperature=0.0)
    msg = f"Diseases: {disease}"
    input = {"messages": [HumanMessage(content=msg)]}
    return await workflow.ainvoke(input)

def load_checkpoint(checkpoint_file: str) -> Optional[pd.DataFrame]:
    """Load checkpoint data if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            return pd.read_csv(checkpoint_file)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not load checkpoint: {e}")
    return None

def save_checkpoint(data: pd.DataFrame, checkpoint_file: str) -> None:
    """Save checkpoint data to file."""
    try:
        data.to_csv(checkpoint_file, index=False)
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not save checkpoint: {e}")

async def process_diseases(args):
    """Main processing function - now async."""
    # Setup checkpoint file
    checkpoint_file = f"{args.output_csv}.checkpoint"
    
    # Connect to database and get records missing tissue ontology
    target_records = pd.read_csv(args.disease_csv)
    req_cols = ["disease"]
    for col in req_cols:
        if col not in target_records.columns:
            console.print(f"[red]Error:[/red] {col} column not found in input CSV", file=sys.stderr)
            sys.exit(1)
    # remove duplcate rows for req_cols
    target_records = target_records[req_cols].drop_duplicates()

    # limit to --limit
    if args.limit:
        target_records = target_records.head(args.limit)
    
    # Check for existing checkpoint
    if args.force_restart:
        checkpoint_data = None
    else:
        checkpoint_data = load_checkpoint(checkpoint_file)

    start_idx = 0
    if checkpoint_data is not None:
        console.print(f"[blue]Info:[/blue] Found checkpoint with {len(checkpoint_data)} processed records")
        # Merge checkpoint data with target records to find unprocessed records
        processed_diseases = set(checkpoint_data['disease'].tolist())
        unprocessed_mask = ~target_records['disease'].isin(processed_diseases)
        target_records = target_records[unprocessed_mask].reset_index(drop=True)
        console.print(f"[blue]Info:[/blue] Resuming with {len(target_records)} remaining records")
    
    console.print(f"Number of records to process: [bold]{len(target_records)}[/bold]")
    
    if len(target_records) == 0:
        console.print("[green]All records already processed![/green]")
        if checkpoint_data is not None:
            checkpoint_data.to_csv(args.output_csv, index=False)
            console.print(f"[green]Final results written to {args.output_csv}[/green]")
        return

    # get disease ontology with progress bar
    results_list = []
    if checkpoint_data is not None:
        results_list = checkpoint_data.to_dict('records')
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing diseases...", total=len(target_records))
        
        for idx, (_, row) in enumerate(target_records.iterrows()):
            try:
                results = await get_disease_ontology(row["disease"])
                ontology_ids = ";".join(results)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                ontology_ids = None
            
            # Add result to list
            result_row = row.to_dict()
            result_row["disease_ontology_term_id"] = ontology_ids
            results_list.append(result_row)
            
            progress.advance(task)
            
            # Save checkpoint every N records
            if (idx + 1) % args.checkpoint_freq == 0:
                checkpoint_df = pd.DataFrame(results_list)
                save_checkpoint(checkpoint_df, checkpoint_file)
                console.print(f"[blue]Checkpoint saved[/blue] ({idx + 1} records processed)")
    
    # Final save
    final_results = pd.DataFrame(results_list)
    final_results.to_csv(args.output_csv, index=False)
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    console.print(f"[green]Success![/green] Wrote [bold]{len(final_results)}[/bold] records to [bold]{args.output_csv}[/bold]")

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Run the async processing function
    asyncio.run(process_diseases(args))

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()