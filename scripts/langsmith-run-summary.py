#!/usr/bin/env python3
"""
LangSmith Run Metadata Extractor

A comprehensive script to extract run metadata from LangSmith projects
with flexible filtering and export options for both tool calls and LLM runs.

Usage:
    python langsmith_run_extractor.py --project "my-project" --days 7 --run-type tool
    python langsmith_run_extractor.py --project "my-project" --days 7 --run-type llm
    python langsmith_run_extractor.py --project "my-project" --start-date "2025-01-01" --end-date "2025-01-31" --run-type llm
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from langsmith import Client


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract run metadata from LangSmith projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --project "my-agent-project" --days 7 --run-type tool
  %(prog)s --project "my-agent-project" --days 7 --run-type llm
  %(prog)s --project "my-agent-project" --start-date "2025-01-01" --end-date "2025-01-31" --run-type llm
  %(prog)s --project "my-agent-project" --days 30 --run-type tool --output-format json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="LangSmith project name to extract data from"
    )
    
    parser.add_argument(
        "--run-type", "-r",
        choices=["tool", "llm"],
        default="tool",
        help="Type of runs to extract (default: tool)"
    )
    
    # Time filtering options
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--days", "-d",
        type=int,
        help="Number of days to look back from now (e.g., 7 for last 7 days)"
    )
    time_group.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (requires --end-date)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (used with --start-date)"
    )
    
    # Output options
    parser.add_argument(
        "--output-format", "-f",
        choices=["csv", "json", "excel"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Output filename (auto-generated if not specified)"
    )
    
    # Filtering options
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include runs that resulted in errors"
    )
    
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum run duration in seconds to include"
    )
    
    parser.add_argument(
        "--max-duration", 
        type=float,
        help="Maximum run duration in seconds to include"
    )
    
    parser.add_argument(
        "--tool-names",
        nargs="+",
        help="Specific tool names to filter for (space-separated, only for tool runs)"
    )
    
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Specific model names to filter for (space-separated, only for LLM runs)"
    )
    
    # Analysis options
    parser.add_argument(
        "--include-workflow-analysis",
        action="store_true",
        help="Include workflow-level usage analysis"
    )
    
    parser.add_argument(
        "--include-performance-stats",
        action="store_true", 
        help="Include performance statistics summary"
    )
    
    # Other options
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of runs to retrieve (useful for testing)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Validate date arguments
    if args.start_date and not args.end_date:
        raise ValueError("--end-date is required when using --start-date")
    
    if args.end_date and not args.start_date:
        raise ValueError("--start-date is required when using --end-date")
    
    # Validate date formats
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("--start-date must be in YYYY-MM-DD format")
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("--end-date must be in YYYY-MM-DD format")
    
    # Validate duration filters
    if args.min_duration and args.min_duration < 0:
        raise ValueError("--min-duration must be positive")
    
    if args.max_duration and args.max_duration < 0:
        raise ValueError("--max-duration must be positive")
    
    if (args.min_duration and args.max_duration and 
        args.min_duration > args.max_duration):
        raise ValueError("--min-duration cannot be greater than --max-duration")


def get_time_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Get start and end times based on arguments."""
    if args.days:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=args.days)
    else:
        start_time = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_time = datetime.strptime(args.end_date, "%Y-%m-%d")
        # Set end time to end of day
        end_time = end_time.replace(hour=23, minute=59, second=59)
    
    return start_time, end_time


def extract_tool_metadata(run, trace_runs_cache: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive metadata from a tool run."""
    # Calculate duration
    duration_seconds = None
    if run.end_time and run.start_time:
        duration_seconds = (run.end_time - run.start_time).total_seconds()
    
    # Get parent trace information (cached for performance)
    parent_info = {}
    if run.trace_id and run.trace_id not in trace_runs_cache:
        # This would require an additional API call in practice
        # For now, we'll use available information
        trace_runs_cache[run.trace_id] = {"cached": True}
    
    # Extract input/output information
    input_info = {}
    output_info = {}
    
    if run.inputs:
        input_info = {
            "input_keys": list(run.inputs.keys()),
            "input_size_chars": len(str(run.inputs)),
            "input_summary": str(run.inputs)[:200] + "..." if len(str(run.inputs)) > 200 else str(run.inputs)
        }
    
    if run.outputs:
        output_info = {
            "output_keys": list(run.outputs.keys()),
            "output_size_chars": len(str(run.outputs)),
            "output_summary": str(run.outputs)[:200] + "..." if len(str(run.outputs)) > 200 else str(run.outputs)
        }
    
    # Compile metadata
    metadata = {
        # Basic run information
        "run_id": str(run.id),
        "trace_id": str(run.trace_id),
        "tool_name": run.name,
        "run_type": run.run_type,
        
        # Timing information
        "start_time": run.start_time.isoformat() if run.start_time else None,
        "end_time": run.end_time.isoformat() if run.end_time else None,
        "duration_seconds": duration_seconds,
        "duration_ms": duration_seconds * 1000 if duration_seconds else None,
        
        # Status information
        "has_error": run.error is not None,
        "error_message": str(run.error) if run.error else None,
        "status": "error" if run.error else "success",
        
        # Input/Output information
        **input_info,
        **output_info,
        
        # Custom metadata
        "custom_metadata": run.metadata or {},
        "tags": run.tags or [],
        
        # Additional metrics
        "total_tokens": getattr(run, 'total_tokens', None),
        "prompt_tokens": getattr(run, 'prompt_tokens', None),
        "completion_tokens": getattr(run, 'completion_tokens', None),
    }
    
    return metadata


def extract_llm_metadata(run, trace_runs_cache: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive metadata from an LLM run."""
    # Calculate duration
    duration_seconds = None
    if run.end_time and run.start_time:
        duration_seconds = (run.end_time - run.start_time).total_seconds()
    
    # Get parent trace information (cached for performance)
    parent_info = {}
    if run.trace_id and run.trace_id not in trace_runs_cache:
        # This would require an additional API call in practice
        # For now, we'll use available information
        trace_runs_cache[run.trace_id] = {"cached": True}
    
    # Extract input/output information
    input_info = {}
    output_info = {}
    
    if run.inputs:
        input_info = {
            "input_keys": list(run.inputs.keys()),
            "input_size_chars": len(str(run.inputs)),
            "input_summary": str(run.inputs)[:200] + "..." if len(str(run.inputs)) > 200 else str(run.inputs)
        }
    
    if run.outputs:
        output_info = {
            "output_keys": list(run.outputs.keys()),
            "output_size_chars": len(str(run.outputs)),
            "output_summary": str(run.outputs)[:200] + "..." if len(str(run.outputs)) > 200 else str(run.outputs)
        }
    
    # Extract LLM-specific information
    llm_info = {}
    if hasattr(run, 'extra') and run.extra:
        invocation_params = run.extra.get('invocation_params', {})
        llm_info = {
            "model_name": invocation_params.get('model_name') or invocation_params.get('model'),
            "temperature": invocation_params.get('temperature'),
            "max_tokens": invocation_params.get('max_tokens'),
            "top_p": invocation_params.get('top_p'),
            "frequency_penalty": invocation_params.get('frequency_penalty'),
            "presence_penalty": invocation_params.get('presence_penalty'),
        }
    
    # Compile metadata
    metadata = {
        # Basic run information
        "run_id": str(run.id),
        "trace_id": str(run.trace_id),
        "run_name": run.name,
        "run_type": run.run_type,
        
        # Timing information
        "start_time": run.start_time.isoformat() if run.start_time else None,
        "end_time": run.end_time.isoformat() if run.end_time else None,
        "duration_seconds": duration_seconds,
        "duration_ms": duration_seconds * 1000 if duration_seconds else None,
        
        # Status information
        "has_error": run.error is not None,
        "error_message": str(run.error) if run.error else None,
        "status": "error" if run.error else "success",
        
        # Input/Output information
        **input_info,
        **output_info,
        
        # LLM-specific information
        **llm_info,
        
        # Token usage information
        "total_tokens": getattr(run, 'total_tokens', None),
        "prompt_tokens": getattr(run, 'prompt_tokens', None),
        "completion_tokens": getattr(run, 'completion_tokens', None),
        "prompt_cost": getattr(run, 'prompt_cost', None),
        "completion_cost": getattr(run, 'completion_cost', None),
        "total_cost": getattr(run, 'total_cost', None),
        
        # Custom metadata
        "custom_metadata": run.metadata or {},
        "tags": run.tags or [],
    }
    
    return metadata


def filter_run_data(run_data: List[Dict], args: argparse.Namespace) -> List[Dict]:
    """Apply filters to run data based on arguments."""
    filtered_data = run_data
    
    # Filter by error status
    if not args.include_errors:
        filtered_data = [r for r in filtered_data if not r["has_error"]]
        logging.info(f"Filtered out {len(run_data) - len(filtered_data)} error runs")
    
    # Filter by duration
    if args.min_duration:
        before_count = len(filtered_data)
        filtered_data = [r for r in filtered_data 
                        if r["duration_seconds"] and r["duration_seconds"] >= args.min_duration]
        logging.info(f"Filtered out {before_count - len(filtered_data)} runs below {args.min_duration}s")
    
    if args.max_duration:
        before_count = len(filtered_data)
        filtered_data = [r for r in filtered_data 
                        if r["duration_seconds"] and r["duration_seconds"] <= args.max_duration]
        logging.info(f"Filtered out {before_count - len(filtered_data)} runs above {args.max_duration}s")
    
    # Filter by tool names (for tool runs)
    if args.tool_names and args.run_type == "tool":
        before_count = len(filtered_data)
        filtered_data = [r for r in filtered_data if r.get("tool_name") in args.tool_names]
        logging.info(f"Filtered to {len(filtered_data)} runs matching tool names: {args.tool_names}")
    
    # Filter by model names (for LLM runs)
    if args.model_names and args.run_type == "llm":
        before_count = len(filtered_data)
        filtered_data = [r for r in filtered_data if r.get("model_name") and r.get("model_name") in args.model_names]
        logging.info(f"Filtered to {len(filtered_data)} runs matching model names: {args.model_names}")
    
    return filtered_data


def generate_workflow_analysis(run_data: List[Dict], run_type: str) -> Dict[str, Any]:
    """Generate workflow-level analysis of run usage."""
    workflows = defaultdict(list)
    
    # Group runs by trace_id (workflow)
    for run in run_data:
        workflows[run["trace_id"]].append(run)
    
    workflow_stats = []
    for trace_id, runs in workflows.items():
        # Sort runs by start time to get execution order
        runs_sorted = sorted([r for r in runs if r["start_time"]], 
                           key=lambda x: x["start_time"])
        
        total_duration = sum(r["duration_seconds"] for r in runs if r["duration_seconds"])
        
        workflow_stat = {
            "trace_id": trace_id,
            f"{run_type}_count": len(runs),
            "total_duration": total_duration,
            "avg_duration": total_duration / len(runs) if runs else 0,
            "error_count": sum(1 for r in runs if r["has_error"]),
            "success_rate": (len(runs) - sum(1 for r in runs if r["has_error"])) / len(runs) if runs else 0
        }
        
        if run_type == "tool":
            workflow_stat.update({
                "unique_tools": len(set(r["tool_name"] for r in runs if r.get("tool_name"))),
                "tool_sequence": [r["tool_name"] for r in runs_sorted if r.get("tool_name")],
            })
        elif run_type == "llm":
            workflow_stat.update({
                "unique_models": len(set(r["model_name"] for r in runs if r.get("model_name"))),
                "total_tokens": sum(r["total_tokens"] for r in runs if r.get("total_tokens")),
                "total_cost": sum(r["total_cost"] for r in runs if r.get("total_cost")),
            })
        
        if runs_sorted:
            workflow_stat.update({
                f"first_{run_type}": runs_sorted[0].get("tool_name" if run_type == "tool" else "run_name"),
                f"last_{run_type}": runs_sorted[-1].get("tool_name" if run_type == "tool" else "run_name"),
                "workflow_start": runs_sorted[0]["start_time"],
                "workflow_end": runs_sorted[-1]["end_time"] if runs_sorted[-1]["end_time"] else None
            })
        
        workflow_stats.append(workflow_stat)
    
    return {
        "total_workflows": len(workflows),
        f"total_{run_type}s": len(run_data),
        f"avg_{run_type}s_per_workflow": len(run_data) / len(workflows) if workflows else 0,
        "workflows": workflow_stats
    }


def generate_performance_stats(run_data: List[Dict], run_type: str) -> Dict[str, Any]:
    """Generate performance statistics for runs."""
    if not run_data:
        return {}
    
    df = pd.DataFrame(run_data)
    
    # Overall stats
    overall_stats = {
        f"total_{run_type}_calls": len(run_data),
        "error_rate": df["has_error"].mean(),
        "avg_duration": df["duration_seconds"].mean() if "duration_seconds" in df else None,
        "median_duration": df["duration_seconds"].median() if "duration_seconds" in df else None,
    }
    
    if run_type == "tool":
        overall_stats["unique_tools"] = df.get("tool_name", pd.Series()).nunique()
    elif run_type == "llm":
        overall_stats.update({
            "unique_models": df.get("model_name", pd.Series()).nunique(),
            "total_tokens": df.get("total_tokens", pd.Series()).sum(),
            "avg_tokens": df.get("total_tokens", pd.Series()).mean(),
            "total_cost": df.get("total_cost", pd.Series()).sum(),
            "avg_cost": df.get("total_cost", pd.Series()).mean(),
        })
    
    # Per-item stats (tools or models)
    item_stats = []
    groupby_field = "tool_name" if run_type == "tool" else "model_name"
    
    if groupby_field in df.columns and df[groupby_field].notna().any():
        for item_name in df[groupby_field].dropna().unique():
            item_df = df[df[groupby_field] == item_name]
            duration_data = item_df["duration_seconds"].dropna()
            
            item_stat = {
                f"{run_type}_name": item_name,
                "call_count": len(item_df),
                "error_count": item_df["has_error"].sum(),
                "error_rate": item_df["has_error"].mean(),
                "avg_duration": duration_data.mean() if len(duration_data) > 0 else None,
                "median_duration": duration_data.median() if len(duration_data) > 0 else None,
                "min_duration": duration_data.min() if len(duration_data) > 0 else None,
                "max_duration": duration_data.max() if len(duration_data) > 0 else None,
                "std_duration": duration_data.std() if len(duration_data) > 0 else None,
            }
            
            if run_type == "llm":
                token_data = item_df["total_tokens"].dropna()
                cost_data = item_df["total_cost"].dropna()
                item_stat.update({
                    "total_tokens": token_data.sum() if len(token_data) > 0 else None,
                    "avg_tokens": token_data.mean() if len(token_data) > 0 else None,
                    "total_cost": cost_data.sum() if len(cost_data) > 0 else None,
                    "avg_cost": cost_data.mean() if len(cost_data) > 0 else None,
                })
            
            item_stats.append(item_stat)
    
    return {
        "overall": overall_stats,
        f"by_{run_type}": sorted(item_stats, key=lambda x: x["call_count"], reverse=True)
    }


def save_data(data: Dict[str, Any], args: argparse.Namespace) -> str:
    """Save data to file in specified format."""
    # Generate filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"langsmith_{args.run_type}s_{args.project}_{timestamp}"
        
        if args.output_format == "csv":
            filename = f"{base_name}.csv"
        elif args.output_format == "json":
            filename = f"{base_name}.json"
        elif args.output_format == "excel":
            filename = f"{base_name}.xlsx"
    else:
        filename = args.output_file
    
    # Save based on format
    if args.output_format == "csv":
        df = pd.DataFrame(data["run_data"])
        df.to_csv(filename, index=False)
        
    elif args.output_format == "json":
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    elif args.output_format == "excel":
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main run data
            pd.DataFrame(data["run_data"]).to_excel(writer, sheet_name=f"{args.run_type.title()}_Data", index=False)
            
            # Performance stats if available
            if "performance_stats" in data and data["performance_stats"]:
                if f"by_{args.run_type}" in data["performance_stats"]:
                    pd.DataFrame(data["performance_stats"][f"by_{args.run_type}"]).to_excel(
                        writer, sheet_name="Performance_Stats", index=False
                    )
            
            # Workflow analysis if available
            if "workflow_analysis" in data and "workflows" in data["workflow_analysis"]:
                pd.DataFrame(data["workflow_analysis"]["workflows"]).to_excel(
                    writer, sheet_name="Workflow_Analysis", index=False
                )
    
    return filename


def main():
    """Main function."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        setup_logging(args.verbose)
        
        logging.info(f"Starting {args.run_type} metadata extraction for project: {args.project}")
        
        # Initialize LangSmith client
        try:
            client = Client()
            logging.info("Successfully connected to LangSmith")
        except Exception as e:
            logging.error(f"Failed to initialize LangSmith client: {e}")
            logging.error("Make sure LANGSMITH_API_KEY environment variable is set")
            sys.exit(1)
        
        # Get time range
        start_time, end_time = get_time_range(args)
        logging.info(f"Extracting data from {start_time} to {end_time}")
        
        # Build query parameters
        query_params = {
            "project_name": args.project,
            "run_type": args.run_type,
            "start_time": start_time,
        }
        
        if args.limit:
            query_params["limit"] = args.limit
        
        # Query runs
        logging.info(f"Querying LangSmith for {args.run_type} runs...")
        runs = client.list_runs(**query_params)
        
        # Convert to list and process
        runs_list = list(runs)
        logging.info(f"Retrieved {len(runs_list)} {args.run_type} runs")
        
        if not runs_list:
            logging.warning(f"No {args.run_type} runs found for the specified criteria")
            sys.exit(0)
        
        # Extract metadata
        logging.info(f"Extracting {args.run_type} metadata...")
        trace_cache = {}
        run_data = []
        
        for run in runs_list:
            try:
                if args.run_type == "tool":
                    metadata = extract_tool_metadata(run, trace_cache)
                else:  # llm
                    metadata = extract_llm_metadata(run, trace_cache)
                run_data.append(metadata)
            except Exception as e:
                logging.warning(f"Failed to extract metadata for run {run.id}: {e}")
        
        logging.info(f"Successfully extracted metadata for {len(run_data)} {args.run_type} runs")
        
        # Apply filters
        if (args.min_duration or args.max_duration or args.tool_names or 
            args.model_names or not args.include_errors):
            logging.info("Applying filters...")
            run_data = filter_run_data(run_data, args)
            logging.info(f"After filtering: {len(run_data)} {args.run_type} runs remain")
        
        if not run_data:
            logging.warning(f"No {args.run_type} runs remain after filtering")
            sys.exit(0)
        
        # Prepare output data
        output_data = {
            "metadata": {
                "project": args.project,
                "run_type": args.run_type,
                "extraction_time": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                f"total_{args.run_type}_calls": len(run_data),
                "filters_applied": {
                    "include_errors": args.include_errors,
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "tool_names": args.tool_names,
                    "model_names": args.model_names
                }
            },
            "run_data": run_data
        }
        
        # Generate additional analyses if requested
        if args.include_workflow_analysis:
            logging.info("Generating workflow analysis...")
            output_data["workflow_analysis"] = generate_workflow_analysis(run_data, args.run_type)
        
        if args.include_performance_stats:
            logging.info("Generating performance statistics...")
            output_data["performance_stats"] = generate_performance_stats(run_data, args.run_type)
        
        # Save data
        logging.info(f"Saving data in {args.output_format} format...")
        output_file = save_data(output_data, args)
        
        # Print summary
        print(f"\n✅ Successfully extracted {args.run_type} metadata!")
        print(f"📊 Total {args.run_type} calls: {len(run_data)}")
        
        if args.run_type == "tool":
            print(f"🔧 Unique tools: {len(set(r.get('tool_name') for r in run_data if r.get('tool_name')))}")
        else:  # llm
            print(f"🤖 Unique models: {len(set(r.get('model_name') for r in run_data if r.get('model_name')))}")
            total_tokens = sum(r.get('total_tokens', 0) for r in run_data if r.get('total_tokens'))
            total_cost = sum(r.get('total_cost', 0) for r in run_data if r.get('total_cost'))
            if total_tokens > 0:
                print(f"🎯 Total tokens: {total_tokens:,}")
            if total_cost > 0:
                print(f"💰 Total cost: ${total_cost:.4f}")
        
        print(f"⏱️  Time range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        print(f"💾 Output saved to: {output_file}")
        
        if args.include_performance_stats and "performance_stats" in output_data:
            stats = output_data["performance_stats"]["overall"]
            print(f"📈 Error rate: {stats.get('error_rate', 0):.2%}")
            if stats.get('avg_duration'):
                print(f"⏱️  Average duration: {stats['avg_duration']:.2f}s")
        
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # check for LANGSMITH_API_KEY
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is not set")
    main()