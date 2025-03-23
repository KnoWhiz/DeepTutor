#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download recent arXiv papers.
This script allows users to download a specified number of the most recent arXiv papers
to a specified directory. The script tracks previously downloaded papers to ensure each run
retrieves different papers. Each run selects papers from a random date range.
"""

print("Starting arXiv loader script...")

import os
import arxiv
import datetime
import argparse
import time
import json
import random
from typing import List, Optional, Set, Dict, Any, Tuple

print("Imports completed successfully")


def load_download_history(history_file: str) -> Set[str]:
    """
    Load the history of previously downloaded papers.

    Args:
        history_file: Path to the download history file

    Returns:
        Set of previously downloaded paper IDs
    """
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
                return set(history_data.get("downloaded_ids", []))
        except Exception as e:
            print(f"Error reading download history: {str(e)}")
            return set()
    return set()


def save_download_history(history_file: str, downloaded_ids: Set[str]) -> None:
    """
    Save the updated history of downloaded papers.

    Args:
        history_file: Path to the download history file
        downloaded_ids: Set of all downloaded paper IDs
    """
    try:
        history_data: Dict[str, Any] = {"downloaded_ids": list(downloaded_ids)}
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)
    except Exception as e:
        print(f"Error saving download history: {str(e)}")


def generate_random_date_range() -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Generate a random date range within the last 10 years.

    Returns:
        Tuple of (start_date, end_date) for the random date range
    """
    # Get current date
    today = datetime.datetime.now()
    
    # Set the maximum range to 10 years ago
    max_days_ago = 365 * 10
    
    # Select a random date between 10 years ago and yesterday
    random_days_ago = random.randint(1, max_days_ago)
    random_date = today - datetime.timedelta(days=random_days_ago)
    
    # Create a date range of one month around the random date
    start_date = random_date - datetime.timedelta(days=15)
    end_date = random_date + datetime.timedelta(days=15)
    
    print(f"Generated random date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    return start_date, end_date


def download_recent_papers(num_papers: int, output_dir: str, history_file: str = "") -> List[str]:
    """
    Download arXiv papers from a random date range that haven't been downloaded before.

    Args:
        num_papers: Number of papers to download
        output_dir: Directory to save papers to
        history_file: Path to the download history file

    Returns:
        List of paths to downloaded papers
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")
    
    # Set default history file if not provided
    if not history_file:
        history_file = os.path.join(output_dir, "download_history.json")
    
    # Load download history
    downloaded_ids = load_download_history(history_file)
    print(f"Found {len(downloaded_ids)} previously downloaded papers")
    
    # Generate a random date range
    start_date, end_date = generate_random_date_range()
    
    # Format dates for the arXiv query
    # ArXiv API uses format YYYYMMDDHHMMSS
    start_date_str = start_date.strftime("%Y%m%d%H%M%S")
    end_date_str = end_date.strftime("%Y%m%d%H%M%S")
    
    # Set up search query for submissions in the random date range
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=5
    )
    
    # Use date range in the query
    date_query = f"submittedDate:[{start_date_str} TO {end_date_str}]"
    search = arxiv.Search(
        query=f"({date_query}) AND (cat:cs.* OR cat:math.* OR cat:physics.*)",
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    downloaded_paths = []
    downloaded_count = 0
    new_downloaded_ids = set()
    
    print(f"Downloading up to {num_papers} papers from random date range to {output_dir}")
    print("Fetching results from arXiv API...")
    
    try:
        # Fetch papers with better pagination handling
        results = []
        for result in client.results(search):
            results.append(result)
            # If we have enough results after filtering out duplicates, stop fetching
            if len(results) >= num_papers * 3:
                break
                
        print(f"Found {len(results)} papers from arXiv API")
        
        # Shuffle the results to get random papers from the date range
        random.shuffle(results)
        
        for i, result in enumerate(results):
            if downloaded_count >= num_papers:
                break
                
            paper_id = result.get_short_id()
            
            # Skip if already downloaded
            if paper_id in downloaded_ids:
                print(f"Skipping already downloaded paper: {result.title} (ID: {paper_id})")
                continue
                
            print(f"Processing paper {i+1}/{len(results)}: {result.title} (ID: {paper_id})")
            
            # Format paper title for filename
            paper_title = result.title.replace(" ", "_").replace("/", "-").replace(":", "")[:100]
            paper_title = "".join(c for c in paper_title if c.isalnum() or c in "_-")
            filename = f"{paper_id}_{paper_title}.pdf"
            output_path = os.path.join(output_dir, filename)
            
            try:
                print(f"Attempting to download: {result.title}")
                result.download_pdf(dirpath=output_dir, filename=filename)
                print(f"Download successful: {output_path}")
                downloaded_paths.append(output_path)
                downloaded_count += 1
                downloaded_ids.add(paper_id)
                new_downloaded_ids.add(paper_id)
                print(f"Downloaded ({downloaded_count}/{num_papers}): {result.title}")
                # Add delay to prevent rate limiting
                time.sleep(1)
            except Exception as e:
                print(f"Error downloading {result.title}: {str(e)}")
    except Exception as e:
        print(f"Error fetching papers from arXiv: {str(e)}")
        # Try with a more conservative approach if the first attempt fails
        if "unexpectedly empty" in str(e) or "DateParseError" in str(e):
            print("Retrying with more conservative parameters...")
            try:
                # Use a more conservative search without date restrictions
                conservative_search = arxiv.Search(
                    query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
                    max_results=50,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                results = list(client.results(conservative_search))
                # Shuffle results for randomness
                random.shuffle(results)
                print(f"Found {len(results)} papers from conservative arXiv API search")
                
                for i, result in enumerate(results):
                    if downloaded_count >= num_papers:
                        break
                        
                    paper_id = result.get_short_id()
                    
                    # Skip if already downloaded
                    if paper_id in downloaded_ids:
                        continue
                        
                    print(f"Processing paper {i+1}/{len(results)}: {result.title} (ID: {paper_id})")
                    
                    # Format paper title for filename
                    paper_title = result.title.replace(" ", "_").replace("/", "-").replace(":", "")[:100]
                    paper_title = "".join(c for c in paper_title if c.isalnum() or c in "_-")
                    filename = f"{paper_id}_{paper_title}.pdf"
                    output_path = os.path.join(output_dir, filename)
                    
                    try:
                        result.download_pdf(dirpath=output_dir, filename=filename)
                        downloaded_paths.append(output_path)
                        downloaded_count += 1
                        downloaded_ids.add(paper_id)
                        new_downloaded_ids.add(paper_id)
                        print(f"Downloaded ({downloaded_count}/{num_papers}): {result.title}")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error downloading {result.title}: {str(e)}")
            except Exception as fallback_e:
                print(f"Fallback approach also failed: {str(fallback_e)}")
    
    # Save updated download history
    if new_downloaded_ids:
        save_download_history(history_file, downloaded_ids)
        print(f"Updated download history with {len(new_downloaded_ids)} new papers")
    
    print(f"Successfully downloaded {len(downloaded_paths)} papers to {output_dir}")
    return downloaded_paths


def main(num_papers: Optional[int] = None, output_dir: Optional[str] = None, history_file: Optional[str] = None) -> None:
    """
    Main function to parse arguments and download papers.

    Args:
        num_papers: Optional number of papers to download (default: 20)
        output_dir: Optional directory to save papers to
        history_file: Optional path to the download history file
    """
    print(f"Running main function with default params: num_papers={num_papers}, output_dir={output_dir}")
    parser = argparse.ArgumentParser(description="Download arXiv papers from random date ranges")
    parser.add_argument("--num-papers", type=int, default=20 if num_papers is None else num_papers,
                       help="Number of papers to download (default: 20)")
    parser.add_argument("--output-dir", type=str, 
                       default="/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/" if output_dir is None else output_dir,
                       help="Directory to save papers to")
    parser.add_argument("--history-file", type=str, default=history_file,
                       help="Path to download history file (default: download_history.json in output directory)")
    
    args = parser.parse_args()
    print(f"Arguments parsed: num_papers={args.num_papers}, output_dir={args.output_dir}, history_file={args.history_file}")
    download_recent_papers(args.num_papers, args.output_dir, args.history_file)


if __name__ == "__main__":
    print("Script executed directly")
    dir_path = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/"
    # dir_path = "./papers/"
    main(10, dir_path)
    print("Script completed")
