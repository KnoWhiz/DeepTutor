#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download recent arXiv papers.
This script allows users to download a specified number of the most recent arXiv papers
to a specified directory.
"""

print("Starting arXiv loader script...")

import os
import arxiv
import datetime
import argparse
import time
from typing import List, Optional

print("Imports completed successfully")


def download_recent_papers(num_papers: int, output_dir: str) -> List[str]:
    """
    Download recent arXiv papers.

    Args:
        num_papers: Number of papers to download
        output_dir: Directory to save papers to

    Returns:
        List of paths to downloaded papers
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")
    
    # Set up search query for recent submissions
    # Use a broader search query with multiple categories
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3
    )
    
    # Use a more general query to get recent papers
    search = arxiv.Search(
        query="cat:cs.* OR cat:math.* OR cat:physics.*",  # Search across multiple categories
        max_results=100,  # Get more results than needed
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    downloaded_paths = []
    downloaded_count = 0
    
    print(f"Downloading up to {num_papers} recent arXiv papers to {output_dir}")
    print("Fetching results from arXiv API...")
    
    try:
        # Fetch and download papers
        results = list(client.results(search))
        print(f"Found {len(results)} papers from arXiv API")
        
        for i, result in enumerate(results):
            if downloaded_count >= num_papers:
                break
                
            print(f"Processing paper {i+1}/{len(results)}: {result.title} (ID: {result.get_short_id()})")
            
            # Format paper title for filename
            paper_title = result.title.replace(" ", "_").replace("/", "-").replace(":", "")[:100]
            paper_title = "".join(c for c in paper_title if c.isalnum() or c in "_-")
            filename = f"{result.get_short_id()}_{paper_title}.pdf"
            output_path = os.path.join(output_dir, filename)
            
            try:
                print(f"Attempting to download: {result.title}")
                result.download_pdf(dirpath=output_dir, filename=filename)
                print(f"Download successful: {output_path}")
                downloaded_paths.append(output_path)
                downloaded_count += 1
                print(f"Downloaded ({downloaded_count}/{num_papers}): {result.title}")
                # Add delay to prevent rate limiting
                time.sleep(1)
            except Exception as e:
                print(f"Error downloading {result.title}: {str(e)}")
    except Exception as e:
        print(f"Error fetching papers from arXiv: {str(e)}")
    
    print(f"Successfully downloaded {len(downloaded_paths)} papers to {output_dir}")
    return downloaded_paths


def main(num_papers: Optional[int] = None, output_dir: Optional[str] = None) -> None:
    """
    Main function to parse arguments and download papers.

    Args:
        num_papers: Optional number of papers to download (default: 20)
        output_dir: Optional directory to save papers to
    """
    print(f"Running main function with default params: num_papers={num_papers}, output_dir={output_dir}")
    parser = argparse.ArgumentParser(description="Download recent arXiv papers")
    parser.add_argument("--num-papers", type=int, default=20 if num_papers is None else num_papers,
                       help="Number of papers to download (default: 20)")
    parser.add_argument("--output-dir", type=str, 
                       default="/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/" if output_dir is None else output_dir,
                       help="Directory to save papers to")
    
    args = parser.parse_args()
    print(f"Arguments parsed: num_papers={args.num_papers}, output_dir={args.output_dir}")
    download_recent_papers(args.num_papers, args.output_dir)


if __name__ == "__main__":
    print("Script executed directly")
    main()
    print("Script completed")
