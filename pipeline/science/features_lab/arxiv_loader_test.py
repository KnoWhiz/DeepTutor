#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download recent arXiv papers.
This script allows users to download a specified number of the most recent arXiv papers
to a specified directory.
"""

import os
import arxiv
import datetime
import argparse
from typing import List, Optional


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
    
    # Get today's date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Set up search query for recent submissions
    # Note: arXiv API does not allow direct filtering by submission date
    # So we'll fetch recent papers and then filter
    client = arxiv.Client()
    search = arxiv.Search(
        query="",  # Empty query to get all papers
        max_results=num_papers * 5,  # Get more results to account for filtering
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    downloaded_paths = []
    downloaded_count = 0
    
    print(f"Downloading up to {num_papers} recent arXiv papers to {output_dir}")
    
    # Fetch and download papers
    for result in client.results(search):
        if downloaded_count >= num_papers:
            break
            
        # Format paper title for filename
        paper_title = result.title.replace(" ", "_").replace("/", "-").replace(":", "")[:100]
        paper_title = "".join(c for c in paper_title if c.isalnum() or c in "_-")
        filename = f"{result.get_short_id()}_{paper_title}.pdf"
        output_path = os.path.join(output_dir, filename)
        
        try:
            result.download_pdf(dirpath=output_dir, filename=filename)
            downloaded_paths.append(output_path)
            downloaded_count += 1
            print(f"Downloaded ({downloaded_count}/{num_papers}): {result.title}")
        except Exception as e:
            print(f"Error downloading {result.title}: {str(e)}")
    
    print(f"Successfully downloaded {len(downloaded_paths)} papers to {output_dir}")
    return downloaded_paths


def main(num_papers: Optional[int] = None, output_dir: Optional[str] = None) -> None:
    """
    Main function to parse arguments and download papers.

    Args:
        num_papers: Optional number of papers to download (default: 20)
        output_dir: Optional directory to save papers to
    """
    parser = argparse.ArgumentParser(description="Download recent arXiv papers")
    parser.add_argument("--num-papers", type=int, default=20 if num_papers is None else num_papers,
                       help="Number of papers to download (default: 20)")
    parser.add_argument("--output-dir", type=str, 
                       default="/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers" if output_dir is None else output_dir,
                       help="Directory to save papers to")
    
    args = parser.parse_args()
    download_recent_papers(args.num_papers, args.output_dir)


if __name__ == "__main__":
    main()
