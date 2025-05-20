#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to estimate the number of tokens per page in PDF papers.
This script downloads random arXiv papers, processes them using two different PDF loaders,
and calculates the average number of tokens per page.
"""

import os
import io
import arxiv
import random
import time
import tempfile
import tiktoken
import logging
import fitz
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import tiktoken for token counting
try:
    import tiktoken
except ImportError:
    logger.error("tiktoken not found. Please install it with 'pip install tiktoken'")
    sys.exit(1)

# Define count_tokens function locally to avoid import issues
def count_tokens(text, model_name='gpt-4o'):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text, disallowed_special=(encoding.special_tokens_set - {'<|endoftext|>'}))
        length = len(tokens)
        logger.info(f"Length of tokens: {length}")
        return length
    except Exception as e:
        logger.exception(f"Error counting tokens: {str(e)}")
        length = len(text.split())
        logger.info(f"Length of text: {length}")
        return length

def download_random_papers(num_papers: int = 10, temp_dir: Optional[str] = None) -> List[str]:
    """
    Download a specified number of random arXiv papers.
    
    Args:
        num_papers: Number of papers to download
        temp_dir: Directory to save papers to (uses temporary directory if None)
        
    Returns:
        List of paths to downloaded papers
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)
        
    logger.info(f"Downloading {num_papers} random papers to {temp_dir}")
    
    # Categories to sample from
    categories = [
        "cs.AI", "cs.CL", "cs.LG",   # Computer Science
        "math.ST", "math.PR",         # Mathematics
        "physics.comp-ph",            # Physics
        "q-bio.QM",                   # Quantitative Biology
        "q-fin.PM",                   # Quantitative Finance
        "stat.ML"                     # Statistics
    ]
    
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=5
    )
    
    downloaded_paths = []
    
    # Try each category until we have enough papers
    random.shuffle(categories)
    for category in categories:
        if len(downloaded_paths) >= num_papers:
            break
            
        logger.info(f"Searching for papers in category: {category}")
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=50,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        try:
            results = list(client.results(search))
            random.shuffle(results)
            
            for result in results:
                if len(downloaded_paths) >= num_papers:
                    break
                    
                paper_id = result.get_short_id()
                paper_title = result.title.replace(" ", "_").replace("/", "-").replace(":", "")[:50]
                paper_title = "".join(c for c in paper_title if c.isalnum() or c in "_-")
                filename = f"{paper_id}_{paper_title}.pdf"
                output_path = os.path.join(temp_dir, filename)
                
                try:
                    logger.info(f"Downloading: {result.title} (ID: {paper_id})")
                    result.download_pdf(dirpath=temp_dir, filename=filename)
                    downloaded_paths.append(output_path)
                    logger.info(f"Downloaded ({len(downloaded_paths)}/{num_papers}): {output_path}")
                    time.sleep(1)  # Avoid rate limiting
                except Exception as e:
                    logger.error(f"Error downloading {result.title}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error fetching papers from category {category}: {str(e)}")
            continue
    
    logger.info(f"Successfully downloaded {len(downloaded_paths)} papers")
    return downloaded_paths

def extract_document_with_pymupdf_direct(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract document using PyMuPDF directly without LangChain.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of document pages with content
    """
    doc = fitz.open(file_path)
    pages = []
    
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_content": text,
            "metadata": {
                "source": file_path,
                "page": i
            }
        })
    
    return pages

def extract_document_with_fitz(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract document using fitz.open with BytesIO.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of page contents
    """
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
    
    doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    pages = []
    
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_content": text,
            "metadata": {
                "source": file_path,
                "page": i
            }
        })
    
    return pages

def count_tokens_per_page(document_pages: List[Dict[str, Any]], model_name: str = "gpt-4o") -> List[int]:
    """
    Count tokens for each page in the document.
    
    Args:
        document_pages: List of document pages with content
        model_name: Name of the model to use for token counting
        
    Returns:
        List of token counts per page
    """
    token_counts = []
    
    for page in document_pages:
        page_content = page["page_content"]
        num_tokens = count_tokens(page_content, model_name)
        token_counts.append(num_tokens)
        
    return token_counts

def analyze_papers_token_counts(paper_paths: List[str], model_name: str = "gpt-4o") -> Dict[str, Any]:
    """
    Analyze token counts per page for a list of papers using both loaders.
    
    Args:
        paper_paths: List of paths to PDF papers
        model_name: Name of the model to use for token counting
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "pymupdf": {"total_pages": 0, "total_tokens": 0, "per_paper": []},
        "fitz": {"total_pages": 0, "total_tokens": 0, "per_paper": []}
    }
    
    for i, paper_path in enumerate(paper_paths):
        logger.info(f"Processing paper {i+1}/{len(paper_paths)}: {os.path.basename(paper_path)}")
        
        # Process with PyMuPDF directly
        try:
            document_pymupdf = extract_document_with_pymupdf_direct(paper_path)
            page_count_pymupdf = len(document_pymupdf)
            token_counts_pymupdf = count_tokens_per_page(document_pymupdf, model_name)
            total_tokens_pymupdf = sum(token_counts_pymupdf)
            
            results["pymupdf"]["total_pages"] += page_count_pymupdf
            results["pymupdf"]["total_tokens"] += total_tokens_pymupdf
            results["pymupdf"]["per_paper"].append({
                "path": paper_path,
                "pages": page_count_pymupdf,
                "tokens": total_tokens_pymupdf,
                "tokens_per_page": [int(count) for count in token_counts_pymupdf],
                "avg_tokens_per_page": total_tokens_pymupdf / max(1, page_count_pymupdf)
            })
            
            logger.info(f"PyMuPDF direct: {page_count_pymupdf} pages, {total_tokens_pymupdf} tokens")
        except Exception as e:
            logger.error(f"Error processing {paper_path} with PyMuPDF direct: {str(e)}")
        
        # Process with fitz using BytesIO
        try:
            document_fitz = extract_document_with_fitz(paper_path)
            page_count_fitz = len(document_fitz)
            token_counts_fitz = count_tokens_per_page(document_fitz, model_name)
            total_tokens_fitz = sum(token_counts_fitz)
            
            results["fitz"]["total_pages"] += page_count_fitz
            results["fitz"]["total_tokens"] += total_tokens_fitz
            results["fitz"]["per_paper"].append({
                "path": paper_path,
                "pages": page_count_fitz,
                "tokens": total_tokens_fitz,
                "tokens_per_page": [int(count) for count in token_counts_fitz],
                "avg_tokens_per_page": total_tokens_fitz / max(1, page_count_fitz)
            })
            
            logger.info(f"fitz.open with BytesIO: {page_count_fitz} pages, {total_tokens_fitz} tokens")
        except Exception as e:
            logger.error(f"Error processing {paper_path} with fitz.open: {str(e)}")
    
    # Calculate averages
    for loader in ["pymupdf", "fitz"]:
        total_pages = results[loader]["total_pages"]
        total_tokens = results[loader]["total_tokens"]
        
        if total_pages > 0:
            results[loader]["avg_tokens_per_page"] = total_tokens / total_pages
        else:
            results[loader]["avg_tokens_per_page"] = 0
    
    return results

def main():
    """
    Main function to download papers and analyze token counts.
    """
    parser = argparse.ArgumentParser(description="Estimate tokens per page in PDF papers")
    parser.add_argument("--num-papers", type=int, default=10,
                       help="Number of papers to analyze (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save papers (default: temporary directory)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model to use for token counting (default: gpt-4o)")
    
    args = parser.parse_args()
    
    # Download random papers
    paper_paths = download_random_papers(args.num_papers, args.output_dir)
    
    if not paper_paths:
        logger.error("Failed to download any papers. Exiting.")
        return
    
    # Analyze token counts
    results = analyze_papers_token_counts(paper_paths, args.model)
    
    # Print summary
    print("\n--- Token Count Summary ---")
    for loader in ["pymupdf", "fitz"]:
        avg_tokens = results[loader]["avg_tokens_per_page"]
        total_pages = results[loader]["total_pages"]
        print(f"\n{loader.upper()} Loader:")
        print(f"Total pages analyzed: {total_pages}")
        print(f"Average tokens per page: {avg_tokens:.2f}")
        
        # Per-paper breakdown
        print("\nPer-paper breakdown:")
        for paper_result in results[loader]["per_paper"]:
            paper_name = os.path.basename(paper_result["path"])
            avg_tokens_paper = paper_result["avg_tokens_per_page"]
            print(f"  {paper_name}: {paper_result['pages']} pages, {avg_tokens_paper:.2f} tokens/page")

if __name__ == "__main__":
    main()
