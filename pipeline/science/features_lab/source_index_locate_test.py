import io
import fitz  # PyMuPDF
import random
import re

def locate_chunk_in_pdf(chunk: str, pdf_path: str) -> dict:
    """
    Locates a text chunk within a PDF file and returns its position information.
    
    Args:
        chunk: A string of text to locate within the PDF
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary containing:
            - page_num: The page number where the chunk was found (0-indexed)
            - start_char: The starting character position in the page
            - end_char: The ending character position in the page
            - success: Boolean indicating if the chunk was found
    """
    result = {
        "page_num": -1,
        "start_char": -1,
        "end_char": -1,
        "success": False
    }
    
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Search for the chunk in each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Look for the chunk in the page text
            start_pos = text.find(chunk)
            if start_pos != -1:
                end_pos = start_pos + len(chunk)
                result["page_num"] = page_num
                result["start_char"] = start_pos
                result["end_char"] = end_pos
                result["success"] = True
                break
        
        doc.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
    
    return result

# Test function with 5 randomly selected chunks
def test_locate_chunks():
    pdf_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf"
    
    # Open the PDF to extract test chunks
    doc = fitz.open(pdf_path)
    
    # Get total number of pages
    total_pages = len(doc)
    print(f"PDF has {total_pages} pages")
    
    # Extract 5 different chunks from random positions and store original locations
    test_chunks = []
    original_locations = []
    num_chunks = 5
    
    # Try to select chunks from different parts of the document
    if total_pages >= 5:
        # If we have enough pages, try to distribute across the document
        page_indices = random.sample(range(total_pages), min(num_chunks, total_pages))
    else:
        # Otherwise, we might need to use some pages multiple times
        page_indices = [random.randint(0, total_pages-1) for _ in range(num_chunks)]
    
    for page_idx in page_indices:
        page = doc[page_idx]
        text = page.get_text()
        
        if len(text) < 100:  # Skip pages with very little text
            continue
            
        # Pick a random position in the page for the chunk start
        chunk_start = random.randint(0, max(0, len(text) - 100))
        
        # Random chunk size between 20 and 80 characters (but don't exceed text length)
        chunk_size = random.randint(20, min(80, len(text) - chunk_start))
        chunk_end = chunk_start + chunk_size
        
        chunk_text = text[chunk_start:chunk_end]
        
        # Skip chunks that are just whitespace or too short
        if len(chunk_text.strip()) < 10:
            continue
            
        test_chunks.append(chunk_text)
        original_locations.append({
            "page_num": page_idx,
            "start_char": chunk_start,
            "end_char": chunk_end
        })
        
        print(f"Selected chunk from page {page_idx}: pos {chunk_start}-{chunk_end}, length: {chunk_size}")
    
    # If we didn't get enough chunks, try again with different approach
    if len(test_chunks) < num_chunks:
        # Try to get chunks from sections with more text content
        for i in range(min(total_pages, 3)):  # Check first few pages
            page = doc[i]
            text = page.get_text()
            
            if len(text) < 200:
                continue
                
            # Look for paragraphs by finding sections with multiple sentences
            for j in range(3):  # Try a few different positions
                pos = random.randint(0, max(0, len(text) - 200))
                section = text[pos:pos+200]
                
                # Find a period followed by space then capital letter
                periods = [m.start() for m in re.finditer(r'\. [A-Z]', section)]
                if periods and len(periods) > 1:
                    # Extract a chunk between two periods
                    period_idx = random.choice(periods[:-1])
                    next_period_idx = periods[periods.index(period_idx) + 1]
                    
                    chunk_start = pos + period_idx + 2  # Skip period and space
                    chunk_end = pos + next_period_idx + 1  # Include the period
                    
                    # Make sure chunk isn't too long
                    if chunk_end - chunk_start > 80:
                        chunk_end = chunk_start + 80
                        
                    chunk_text = text[chunk_start:chunk_end]
                    test_chunks.append(chunk_text)
                    original_locations.append({
                        "page_num": i,
                        "start_char": chunk_start,
                        "end_char": chunk_end
                    })
                    
                    if len(test_chunks) >= num_chunks:
                        break
            
            if len(test_chunks) >= num_chunks:
                break
    
    # If we still don't have enough chunks, add some from the beginning of pages
    while len(test_chunks) < num_chunks and total_pages > 0:
        page_idx = random.randint(0, total_pages-1)
        page = doc[page_idx]
        text = page.get_text()
        
        if len(text) < 50:
            continue
            
        chunk_start = 0
        chunk_end = min(50, len(text))
        chunk_text = text[chunk_start:chunk_end]
        
        if chunk_text.strip():  # Make sure it's not just whitespace
            test_chunks.append(chunk_text)
            original_locations.append({
                "page_num": page_idx,
                "start_char": chunk_start,
                "end_char": chunk_end
            })
    
    doc.close()
    
    print(f"\nTesting with {len(test_chunks)} chunks\n")
    
    # Test each chunk
    for i, chunk in enumerate(test_chunks):
        print(f"\nTest Chunk {i+1}:")
        print(f"Chunk text: '{chunk[:30]}...'")
        print(f"Chunk length: {len(chunk)} characters")
        
        # Original location
        original = original_locations[i]
        print(f"Original location: Page {original['page_num']}, Chars {original['start_char']}-{original['end_char']}")
        
        # Test the locate function
        result = locate_chunk_in_pdf(chunk, pdf_path)
        print(f"Found location: Page {result['page_num']}, Chars {result['start_char']}-{result['end_char']}")
        
        # Validate results
        is_page_correct = result["page_num"] == original["page_num"]
        is_start_correct = result["start_char"] == original["start_char"]
        is_end_correct = result["end_char"] == original["end_char"]
        
        if is_page_correct and is_start_correct and is_end_correct:
            print("✅ Location match: EXACT MATCH")
        else:
            print("⚠️ Location match: MISMATCH")
            if not is_page_correct:
                print(f"  - Page numbers don't match: Expected {original['page_num']}, Got {result['page_num']}")
            if not is_start_correct:
                print(f"  - Start positions don't match: Expected {original['start_char']}, Got {result['start_char']}")
            if not is_end_correct:
                print(f"  - End positions don't match: Expected {original['end_char']}, Got {result['end_char']}")

if __name__ == "__main__":
    test_locate_chunks()
