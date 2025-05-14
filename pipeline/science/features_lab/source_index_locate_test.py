import io
import fitz  # PyMuPDF
import random
import re
from difflib import SequenceMatcher
import os

def normalize_text(text, remove_linebreaks=True):
    """
    Normalize text by removing excessive whitespace and standardizing common special characters.
    
    Args:
        text: Text to normalize
        remove_linebreaks: If True, replace line breaks with empty string, otherwise keep current behavior
        
    Returns:
        Normalized text
    """
    if remove_linebreaks:
        # Replace line breaks with empty string
        text = re.sub(r'[\n\r]+', '', text)
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize special characters that might appear differently in PDFs
    text = text.replace('−', '-')  # Replace Unicode minus with hyphen
    text = text.replace('∼', '~')  # Replace tilde approximation
    
    # Handle commonly misrecognized math symbols
    text = re.sub(r'\|\s*↓\s*⟩', '|↓⟩', text)
    text = re.sub(r'\|\s*↑\s*⟩', '|↑⟩', text)
    
    # Clean up spaces around symbols
    text = re.sub(r'\s*\[\s*(\d+)\s*\]', r'[\1]', text)  # [39] -> [39]
    
    return text.strip()

# Test function for normalize_text
def test_normalize_text():
    """
    Test the normalize_text function with different input types and settings.
    """
    print("\nTesting normalize_text function:")
    
    # Test case 1: Text with line breaks
    test_case_1 = "This is a test\nwith line\r\nbreaks."
    result_1a = normalize_text(test_case_1, remove_linebreaks=True)
    result_1b = normalize_text(test_case_1, remove_linebreaks=False)
    print(f"Original text: '{test_case_1}'")
    print(f"With remove_linebreaks=True: '{result_1a}'")
    print(f"With remove_linebreaks=False: '{result_1b}'")
    assert result_1a == "This is a testwith linebreaks."  # No spaces when line breaks are removed
    assert result_1b == "This is a test with line breaks."  # Line breaks become spaces with remove_linebreaks=False
    
    # Test case 2: Text with multiple spaces and special characters
    test_case_2 = "Text with    multiple  spaces and − special ∼ characters"
    result_2 = normalize_text(test_case_2)
    print(f"Original text: '{test_case_2}'")
    print(f"Normalized: '{result_2}'")
    assert result_2 == "Text with multiple spaces and - special ~ characters"
    
    # Test case 3: Text with math symbols
    test_case_3 = "Quantum state | ↓ ⟩ and | ↑ ⟩"
    result_3 = normalize_text(test_case_3)
    print(f"Original text: '{test_case_3}'")
    print(f"Normalized: '{result_3}'")
    assert result_3 == "Quantum state |↓⟩ and |↑⟩"
    
    # Test case 4: Text with citations
    test_case_4 = "Reference [ 39 ] and [ 42 ]"
    result_4 = normalize_text(test_case_4)
    print(f"Original text: '{test_case_4}'")
    print(f"Normalized: '{result_4}'")
    assert result_4 == "Reference[39] and[42]"  # There's no space between Reference and [39]
    
    # Test case 5: Line breaks with other normalization
    test_case_5 = "Line 1\nLine 2 with [ 39 ] and | ↓ ⟩"
    result_5a = normalize_text(test_case_5, remove_linebreaks=True)
    result_5b = normalize_text(test_case_5, remove_linebreaks=False)
    print(f"Original text: '{test_case_5}'")
    print(f"With remove_linebreaks=True: '{result_5a}'")
    print(f"With remove_linebreaks=False: '{result_5b}'")
    assert result_5a == "Line 1Line 2 with[39] and |↓⟩"
    assert result_5b == "Line 1 Line 2 with[39] and |↓⟩"
    
    print("All normalize_text tests passed!\n")


def locate_chunk_in_pdf(chunk: str, pdf_path: str, similarity_threshold: float = 0.8, remove_linebreaks: bool = True) -> dict:
    """
    Locates a text chunk within a PDF file and returns its position information.
    Uses both exact matching and fuzzy matching for robustness.
    
    Args:
        chunk: A string of text to locate within the PDF
        pdf_path: Path to the PDF file
        similarity_threshold: Threshold for fuzzy matching (0.0-1.0)
        remove_linebreaks: If True, remove line breaks during text normalization
    
    Returns:
        Dictionary containing:
            - page_num: The page number where the chunk was found (0-indexed)
            - start_char: The starting character position in the page
            - end_char: The ending character position in the page
            - success: Boolean indicating if the chunk was found
            - similarity: Similarity score if found by fuzzy matching
    """
    result = {
        "page_num": -1,
        "start_char": -1,
        "end_char": -1,
        "success": False,
        "similarity": 0.0
    }
    
    try:
        # Normalize the search chunk
        normalized_chunk = normalize_text(chunk, remove_linebreaks)
        chunk_words = normalized_chunk.split()
        min_match_length = min(100, len(normalized_chunk))  # For long chunks, we'll use word-based search
        
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # First try exact matching
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            normalized_text = normalize_text(text, remove_linebreaks)
            
            # Try exact match first
            start_pos = normalized_text.find(normalized_chunk)
            if start_pos != -1:
                end_pos = start_pos + len(normalized_chunk)
                result["page_num"] = page_num
                result["start_char"] = start_pos
                result["end_char"] = end_pos
                result["success"] = True
                result["similarity"] = 1.0
                break
            
            # If chunk is long, try matching the first N words
            if len(chunk_words) > 10:
                first_words = " ".join(chunk_words[:10])
                start_pos = normalized_text.find(first_words)
                
                if start_pos != -1:
                    # Found beginning of the chunk, now check similarity
                    potential_match = normalized_text[start_pos:start_pos + len(normalized_chunk)]
                    
                    # Check if lengths are comparable
                    if abs(len(potential_match) - len(normalized_chunk)) < 0.2 * len(normalized_chunk):
                        # Calculate similarity
                        similarity = SequenceMatcher(None, normalized_chunk, potential_match).ratio()
                        
                        if similarity > similarity_threshold:
                            result["page_num"] = page_num
                            result["start_char"] = start_pos
                            result["end_char"] = start_pos + len(potential_match)
                            result["success"] = True
                            result["similarity"] = similarity
                            break
        
        # If not found, try sliding window approach on pages
        if not result["success"]:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                normalized_text = normalize_text(text, remove_linebreaks)
                
                # For very long chunks, we'll use a sliding window approach
                # to find the most similar section
                if len(normalized_chunk) > min_match_length:
                    best_similarity = 0
                    best_start = -1
                    
                    # Try to match beginning of chunk with sliding windows
                    first_words = " ".join(chunk_words[:10])
                    for match in re.finditer(re.escape(chunk_words[0]), normalized_text):
                        start_pos = match.start()
                        
                        # Skip if there's not enough text left
                        if start_pos + len(normalized_chunk) > len(normalized_text):
                            continue
                        
                        # Extract a section of text of similar length
                        window_size = min(len(normalized_chunk) + 100, len(normalized_text) - start_pos)
                        window_text = normalized_text[start_pos:start_pos + window_size]
                        
                        # Compare beginning of window with beginning of chunk
                        window_start = window_text[:len(first_words)]
                        similarity = SequenceMatcher(None, first_words, window_start).ratio()
                        
                        if similarity > 0.8:  # If beginning matches well, check the whole chunk
                            chunk_similarity = SequenceMatcher(None, normalized_chunk, 
                                                             window_text[:len(normalized_chunk)]).ratio()
                            if chunk_similarity > best_similarity:
                                best_similarity = chunk_similarity
                                best_start = start_pos
                    
                    if best_similarity > similarity_threshold:
                        result["page_num"] = page_num
                        result["start_char"] = best_start
                        result["end_char"] = best_start + len(normalized_chunk)
                        result["success"] = True
                        result["similarity"] = best_similarity
                        break
        
        doc.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
    
    return result

# Test function with 5 randomly selected chunks
def test_locate_chunks():
    # Check for file existence first
    pdf_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Skipping PDF-based tests")
        
        # Run a simple mock test of the locate function instead
        print("\nRunning mock test of locate_chunk_in_pdf:")
        
        # Create a MockPDF class to simulate the PyMuPDF behavior
        class MockDocument:
            def __init__(self):
                self.pages = [MockPage("Page 1 content with some test text."), 
                              MockPage("Another page with different content.")]
            
            def __len__(self):
                return len(self.pages)
            
            def __getitem__(self, idx):
                return self.pages[idx]
            
            def close(self):
                pass
                
        class MockPage:
            def __init__(self, text):
                self.text = text
                
            def get_text(self):
                return self.text
        
        # Monkey patch fitz.open to return our mock
        original_open = fitz.open
        fitz.open = lambda path: MockDocument()
        
        try:
            # Test with a chunk that should be found
            mock_chunk = "test text"
            result = locate_chunk_in_pdf(mock_chunk, "mock_path.pdf")
            print(f"Mock test result: {result}")
            assert result['success'] == True
            assert result['page_num'] == 0  # Should find it on the first page
            
            # Test with a chunk that shouldn't be found
            mock_chunk2 = "not present chunk"
            result2 = locate_chunk_in_pdf(mock_chunk2, "mock_path.pdf")
            print(f"Mock test result (not found): {result2}")
            assert result2['success'] == False
            
            # Test with line break handling
            mock_chunk3 = "test\ntext"
            result3a = locate_chunk_in_pdf(mock_chunk3, "mock_path.pdf", remove_linebreaks=True)
            result3b = locate_chunk_in_pdf(mock_chunk3, "mock_path.pdf", remove_linebreaks=False)
            print(f"Mock test with line breaks removed: {result3a}")
            print(f"Mock test with line breaks preserved: {result3b}")
            
            print("All mock tests passed!")
        finally:
            # Restore the original function
            fitz.open = original_open
            
        return
    
    # If file exists, proceed with the original test
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
    
    # Test each chunk
    print(f"\nTesting with {len(test_chunks)} random chunks\n")
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
    
    # Test without line break removal
    print("\nTesting with random chunks and line break preservation (remove_linebreaks=False):")
    if len(test_chunks) > 0:
        chunk = test_chunks[0]
        result_default = locate_chunk_in_pdf(chunk, pdf_path)
        result_preserve = locate_chunk_in_pdf(chunk, pdf_path, remove_linebreaks=False)
        
        print(f"Chunk text: '{chunk[:30]}...'")
        print(f"Default (remove_linebreaks=True): Page {result_default['page_num']}, Success: {result_default['success']}")
        print(f"With line breaks preserved: Page {result_preserve['page_num']}, Success: {result_preserve['success']}")
    
    # Add specific chunks for testing
    print("\n\nTesting with specific chunks:")
    specific_chunks = [
        "addressing beam. The results indicate the major source of residual correlation is addressing crosstalk [39]. This can be mitigated by coupling the single photons into a single-mode fiber or improving the optical quality of the excitation beam. After characterizing the single-photon nature of the transport-multiplexing scheme, we characterize the motion of the ions introduced by shuttling. This is important as the quality of subsequent quantum operations on the ions or ion-photon entanglement will depend on the ions' motional states. We further explore the system performance by measuring the motional heating from the ion transport. To do this, we first perform sideband cooling for all axial modes sequentially using the method similar to that in [41] and prepare the ion in the state |↓⟩= |42S1/2, mJ = −1/2⟩. We compare the |↓⟩↔|↑⟩= |32D5/2, mJ = −1/2⟩carrier transition before and after transport with a global 729 nm beam along the axial direction to determine how the transport affects the ion-motion (Fig. 4). The carrier Rabi flopping is motional state sensitive, and the Hamiltonian has the form of [42, 43]",
        "In this work, we demonstrate a temporal multiplexing scheme based on the transport of an ion-chain for improving the rate of ion-photon entanglement over long distances. In our experiments, we generate on-demand single photons by shuttling a nine-ion chain across the focus of a single-ion addressing beam. This scheme is expected to lead to a nearly nine-fold increase in attempt rate of the entanglement generation for quantum repeater nodes separated by >100 km. We verify the single-photon nature of the photon trains by measuring a second-order time correlation of g(2)(0) = 0.060(13) without background subtraction. Furthermore, we address the problem of motional excitation during the transport, which is detrimental to local entangling operations [34] and in the case of using a cavity for stimulating the photons would lead to uncertainty in the coupling strength. [35]. Using a shuttling function designed to mitigate motional excitation, we find coherent excitation as high as ¯nα ∼110 on the center-of-mass (COM) mode during one round of ion chain transport. These results show that the proposed multiplexing scheme can be scaled up to higher rates provided that more optimal transport methods are",
        "High-rate remote entanglement between photon and matter-based qubits is essential for distributed quantum information processing. A key technique to increase the modest entangling rates of existing long-distance quantum networking approaches is multiplexing. Here, we demonstrate a temporally multiplexed ion-photon interface via rapid transport of a chain of nine calcium ions across 74 µm within 86 µs. The non-classical nature of the multiplexed photons is verified by measuring the second-order correlation function with an average value of g(2)(0) = 0.060(13). This indicates low crosstalk between the multiplexed modes, and can be reduced to negligible level once fiber coupling of single photons is incorporated. In addition, we characterize the motional degree-of-freedom of the ion crystal after transport and find that it is coherently excited to as much as ¯nα ≈110 for the center-of-mass mode. Our proof-of-principle implementation paves the way for large-scale quantum networking with trapped ions, but highlights some challenges that must be overcome.",
        "FIG. 4. |↓⟩↔|↑⟩carrier excitation of nine-ion chain before and after shuttling. The horizontal axis is the global 729 nm beam probe time, and the vertical axis is the average ion excitation on the |↑⟩state. Error bars denote one standard deviation of the quantum projection noise. (a) Rabi oscillations of the sideband-cooled ions (red dots). The red line is a numerical simulation of thermal distribution with ¯nth = 4.0 ± 3.0. (b) Rabi oscillation after the transport at half speed of the transport function in Fig. 2(a). The blue line is a numerical simulation with with ¯nth = 4, ¯nα = 50 ± 5. (c) Rabi oscillation after the transport at full speed. The green line is a numerical simulation with with ¯nth = 4, ¯nα = 110 ± 5."
    ]
    
    for i, chunk in enumerate(specific_chunks):
        print(f"\nSpecific Test Chunk {i+1}:")
        print(f"Chunk text: '{chunk[:50]}...'")
        print(f"Chunk length: {len(chunk)} characters")
        
        # Test the locate function with default settings
        result = locate_chunk_in_pdf(chunk, pdf_path)
        
        if result["success"]:
            print(f"Found location: Page {result['page_num']}, Chars {result['start_char']}-{result['end_char']}")
            print(f"Match similarity: {result['similarity']:.2f}")
            print("✅ Chunk found in document")
        else:
            print("❌ Chunk NOT found in document")
            
        # Test with line break preservation
        result_preserve = locate_chunk_in_pdf(chunk, pdf_path, remove_linebreaks=False)
        print(f"With line breaks preserved: Success: {result_preserve['success']}, Similarity: {result_preserve['similarity']:.2f}")
    
    doc.close()

if __name__ == "__main__":
    # Test normalize_text function
    test_normalize_text()
    
    # Test chunk location function
    test_locate_chunks()
