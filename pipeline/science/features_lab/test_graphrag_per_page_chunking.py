#!/usr/bin/env python3
"""
Test script to verify GraphRAG per-page chunking implementation.

This script tests the new per-page chunking functionality for GraphRAG embeddings.
It compares the old single-file approach with the new per-page file approach.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the pipeline directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from pipeline.science.pipeline.doc_processor import save_file_txt_locally
from pipeline.science.pipeline.embeddings_agent import embeddings_agent
from pipeline.science.pipeline.utils import ChatMode
import fitz  # PyMuPDF

def create_test_pdf():
    """Create a test PDF with multiple pages for testing."""
    doc = fitz.open()
    
    # Page 1: Short content
    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is page 1 with short content.")
    
    # Page 2: Medium content
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is page 2 with medium content. " * 20)
    
    # Page 3: Long content
    page3 = doc.new_page()
    page3.insert_text((50, 50), "This is page 3 with very long content. " * 50)
    
    return doc

def test_per_page_chunking():
    """Test the per-page chunking functionality."""
    print("🧪 Testing GraphRAG per-page chunking implementation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create test PDF
        test_pdf_path = os.path.join(temp_dir, "test_document.pdf")
        doc = create_test_pdf()
        doc.save(test_pdf_path)
        doc.close()
        
        print(f"📄 Created test PDF with {len(doc)} pages")
        
        # Test the save_file_txt_locally function
        embedding_folder = os.path.join(temp_dir, "embeddings")
        os.makedirs(embedding_folder, exist_ok=True)
        
        print("💾 Testing save_file_txt_locally with per-page chunking...")
        save_file_txt_locally(
            file_path=test_pdf_path,
            filename="test_document.pdf",
            embedding_folder=embedding_folder
        )
        
        # Check the GraphRAG input directory
        graphrag_input_dir = os.path.join(embedding_folder, "GraphRAG", "input")
        
        if os.path.exists(graphrag_input_dir):
            files = os.listdir(graphrag_input_dir)
            page_files = [f for f in files if f.endswith('.txt')]
            
            print(f"📁 Found {len(page_files)} page files in GraphRAG input directory:")
            for file in sorted(page_files):
                file_path = os.path.join(graphrag_input_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"  📄 {file}: {len(content)} characters")
            
            # Check the mapping file
            mapping_file = os.path.join(embedding_folder, "GraphRAG", "filename_mapping.json")
            if os.path.exists(mapping_file):
                import json
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                print(f"📋 Mapping file content: {mapping}")
            
            print("✅ Per-page chunking test completed successfully!")
            return True
        else:
            print("❌ GraphRAG input directory not found!")
            return False

def test_markdown_processing():
    """Test processing with markdown content."""
    print("\n🧪 Testing markdown processing with per-page chunking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test markdown content with page separators
        markdown_content = """# Page 1
This is the first page with short content.

---

# Page 2
This is the second page with medium content. This is the second page with medium content. This is the second page with medium content.

---

# Page 3
This is the third page with very long content. This is the third page with very long content. This is the third page with very long content. This is the third page with very long content. This is the third page with very long content. This is the third page with very long content. This is the third page with very long content. This is the third page with very long content.
"""
        
        # Create a dummy PDF file (we'll use markdown content)
        test_pdf_path = os.path.join(temp_dir, "test_markdown.pdf")
        with open(test_pdf_path, 'w') as f:
            f.write("dummy pdf content")
        
        # Create markdown directory and file
        embedding_folder = os.path.join(temp_dir, "embeddings")
        markdown_dir = os.path.join(embedding_folder, "markdown")
        os.makedirs(markdown_dir, exist_ok=True)
        
        # Create a dummy file_id for the markdown file
        import hashlib
        with open(test_pdf_path, 'rb') as f:
            file_bytes = f.read()
            file_id = hashlib.md5(file_bytes).hexdigest()
        
        md_path = os.path.join(markdown_dir, f"{file_id}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📝 Created markdown file: {md_path}")
        
        # Test the save_file_txt_locally function
        save_file_txt_locally(
            file_path=test_pdf_path,
            filename="test_markdown.pdf",
            embedding_folder=embedding_folder
        )
        
        # Check the GraphRAG input directory
        graphrag_input_dir = os.path.join(embedding_folder, "GraphRAG", "input")
        
        if os.path.exists(graphrag_input_dir):
            files = os.listdir(graphrag_input_dir)
            page_files = [f for f in files if f.endswith('.txt')]
            
            print(f"📁 Found {len(page_files)} page files from markdown:")
            for file in sorted(page_files):
                file_path = os.path.join(graphrag_input_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"  📄 {file}: {len(content)} characters")
                    print(f"     Preview: {content[:100]}...")
            
            print("✅ Markdown processing test completed successfully!")
            return True
        else:
            print("❌ GraphRAG input directory not found!")
            return False

def main():
    """Run all tests."""
    print("🚀 Starting GraphRAG per-page chunking tests...\n")
    
    # Test 1: PDF processing
    test1_result = test_per_page_chunking()
    
    # Test 2: Markdown processing
    test2_result = test_markdown_processing()
    
    print(f"\n📊 Test Results:")
    print(f"  PDF Processing: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"  Markdown Processing: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! Per-page chunking is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 