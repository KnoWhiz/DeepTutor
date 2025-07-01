#https://docs.anthropic.com/en/docs/build-with-claude/citations

import anthropic
import os
import base64
from dotenv import load_dotenv

# Load environment variables from .env file (override any existing ones)
load_dotenv(override=True)

client = anthropic.Anthropic()

# Path to the real PDF file
pdf_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf"

try:
    # Read and encode the PDF file
    with open(pdf_path, "rb") as pdf_file:
        pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")

    print(f"Successfully loaded PDF file: {pdf_path}")
    print(f"PDF file size: {len(pdf_data)} characters (base64 encoded)")

    print("\nSending request to Claude API...")
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        },
                        "citations": {"enabled": True},
                        "cache_control": {"type": "ephemeral"}  # Cache the document content
                    },
                    {
                        "type": "text",
                        "text": "What is the reason for causing the non-zero g2 measurement?"
                    }
                ]
            }
        ]
    )

    print("Response received!")
    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)
    
    # Check response structure and print ALL content blocks
    if hasattr(response, 'content') and response.content:
        if isinstance(response.content, list):
            for i, content_block in enumerate(response.content):
                if hasattr(content_block, 'text') and content_block.text:
                    print(f"--- Content Block {i} ---")
                    print(content_block.text)
                    print()  # Add spacing between blocks
        else:
            print("Content exists but structure is unexpected:")
            print(response.content)
    else:
        print("No content returned")
        print("Full response object:")
        print(response)

    # Print citations if available
    print("\n" + "="*80)
    print("CHECKING FOR CITATIONS:")
    print("="*80)
    
    # Check different possible citation locations
    if hasattr(response, 'citations') and response.citations:
        print("Found citations in response.citations:")
        for i, citation in enumerate(response.citations, 1):
            print(f"Citation {i}: {citation}")
    elif hasattr(response, 'content') and response.content:
        for i, content_block in enumerate(response.content):
            if hasattr(content_block, 'citations') and content_block.citations:
                print(f"Found citations in content block {i}:")
                for j, citation in enumerate(content_block.citations, 1):
                    print(f"Citation {j}: {citation}")
    else:
        print("No citations found")

except FileNotFoundError:
    print(f"Error: PDF file not found at {pdf_path}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()