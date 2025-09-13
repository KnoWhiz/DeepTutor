"""
PDF to PDF Translation using DeepL API

This module provides functionality to translate PDF files from English to Chinese
using DeepL's Document Translation API.

Prerequisites:
1. DeepL API account with document translation enabled
2. Environment variable configured in .env file:
   - DeepL_API_Key

Usage:
    from deepl_pdf_translator import translate_pdf_to_chinese
    
    # Translate a PDF to Chinese
    result_path = translate_pdf_to_chinese("input.pdf", "./output/")
    print(f"Translated PDF saved to: {result_path}")

API Reference:
- Upload: https://developers.deepl.com/api-reference/document/upload-and-translate-a-document
- Status: https://developers.deepl.com/api-reference/document/check-document-status
- Download: https://developers.deepl.com/api-reference/document/download-translated-document
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLPDFTranslator:
    """
    A class to handle PDF to PDF translation using DeepL Document Translation API.
    
    This class provides functionality to:
    1. Upload PDF files to DeepL for translation
    2. Check translation status
    3. Download translated PDFs
    """
    
    def __init__(self):
        """Initialize the DeepL PDF translator with API credentials."""
        self.api_key = os.getenv("DeepL_API_Key")
        
        # Validate required credentials
        self._validate_credentials()
        
        # DeepL API endpoints - determine if using free or paid API based on key
        if self.api_key.endswith(":fx"):
            self.base_url = "https://api-free.deepl.com/v2"  # Free API
            logger.info("Using DeepL Free API")
        else:
            self.base_url = "https://api.deepl.com/v2"  # Paid API
            logger.info("Using DeepL Paid API")
        self.upload_url = f"{self.base_url}/document"
        
        # Common headers for API requests
        self.headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}"
        }
        
        logger.info("DeepL PDF Translator initialized successfully")
    
    def _validate_credentials(self) -> None:
        """Validate that all required environment variables are set."""
        if not self.api_key:
            raise ValueError("Missing required environment variable: DeepL_API_Key")
        
        if not self.api_key.strip():
            raise ValueError("DeepL_API_Key environment variable is empty")
    
    def _upload_document(self, file_path: str, target_language: str = "ZH") -> Tuple[str, str]:
        """
        Upload a document to DeepL for translation.
        
        Args:
            file_path: Path to the PDF file to translate
            target_language: Target language code (ZH for Chinese)
            
        Returns:
            Tuple[str, str]: document_id and document_key for tracking translation
        """
        try:
            logger.info(f"Uploading document: {file_path}")
            
            # Prepare the file for upload
            with open(file_path, "rb") as file:
                files = {
                    "file": (Path(file_path).name, file, "application/pdf")
                }
                
                data = {
                    "target_lang": target_language
                }
                
                # Make the upload request
                response = requests.post(
                    self.upload_url,
                    headers=self.headers,
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                document_id = result["document_id"]
                document_key = result["document_key"]
                
                logger.info(f"Document uploaded successfully. ID: {document_id}")
                return document_id, document_key
            else:
                error_message = f"Upload failed with status {response.status_code}: {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
                
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise
    
    def _check_translation_status(self, document_id: str, document_key: str) -> Dict[str, Any]:
        """
        Check the translation status of a document.
        
        Args:
            document_id: The document ID returned from upload
            document_key: The document key returned from upload
            
        Returns:
            Dict[str, Any]: Status information including current status and remaining time
        """
        try:
            status_url = f"{self.upload_url}/{document_id}"
            
            data = {
                "document_key": document_key
            }
            
            headers = {
                **self.headers,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            response = requests.post(status_url, headers=headers, data=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_message = f"Status check failed with status {response.status_code}: {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
                
        except Exception as e:
            logger.error(f"Failed to check translation status: {e}")
            raise
    
    def _wait_for_completion(
        self, 
        document_id: str, 
        document_key: str, 
        timeout_minutes: int = 30,
        check_interval_seconds: int = 5
    ) -> None:
        """
        Wait for the translation to complete.
        
        Args:
            document_id: The document ID
            document_key: The document key
            timeout_minutes: Maximum time to wait in minutes
            check_interval_seconds: How often to check status in seconds
        """
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        logger.info("Waiting for translation to complete...")
        
        while True:
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Translation timed out after {timeout_minutes} minutes")
            
            # Check the current status
            status_info = self._check_translation_status(document_id, document_key)
            current_status = status_info.get("status", "unknown")
            
            logger.info(f"Translation status: {current_status}")
            
            if current_status == "done":
                logger.info("Translation completed successfully!")
                return
            elif current_status == "error":
                error_message = status_info.get("message", "Unknown error occurred during translation")
                raise Exception(f"Translation failed: {error_message}")
            elif current_status in ["queued", "translating"]:
                # Still in progress
                seconds_remaining = status_info.get("seconds_remaining")
                if seconds_remaining:
                    logger.info(f"Estimated time remaining: {seconds_remaining} seconds")
                
                time.sleep(check_interval_seconds)
            else:
                # Unknown status, wait and try again
                logger.warning(f"Unknown status: {current_status}, continuing to wait...")
                time.sleep(check_interval_seconds)
    
    def _download_translated_document(
        self, 
        document_id: str, 
        document_key: str, 
        output_path: str
    ) -> None:
        """
        Download the translated document.
        
        Args:
            document_id: The document ID
            document_key: The document key
            output_path: Local path where the translated file should be saved
        """
        try:
            download_url = f"{self.upload_url}/{document_id}/result"
            
            data = {
                "document_key": document_key
            }
            
            headers = {
                **self.headers,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            logger.info(f"Downloading translated document to: {output_path}")
            
            response = requests.post(download_url, headers=headers, data=data)
            
            if response.status_code == 200:
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the translated document
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Translated document saved successfully to: {output_path}")
            else:
                error_message = f"Download failed with status {response.status_code}: {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
                
        except Exception as e:
            logger.error(f"Failed to download translated document: {e}")
            raise
    
    def translate_pdf_to_chinese(
        self,
        input_pdf_path: str,
        output_folder: str = "./translated_pdfs/",
        target_language: str = "ZH"
    ) -> str:
        """
        Translate a PDF file to Chinese using DeepL Document Translation API.
        
        Args:
            input_pdf_path: Path to the input PDF file
            output_folder: Local folder to save the translated PDF
            target_language: Target language code (ZH for Chinese)
            
        Returns:
            str: Path to the translated PDF file
        """
        try:
            # Validate input file exists
            if not os.path.exists(input_pdf_path):
                raise FileNotFoundError(f"Input PDF file not found: {input_pdf_path}")
            
            input_filename = Path(input_pdf_path).name
            output_filename = f"translated_chinese_{input_filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            logger.info(f"Starting PDF translation: {input_filename} -> Chinese")
            
            # Step 1: Upload document for translation
            logger.info("Step 1: Uploading document to DeepL...")
            document_id, document_key = self._upload_document(input_pdf_path, target_language)
            
            # Step 2: Wait for translation to complete
            logger.info("Step 2: Waiting for translation to complete...")
            self._wait_for_completion(document_id, document_key)
            
            # Step 3: Download the translated document
            logger.info("Step 3: Downloading translated document...")
            self._download_translated_document(document_id, document_key, output_path)
            
            logger.info("PDF translation completed successfully!")
            logger.info(f"Translated PDF saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"PDF translation failed: {e}")
            raise


def translate_pdf_to_chinese(
    input_pdf_path: str, 
    output_folder: str = "./translated_pdfs/"
) -> str:
    """
    Convenience function to translate a PDF to Chinese using DeepL.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Local folder to save the translated PDF
        
    Returns:
        str: Path to the translated PDF file
    """
    translator = DeepLPDFTranslator()
    return translator.translate_pdf_to_chinese(input_pdf_path, output_folder)


if __name__ == "__main__":
    # Example usage and test
    print("DeepL PDF Translation Tool")
    print("=" * 40)
    
    try:
        # Test with the provided test document
        test_pdf_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/PDF_translate_test/test_document.pdf"
        
        if os.path.exists(test_pdf_path):
            print(f"Starting translation of: {test_pdf_path}")
            result_path = translate_pdf_to_chinese(test_pdf_path)
            print("\n✅ Translation completed successfully!")
            print(f"📄 Translated PDF saved to: {result_path}")
        else:
            print(f"❌ Test PDF file not found at: {test_pdf_path}")
            print("Please ensure the test document exists or update the file path.")
            
    except Exception as e:
        print(f"\n❌ Error during translation: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check your .env file contains DeepL_API_Key")
        print("2. Ensure your DeepL API key is valid and has document translation enabled")
        print("3. Verify you have sufficient quota in your DeepL account")
        print("4. Check that the input PDF file exists and is readable")
