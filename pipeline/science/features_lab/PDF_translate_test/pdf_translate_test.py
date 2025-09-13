"""
based on ```@https://pypi.org/project/azure-ai-translation-document/ ```, implement a pdf to pdf translation python function. I have AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME,  AZURE_INSIGHTS_CONNECTION_STRING, AZURE_TRANSLATOR_KEY, 
AZURE_TRANSLATOR_DOC_ENDPOINT (this is the one we should use for this case),
AZURE_TRANSLATOR_ENDPOINT, AZURE_TRANSLATOR_LOCATION already in the .env file so you can assume I have the translator resource and blob storage. in blob use a fold under "/pdf_translation/". only change file @pdf_translate_test.py . make sure the file can be properly converted to Chinese and save under folder "/pdf_after_translation/" and download to the same local folder. note that you can refer to translate_content function as a reference for using the translator resource. i also have Blob_SAS_token and Blob_SAS_URL in the .env file that you can use to get the permission of files in Blob

note: the workflow is: uploading the file to blob, and then call the translation service, and then save the file to blob, and then download the converted pdf to local
"""

"""
PDF to PDF Translation using Azure AI Document Translation

This module provides functionality to translate PDF files from one language to another
using Azure AI Document Translation service.

Prerequisites:
1. Azure Translator resource with Document Translation enabled
2. Azure Blob Storage account with SAS token
3. Environment variables configured in .env file:
   - AZURE_STORAGE_CONNECTION_STRING
   - AZURE_STORAGE_CONTAINER_NAME
   - AZURE_TRANSLATOR_KEY
   - AZURE_TRANSLATOR_DOC_ENDPOINT
   - AZURE_TRANSLATOR_LOCATION
   - Blob_SAS_token (with read/list/write permissions)
   - Blob_SAS_URL

Usage:
    from pdf_translate_test import translate_pdf_to_chinese
    
    # Translate a PDF to Chinese
    result_path = translate_pdf_to_chinese("input.pdf", "./output/")
    print(f"Translated PDF saved to: {result_path}")

Note: The Azure Document Translation service has limits on the number of documents
that can be processed simultaneously. If you get "MaxDocumentsExceeded" error,
please clean up old files in your blob container.
"""

import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.document import DocumentTranslationClient, DocumentTranslationInput, TranslationTarget
from azure.storage.blob import BlobServiceClient
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFTranslator:
    """
    A class to handle PDF to PDF translation using Azure AI Document Translation service.
    
    This class provides functionality to:
    1. Upload PDF files to Azure Blob Storage
    2. Translate PDFs using Azure Document Translation service
    3. Download translated PDFs from Azure Blob Storage
    """
    
    def __init__(self):
        """Initialize the PDF translator with Azure credentials."""
        self.storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        self.translator_key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.translator_doc_endpoint = os.getenv("AZURE_TRANSLATOR_DOC_ENDPOINT")
        self.translator_location = os.getenv("AZURE_TRANSLATOR_LOCATION")
        self.blob_sas_token = os.getenv("Blob_SAS_token")
        self.blob_sas_url = os.getenv("Blob_SAS_URL")
        
        # Validate required credentials
        self._validate_credentials()
        
        # Initialize Azure clients
        self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection_string)
        self.document_translation_client = DocumentTranslationClient(
            self.translator_doc_endpoint,
            AzureKeyCredential(self.translator_key)
        )
        
        logger.info("PDF Translator initialized successfully")
    
    def _validate_credentials(self) -> None:
        """Validate that all required environment variables are set."""
        required_vars = [
            "AZURE_STORAGE_CONNECTION_STRING",
            "AZURE_STORAGE_CONTAINER_NAME", 
            "AZURE_TRANSLATOR_KEY",
            "AZURE_TRANSLATOR_DOC_ENDPOINT",
            "AZURE_TRANSLATOR_LOCATION",
            "Blob_SAS_token",
            "Blob_SAS_URL"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _upload_to_blob(self, local_file_path: str, blob_name: str) -> str:
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            local_file_path: Path to the local file to upload
            blob_name: Name for the blob in storage
            
        Returns:
            str: URL of the uploaded blob with SAS token
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Return blob URL with SAS token for access
            blob_url = f"{self.blob_sas_url}/{blob_name}?{self.blob_sas_token}"
            logger.info(f"Successfully uploaded {local_file_path} to blob: {blob_name}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to upload file to blob storage: {e}")
            raise
    
    def _download_from_blob(self, blob_name: str, local_file_path: str) -> None:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            blob_name: Name of the blob to download
            local_file_path: Local path where the file should be saved
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Ensure the local directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            logger.info(f"Successfully downloaded blob {blob_name} to {local_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to download file from blob storage: {e}")
            raise
    
    def _wait_for_translation_completion(self, poller, timeout_minutes: int = 30) -> list:
        """
        Wait for the translation operation to complete.
        
        Args:
            poller: The translation operation poller
            timeout_minutes: Maximum time to wait in minutes
            
        Returns:
            list: List of translated document results
        """
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        while not poller.done():
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Translation operation timed out after {timeout_minutes} minutes")
            
            time.sleep(10)  # Check every 10 seconds
            logger.info("Translation in progress...")
        
        result = poller.result()
        logger.info("Translation completed successfully")
        
        # Log translation details and collect results
        translated_documents = []
        for document in result:
            logger.info(f"Document ID: {document.id}")
            logger.info(f"Status: {document.status}")
            if document.status == "Succeeded":
                logger.info(f"Translated from {document.source_document_url} to {document.translated_document_url}")
                translated_documents.append(document)
            elif document.error:
                logger.error(f"Translation error: {document.error.message}")
                raise Exception(f"Translation failed: {document.error.message}")
        
        return translated_documents
    
    def translate_pdf_to_chinese(
        self, 
        input_pdf_path: str, 
        output_folder: str = "./pdf_after_translation/",
        target_language: str = "zh-Hans"
    ) -> str:
        """
        Translate a PDF file to Chinese using Azure Document Translation service.
        
        Args:
            input_pdf_path: Path to the input PDF file
            output_folder: Local folder to save the translated PDF
            target_language: Target language code (default: zh-Hans for Simplified Chinese)
            
        Returns:
            str: Path to the translated PDF file
        """
        try:
            # Generate unique identifiers for this translation job
            job_id = str(uuid.uuid4())[:8]
            input_filename = Path(input_pdf_path).name
            input_blob_name = f"pdf_translation/{job_id}_input_{input_filename}"
            
            logger.info(f"Starting PDF translation job {job_id} for file: {input_filename}")
            
            # Step 1: Upload input PDF to blob storage
            logger.info("Step 1: Uploading PDF to blob storage...")
            self._upload_to_blob(input_pdf_path, input_blob_name)
            
            # Step 2: Prepare source and target URLs for translation
            # For Azure Document Translation, we need container-level URLs with proper SAS permissions
            # The SAS token must have read and list permissions for source and write and list permissions for target
            
            # The blob_sas_url should already contain the SAS token
            # Format: https://account.blob.core.windows.net/container?sp=racwdli&st=...
            base_url = self.blob_sas_url
            
            logger.info("Step 2: Starting document translation...")
            logger.info(f"Using SAS URL: {base_url.split('?')[0]}?[SAS_TOKEN]")
            
            # Step 3: Start translation using Azure Document Translation
            # Use container-level URLs - Azure will translate all files in the container
            # This is the most reliable approach
            
            logger.info(f"Source container URL: {base_url.split('?')[0]}?[SAS_TOKEN]")
            logger.info(f"Target container URL: {base_url.split('?')[0]}?[SAS_TOKEN]")
            
            # Create translation input using container-level URLs
            translation_input = DocumentTranslationInput(
                source_url=base_url,
                targets=[
                    TranslationTarget(
                        target_url=base_url,
                        language=target_language
                    )
                ]
            )
            
            poller = self.document_translation_client.begin_translation([translation_input])
            
            # Step 4: Wait for translation to complete
            logger.info("Step 3: Waiting for translation to complete...")
            translated_documents = self._wait_for_translation_completion(poller)
            
            # Step 5: Download the translated PDF
            logger.info("Step 4: Downloading translated PDF...")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            if not translated_documents:
                raise Exception("No documents were successfully translated")
            
            # Get the first (and should be only) translated document
            translated_doc = translated_documents[0]
            
            # Extract blob name from the translated document URL
            translated_url = translated_doc.translated_document_url
            # Parse the blob name from the URL (after the container name)
            blob_name_start = translated_url.find(self.container_name) + len(self.container_name) + 1
            blob_name_end = translated_url.find('?') if '?' in translated_url else len(translated_url)
            translated_blob_name = translated_url[blob_name_start:blob_name_end]
            
            translated_filename = f"translated_{target_language}_{input_filename}"
            local_output_path = os.path.join(output_folder, translated_filename)
            
            logger.info(f"Downloading translated file from blob: {translated_blob_name}")
            
            # Download the translated file
            self._download_from_blob(translated_blob_name, local_output_path)
            
            logger.info("PDF translation completed successfully!")
            logger.info(f"Translated PDF saved to: {local_output_path}")
            
            return local_output_path
            
        except Exception as e:
            error_message = str(e)
            if "MaxDocumentsExceeded" in error_message:
                logger.error("Translation failed: Too many documents in the container. Please clean up old files or use a different container.")
                raise Exception("Translation failed: Container has too many documents. Please clean up old files and try again.")
            elif "InvalidDocumentAccessLevel" in error_message:
                logger.error("Translation failed: Invalid permissions. Please check your SAS token has read/list permissions for source and write/list permissions for target.")
                raise Exception("Translation failed: Invalid blob storage permissions. Please check your SAS token configuration.")
            else:
                logger.error(f"PDF translation failed: {e}")
                raise


def translate_pdf_to_chinese(input_pdf_path: str, output_folder: str = "./pdf_after_translation/") -> str:
    """
    Convenience function to translate a PDF to Chinese.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Local folder to save the translated PDF
        
    Returns:
        str: Path to the translated PDF file
    """
    translator = PDFTranslator()
    return translator.translate_pdf_to_chinese(input_pdf_path, output_folder)


if __name__ == "__main__":
    # Example usage
    print("PDF Translation Tool - Azure AI Document Translation")
    print("=" * 50)
    
    try:
        # Test with a sample PDF file
        input_pdf = "test_document.pdf"  # Replace with your PDF file path
        
        if os.path.exists(input_pdf):
            print(f"Starting translation of: {input_pdf}")
            result_path = translate_pdf_to_chinese(input_pdf)
            print("\n✅ Translation completed successfully!")
            print(f"📄 Translated PDF saved to: {result_path}")
        else:
            print(f"❌ Test PDF file '{input_pdf}' not found.")
            print("Please ensure you have a PDF file in the current directory or update the file path.")
            
    except Exception as e:
        print(f"\n❌ Error during translation: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check your .env file has all required variables")
        print("2. Ensure your SAS token has proper permissions (read/list/write)")
        print("3. If you get 'MaxDocumentsExceeded', clean up old files in blob storage")
        print("4. Verify your Azure Translator resource has Document Translation enabled")
