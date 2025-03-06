from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
load_dotenv()

import logging
logger = logging.getLogger("tutorpipeline.science.helper.azure_blob")


class AzureBlobHelper(object):
    def __init__(self):
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def download(self, blob_name: str, output_name: str, container_name: str):
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(output_name, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logger.info(f"Downloaded {blob_name} to {output_name}")
            return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        except Exception as e:
            logger.info(f"Error downloading blob: {e}")
            raise

    def upload(self, file_path: str, blob_name: str, container_name: str):
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Uploaded {file_path} as {blob_name} to container {container_name}")
            logger.info(f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}")
        except Exception as e:
            logger.info(f"Error uploading blob: {e}")
            raise


if __name__ == "__main__":
    azure_blob_helper = AzureBlobHelper()
    azure_blob_helper.upload("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/input_files/(LEC#5 Optional) Can Language Models Encode Perceptual Structure Without Grounding A Case Study in Color.pdf", "graphrag_index/666f4fd662e98b4ac4ccbf54/test.pdf", "knowhiztutorrag")
    # azure_blob_helper.download("graphrag_index/666f4fd662e98b4ac4ccbf54/test.pdf", "test.pdf", "knowhiztutorrag")