import os
import shutil
from pipeline.science.pipeline.helper.azure_blob import AzureBlobHelper
from pipeline.science.pipeline.utils import file_check_list

import logging
logger = logging.getLogger("tutorpipeline.science.helper.index_files_saving")


def literag_index_files_decompress(embedding_folder):
    """
    Function to decompress the LiteRAG index files and download them from Azure Blob Storage
    :param embedding_folder: The path to the embedding folder
    :return: True if the index files are ready now in embedding_folder, False otherwise
    """
    lite_embedding_folder = os.path.join(embedding_folder, "lite_embedding")
    faiss_path = os.path.join(lite_embedding_folder, "index.faiss")
    pkl_path = os.path.join(lite_embedding_folder, "index.pkl")
    path_list = [faiss_path, pkl_path]
    all_files_exist = True
    for path in path_list:
        if not os.path.exists(path):
            all_files_exist = False
            logger.info(f"Missing directory: {path}")
    return all_files_exist


def graphrag_index_files_check(embedding_folder):
    """
    Function to check if all necessary files exist to load the embeddings
    :param embedding_folder: The path to the embedding folder
    :return: True if all necessary files exist, False otherwise
    """
    # Define the index files path for GraphRAG embedding
    GraphRAG_embedding_folder, path_list = file_check_list(embedding_folder)

    # Define the index files path for VectorRAG embedding
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")

    path_list.extend([
        faiss_path,
        pkl_path,
        document_summary_path
    ])

    # Check if all necessary files exist to load the embeddings and logger.info the directories that are missing
    all_files_exist = True
    for path in path_list:
        if not os.path.exists(path):
            all_files_exist = False
            logger.info(f"Missing directory: {path}")
    
    # If there is "I'm sorry" in documents_summary.txt, return False
    if os.path.exists(document_summary_path):
        with open(document_summary_path, "r") as file:
            if "I'm sorry" in file.read():
                all_files_exist = False
                logger.info(f"GraphRAG index files status: {all_files_exist}")
                return False
    else:
        all_files_exist = False
        logger.info(f"GraphRAG index files status: {all_files_exist}")
        return False
    
    logger.info(f"GraphRAG index files status: {all_files_exist}")
    return all_files_exist


def graphrag_index_files_compress(embedding_folder):
    """
    Function to compress the index files and upload them to Azure Blob Storage
    :param embedding_folder: The path to the embedding folder
    :return: True if the index files are ready now in embedding_folder and uploaded to Azure blob, False otherwise
    """
    # Decompress the zip file to the folder named with file_id under the parent folder
    if embedding_folder.endswith("/"):
        folder = embedding_folder[:-1]
    else:
        folder = embedding_folder
    file_id = os.path.basename(folder)
    parent_folder = folder.replace(file_id, "").rstrip(os.sep)
    compressed_file = os.path.join(parent_folder, file_id)

    if graphrag_index_files_check(embedding_folder):
        logger.info("Index files are already ready to be compressed!")

        shutil.make_archive(compressed_file, 'zip', folder)
        logger.info(f"Compressed the embedding folder to {compressed_file}")

        # Upload the compressed zip file to Azure Blob Storage
        azure_blob_helper = AzureBlobHelper()
        azure_blob_helper.upload(compressed_file + ".zip", f"graphrag_index/{file_id}.zip", "knowhiztutorrag")
        logger.info(f"Uploaded the compressed {file_id}.zip file to Azure Blob Storage")
        return True
    else:
        # CLEANUP: If the files are not ready, clear the compressed zip file and the corresponding folder
        if os.path.exists(compressed_file + ".zip"):
            os.remove(compressed_file + ".zip")
        if os.path.exists(folder):
            shutil.rmtree(folder)
        logger.info("Index files are not ready to be compressed!")
        return False


def graphrag_index_files_decompress(embedding_folder):
    """
    Function to decompress the index files and download them from Azure Blob Storage
    :param embedding_folder: The path to the embedding folder
    :return: True if the index files are ready now in embedding_folder, False otherwise
    """
    # Decompress the zip file to the folder named with file_id under the parent folder
    if embedding_folder.endswith("/"):
        folder = embedding_folder[:-1]
    else:
        folder = embedding_folder
    file_id = os.path.basename(folder)
    parent_folder = folder.replace(file_id, "").rstrip(os.sep)
    compressed_file = os.path.join(parent_folder, file_id)
    compressed_file_blob = f"graphrag_index/{file_id}.zip"

    # Try No.1: Check if the index files are already ready
    if graphrag_index_files_check(embedding_folder):
        logger.info("Index files are already ready!")
        return True
    else:
        # CLEANUP: Clear the existing folder if the index files are not ready yet
        if os.path.exists(compressed_file + ".zip"):
            os.remove(compressed_file + ".zip")
        if os.path.exists(folder):
            shutil.rmtree(folder)
        logger.info("Index files are not ready yet!")

    # Try No.2: Download the compressed zip file from Azure Blob Storage and decompress it
    try:
        # Download the compressed zip file from Azure Blob Storage to the parent folder
        azure_blob_helper = AzureBlobHelper()
        azure_blob_helper.download(compressed_file_blob, compressed_file + ".zip", "knowhiztutorrag")

        # Decompress the zip file and overwrite the existing folder
        shutil.unpack_archive(compressed_file + ".zip", folder)

        logger.info(f"Decompressed the zip file to {folder}")

        if graphrag_index_files_check(embedding_folder):
            logger.info("Index files are already ready after being decompressed!")
            return True
        else:
            logger.info("Index files are not ready after being decompressed, zip file in Azure blob may be unhealthy!")

            # CLEANUP: Clear the downloaded zip file and the corresponding folder
            if os.path.exists(compressed_file + ".zip"):
                os.remove(compressed_file + ".zip")
            if os.path.exists(folder):
                shutil.rmtree(folder)

            return False
    except Exception as e:
        # CLEANUP: Clear the downloaded zip file and the corresponding folder if an error occurs
        if os.path.exists(compressed_file + ".zip"):
            os.remove(compressed_file + ".zip")
        if os.path.exists(folder):
            shutil.rmtree(folder)

        logger.info(f"Error downloading the zip file: {e}")
        return False


def vectorrag_index_files_check(embedding_folder):
    """
    Function to check if all necessary files exist to load the VectorRAG embeddings
    :param embedding_folder: The path to the embedding folder
    :return: True if all necessary files exist, False otherwise
    """
    markdown_embedding_folder = os.path.join(embedding_folder, "markdown")
    
    # Define the index files path for VectorRAG embedding
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
    markdown_faiss_path = os.path.join(markdown_embedding_folder, "index.faiss")
    markdown_pkl_path = os.path.join(markdown_embedding_folder, "index.pkl")

    path_list = [
        faiss_path,
        pkl_path,
        document_summary_path,
        markdown_faiss_path,
        markdown_pkl_path
    ]

    # Check if all necessary files exist to load the embeddings and logger.info the directories that are missing
    all_files_exist = True
    for path in path_list:
        if not os.path.exists(path):
            all_files_exist = False
            logger.info(f"Missing directory: {path}")

    # If there is "I'm sorry" in documents_summary.txt, return False
    if os.path.exists(document_summary_path):
        with open(document_summary_path, "r") as file:
            if "I'm sorry" in file.read():
                all_files_exist = False
                logger.info(f"VectorRAG index files status: {all_files_exist}")
                return False
    else:
        all_files_exist = False
        logger.info(f"VectorRAG index files status: {all_files_exist}")
        return False
    
    logger.info(f"VectorRAG index files status: {all_files_exist}")

    return all_files_exist


def vectorrag_index_files_compress(embedding_folder):
    """
    Function to compress the VectorRAG index files and upload them to Azure Blob Storage
    :param embedding_folder: The path to the embedding folder
    :return: True if the index files are ready now in embedding_folder and uploaded to Azure blob, False otherwise
    """
    # Prepare paths
    if embedding_folder.endswith("/"):
        folder = embedding_folder[:-1]
    else:
        folder = embedding_folder
    file_id = os.path.basename(folder)
    parent_folder = folder.replace(file_id, "").rstrip(os.sep)
    compressed_file = os.path.join(parent_folder, f"vectorrag_{file_id}")

    if vectorrag_index_files_check(embedding_folder):
        logger.info("VectorRAG index files are ready to be compressed!")

        shutil.make_archive(compressed_file, 'zip', folder)
        logger.info(f"Compressed the VectorRAG embedding folder to {compressed_file}")

        # Upload the compressed zip file to Azure Blob Storage
        azure_blob_helper = AzureBlobHelper()
        azure_blob_helper.upload(compressed_file + ".zip", f"vectorrag_index/{file_id}.zip", "knowhiztutorrag")
        logger.info(f"Uploaded the compressed vectorrag_{file_id}.zip file to Azure Blob Storage")
        return True
    else:
        # CLEANUP: If the files are not ready, clear the compressed zip file
        if os.path.exists(compressed_file + ".zip"):
            os.remove(compressed_file + ".zip")
        logger.info("VectorRAG index files are not ready to be compressed!")
        return False


def vectorrag_index_files_decompress(embedding_folder):
    """
    Function to decompress the VectorRAG index files and download them from Azure Blob Storage
    :param embedding_folder: The path to the embedding folder
    :return: True if the index files are ready now in embedding_folder, False otherwise
    """
    markdown_embedding_folder = os.path.join(embedding_folder, "markdown")

    # Prepare paths
    if embedding_folder.endswith("/"):
        folder = embedding_folder[:-1]
    else:
        folder = embedding_folder
    file_id = os.path.basename(folder)
    parent_folder = folder.replace(file_id, "").rstrip(os.sep)
    compressed_file = os.path.join(parent_folder, f"vectorrag_{file_id}")
    compressed_file_blob = f"vectorrag_index/{file_id}.zip"

    # Try No.1: Check if the index files are already ready
    if vectorrag_index_files_check(embedding_folder):
        logger.info("VectorRAG index files are locally ready!")
        return True
    else:
        # CLEANUP: Clear the existing files if they're not complete
        faiss_path = os.path.join(embedding_folder, "index.faiss")
        pkl_path = os.path.join(embedding_folder, "index.pkl")
        document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
        markdown_faiss_path = os.path.join(markdown_embedding_folder, "index.faiss")
        markdown_pkl_path = os.path.join(markdown_embedding_folder, "index.pkl")
        
        for path in [faiss_path, pkl_path, document_summary_path, markdown_faiss_path, markdown_pkl_path]:
            if os.path.exists(path):
                os.remove(path)
        logger.info("VectorRAG index files are not locally ready yet!")

    # Try No.2: Download the compressed zip file from Azure Blob Storage and decompress it
    try:
        # Download the compressed zip file from Azure Blob Storage to the parent folder
        azure_blob_helper = AzureBlobHelper()
        azure_blob_helper.download(compressed_file_blob, compressed_file + ".zip", "knowhiztutorrag")

        # Decompress the zip file and overwrite any existing files
        shutil.unpack_archive(compressed_file + ".zip", folder)
        logger.info(f"Decompressed the zip file to {folder}")

        # Clean up the downloaded zip file
        if os.path.exists(compressed_file + ".zip"):
            os.remove(compressed_file + ".zip")

        if vectorrag_index_files_check(embedding_folder):
            logger.info("VectorRAG index files are ready after being decompressed!")
            return True
        else:
            logger.info("VectorRAG index files are not ready after being decompressed, zip file in Azure blob may be unhealthy!")

            # CLEANUP: Clear the downloaded zip file and the corresponding folder
            if os.path.exists(compressed_file + ".zip"):
                os.remove(compressed_file + ".zip")
            if os.path.exists(folder):
                shutil.rmtree(folder)
            return False
          
    except Exception as e:
        # CLEANUP: Clear the downloaded zip file if an error occurs
        if os.path.exists(compressed_file + ".zip"):
            os.remove(compressed_file + ".zip")
        logger.info(f"Error downloading the VectorRAG zip file: {e}")
        return False


if __name__ == "__main__":
    embedding_folder = "../../embedded_content/be5a180265450fcb5959618dc94d7186"

    # Upload the index files to Azure Blob Storage
    graphrag_index_files_compress(embedding_folder)

    # # Download the index files from Azure Blob Storage
    # graphrag_index_files_decompress(embedding_folder)