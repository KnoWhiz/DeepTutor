# With given index file address (FAISS format and pkl file), check if the index file is valid and display the information.

import os
import faiss
import pickle
from typing import Tuple, Dict, Any
from langchain_community.docstore.in_memory import InMemoryDocstore

import logging
logger = logging.getLogger("index_files_checker.py")
# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def check_index_file(index_file_path: str, pkl_file_path: str) -> None:
    """
    Check and display information about FAISS index file and its associated pickle file.

    Args:
        index_file_path: Path to the FAISS index file
        pkl_file_path: Path to the pickle file containing docstore and id_to_uuid mapping
    """
    # Check if the files exist
    if not os.path.exists(index_file_path):
        logger.info(f"Error: Index file not found at {index_file_path}")
        return

    if not os.path.exists(pkl_file_path):
        logger.info(f"Error: Pickle file not found at {pkl_file_path}")
        return

    try:
        # Load the index file
        index = faiss.read_index(index_file_path)

        # Load the pkl file
        with open(pkl_file_path, "rb") as f:
            pkl_data: Tuple[InMemoryDocstore, Dict[int, str]] = pickle.load(f)

        # Unpack the tuple
        docstore, id_to_uuid_map = pkl_data

        # Display basic information
        logger.info(f"\nFile Information:")
        logger.info(f"Index file: {index_file_path}")
        logger.info(f"Pkl file: {pkl_file_path}")

        logger.info(f"\nType Information:")
        logger.info(f"Index file type: {type(index)}")
        logger.info(f"Docstore type: {type(docstore)}")
        logger.info(f"ID to UUID map type: {type(id_to_uuid_map)}")

        # Display the information of the pkl file components
        logger.info(f"\nPickle File Contents:")
        logger.info(f"Number of document in docstore: {len(docstore._dict)}")
        logger.info(f"Number of UUID mappings: {len(id_to_uuid_map)}")

        # Display sample of UUID mappings
        logger.info(f"\nSample UUID mappings (first 5):")
        for idx, uuid in list(id_to_uuid_map.items())[:5]:
            logger.info(f"Index {idx} -> UUID: {uuid}")

        # Display FAISS index information
        logger.info(f"\nFAISS Index Information:")
        logger.info(f"Total number of vectors: {index.ntotal}")
        logger.info(f"Vector dimension: {index.d}")
        logger.info(f"Is index trained: {index.is_trained}")

        # Display the first and last items of the docstore
        logger.info(f"\nDocstore Information:")
        logger.info(f"First item: {list(docstore._dict.items())[0]}")
        logger.info(f"Last item: {list(docstore._dict.items())[-1]}")

        # # Display the first and last items of FAISS index
        # logger.info(f"\nFAISS Index Information:")
        # logger.info(f"First item: {index.get_items(0)}")
        # logger.info(f"Last item: {index.get_items(index.ntotal - 1)}")

    except Exception as e:
        logger.info(f"Error occurred while processing files: {str(e)}")

if __name__ == "__main__":
    # Use raw strings for Windows paths to avoid escape character issues
    index_file_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/1eecd3da808d385834c966650074b676/index.faiss"
    pkl_file_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/1eecd3da808d385834c966650074b676/index.pkl"
    check_index_file(index_file_path, pkl_file_path)
