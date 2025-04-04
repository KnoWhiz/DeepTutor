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

        # Display all items of the docstore
        logger.info(f"\nDocstore Information:")
        if len(docstore._dict) > 100:
            logger.info(f"Warning: Docstore contains {len(docstore._dict)} items. Displaying all items may produce a lot of output.")
        
        for i, (key, value) in enumerate(docstore._dict.items()):
            logger.info(f"Item {i}: {key} -> {value}")

        # # Display the first and last items of FAISS index
        # logger.info(f"\nFAISS Index Information:")
        # logger.info(f"First item: {index.get_items(0)}")
        # logger.info(f"Last item: {index.get_items(index.ntotal - 1)}")

    except Exception as e:
        logger.info(f"Error occurred while processing files: {str(e)}")

def display_db_file(file_path: str) -> None:
    """
    Display the content of a database file (either FAISS index or pickle file)
    
    Args:
        file_path: Path to the database file (.faiss or .pkl)
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.info(f"Error: File not found at {file_path}")
        return
    
    # Determine file type based on extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.faiss':
            # Handle FAISS index file
            index = faiss.read_index(file_path)
            
            logger.info(f"\nFAISS Index Information:")
            logger.info(f"File path: {file_path}")
            logger.info(f"Total number of vectors: {index.ntotal}")
            logger.info(f"Vector dimension: {index.d}")
            logger.info(f"Is index trained: {index.is_trained}")
            
        elif file_ext == '.pkl':
            # Handle pickle file
            with open(file_path, "rb") as f:
                pkl_data = pickle.load(f)
            
            logger.info(f"\nPickle File Information:")
            logger.info(f"File path: {file_path}")
            
            # Check if the pickle contains a docstore and id_to_uuid_map
            if isinstance(pkl_data, tuple) and len(pkl_data) == 2:
                docstore, id_to_uuid_map = pkl_data
                
                if isinstance(docstore, InMemoryDocstore) and isinstance(id_to_uuid_map, dict):
                    logger.info(f"Content type: FAISS-compatible docstore and UUID mapping")
                    logger.info(f"Number of document in docstore: {len(docstore._dict)}")
                    logger.info(f"Number of UUID mappings: {len(id_to_uuid_map)}")
                    
                    # Display sample of UUID mappings
                    logger.info(f"\nSample UUID mappings (first 5):")
                    for idx, uuid in list(id_to_uuid_map.items())[:5]:
                        logger.info(f"Index {idx} -> UUID: {uuid}")
                    
                    # Display all items of the docstore
                    logger.info(f"\nDocstore Information:")
                    if len(docstore._dict) > 100:
                        logger.info(f"Warning: Docstore contains {len(docstore._dict)} items. Displaying all items may produce a lot of output.")
                    
                    for i, (key, value) in enumerate(docstore._dict.items()):
                        logger.info(f"Item {i}: {key} -> {value}")
                else:
                    logger.info(f"Content type: Other")
                    logger.info(f"Content structure: {type(pkl_data)}")
                    logger.info(f"Content: {pkl_data}")
            else:
                logger.info(f"Content type: Other")
                logger.info(f"Content structure: {type(pkl_data)}")
                logger.info(f"Content: {pkl_data}")
                
        else:
            logger.info(f"Unsupported file type: {file_ext}. Expected .faiss or .pkl")
            
    except Exception as e:
        logger.info(f"Error occurred while processing file: {str(e)}")

if __name__ == "__main__":
    # Use raw strings for Windows paths to avoid escape character issues
    index_file_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/1eecd3da808d385834c966650074b676/markdown/index.faiss"
    pkl_file_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/embedded_content/1eecd3da808d385834c966650074b676/markdown/index.pkl"
    
    # Check both files together
    check_index_file(index_file_path, pkl_file_path)
    
    # Example of using the display_db_file function with a single file
    # Uncomment the line below to display only the FAISS index file
    # display_db_file(index_file_path)
    
    # Uncomment the line below to display only the pickle file
    # display_db_file(pkl_file_path)
