# With given index file address (FAISS format and pkl file), check if the index file is valid and display the information.

import os
import faiss
import pickle
from typing import Tuple, Dict, Any
from langchain_community.docstore.in_memory import InMemoryDocstore

def check_index_file(index_file_path: str, pkl_file_path: str) -> None:
    """
    Check and display information about FAISS index file and its associated pickle file.

    Args:
        index_file_path: Path to the FAISS index file
        pkl_file_path: Path to the pickle file containing docstore and id_to_uuid mapping
    """
    # Check if the files exist
    if not os.path.exists(index_file_path):
        print(f"Error: Index file not found at {index_file_path}")
        return

    if not os.path.exists(pkl_file_path):
        print(f"Error: Pickle file not found at {pkl_file_path}")
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
        print(f"\nFile Information:")
        print(f"Index file: {index_file_path}")
        print(f"Pkl file: {pkl_file_path}")

        print(f"\nType Information:")
        print(f"Index file type: {type(index)}")
        print(f"Docstore type: {type(docstore)}")
        print(f"ID to UUID map type: {type(id_to_uuid_map)}")

        # Display the information of the pkl file components
        print(f"\nPickle File Contents:")
        print(f"Number of document in docstore: {len(docstore._dict)}")
        print(f"Number of UUID mappings: {len(id_to_uuid_map)}")

        # Display sample of UUID mappings
        print(f"\nSample UUID mappings (first 5):")
        for idx, uuid in list(id_to_uuid_map.items())[:5]:
            print(f"Index {idx} -> UUID: {uuid}")

        # Display FAISS index information
        print(f"\nFAISS Index Information:")
        print(f"Total number of vectors: {index.ntotal}")
        print(f"Vector dimension: {index.d}")
        print(f"Is index trained: {index.is_trained}")

        # Display the first and last items of the docstore
        print(f"\nDocstore Information:")
        print(f"First item: {list(docstore._dict.items())[0]}")
        print(f"Last item: {list(docstore._dict.items())[-1]}")

        # # Display the first and last items of FAISS index
        # print(f"\nFAISS Index Information:")
        # print(f"First item: {index.get_items(0)}")
        # print(f"Last item: {index.get_items(index.ntotal - 1)}")

    except Exception as e:
        print(f"Error occurred while processing files: {str(e)}")

if __name__ == "__main__":
    # Use raw strings for Windows paths to avoid escape character issues
    index_file_path = r"/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/c8773c4a9a62ca3bafd2010d3d0093f5/index.faiss"
    pkl_file_path = r"/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/c8773c4a9a62ca3bafd2010d3d0093f5/index.pkl"
    check_index_file(index_file_path, pkl_file_path)
