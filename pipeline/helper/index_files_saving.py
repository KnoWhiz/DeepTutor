import os
import shutil
from azure_blob import AzureBlobHelper


def index_files_check(embedding_folder):
    # Define the index files path for GraphRAG embedding
    GraphRAG_embedding_folder = os.path.join(embedding_folder, "GraphRAG/")
    create_final_community_reports_path = GraphRAG_embedding_folder + "output/create_final_community_reports.parquet"
    create_final_covariates_path = GraphRAG_embedding_folder + "output/create_final_covariates.parquet"
    create_final_documents_path = GraphRAG_embedding_folder + "output/create_final_documents.parquet"
    create_final_entities_path = GraphRAG_embedding_folder + "output/create_final_entities.parquet"
    create_final_nodes_path = GraphRAG_embedding_folder + "output/create_final_nodes.parquet"
    create_final_relationships_path = GraphRAG_embedding_folder + "output/create_final_relationships.parquet"
    create_final_text_units_path = GraphRAG_embedding_folder + "output/create_final_text_units.parquet"
    create_final_communities_path = GraphRAG_embedding_folder + "output/create_final_communities.parquet"
    lancedb_path = GraphRAG_embedding_folder + "output/lancedb/"

    # Define the index files path for VectorRAG embedding
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")

    path_list = [
        create_final_community_reports_path,
        create_final_covariates_path,
        create_final_documents_path,
        create_final_entities_path,
        create_final_nodes_path,
        create_final_relationships_path,
        create_final_text_units_path,
        create_final_communities_path,
        lancedb_path,
        faiss_path,
        pkl_path,
        documents_summary_path
    ]

    # Check if all necessary files exist to load the embeddings and print the directories that are missing
    all_files_exist = True
    for path in path_list:
        if not os.path.exists(path):
            all_files_exist = False
            print(f"Missing directory: {path}")
    return all_files_exist
    

def index_files_compress(embedding_folder):
    # Compress the entire embedding folder to a zip file under the folder one level up (the parent folder outside folder named with course_id). The zip file should be named as course_id.zip
    # course_id is the latest part of the embedding_folder path
    # parent_folder is the parent folder of the embedding_folder
    # remove the ""/"" at the end of the parent_folder if it exists
    if embedding_folder.endswith("/"):
        folder = embedding_folder[:-1]
    course_id = os.path.basename(folder)
    parent_folder = folder.replace(course_id, "").rstrip(os.sep)
    compressed_file = os.path.join(parent_folder, course_id)
    shutil.make_archive(compressed_file, 'zip', folder)
    print(f"Compressed the embedding folder to {compressed_file}")

    # Upload the compressed zip file to Azure Blob Storage
    azure_blob_helper = AzureBlobHelper()
    azure_blob_helper.upload(compressed_file + ".zip", f"graphrag_index/{course_id}.zip", "knowhiztutorrag")


def index_files_decompress(embedding_folder):
    # Decompress the zip file to the folder named with course_id under the parent folder
    if embedding_folder.endswith("/"):
        folder = embedding_folder[:-1]
    course_id = os.path.basename(folder)
    parent_folder = folder.replace(course_id, "").rstrip(os.sep)
    compressed_file = os.path.join(parent_folder, course_id)
    compressed_file_blob = f"graphrag_index/{course_id}.zip"

    # Download the compressed zip file from Azure Blob Storage to the parent folder
    azure_blob_helper = AzureBlobHelper()
    azure_blob_helper.download(compressed_file_blob, compressed_file + ".zip", "knowhiztutorrag")

    # Decompress the zip file and overwrite the existing folder
    shutil.unpack_archive(compressed_file + ".zip", folder)
    print(f"Decompressed the zip file to {folder}")


if __name__ == "__main__":
    embedding_folder = "../../embedded_content/be5a180265450fcb5959618dc94d7186/"
    
    # Upload the index files to Azure Blob Storage
    if index_files_check(embedding_folder):
        index_files_compress(embedding_folder)
        print("Index files have been compressed and uploaded to Azure Blob Storage!")
    else:
        print("Index files are not ready to be compressed!")

    # Download the index files from Azure Blob Storage
    try:
        index_files_decompress(embedding_folder)
        print("Index files have been downloaded and decompressed!")
    except Exception as e:
        print(f"Error decompressing index files: {e}")
        raise