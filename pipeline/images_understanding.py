import io
import fitz
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st


def extract_images_with_context(pdf_file):
    """
    Extract images from a PDF along with their surrounding text for context.
    """
    # Convert io.BytesIO object to bytes if needed
    if isinstance(pdf_file, io.BytesIO):
        pdf_file = pdf_file.getvalue()

    doc = fitz.open(stream=pdf_file, filetype="pdf")  # Open the PDF using bytes stream
    images_with_context = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get all images on the page
        images = page.get_images(full=True)
        # Extract the page text
        text = page.get_text("text")

        for img in images:
            xref = img[0]  # Reference to the image
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Append image with its context
            images_with_context.append({
                "page": page_num + 1,
                "image_bytes": image_bytes,
                "ext": image_ext,
                "context": text.strip()  # Use the entire page text as context
            })
    
    return images_with_context


def save_images_temp(images_with_context):
    """
    Save extracted images to temporary files and return their paths.
    """
    temp_files = []
    for img in images_with_context:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img['ext']}") as temp_file:
            temp_file.write(img["image_bytes"])
            temp_files.append({
                "path": temp_file.name,
                "page": img["page"],
                "context": img["context"]  # Retain the context for relevance matching
            })
    return temp_files


def get_relevant_images(query, images_with_context):
    """
    Match the query to images based on semantic similarity to their context.
    """
    # Load a text embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)

    relevant_images = []
    for img in images_with_context:
        # Embed the image context
        context_embedding = embedding_model.encode([img["context"]], convert_to_tensor=True)
        # Compute similarity
        similarity = cosine_similarity(query_embedding, context_embedding)
        if similarity[0][0] > 0.1:  # Adjust threshold for relevance
            relevant_images.append(img)
    
    return relevant_images


def display_relevant_images(query):
    if "images_with_context" not in st.session_state:
        return  # Do nothing if images are not extracted yet

    # Retrieve relevant images
    relevant_images = get_relevant_images(query, st.session_state.images_with_context)
    if not relevant_images:
        return  # Leave the space blank if no relevant images are found

    # Display only the images
    for img in relevant_images:
        # Convert image bytes to displayable format
        image_data = io.BytesIO(img['image_bytes'])
        st.image(image_data, caption=f"Page {img['page']}", use_column_width=True)