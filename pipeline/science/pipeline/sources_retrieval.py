import os
import pprint
import json

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    get_llm,
    robust_search_for,
)
from pipeline.science.pipeline.embeddings import (
    load_embeddings,
)
from pipeline.science.pipeline.embeddings_agent import embeddings_agent


import logging
logger = logging.getLogger("tutorpipeline.science.sources_retrieval")

def get_response_source(mode, _doc, _document, file_path, user_input, answer, chat_history, embedding_folder):
    """
    Get the sources for the response
    Return a dictionary of sources with scores and metadata
    The scores are normalized to 0-1 range
    The metadata includes the page number, chunk index, and block bounding box coordinates
    The sources are refined by checking if they can be found in the document
    Only get first 20 sources
    Show them in the order they are found in the document
    Preserve image filenames but filter them based on context relevance using LLM
    """
    config = load_config()
    para = config['llm']

    # Load image context
    logger.info(f"TEST: loading from embedding_folder: {embedding_folder}")
    image_context_path = os.path.join(embedding_folder, "markdown/image_context.json")
    if os.path.exists(image_context_path):
        with open(image_context_path, 'r') as f:
            image_context = json.loads(f.read())
    else:
        print("image_context_path does not exist")
        image_context = {}
        with open(image_context_path, 'w') as f:
            json.dump(image_context, f)

    # Create reverse mapping from description to image name
    image_mapping = {}
    for image_name, descriptions in image_context.items():
        for desc in descriptions:
            image_mapping[desc] = image_name

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        # Load existing embeddings
        # print("Loading existing embeddings...")
        logger.info("Loading existing embeddings...")
        db = load_embeddings(embedding_folder, 'default')
    else:
        print("No existing embeddings found, creating new ones...")
        embeddings_agent(mode, _document, _doc, file_path, embedding_folder)
        db = load_embeddings(embedding_folder, 'default')

    # Get relevant chunks for both question and answer with scores
    question_chunks_with_scores = db.similarity_search_with_score(user_input, k=config['sources_retriever']['k'])
    answer_chunks_with_scores = db.similarity_search_with_score(answer, k=config['sources_retriever']['k'])

    # The total list of sources chunks
    sources_chunks = []
    for chunk in question_chunks_with_scores:
        sources_chunks.append(chunk[0])
    for chunk in answer_chunks_with_scores:
        sources_chunks.append(chunk[0])

    # Get source pages dictionary, which maps each source to the page number it is found in. the page number is in the metadata of the document chunks
    source_pages = {}
    for chunk in sources_chunks:
        try:
            source_pages[chunk.page_content] = chunk.metadata['page']
        except KeyError:
            print(f"Error getting source pages for {chunk.page_content}")
            print(f"Chunk metadata: {chunk.metadata}")
            source_pages[chunk.page_content] = 1

    # Extract page content and scores, normalize scores to 0-1 range
    max_score = max(max(score for _, score in question_chunks_with_scores), 
                   max(score for _, score in answer_chunks_with_scores))
    min_score = min(min(score for _, score in question_chunks_with_scores), 
                   min(score for _, score in answer_chunks_with_scores))
    score_range = max_score - min_score if max_score != min_score else 1

    # Get sources_with_scores dictionary, which maps each source to the score it has
    sources_with_scores = {}
    # Process question chunks
    for chunk, score in question_chunks_with_scores:
        normalized_score = 1 - (score - min_score) / score_range  # Invert because lower distance = higher relevance
        sources_with_scores[chunk.page_content] = max(normalized_score, sources_with_scores.get(chunk.page_content, 0))
    # Process answer chunks
    for chunk, score in answer_chunks_with_scores:
        normalized_score = 1 - (score - min_score) / score_range  # Invert because lower distance = higher relevance
        sources_with_scores[chunk.page_content] = max(normalized_score, sources_with_scores.get(chunk.page_content, 0))

    # Replace matching sources with image names while preserving scores
    sources_with_scores = {image_mapping.get(source, source): score 
                         for source, score in sources_with_scores.items()}
    source_pages = {image_mapping.get(source, source): page 
                         for source, page in source_pages.items()}

    # TEST
    logger.info("TEST: sources before refine:")
    logger.info(f"TEST: length of sources before refine: {len(sources_with_scores)}")

    # Refine and limit sources while preserving scores
    markdown_dir = os.path.join(embedding_folder, "markdown")
    sources_with_scores = refine_sources(_doc, _document, sources_with_scores, markdown_dir, user_input)

    # Refine source pages while preserving scores
    refined_source_pages = {}
    for source, page in source_pages.items():
        if source in sources_with_scores:
            refined_source_pages[source] = page + 1

    # TEST
    logger.info("TEST: sources after refine:")
    for source, score in sources_with_scores.items():
        logger.info(f"{source}: {score}")
    logger.info(f"TEST: length of sources after refine: {len(sources_with_scores)}")

    return sources_with_scores, source_pages, refined_source_pages


def refine_sources(_doc, _document, sources_with_scores, markdown_dir, user_input):
    """
    Refine sources by checking if they can be found in the document
    Only get first 20 sources
    Show them in the order they are found in the document
    Preserve image filenames but filter them based on context relevance using LLM
    """
    config = load_config()
    refined_sources = {}
    image_sources = {}

    # Load image context
    image_context_path = os.path.join(markdown_dir, "image_context.json")
    if os.path.exists(image_context_path):
        with open(image_context_path, 'r') as f:
            image_context = json.load(f)
    else:
        print("image_context_path does not exist")
        image_context = {}
        with open(image_context_path, 'w') as f:
            json.dump(image_context, f)

    # First separate image sources from text sources
    text_sources = {}
    for source, score in sources_with_scores.items():
        # Check if source looks like an image filename (has image extension)
        if any(source.lower().endswith(ext) for ext in config["image_extensions"]):
            image_sources[source] = score
        else:
            text_sources[source] = score

    # TEST
    logger.info("TEST: image sources before refine:")
    logger.info(f"TEST: length of image sources before refine: {len(image_sources)}")
    logger.info(f"TEST: text sources before refine: {text_sources}")
    logger.info(f"TEST: length of text sources before refine: {len(text_sources)}")

    # Filter image sources based on LLM evaluation
    filtered_images = {}
    if image_sources:
        # Initialize LLM for relevance evaluation
        config = load_config()
        para = config['llm']
        llm = get_llm('Basic', para)
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

        # Create prompt for image relevance evaluation
        system_prompt = """
        You are an expert at evaluating the relevance between a user's question and image descriptions.
        Given a user's question and descriptions of an image, determine if the image is relevant and provide a relevance score.

        First, analyze the image descriptions to identify the actual figure number in the document (e.g., "Figure 1", "Fig. 2", etc.).
        Then evaluate the relevance considering both the actual figure number and the content.

        Organize your response in the following JSON format:
        ```json
        {{
            "actual_figure_number": "<extracted figure number from descriptions, e.g. 'Figure 1', 'Fig. 2', etc.>",
            "is_relevant": <Boolean, True/False>,
            "relevance_score": <float between 0 and 1>,
            "explanation": "<brief explanation including actual figure number and why this image is or isn't relevant>"
        }}
        ```

        Pay special attention to:
        1. If the user asks about a specific figure number (e.g., "Figure 1"), prioritize matching the ACTUAL figure number from descriptions, NOT the filename
        2. The semantic meaning and context of both the question and image descriptions
        3. Whether the image would help answer the user's question
        4. Look for figure number mentions in the descriptions like "Figure X", "Fig. X", "Figure-X", etc.
        """

        human_prompt = """
        User's question: {question}
        Image descriptions:
        {descriptions}

        Note: The image filename may not reflect the actual figure number in the document. Please extract the actual figure number from the descriptions.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

        chain = prompt | llm | error_parser

        # Evaluate each image
        image_scores = []
        for image, score in image_sources.items():
            if image in image_context:
                # Get all context descriptions for this image
                descriptions = image_context[image]
                descriptions_text = "\n".join([f"- {desc}" for desc in descriptions])

                # Evaluate relevance using LLM
                try:
                    result = chain.invoke({
                        "question": user_input,
                        "descriptions": descriptions_text
                    })

                    # Store both the actual figure number and score
                    if result["is_relevant"]:
                        # Combine vector similarity score with LLM relevance score
                        combined_score = (score + result["relevance_score"]) / 2
                        image_scores.append((
                            image,
                            combined_score,
                            result["actual_figure_number"],
                            result["explanation"]
                        ))
                        # # TEST
                        # print(f"image_scores for {image}: {image_scores}")
                        # print(f"result for {image}: {result}")
                except Exception as e:
                    logger.exception(f"Error evaluating image {image}: {e}")
                    continue

        # Sort images by relevance score
        image_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter images with high relevance score (score > 0.2)
        filtered_images = {img: score for img, score, fig_num, expl in image_scores if score > 0.5}

        if filtered_images:
            # If asking about a specific figure, prioritize exact figure number match
            import re
            figure_pattern = re.compile(r'fig(?:ure)?\.?\s*(\d+)', re.IGNORECASE)
            user_figure_match = figure_pattern.search(user_input)

            if user_figure_match:
                user_figure_num = user_figure_match.group(1)
                # Look for exact figure number match first
                exact_matches = {
                    img: score for img, score, fig_num, expl in image_scores 
                    if re.search(rf'(?:figure|fig)\.?\s*{user_figure_num}\b', fig_num, re.IGNORECASE)
                }
                if exact_matches:
                    # Take the highest scored exact match
                    highest_match = max(exact_matches.items(), key=lambda x: x[1])
                    filtered_images = {highest_match[0]: highest_match[1]}
                else:
                    # If no exact match found, include images with scores close to the highest score
                    if filtered_images:
                        # Get the highest score
                        highest_score = max(filtered_images.values())
                        # Keep images with scores within 10% of the highest score
                        score_threshold = highest_score * 0.9
                        filtered_images = {img: score for img, score in filtered_images.items() if score >= score_threshold}
            else:
                # If no specific figure was asked for, include images with scores close to the highest score
                if filtered_images:
                    # Get the highest score
                    highest_score = max(filtered_images.values())
                    # Keep images with scores within 10% of the highest score
                    score_threshold = highest_score * 0.9
                    filtered_images = {img: score for img, score in filtered_images.items() if score >= score_threshold}

    # Process text sources as before
    for page in _doc:
        for source, score in text_sources.items():
            text_instances = robust_search_for(page, source)
            if text_instances:
                refined_sources[source] = score

    # Combine filtered image sources with refined text sources
    final_sources = {**filtered_images, **refined_sources}

    # Sort by score
    sorted_sources = dict(sorted(final_sources.items(), key=lambda x: x[1], reverse=True))
    
    # Keep only the top 50% of sources by score
    num_sources_to_keep = max(1, len(sorted_sources) // 2)  # Keep at least 1 source
    sorted_sources = dict(list(sorted_sources.items())[:num_sources_to_keep])
    
    # Further limit to top 20 if needed
    sorted_sources = dict(list(sorted_sources.items())[:20])
    return sorted_sources


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0


class PageAwareTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter that respects page boundaries"""

    def split_document(self, document):
        """Split document while respecting page boundaries"""
        final_chunks = []

        for doc in document:
            # Get the page number from the metadata
            page_num = doc.metadata.get("page", 0)
            text = doc.page_content

            # Use parent class's splitting logic first
            chunks = super().split_text(text)

            # Create new document for each chunk with original metadata
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                # Update metadata to indicate chunk position
                metadata["chunk_index"] = i
                final_chunks.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))

        # Sort chunks by page number and then by chunk index
        final_chunks.sort(key=lambda x: (
            x.metadata.get("page", 0),
            x.metadata.get("chunk_index", 0)
        ))

        return final_chunks