import os
import copy
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_document,
    get_llm,
    count_tokens,
)
from pipeline.science.pipeline.doc_processor import process_pdf_file
from pipeline.science.pipeline.get_rag_response import get_embedding_folder_rag_response
from pipeline.science.pipeline.api_handler import ApiHandler

import logging
logger = logging.getLogger("tutorpipeline.science.get_doc_summary")

load_dotenv()


def refine_document_summary(markdown_summary, llm):
    """
    Refine the document summary to remove duplicated titles and redundant information,
    creating a concise and coherent document summary with brief bullet points.
    Aggressively merge or remove sections with redundant information while preserving core sections.

    Args:
        markdown_summary: The markdown summary to refine.
        llm: The language model to use for refinement.

    Returns:
        The refined markdown summary.
    """
    # First step: Analyze and identify redundancies
    analysis_prompt = """
    Analyze this document summary, identifying all redundancies with special attention to section consolidation:

    1. Identify specific facts that appear in multiple sections
    2. Note sections with overlapping content that should be merged or removed
    3. List repeated metrics or measurements (like "$g^{(2)}(0) = 0.060(13)$")
    4. Flag concepts explained multiple times across different sections
    5. Identify all bullet points exceeding 15 words
    6. Catalog phrases that could be expressed more concisely

    Document to analyze:
    {summary}

    Return your analysis in JSON format with these keys:
    - "redundant_facts": [list of specific facts/phrases that appear multiple times]
    - "overlapping_sections": [pairs of section names that have significant overlap]
    - "sections_to_remove": [sections that can be removed entirely as their content is covered elsewhere]
    - "sections_to_merge": [{"new_section_name": "name", "sections_to_combine": ["section1", "section2"]}]
    - "repeated_metrics": [specific measurements/numbers that are mentioned multiple times]
    - "verbose_bullets": [bullet points exceeding 15 words, with word count noted]
    - "wordy_phrases": [common phrases that could be expressed more concisely]
    - "consolidated_structure": [brief outline of how the summary should be restructured with minimal sections]
    """

    analysis_prompt_template = ChatPromptTemplate.from_template(analysis_prompt)
    json_parser = JsonOutputParser()
    # Add error handling with a fixing parser
    fixing_parser = OutputFixingParser.from_llm(parser=json_parser, llm=llm)
    analysis_chain = analysis_prompt_template | llm | fixing_parser

    try:
        analysis_result = analysis_chain.invoke({"summary": markdown_summary})
    except Exception as e:
        logger.warning(f"Error during summary analysis: {e}. Proceeding with basic refinement.")
        analysis_result = {
            "redundant_facts": [],
            "overlapping_sections": [],
            "sections_to_remove": [],
            "sections_to_merge": [],
            "repeated_metrics": [],
            "verbose_bullets": [],
            "wordy_phrases": [],
            "consolidated_structure": "Use original structure but remove redundancies"
        }

    # Second step: Create refined summary based on analysis
    refine_prompt = """
    Refine this document summary to create an extremely concise final version. Be aggressive about combining and removing sections.

    DOCUMENT ANALYSIS:
    {analysis}

    CORE SECTIONS TO PRESERVE:
    - "👋 Welcome to DeepTutor!" (Keep this exactly as is)
    - "💡 Key Takeaway" (Keep this exactly as is)
    - "📚 Document Overview" (Keep this section but refine content)
    - "💬 Ask Me Anything!" (Keep this exactly as is at the end)

    REFINEMENT RULES:
    1. PRESERVE ONLY CORE SECTIONS:
       - Keep the four core sections listed above
       - AGGRESSIVELY merge or completely remove all other sections
       - Move unique content from removed sections into "Document Overview" or merged sections

    2. ELIMINATE ALL REDUNDANCY:
       - Include each piece of information EXACTLY ONCE in the most relevant section
       - Combine all similar topics into a single section with a general title
       - Use each metric/measurement only once

    3. BULLET POINT CONSTRAINTS:
       - STRICT LIMIT: Each bullet point MUST be 15 words or fewer without exception
       - Remove all introductory phrases like "We demonstrate", "This study shows", etc.
       - Use active voice and technical terminology to reduce word count
       - Break longer points into multiple concise bullets if necessary
       - Eliminate all filler words (quickly, effectively, significantly, etc.)

    4. CONTENT TRANSFORMATION:
       - Convert verbose explanations to precise technical statements
       - Replace wordy descriptions with specific metrics where available
       - Remove adjectives and adverbs unless absolutely necessary for meaning
       - Use technical abbreviations when appropriate

    5. CRITICAL REQUIREMENTS:
       - Preserve all unique information while eliminating redundancy
       - Maintain the markdown formatting style
       - Do not add new information not present in the original
       - EVERY bullet point MUST be 15 words or fewer
       - The final summary should have MINIMAL sections beyond the core sections

    Original document to refine:
    {summary}

    Return the complete refined summary with dramatically reduced word count and minimal sections.
    """

    refine_prompt_template = ChatPromptTemplate.from_template(refine_prompt)
    str_parser = StrOutputParser()
    refine_chain = refine_prompt_template | llm | str_parser

    refined_markdown_summary = refine_chain.invoke({
        "summary": markdown_summary,
        "analysis": analysis_result
    })

    return refined_markdown_summary


async def generate_document_summary(file_path, embedding_folder, md_document=None):
    """
    Given a file path, generate a comprehensive markdown-formatted summary of the document using multiple LLM calls.
    The document can be a PDF file or a markdown file.
    If the document is a PDF file, the content will be extracted from the PDF file embedded in the embedding folder.
    If the document is a markdown file, the content will be read from the markdown file.
    """
    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)  # Using Advanced model for better quality
    api = ApiHandler(para)
    max_tokens = int(api.models['advanced']['context_window']/2)
    # max_tokens = int(65536/3)
    default_topics = config['default_topics']

    # First try to get content from markdown document
    combined_content = ""
    if md_document:
        logger.info("Using provided markdown content...")
        combined_content = md_document

    # If no content in markdown document, fall back to document content
    if not combined_content:
        logger.info("Using document loaded content as source instead of markdown file...")
        _document, _doc = process_pdf_file(file_path)
        combined_content = "\n\n".join(doc.page_content for doc in _document)

    if not combined_content:
        raise ValueError("No content available from either markdown document or document")

    # First generate the take-home message
    takehome_prompt = """
    Provide a single, impactful sentence that captures the most important takeaway from this document.

    Guidelines:
    - Be extremely concise and specific (maximum 15 words)
    - Focus on the main contribution or finding
    - Use bold for key terms
    - Keep it to one sentence
    - Add a relevant emoji at the start of bullet points or the first sentence
    - For inline formulas use single $ marks. For example, $E = mc^2$
    - For block formulas use double $$ marks. For example,
    $$
    F = ma (just an example, may not be a real formula in the doc)
    $$

    Document: {context}
    """

    # Topics extraction
    topics_prompt = """
    Identify only the most essential topics/sections from this document. Be extremely selective and concise - only include major components.

    Return format:
    {{"topics": ["topic1", "topic2", ...]}}

    Guidelines:
    - Focus only on critical sections
    - Use short, descriptive names (1-2 words per topic)
    - Avoid overlapping topics
    - Consider if topics can be covered in the Document Overview instead

    Document: {context}
    """

    # Generate overview
    overview_prompt = """
    Provide a clear and engaging overview using bullet points that summarizes all key aspects of the document.

    Guidelines:
    - Use 4-5 concise bullet points
    - **Bold** for key terms
    - STRICT LIMIT: Each bullet point MUST be 15 words or fewer
    - Focus on technical accuracy over narrative flow
    - Remove all introductory phrases, filler words, and unnecessary adjectives
    - Cover the most important aspects from all parts of the document
    - For inline formulas use single $ marks. For example, $E = mc^2$
    - For block formulas use double $$ marks. For example,
    $$
    F = ma (just an example, may not be a real formula in the doc)
    $$

    Document: {context}
    """

    # Generate summaries for each topic
    topic_prompt = """
    Provide an engaging summary for the topic "{topic}" using bullet points.

    Guidelines:
    - Use 2-3 bullet points
    - STRICT LIMIT: Each bullet point MUST be 15 words or fewer
    - **Bold** for key terms
    - Use technical terminology to reduce word count
    - Include specific metrics only if crucial
    - Remove all introductory phrases, filler words, and unnecessary adjectives
    - Ensure no overlap with content in the Document Overview
    - For inline formulas use single $ marks. For example, $E = mc^2$
    - For block formulas use double $$ marks. For example,
    $$
    F = ma (just an example, may not be a real formula in the doc)
    $$

    Document: {context}
    """

    # If the document length is within the token limit, we can directly use the document content
    token_length = count_tokens(combined_content)
    logger.info(f"Document token length: {token_length}")
    if token_length < max_tokens:
        logger.info("Document length is within the token limit, using the document content directly...")

        # First generate the take-home message
        takehome_prompt = ChatPromptTemplate.from_template(takehome_prompt)
        str_parser = StrOutputParser()
        takehome_chain = takehome_prompt | llm | str_parser
        takehome = takehome_chain.invoke({"context": truncate_document(combined_content)})

        # Topics extraction
        topics_prompt = ChatPromptTemplate.from_template(topics_prompt)
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        topics_chain = topics_prompt | llm | error_parser
        topics_result = topics_chain.invoke({"context": truncate_document(combined_content)})

        try:
            topics = topics_result.get("topics", [])
        except AttributeError:
            logger.exception("Warning: Failed to get topics. Using default topics.")
            topics = ["Overview", "Methods", "Results", "Discussion"]

        if len(topics) >= 10:
            logger.info("Number of topics is greater than 10, using default topics...")
            topics = default_topics

        # Generate overview
        overview_prompt = ChatPromptTemplate.from_template(overview_prompt)
        overview_chain = overview_prompt | llm | str_parser
        overview = overview_chain.invoke({"context": truncate_document(combined_content)})

        # Generate summaries for each topic
        summaries = []
        for topic in topics:
            topic_prompt_copy = copy.deepcopy(topic_prompt)
            topic_prompt_template = ChatPromptTemplate.from_template(topic_prompt_copy)
            topic_chain = topic_prompt_template | llm | str_parser
            topic_summary = topic_chain.invoke({
                "topic": topic,
                "context": truncate_document(combined_content)
            })
            summaries.append((topic, topic_summary))

        # Combine everything into markdown format with welcome message and take-home message
        markdown_summary = f"""### 👋 Welcome to DeepTutor!

I'm your AI tutor 🤖 ready to help you understand this document.

### 💡 Key Takeaway
{takehome}

### 📚 Document Overview
{overview}

"""

        # Add emojis for common topic titles
        topic_emojis = config['topic_emojis']

        for topic, summary in summaries:
            # Get emoji based on topic, defaulting to 📌 if not found
            topic_lower = topic.lower()
            emoji = next((v for k, v in topic_emojis.items() if k in topic_lower), topic_emojis["default"])

            markdown_summary += f"""### {emoji} {topic}
{summary}

"""

        markdown_summary += """
---
### 💬 Ask Me Anything!
Feel free to ask me any questions about the document! I'm here to help! ✨
"""

        # Refine the document summary to remove duplicated titles
        refined_markdown_summary = refine_document_summary(markdown_summary, llm=get_llm("advanced", config['llm']))

        # Use the refined version
        document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
        with open(document_summary_path, "w", encoding='utf-8') as f:
            f.write(refined_markdown_summary)

        return refined_markdown_summary

    else:
        # If the document length is beyond the token limit, we need to do RAG for each query
        logger.info("Document length is beyond the token limit, doing RAG for each query...")

        # First generate the take-home message, user input is the prompt's first line
        try:
            takehome = await get_embedding_folder_rag_response(
                prompt_string=takehome_prompt,
                user_input=takehome_prompt.split("\n")[0],
                chat_history="",
                embedding_folder=os.path.join(embedding_folder, 'markdown'),
                embedding_type='default',
            )
        except Exception as e:
            logger.exception(f"Failed to generate take-home message: {str(e)}")
            takehome = await get_embedding_folder_rag_response(
                prompt_string=takehome_prompt,
                user_input=takehome_prompt.split("\n")[0],
                chat_history="",
                embedding_folder=embedding_folder,
                embedding_type='default'
            )

        # Generate overview
        try:
            overview = await get_embedding_folder_rag_response(
                prompt_string=overview_prompt,
                user_input=overview_prompt.split("\n")[0],
                chat_history="",
                embedding_folder=os.path.join(embedding_folder, 'markdown'),
                embedding_type='default'
            )
        except Exception as e:
            logger.exception(f"Failed to generate overview: {str(e)}")
            overview = await get_embedding_folder_rag_response(
                prompt_string=overview_prompt,
                user_input=overview_prompt.split("\n")[0],
                chat_history="",
                embedding_folder=embedding_folder,
                embedding_type='default'
            )

        # Generate summaries for each topic
        try:
            summaries = []
            for topic in topics:
                topic_prompt_copy = copy.deepcopy(topic_prompt)
                # Fill in the topic in the prompt
                topic_prompt_copy = topic_prompt_copy.format(topic=topic)

                logger.info(f"Generating summary for topic: {topic}")
                logger.info(f"Prompt: {topic_prompt_copy}")
                try:
                    summary = await get_embedding_folder_rag_response(
                        prompt_string=topic_prompt_copy,
                        user_input=topic_prompt_copy.split("\n")[0],
                        chat_history="",
                        embedding_folder=os.path.join(embedding_folder, 'markdown'),
                        embedding_type='default'
                    )
                except Exception as e:
                    logger.exception(f"Failed to generate summary for topic: {topic}, error: {str(e)}")
                    summary = await get_embedding_folder_rag_response(
                        prompt_string=topic_prompt_copy,
                        user_input=topic_prompt_copy.split("\n")[0],
                        chat_history="",
                        embedding_folder=embedding_folder,
                        embedding_type='default'
                    )
                summaries.append((topic, summary))
        except Exception as e:
            logger.exception(f"Failed to load topics: {str(e)}")
            topics = default_topics
            summaries = []
            for topic in topics:
                topic_prompt_copy = copy.deepcopy(topic_prompt)
                # Fill in the topic in the prompt
                topic_prompt_copy = topic_prompt_copy.replace("{topic}", topic)

                logger.info(f"Generating summary for topic: {topic}")
                logger.info(f"Prompt: {topic_prompt_copy}")
                try:
                    summary = await get_embedding_folder_rag_response(
                        prompt_string=topic_prompt_copy,
                        user_input=topic_prompt_copy.split("\n")[0],
                        chat_history="",
                        embedding_folder=os.path.join(embedding_folder, 'markdown'),
                        embedding_type='default'
                    )
                except Exception as e:
                    logger.exception(f"Failed to generate summary for topic: {topic}, error: {str(e)}")
                    summary = await get_embedding_folder_rag_response(
                        prompt_string=topic_prompt_copy,
                        user_input=topic_prompt_copy.split("\n")[0],
                        chat_history="",
                        embedding_folder=embedding_folder,
                        embedding_type='default'
                    )
                summaries.append((topic, summary))

        # Combine everything into markdown format with welcome message and take-home message
        markdown_summary = f"""### 👋 Welcome to DeepTutor!

I'm your AI tutor 🤖 ready to help you understand this document.

### 💡 Key Takeaway
{takehome}

### 📚 Document Overview
{overview}

"""

        # Add emojis for common topic titles
        topic_emojis = config['topic_emojis']

        for topic, summary in summaries:
            # Get emoji based on topic, defaulting to 📌 if not found
            topic_lower = topic.lower()
            emoji = next((v for k, v in topic_emojis.items() if k in topic_lower), topic_emojis["default"])

            markdown_summary += f"""### {emoji} {topic}
{summary}

"""

        markdown_summary += """
---
### 💬 Ask Me Anything!
Feel free to ask me any questions about the document! I'm here to help! ✨
"""

        # Refine the document summary to remove duplicated titles
        refined_markdown_summary = refine_document_summary(markdown_summary, llm=get_llm("advanced", config['llm']))

        # Use the refined version
        document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
        with open(document_summary_path, "w", encoding='utf-8') as f:
            f.write(refined_markdown_summary)

        return refined_markdown_summary