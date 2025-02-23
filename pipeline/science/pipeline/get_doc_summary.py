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
    process_pdf_file,
)
from pipeline.science.pipeline.api_handler import ApiHandler
import logging
logger = logging.getLogger("tutorpipeline.science.get_doc_summary")
load_dotenv()


def generate_document_summary(file_path, embedding_folder, md_document=None):
    """
    Given a file path, generate a comprehensive markdown-formatted summary of the document using multiple LLM calls.
    The document can be a PDF file or a markdown file.
    If the document is a PDF file, the content will be extracted from the PDF file embedded in the embedding folder.
    If the document is a markdown file, the content will be read from the markdown file.
    """
    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)  # Using advanced model for better quality
    api = ApiHandler(para)
    max_tokens = int(api.models['advance']['context_window']/1.2)

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
    - Be extremely concise and specific
    - Focus on the main contribution or finding
    - Use bold for key terms
    - Keep it to one sentence
    - Add a relevant emoji at the start of bullet points or the first sentence
    - For inline formulas use single $ marks. For example, $E = mc^2$
    - For block formulas use double $$ marks. For example,
    $$
    F = ma (just an example, may not be a real formula in the doc)
    $$

    Document: {document}
    """

    # Topics extraction
    topics_prompt = """
    Identify only the most essential topics/sections from this document.
    Be extremely selective and concise - only include major components.

    Return format:
    {{"topics": ["topic1", "topic2", ...]}}

    Guidelines:
    - Include maximum 4-5 topics
    - Focus only on critical sections
    - Use short, descriptive names

    Document: {document}
    """

    # Generate overview
    overview_prompt = """
    Provide a clear and engaging overview using bullet points.

    Guidelines:
    - Use 3-4 concise bullet points
    - **Bold** for key terms
    - Each bullet point should be one short sentence
    - For inline formulas use single $ marks. For example, $E = mc^2$
    - For block formulas use double $$ marks. For example,
    $$
    F = ma (just an example, may not be a real formula in the doc)
    $$

    Document: {document}
    """

    # Generate summaries for each topic
    topic_prompt = """
    Provide an engaging summary for the topic "{topic}" using bullet points.

    Guidelines:
    - Use 2-3 bullet points
    - Each bullet point should be one short sentence
    - **Bold** for key terms
    - Use simple, clear language
    - Include specific metrics only if crucial
    - For inline formulas use single $ marks. For example, $E = mc^2$
    - For block formulas use double $$ marks. For example,
    $$
    F = ma (just an example, may not be a real formula in the doc)
    $$

    Document: {document}
    """

    # If the document length is within the token limit, we can directly use the document content
    if len(combined_content) < 2 * max_tokens:
        logger.info("Document length is within the token limit, using the document content directly...")

        # First generate the take-home message
        takehome_prompt = ChatPromptTemplate.from_template(takehome_prompt)
        str_parser = StrOutputParser()
        takehome_chain = takehome_prompt | llm | str_parser
        takehome = takehome_chain.invoke({"document": truncate_document(combined_content)})

        # Topics extraction
        topics_prompt = ChatPromptTemplate.from_template(topics_prompt)
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        topics_chain = topics_prompt | llm | error_parser
        topics_result = topics_chain.invoke({"document": truncate_document(combined_content)})

        try:
            topics = topics_result.get("topics", [])
        except AttributeError:
            logger.exception("Warning: Failed to get topics. Using default topics.")
            topics = ["Overview", "Methods", "Results", "Discussion"]

        if len(topics) >= 10:
            logger.info("Number of topics is greater than 10, using only the first 10 topics...")
            topics = topics[:10]

        # Generate overview
        overview_prompt = ChatPromptTemplate.from_template(overview_prompt)
        overview_chain = overview_prompt | llm | str_parser
        overview = overview_chain.invoke({"document": truncate_document(combined_content)})

        # Generate summaries for each topic
        summaries = []
        for topic in topics:
            topic_prompt_copy = copy.deepcopy(topic_prompt)
            topic_prompt_template = ChatPromptTemplate.from_template(topic_prompt_copy)
            topic_chain = topic_prompt_template | llm | str_parser
            topic_summary = topic_chain.invoke({
                "topic": topic,
                "document": truncate_document(combined_content)
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

        document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
        with open(document_summary_path, "w", encoding='utf-8') as f:
            f.write(markdown_summary)

        return markdown_summary

    else:
        # If the document length is beyond the token limit, we need to do RAG for each query
        logger.info("Document length is beyond the token limit, doing RAG for each query...")

        # First generate the take-home message
        takehome_prompt = ChatPromptTemplate.from_template(takehome_prompt)
        str_parser = StrOutputParser()
        takehome_chain = takehome_prompt | llm | str_parser
        takehome = takehome_chain.invoke({"document": truncate_document(combined_content)})

        # Generate overview
        
