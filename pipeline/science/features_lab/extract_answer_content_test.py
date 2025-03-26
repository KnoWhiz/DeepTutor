import os
import io
import hashlib
# import shutil
import fitz
import tiktoken
import json
import langid
import requests
import base64
import time
from datetime import datetime, UTC
import re

from dotenv import load_dotenv
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
logger = logging.getLogger(__name__)

load_dotenv()
# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


def clean_translation_prefix(text):
    """
    Clean translation prefixes from text in all supported languages.
    
    Args:
        text: The text to clean
        
    Returns:
        The cleaned text with translation prefixes removed
    """
    if not text:
        return text
        
    # First, handle the specific case that's still occurring
    if "当然可以！以下是翻译内容：" in text:
        text = text.replace("当然可以！以下是翻译内容：", "").strip()
    
    # Common starter words in different languages
    starters = (
        # English starters
        r"Sure|Certainly|Of course|Here|Yes|Okay|"
        # Chinese starters
        r"当然|好的|是的|这是|以下是|"
        # Spanish starters
        r"Claro|Seguro|Por supuesto|Aquí|Sí|"
        # French starters
        r"Bien sûr|Certainement|Oui|Voici|Voilà|"
        # German starters
        r"Natürlich|Sicher|Klar|Hier ist|Ja|"
        # Japanese starters
        r"もちろん|はい|ここに|"
        # Korean starters
        r"물론|네|여기|"
        # Hindi starters
        r"ज़रूर|हां|यहां|निश्चित रूप से|"
        # Portuguese starters
        r"Claro|Certamente|Sim|Aqui|"
        # Italian starters
        r"Certo|Sicuro|Sì|Ecco"
    )
    
    # Translation-related terms in different languages
    translation_terms = (
        # English
        r"translation|translated|"
        # Chinese
        r"翻译|译文|"
        # Spanish
        r"traducción|traducido|"
        # French
        r"traduction|traduit|"
        # German
        r"Übersetzung|übersetzt|"
        # Japanese
        r"翻訳|"
        # Korean
        r"번역|"
        # Hindi
        r"अनुवाद|"
        # Portuguese
        r"tradução|traduzido|"
        # Italian
        r"traduzione|tradotto"
    )
    
    # More aggressive regex pattern for translation prefixes that handles newlines better
    pattern = rf'^({starters})[^A-Za-z0-9]*?({translation_terms})[^:]*?:[ \n]*'
    
    # Apply the cleanup with a more permissive DOTALL flag to handle newlines
    result = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
    
    # Additional cleanup for common patterns with newlines
    result = re.sub(r'^[^:]+:[ \n]+', '', result, flags=re.DOTALL)
    
    # Remove any leading newlines after cleanup
    result = result.lstrip('\n')
    
    return result.strip()


def extract_answer_content(message_content):
    sources = {}    # {source_string: source_score}
    source_pages = {}    # {source_page_string: source_page_score}
    source_annotations = {}
    refined_source_pages = {}    # {refined_source_page_string: refined_source_page_score}
    refined_source_index = {}    # {refined_source_index_string: refined_source_index_score}
    follow_up_questions = []

    # Extract the main answer (content between <response> tags)
    # The logic is: if we have <response> tags, we extract the content between them
    # Otherwise, we extract the content between <original_response> and </original_response> tags
    # If we have neither, we extract the content between <thinking> and </thinking> tags
    # If we have none of the above, we return an empty string
    answer = ""
    thinking = ""
    response_match = re.search(r'<response>(.*?)</response>', message_content, re.DOTALL)
    original_response_match = re.search(r'<original_response>(.*?)</original_response>', message_content, re.DOTALL)
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', message_content, re.DOTALL)
    if response_match:
        answer = response_match.group(1).strip()
    elif original_response_match:
        answer = original_response_match.group(1).strip()
    elif thinking_match:
        answer = thinking_match.group(1).strip()
    else:
        answer = ""

    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Extract follow-up questions (content between <followup_question> tags)
    followup_matches = re.finditer(r'<followup_question>(.*?)</followup_question>', message_content, re.DOTALL)
    for match in followup_matches:
        question = match.group(1).strip()
        if question:
            # Remove any residual XML tags
            question = re.sub(r'<followup_question>.*?</followup_question>', '', question)

            # Apply the clean_translation_prefix function
            question = clean_translation_prefix(question)

            follow_up_questions.append(question)

    # Extract sources (content between <source> tags)
    source_matches = re.finditer(r'<source>(.*?)</source>', message_content, re.DOTALL)
    for match in source_matches:
        source_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', source_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                sources[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                sources[key] = value

    # Extract source pages (content between <source_page> tags)
    source_page_matches = re.finditer(r'<source_page>(.*?)</source_page>', message_content, re.DOTALL)
    for match in source_page_matches:
        source_page_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', source_page_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                source_pages[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                source_pages[key] = value

    # Extract refined source pages (content between <refined_source_page> tags)
    refined_source_page_matches = re.finditer(r'<refined_source_page>(.*?)</refined_source_page>', message_content, re.DOTALL)
    for match in refined_source_page_matches:
        refined_source_page_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', refined_source_page_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float
                refined_source_pages[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                refined_source_pages[key] = value

    # Extract refined source index (content between <refined_source_index> tags)
    refined_source_index_matches = re.finditer(r'<refined_source_index>(.*?)</refined_source_index>', message_content, re.DOTALL)
    for match in refined_source_index_matches:
        refined_source_index_content = match.group(1).strip()
        # Extract the key and value using regex pattern {key}{value}
        key_value_match = re.match(r'\{(.*?)\}\{(.*?)\}', refined_source_index_content)
        if key_value_match:
            key = key_value_match.group(1)
            value = key_value_match.group(2)
            try:
                # Convert value to float or int
                refined_source_index[key] = float(value)
            except ValueError:
                try:
                    # Try converting to int if float conversion fails
                    refined_source_index[key] = int(value)
                except ValueError:
                    # If both conversions fail, store as string
                    refined_source_index[key] = value

    return answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking

def test_extract_answer_content():
    """
    Test the extract_answer_content function with various test cases.
    """
    # Test case 1: Basic message with thinking and original_response
    test_message_1 = """
    <thinking>
    Understanding the user input ...
    User input: GAIA是什么意思呢
    Answer planning...
    Understanding the user input done ...
    </thinking>
    <original_response>
    GAIA（General AI Assistant benchmark）是一个专门为评估通用人工智能助手设计的基准测试。
    </original_response>
    <response>
    GAIA（General AI Assistant benchmark）是一个专门为评估通用人工智能助手设计的基准测试。
    </response>
    <followup_question>
    什么是GAIA？
    </followup_question>
    <followup_question>
    什么是GAIA？
    </followup_question>
    <source>
    {paper1}{0.95}
    </source>
    <source_page>
    {page1}{0.85}
    </source_page>
    <refined_source_page>
    {refined1}{0.75}
    </refined_source_page>
    <refined_source_index>
    {index1}{0.65}
    </refined_source_index>
    """
    
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_1)
    
    # Verify thinking content
    assert "Understanding the user input" in thinking
    assert "User input: GAIA是什么意思呢" in thinking
    assert "Answer planning" in thinking
    
    # Verify answer content
    assert "GAIA（General AI Assistant benchmark）" in answer
    
    # Test case 2: Message with sources and source pages
    test_message_2 = """
    <thinking>Test thinking content</thinking>
    <original_response>Test answer</original_response>
    <source>{paper1}{0.95}</source>
    <source_page>{page1}{0.85}</source_page>
    <refined_source_page>{refined1}{0.75}</refined_source_page>
    <refined_source_index>{index1}{0.65}</refined_source_index>
    """
    
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_2)
    
    # Verify sources
    assert sources["paper1"] == 0.95
    
    # Verify source pages
    assert source_pages["page1"] == 0.85
    
    # Verify refined source pages
    assert refined_source_pages["refined1"] == 0.75
    
    # Verify refined source index
    assert refined_source_index["index1"] == 0.65
    
    # Test case 3: Message with follow-up questions
    test_message_3 = """
    <thinking>Test thinking content</thinking>
    <original_response>Test answer</original_response>
    <followup_question>What are the key features of GAIA?</followup_question>
    <followup_question>How does GAIA compare to other benchmarks?</followup_question>
    """
    
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_3)
    
    # Verify follow-up questions
    assert len(follow_up_questions) == 2
    assert "What are the key features of GAIA?" in follow_up_questions
    assert "How does GAIA compare to other benchmarks?" in follow_up_questions
    
    # Test case 4: Empty message
    test_message_4 = ""
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_4)
    
    # Verify empty results
    assert answer == ""
    assert sources == {}
    assert source_pages == {}
    assert source_annotations == {}
    assert refined_source_pages == {}
    assert refined_source_index == {}
    assert follow_up_questions == []
    assert thinking == ""
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_extract_answer_content()