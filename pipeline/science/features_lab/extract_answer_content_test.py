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
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler
from pipeline.science.pipeline.utils import clean_translation_prefix

import logging
logger = logging.getLogger(__name__)

load_dotenv()
# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


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