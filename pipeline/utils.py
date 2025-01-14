import os
import hashlib
import shutil
import fitz
import tiktoken
from pathlib import Path
import json
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from streamlit_float import *
from pipeline.api_handler import ApiHandler


# Generate a unique course ID for the uploaded file
def generate_course_id(file):
    file_hash = hashlib.md5(file).hexdigest()
    return file_hash


# Custom function to extract document objects from uploaded file
def extract_documents_from_file(file, filename):
    input_dir = './input_files/'

    # Create the input_files directory if it doesn't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # Save the file to the input_files directory with the original filename
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(file)

    # Load the document
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


# Find pages with the given excerpts in the document
def find_pages_with_excerpts(doc, excerpts):
    pages_with_excerpts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                pages_with_excerpts.append(page_num)
                break
    return pages_with_excerpts if pages_with_excerpts else [0]


# Get the highlight information for the given excerpts
def get_highlight_info(doc, excerpts):
    annotations = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                for inst in text_instances:
                    annotations.append(
                        {
                            "page": page_num + 1,
                            "x": inst.x0,
                            "y": inst.y0,
                            "width": inst.x1 - inst.x0,
                            "height": inst.y1 - inst.y0,
                            "color": "red",
                        }
                    )
    return annotations


# Add new helper functions
def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def count_tokens(text, model_name='gpt-4o'):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def truncate_chat_history(chat_history, model_name='gpt-4o'):
    """Only keep messages that fit within token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    total_tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        if(message['role'] == 'assistant' or message['role'] == 'user'):
            message = str(message)
            message_tokens = count_tokens(message, model_name)
            if total_tokens + message_tokens > max_tokens:
                break
            truncated_history.insert(0, message)
            total_tokens += message_tokens
    return truncated_history

def truncate_document(_document, model_name='gpt-4o'):
    """Only keep beginning of document that fits token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    _document = str(_document)
    document_tokens = count_tokens(_document, model_name)
    if document_tokens > max_tokens:
        _document = _document[:max_tokens]
    return _document