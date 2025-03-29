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

import requests, uuid, json
import os
from typing import List, Dict, Any, Union, Tuple, Dict
from dotenv import load_dotenv
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


import logging
logger = logging.getLogger("tutorpipeline.science.content_translator")

load_dotenv()
# Control whether to use Marker API or not. Only for local environment we skip Marker API.
SKIP_MARKER_API = True if os.getenv("ENVIRONMENT") == "local" else False
logger.info(f"SKIP_MARKER_API: {SKIP_MARKER_API}")


def get_translation_llm(para):
    """Get LLM instance specifically for translation"""
    api = ApiHandler(para)
    return api.models['basic']['instance']


def detect_language(text):
    """Detect language of the text"""
    # Load languages from config
    config = load_config()
    language_dict = config['languages']
    language_options = list(language_dict.values())
    language_short_dict = config['languages_short']
    language_short_options = list(language_short_dict.keys())

    language = langid.classify(text)[0]
    if language not in language_short_options:
        language = "English"
    else:
        language = language_short_dict[language]
    return language


def translate_content(
    content: str,
    target_lang: str,
    stream: bool = False
) -> str:
    """
    Translates text using the Azure Translator API.
    
    Args:
        content: The text to translate
        target_lang: The target language code (e.g. "zh" for Chinese) or name (e.g. "English")
        stream: Whether to stream the response (not used in Azure implementation)
        
    Returns:
        str: Translated text in the target language
    
    Raises:
        ValueError: If Azure Translator credentials are not properly configured
        requests.RequestException: If the API request fails
    """

    # Map language names to ISO codes
    language_code_map = {
        "English": "en",
        "Chinese": "zh",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Japanese": "ja",
        "Korean": "ko",
        "Hindi": "hi",
        "Portuguese": "pt",
        "Italian": "it"
    }
    
    # If target_lang is a language name, convert to code
    if target_lang in language_code_map:
        target_lang = language_code_map[target_lang]
    
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
    location = os.getenv("AZURE_TRANSLATOR_LOCATION")
    
    if not key or not endpoint or not location:
        raise ValueError("Azure Translator credentials not properly configured. Check your environment variables.")
    
    path = "/translate"
    constructed_url = endpoint + path

    # Detect source language if content is not in English
    source_lang = detect_language(content)
    source_code = language_code_map.get(source_lang, "en")
    
    params = {
        "api-version": "3.0",
        "from": source_code,
        "to": [target_lang]  # Convert single string to list as required by API
    }

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }

    body = [{
        "text": content
    }]

    try:
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        request.raise_for_status()  # Raise exception for HTTP errors
        response = request.json()
        # Extract just the translated text from the response
        if stream:
            def stream_response():
                yield response[0]["translations"][0]["text"]
            return stream_response()
        else:
            return response[0]["translations"][0]["text"]
    except requests.RequestException as e:
        error_msg = f"Translation request failed: {e}"
        logger.error(error_msg)
        # If translation fails, return original content
        if stream:
            def stream_response():
                yield content
            return stream_response()
        else:
            return content