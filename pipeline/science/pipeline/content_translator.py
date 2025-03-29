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


def translate_content_llm(content: str, target_lang: str, stream=False) -> str:
    """
    Translates content from source language to target language using the LLM.

    Args:
        content (str): The text content to translate
        target_lang (str): Target language code (e.g. "en", "zh") 
        stream (bool): Whether to stream the translation

    Returns:
        str: Translated content
    """
    language = detect_language(content)
    if language == target_lang:
        return content

    # Load config and get LLM
    config = load_config()
    para = config['llm']
    llm = get_translation_llm(para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # Create translation prompt
    system_prompt = """
    You are a professional translator.
    Translate the following entire content to {target_lang} language.
    Maintain the meaning and original formatting, including markdown syntax, LaTeX formulas, and emojis.
    Only translate the text content - do not modify any formatting, code, or special syntax.
    Do not include any additional information, such as "Here is the translated content:"
    """

    if target_lang == "zh":
        system_prompt = """
        你是一个专业的英文到中文的翻译者。
        不要引入任何额外的信息，只返回翻译后的内容。
        不要引入任何额外的信息， 比如"以下是翻译内容："
        """

        human_prompt = """
        你是一个专业的英文到中文的翻译者。
        将以下内容全部翻译成中文，包括标题在内的每一句话。
        不要引入任何额外的信息，只返回翻译后的内容。
        不要引入任何额外的信息， 比如"以下是翻译内容："
        并返回翻译后的中文版本内容：
        {content}
        """

    elif target_lang == "es":
        system_prompt = """
        Eres un traductor profesional de español.
        """

        human_prompt = """
        Eres un traductor profesional de español.
        Traduzca el siguiente contenido al español, incluyendo cada oración, incluido el título.
        Y devuelve la versión traducida al español del contenido:
        {content}
        """

    elif target_lang == "fr":
        system_prompt = """
        Vous êtes un traducteur professionnel français.
        """

        human_prompt = """
        Vous êtes un traducteur professionnel français.
        Traduit le contenu suivant en français, y compris chaque phrase, y compris le titre.
        Et renvoie la version française traduite du contenu:
        {content}
        """

    elif target_lang == "de":
        system_prompt = """
        Du bist ein professioneller deutscher Übersetzer.
        """

        human_prompt = """
        Du bist ein professioneller deutscher Übersetzer.
        Übersetze den folgenden Inhalt ins Deutsche, einschließlich jeder Phrase, einschließlich des Titels.
        Und gibt die übersetzte deutsche Version des Inhalts zurück:
        {content}
        """

    elif target_lang == "ja":   
        system_prompt = """
        あなたは日本語の翻訳者です。
        """

        human_prompt = """
        あなたは日本語の翻訳者です。
        以下の内容を日本語に翻訳し、タイトルを含めてすべての句を翻訳してください。
        返答は日本語でお願いします。
        {content}
        """

    elif target_lang == "ko":
        system_prompt = """
        당신은 한국어 번역자입니다.
        """

        human_prompt = """
        당신은 한국어 번역자입니다.
        다음 내용을 한국어로 번역하고 제목을 포함하여 모든 문장을 번역하세요.
        반환은 한국어로 부탁합니다.
        {content}
        """

    elif target_lang == "hi":
        system_prompt = """
        आप हिंदी के अनुवादक हैं।
        """

        human_prompt = """
        आप हिंदी के अनुवादक हैं।
        निम्नलिखित सामग्री को हिंदी में अनुवाद करें और शीर्षक भी शामिल करें।
        इसे हिंदी में वापस दें।
        {content}
        """

    elif target_lang == "pt":
        system_prompt = """
        Você é um tradutor profissional português.
        """

        human_prompt = """
        Você é um tradutor profissional português.  
        Traduz o seguinte conteúdo para o português, incluindo cada frase, incluindo o título.
        Devolva a versão traduzida em português do conteúdo:
        {content}
        """

    elif target_lang == "it":
        system_prompt = """
        Sei un traduttore professionista italiano.
        """

        human_prompt = """
        Sei un traduttore professionista italiano.
        Traduci il seguente contenuto in italiano, incluso ogni frase, incluso il titolo.
        Restituisci la versione italiana tradotta del contenuto:
        {content}
        """

    else:
        system_prompt = """
        You are a professional translator.
        """

        human_prompt = """
        You are a professional translator.
        Translate the following content to {target_lang} language.
        Maintain the meaning and original formatting, including markdown syntax, LaTeX formulas, and emojis.
        Only translate the text content - do not modify any formatting, code, or special syntax.
        {content}
        """


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    # Create and execute translation chain
    translation_chain = prompt | llm | parser   # error_parser

    if stream:
        translated_content = translation_chain.stream({
            "target_lang": target_lang,
            "content": content
        })
    else:
        translated_content = translation_chain.invoke({
            "target_lang": target_lang,
            "content": content
        })

    return translated_content


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

    # Load languages from config
    config = load_config()
    languages_short = config["languages_short"]
    
    # Create reverted map: language names to ISO codes
    language_code_map = {v: k for k, v in languages_short.items()}
    
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