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
    Test the extract_answer_content function by printing extracted content.
    """
    # Test case 1: Basic message with thinking and original_response
    test_message_1 = """
    <think>
    Understanding the user input ...
    User input: GAIA是什么意思呢
    Answer planning...
    Understanding the user input done ...
    </think>

    <followup_question>
    What is GAIA?
    </followup_question>
    <followup_question>
    What is GAIA?
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
    
    print("\nTest Case 1:")
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_1)
    print("Thinking content extracted:", thinking)
    print("Answer content extracted:", answer)
    print("Sources extracted:", sources)
    print("Source Pages extracted:", source_pages)
    print("Refined Source Pages extracted:", refined_source_pages)
    print("Refined Source Index extracted:", refined_source_index)
    print("Follow-up Questions extracted:", follow_up_questions)
    
    # Test case 2: Message with sources and source pages
    test_message_2 = """
    <thinking>Test thinking content</thinking>
    <original_response>Test answer</original_response>
    <source>{paper1}{0.95}</source>
    <source_page>{page1}{0.85}</source_page>
    <refined_source_page>{refined1}{0.75}</refined_source_page>
    <refined_source_index>{index1}{0.65}</refined_source_index>
    """
    
    print("\nTest Case 2:")
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_2)
    print("Thinking content extracted:", thinking)
    print("Answer content extracted:", answer)
    print("Sources extracted:", sources)
    print("Source Pages extracted:", source_pages)
    print("Refined Source Pages extracted:", refined_source_pages)
    print("Refined Source Index extracted:", refined_source_index)
    print("Follow-up Questions extracted:", follow_up_questions)
    
    # Test case 3: Message with follow-up questions
    test_message_3 = """
    <thinking>Test thinking content</thinking>
    <original_response>Test answer</original_response>
    <followup_question>What are the key features of GAIA?</followup_question>
    <followup_question>How does GAIA compare to other benchmarks?</followup_question>
    """
    
    print("\nTest Case 3:")
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_3)
    print("Thinking content extracted:", thinking)
    print("Answer content extracted:", answer)
    print("Sources extracted:", sources)
    print("Source Pages extracted:", source_pages)
    print("Refined Source Pages extracted:", refined_source_pages)
    print("Refined Source Index extracted:", refined_source_index)
    print("Follow-up Questions extracted:", follow_up_questions)
    
    # Test case 4: Empty message
    test_message_4 = """
<thinking>

**Understanding the user input ...**





**User input: s with the recent trend of LLMs outperforming humans on tasks
requiring professional skills in e.g. law or chemistry. GAIA’s philosophy departs from the current
trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit
that the advent of Artificial General Intelligence (AGI) hin**



**Answer planning...**

**Understanding the user input done ...**

<think>
Okay, let's tackle this question. The student wants to know how GAIA's approach differs from current AI benchmarks that focus on making tasks harder for humans, especially since LLMs are beating humans in professional skills like law or chemistry.

First, I need to recall what the paper said about GAIA's design. The paper mentions that GAIA's philosophy is different because it doesn't follow the trend of creating tasks that are increasingly difficult for humans. Instead, GAIA uses real-world questions that are simple for humans but hard for AI. The human success rate is 92%, while even the best LLMs like GPT-4 with plugins only get 15%. That's a big gap.

Now, traditional benchmarks, like MMLU, have tons of questions (like 15,000) but they're multiple-choice. GAIA has fewer questions (466) but they're open-ended and require more complex reasoning. The paper also points out that while LLMs excel in specialized professional tasks, they struggle with everyday problems that humans find easy. So GAIA is testing fundamental abilities like using tools, handling multi-modality, and reasoning, which are more about general intelligence than specialized knowledge.

Another key point is that GAIA's tasks are designed to have a single correct answer, making evaluation straightforward. This contrasts with benchmarks that might have ambiguous answers or require subjective grading. The paper also mentions that GAIA aims to evaluate AI systems as a whole, not just individual components, which is different from some benchmarks that might isolate specific skills.

The student might be thinking that harder tasks for humans mean better benchmarks, but GAIA shows that simplicity for humans can still challenge AI in meaningful ways. The paper emphasizes that AGI should handle tasks with human-like robustness, not just excel in narrow domains. So GAIA's approach is about practical, real-world applicability rather than pushing for tasks that are hard for both humans and AI.

I should also address why this matters. Even though LLMs are good at professional skills, GAIA highlights gaps in their general problem-solving abilities. This suggests that current benchmarks might not fully capture the capabilities needed for true general AI. GAIA's methodology focuses on tasks that require a blend of skills, which is different from traditional benchmarks that might test isolated or specialized knowledge.

In summary, GAIA differs by focusing on real-world, human-easy tasks that test fundamental AI abilities, whereas traditional benchmarks push for increasingly difficult tasks, often in specialized areas where LLMs already surpass humans. GAIA's approach aims to measure progress toward more robust, general <think>
Okay, let's tackle this question. The student wants to know how GAIA's approach differs from current AI benchmarks that focus on making tasks harder for humans, especially since LLMs are beating humans in professional skills like law or chemistry.

First, I need to recall what the paper said about GAIA's design. The paper mentions that GAIA's philosophy is different because it doesn't follow the trend of creating tasks that are increasingly difficult for humans. Instead, GAIA uses real-world questions that are simple for humans but hard for AI. The human success rate is 92%, while even the best LLMs like GPT-4 with plugins only get 15%. That's a big gap.

Now, traditional benchmarks, like MMLU, have tons of questions (like 15,000) but they're multiple-choice. GAIA has fewer questions (466) but they're open-ended and require more complex reasoning. The paper also points out that while LLMs excel in specialized professional tasks, they struggle with everyday problems that humans find easy. So GAIA is testing fundamental abilities like using tools, handling multi-modality, and reasoning, which are more about general intelligence than specialized knowledge.

Another key point is that GAIA's tasks are designed to have a single correct answer, making evaluation straightforward. This contrasts with benchmarks that might have ambiguous answers or require subjective grading. The paper also mentions that GAIA aims to evaluate AI systems as a whole, not just individual components, which is different from some benchmarks that might isolate specific skills.

The student might be thinking that harder tasks for humans mean better benchmarks, but GAIA shows that simplicity for humans can still challenge AI in meaningful ways. The paper emphasizes that AGI should handle tasks with human-like robustness, not just excel in narrow domains. So GAIA's approach is about practical, real-world applicability rather than pushing for tasks that are hard for both humans and AI.

I should also address why this matters. Even though LLMs are good at professional skills, GAIA highlights gaps in their general problem-solving abilities. This suggests that current benchmarks might not fully capture the capabilities needed for true general AI. GAIA's methodology focuses on tasks that require a blend of skills, which is different from traditional benchmarks that might test isolated or specialized knowledge.

In summary, GAIA differs by focusing on real-world, human-easy tasks that test fundamental AI abilities, whereas traditional benchmarks push for increasingly difficult tasks, often in specialized areas where LLMs already surpass humans. GAIA's approach aims to measure progress toward more robust, general AI systems.
</think></thinking>

**Generating the response ...**



GAIA's approach to benchmarking AI fundamentally differs from traditional benchmarks in both philosophy and methodology, particularly in how it addresses the paradox of LLMs excelling in professional skills while struggling with everyday human tasks. Here's a structured comparison:

---

### **1. Philosophical Divergence**
- **Traditional Benchmarks**: Focus on creating tasks that are *increasingly difficult for humans* (e.g., legal reasoning, advanced mathematics), often prioritizing specialized expertise. This aligns with the trend where LLMs outperform humans in narrow professional domains (e.g., law, chemistry).
- **GAIA**: Prioritizes *real-world, human-easy tasks* (e.g., interpreting a spreadsheet, answering practical questions) that require **general robustness** rather than specialized knowledge. Its 92% human success rate vs. 15% for GPT-4 with plugins highlights this gap.

---

### **2. Task Design & Evaluation Goals**
- **Traditional Benchmarks**:  
  - Measure performance on tasks where humans require professional training (e.g., MMLU’s 15,000 multiple-choice questions).  
  - Often use proxy metrics (e.g., multiple-choice accuracy) that may not reflect real-world problem-solving.  
- **GAIA**:  
  - Tests **fundamental abilities** (reasoning, tool use, multi-modality) through 466 open-ended questions with *unique factual answers*.  
  - Focuses on scenarios where humans excel effortlessly (e.g., daily tasks, basic science), exposing AI’s lack of practical robustness.  

---

### **3. Implications for AGI Development**
- **Traditional Benchmarks**: Risk conflating specialized performance with general intelligence. For example, GPT-4’s high scores on bar exams do not imply human-like adaptability.  
- **GAIA**: Proposes that **AGI requires human-level robustness in everyday tasks**, not just narrow expertise. Its design reflects the "competent AGI" framework (Morris et al., 2023), where success implies systems can handle unpredictable real-world challenges.  

---

### **4. Addressing Benchmark Decay**
- **Traditional Benchmarks**: Static datasets (e.g., MMLU) are vulnerable to contamination via training data memorization.  
- **GAIA**: Mitigates decay by:  
  - Requiring **dynamic tool use** (e.g., web browsing, file analysis) rather than regurgitating memorized facts.  
  - Encouraging community extensions to address emerging challenges (e.g., safety, multi-modality).  

---

### **Key Takeaway**
While traditional benchmarks reward mastering narrow domains, GAIA emphasizes **practical, general-purpose intelligence** by testing abilities humans use daily. This shift acknowledges that outperforming humans in professional skills (e.g., law) does not equate to solving the broader challenge of AGI, which demands adaptability and robustness akin to human cognition. GAIA’s approach reorients AI evaluation toward systems that can "think" like humans in unstructured scenarios, not just excel in predefined expert tasks.</response><appendix>

**Retrieving sources ...**

<source>{In spite of being successful at tasks that are difficult for humans, the most capable LLMs do poorly on GAIA. Even equipped with tools, GPT4 does not exceed a 30% success rate for the easiest of our tasks, and 0% for the hardest. In the meantime, the average success rate for human respondents is 92%. Consequently, a system capable of solving GAIA can be assessed in the context of t-AGI,4 noting that humans typically take between 6 minutes for the simplest questions to 17 minutes for the most complex ones. From a related perspective, such system would arguably be a competent General AI within the framework recently proposed in Morris et al. (2023), which also appear to be the next milestone in AI research since ChatGPT (OpenAI, 2023) is one level below. This paper covers the composition of GAIA, its design choices, and explain how to craft questions and the associated challenges so that the community can further extend the benchmark to target emerging questions such as safety associated to tool use, or multi-modality.}{1.0}</source><source>{We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA’s philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system’s capability to exhibit similar robustness as the average human does on such questions. Using GAIA’s methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board hereby accessible.}{0.8950889259576797}</source><source>{Design choices. GAIA results both from the need for revised AI benchmarks, and the observed shortcomings of LLM evaluation.}{0.8250029981136322}</source><source>{We move in that direction by proposing GAIA, a benchmark for General AI Assistants featuring 466 carefully crafted questions and their answer, along with the associated design methodology. Our questions are easy to create, challenging for AI systems—for LLMs, most require complex generations—, yet admit a unique, factual answer, allowing a simple and robust automatic evaluation.}{0.6628082990646362}</source><source>{GAIA: A Benchmark for General AI Assistants}{0.5816324651241302}</source><source_page>{In spite of being successful at tasks that are difficult for humans, the most capable LLMs do poorly on GAIA. Even equipped with tools, GPT4 does not exceed a 30% success rate for the easiest of our tasks, and 0% for the hardest. In the meantime, the average success rate for human respondents is 92%. Consequently, a system capable of solving GAIA can be assessed in the context of t-AGI,4 noting that humans typically take between 6 minutes for the simplest questions to 17 minutes for the most complex ones. From a related perspective, such system would arguably be a competent General AI within the framework recently proposed in Morris et al. (2023), which also appear to be the next milestone in AI research since ChatGPT (OpenAI, 2023) is one level below. This paper covers the composition of GAIA, its design choices, and explain how to craft questions and the associated challenges so that the community can further extend the benchmark to target emerging questions such as safety associated to tool use, or multi-modality.}{2}</source_page><source_page>{We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA’s philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system’s capability to exhibit similar robustness as the average human does on such questions. Using GAIA’s methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board hereby accessible.}{0}</source_page><source_page>{Design choices. GAIA results both from the need for revised AI benchmarks, and the observed shortcomings of LLM evaluation.}{3}</source_page><source_page>{We move in that direction by proposing GAIA, a benchmark for General AI Assistants featuring 466 carefully crafted questions and their answer, along with the associated design methodology. Our questions are easy to create, challenging for AI systems—for LLMs, most require complex generations—, yet admit a unique, factual answer, allowing a simple and robust automatic evaluation.}{1}</source_page><source_page>{GAIA: A Benchmark for General AI Assistants}{0}</source_page><source_page>{What is GAIA and how does it work? GAIA is a benchmark for AI systems proposing general assistant questions. GAIA attempts to circumvent different pitfalls of LLMs evaluation. It is composed of 466 questions designed and annotated by humans. These questions are text-based, and sometimes come with a file (such as an image or a spreadsheet). They cover various assistant use cases such as daily personal tasks, science, or general knowledge. The questions are designed to admit a short, single correct answer, therefore easy to verify. To use GAIA, one only needs to zero-shot prompt an AI assistant with the questions and attached evidence if there are some. Scoring perfectly on GAIA requires a varied set of fundamental abilities (see Section 3.3). We provide questions along various with meta-data in supplementary material.}{3}</source_page><source_page>{While GAIA attempts to circumvent current pitfalls of LLM benchmarks, some limitation remains.}{9}</source_page><source_page>{Static versus dynamic benchmarks. Much like other complex expert datasets, GAIA currently comes with hundreds of questions that have been carefully curated and selected. By comparison, a more massive benchmark such as MMLU has close to 15,000. Yet, MMLU consists of multiple choice questions hence is seemingly easier than our open questions. Questions that admit a single correct answer require care, and we preferred to favour quality over quantity. Moreover, we hope that our insights on question design will help the community to add more questions. GAIA is indeed likely to decay over time, be it via (i) catastrophic contamination of pre-training data or (ii) disappearance from the web of some information required to answer the questions. We are confident that the various mitigations we provide for these problems will help maintaining GAIA relevant until it is solved. Static benchmarks are broken benchmarks in the making, and making GAIA evolve}{8}</source_page><source_page>{Capabilities coverage. Scoring perfectly on GAIA requires advanced reasoning, multi-modality understanding, coding capabilities and generally tool use, e.g web browsing, for which we provide a more precise definition in Appendix C. We also include questions requiring to process varied data modalities such as PDFs, spreadsheets, but also images, videos or audio, whose distribution is reported in Appendix C (Figure 6). Figure 3 (left) is an overview of these capabilities. Although web browsing is a key component of GAIA, we do not require assistants to perform actions other than “clicks” on a website such as uploading a file, post a comment or book a meeting. Testing these capabilities in real environments while avoiding spamming websites requires careful consideration that we leave for future work, and refer the reader to recent works proposing closed environments for LLMs agents (Liu et al., 2023a). We do not provide a more detailed list of required capabilities to solve the benchmark since most questions can be solved equally well via different combinations of capabilities.}{5}</source_page><source_page>{Our third principle is robustness against memorization: GAIA aims to be less gameable than most current benchmarks. To complete a task, a system has to plan and successfully complete some number of steps since the resulting answer is absent by design in plain text from current pre-training data. A progress in accuracy reflects actual system progress. Due to their diversity and the size of the action space, these tasks cannot be brute-forced without cheating, for example by memorizing the ground truth. Although accidental memorization is possible through data contamination, the accuracy required in the answers, their absence from pre-training data, and the possibility to check the reasoning trace mitigate this risk. In contrast, multiple choice answers make contamination assessment difficult since a wrong reasoning trace can still get to the correct choice. If catastrophic memorization happens in spite of these mitigations, it is easy to craft new questions using the guidelines we provide in Section 3.4.}{3}</source_page><refined_source_page>{In spite of being successful at tasks that are difficult for humans, the most capable LLMs do poorly on GAIA. Even equipped with tools, GPT4 does not exceed a 30% success rate for the easiest of our tasks, and 0% for the hardest. In the meantime, the average success rate for human respondents is 92%. Consequently, a system capable of solving GAIA can be assessed in the context of t-AGI,4 noting that humans typically take between 6 minutes for the simplest questions to 17 minutes for the most complex ones. From a related perspective, such system would arguably be a competent General AI within the framework recently proposed in Morris et al. (2023), which also appear to be the next milestone in AI research since ChatGPT (OpenAI, 2023) is one level below. This paper covers the composition of GAIA, its design choices, and explain how to craft questions and the associated challenges so that the community can further extend the benchmark to target emerging questions such as safety associated to tool use, or multi-modality.}{3}</refined_source_page><refined_source_page>{We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA’s philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system’s capability to exhibit similar robustness as the average human does on such questions. Using GAIA’s methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board hereby accessible.}{1}</refined_source_page><refined_source_page>{Design choices. GAIA results both from the need for revised AI benchmarks, and the observed shortcomings of LLM evaluation.}{4}</refined_source_page><refined_source_page>{We move in that direction by proposing GAIA, a benchmark for General AI Assistants featuring 466 carefully crafted questions and their answer, along with the associated design methodology. Our questions are easy to create, challenging for AI systems—for LLMs, most require complex generations—, yet admit a unique, factual answer, allowing a simple and robust automatic evaluation.}{2}</refined_source_page><refined_source_page>{GAIA: A Benchmark for General AI Assistants}{1}</refined_source_page><refined_source_index>{In spite of being successful at tasks that are difficult for humans, the most capable LLMs do poorly on GAIA. Even equipped with tools, GPT4 does not exceed a 30% success rate for the easiest of our tasks, and 0% for the hardest. In the meantime, the average success rate for human respondents is 92%. Consequently, a system capable of solving GAIA can be assessed in the context of t-AGI,4 noting that humans typically take between 6 minutes for the simplest questions to 17 minutes for the most complex ones. From a related perspective, such system would arguably be a competent General AI within the framework recently proposed in Morris et al. (2023), which also appear to be the next milestone in AI research since ChatGPT (OpenAI, 2023) is one level below. This paper covers the composition of GAIA, its design choices, and explain how to craft questions and the associated challenges so that the community can further extend the benchmark to target emerging questions such as safety associated to tool use, or multi-modality.}{1}</refined_source_index><refined_source_index>{We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA’s philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system’s capability to exhibit similar robustness as the average human does on such questions. Using GAIA’s methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board hereby accessible.}{1}</refined_source_index><refined_source_index>{Design choices. GAIA results both from the need for revised AI benchmarks, and the observed shortcomings of LLM evaluation.}{1}</refined_source_index><refined_source_index>{We move in that direction by proposing GAIA, a benchmark for General AI Assistants featuring 466 carefully crafted questions and their answer, along with the associated design methodology. Our questions are easy to create, challenging for AI systems—for LLMs, most require complex generations—, yet admit a unique, factual answer, allowing a simple and robust automatic evaluation.}{1}</refined_source_index><refined_source_index>{GAIA: A Benchmark for General AI Assistants}{1}</refined_source_index>

**Retrieving sources done ...**



**Processing image sources ...**



**Processing image sources done ...**



**Retrieving source annotations done ...**



**Generating follow-up questions ...**

<followup_question>How can GAIA influence AI training strategies?</followup_question>

<followup_question>What are the limitations of current AI training methods?</followup_question>

<followup_question>In what scenarios would GAIA's insights be most beneficial?</followup_question>



**Generating follow-up questions done ...**

</appendix>
"""
    
    print("\nTest Case 4:")
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_4)
    print("Thinking content extracted:", thinking)
    print("Answer content extracted:", answer)
    print("Sources extracted:", sources)
    print("Source Pages extracted:", source_pages)
    print("Refined Source Pages extracted:", refined_source_pages)
    print("Refined Source Index extracted:", refined_source_index)
    print("Follow-up Questions extracted:", follow_up_questions)

if __name__ == "__main__":
    test_extract_answer_content()