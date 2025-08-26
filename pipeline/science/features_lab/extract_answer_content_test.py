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
    source_annotations = {}    # {source_annotation_string: source_annotation_data}
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

    # Extract source annotations (content between <source_annotations> tags)
    source_annotation_matches = re.finditer(r'<source_annotations>(.*?)</source_annotations>', message_content, re.DOTALL)
    for match in source_annotation_matches:
        source_annotation_content = match.group(1).strip()
        # Extract the content and JSON data
        # The format is {text content}{JSON data}
        annotation_match = re.match(r'\{(.*?)\}\{(.*)\}', source_annotation_content, re.DOTALL)
        if annotation_match:
            text = annotation_match.group(1)
            json_str = annotation_match.group(2)
            
            # Make sure the JSON string is valid by checking for proper closing brace
            if not json_str.endswith('}'):
                json_str += '}'
                
            try:
                # Parse the JSON data directly from the string
                # Replace single quotes with double quotes for valid JSON
                # And handle Python bool literals
                processed_json = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
                data = {
                    "page_num": int(re.search(r'"page_num":\s*(\d+)', processed_json).group(1)),
                    "start_char": int(re.search(r'"start_char":\s*(\d+)', processed_json).group(1)),
                    "end_char": int(re.search(r'"end_char":\s*(\d+)', processed_json).group(1)),
                    "success": True if "true" in processed_json else False,
                    "similarity": float(re.search(r'"similarity":\s*([\d\.]+)', processed_json).group(1))
                }
                source_annotations[text] = data
            except Exception as e:
                logger.error(f"Failed to extract source annotation data: {e}")
                # Fall back to storing as string if parsing fails
                source_annotations[text] = json_str

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

    source_objects = []
    for index, (key, value) in enumerate(refined_source_pages.items()):
        source_objects.append({
            "index": index,
            "referenceString": key,
            "page": value,
            "refinedIndex": refined_source_index.get(key, {}),
            "sourceAnnotation": {
                "pageNum": source_annotations.get(key, {}).get("page_num", 0),
                "startChar": source_annotations.get(key, {}).get("start_char", 0),
                "endChar": source_annotations.get(key, {}).get("end_char", 0),
                "success": source_annotations.get(key, {}).get("success", True),
                "similarity": source_annotations.get(key, {}).get("similarity", 0.0)
            }
        })
    
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

    source_objects = []
    for index, (key, value) in enumerate(refined_source_pages.items()):
        source_objects.append({
            "index": index,
            "referenceString": key,
            "page": value,
            "refinedIndex": refined_source_index.get(key, {}),
            "sourceAnnotation": {
                "pageNum": source_annotations.get(key, {}).get("page_num", 0),
                "startChar": source_annotations.get(key, {}).get("start_char", 0),
                "endChar": source_annotations.get(key, {}).get("end_char", 0),
                "success": source_annotations.get(key, {}).get("success", True),
                "similarity": source_annotations.get(key, {}).get("similarity", 0.0)
            }
        })
    
    # Test case 3: Message with follow-up questions
    test_message_3 = """
    <thinking>Test thinking content</thinking>
    <original_response>Test answer</original_response>
    <followup_question>What are the key features of GAIA?</followup_question>
    <followup_question>How does GAIA compare to other benchmarks?</followup_question>
    <source_annotations>{High-rate remote entanglement between photon and matter-based qubits is essential for distributed quantum information processing. A key technique to increase the modest entangling rates of existing long-distance quantum networking approaches is multiplexing. Here, we demonstrate a temporally multiplexed ion-photon interface via rapid transport of a chain of nine calcium ions across 74 µm within 86 µs. The non-classical nature of the multiplexed photons is verified by measuring the second-order correlation function with an average value of g(2)(0) = 0.060(13). This indicates low crosstalk between the multiplexed modes, and can be reduced to negligible level once fiber coupling of single photons is incorporated. In addition, we characterize the motional degree-of-freedom of the ion crystal after transport and find that it is coherently excited to as much as ¯nα ≈110 for the center-of-mass mode. Our proof-of-principle implementation paves the way for large-scale quantum networking with trapped ions, but highlights some challenges that must be overcome.}{{'page_num': 0, 'start_char': 375, 'end_char': 1440, 'success': True, 'similarity': 0.9981220657276996}}</source_annotations><source_annotations>{where Pn is the occupation on the nth number state and encodes a convolution between a thermal and a coherent phonon distribution [44]. Ω(i) n is the Rabi frequency of the ith ion on the nth number state [45]. To verify the effectiveness of our approximation, we probe the sideband-cooled motional spectrum of the nine-ion chain before the transport and verify that only the COM mode is not cooled to near the ground state [39], for which we find a cooling limit of ¯nth = 4.0 ± 3.0. We also measure the electric-field noise induced heating and find a heating rate of 20 quanta / ms.(Fig. 4(a)), indicating that the remaining thermal population is likely limited by the COM mode heating which scales as ion number N [46]. Fig. 4(b) shows the carrier Rabi flopping after ion transport twice as slow as in Fig. 2(a). From numerical simulations (blue line), we find that the data can be explained well by a coherent state ¯nα = |α|2 ≈50 on the COM mode after the transport. Similarly, we perform the full-speed transport and the carrier Rabi flopping matches with COM coherent state with ¯nα ≈110 (Fig. 4(c)). As shown in the Rabi flopping plots, there is mismatch between the experimental data and numerical simulation at full speed, which could be due to thermal and coherent occupation of other modes and will require additional investigation.}{{'page_num': 3, 'start_char': 2547, 'end_char': 3886, 'success': True, 'similarity': 0.9798356982823002}}</source_annotations><source_annotations>{we note that further optimization can be done by energy self-neutral shuttling [31, 44], implementing closed-loop optimization of the shuttling function [48], etc. To summarize, we have presented a multiplexed ionphoton interface by transporting a nine-ion chain with synchronized excitation in sub-hundred µs. The speed is restricted by the motional frequency and can be increased by an order of magnitude, for instance, using a 3D-printed ion trap [49] with radial frequency beyond 10 MHz. The 397 nm photon can be converted to the telecommunication band via a two-step QFC [24]. Once integrated with state preparation on 32D3/2 Zeeman sublevel and photon collection with a single mode fiber, we expect a faster photon extraction rate [50] and negligible ion crosstalk while achieving high fidelity ion-photon entanglement [51, 52]. Our system can also be combined with a miniature cavity [35] for much higher photon extraction efficiency without sacrificing the photon generation rate, while the ion's positional spread caused by coherent excitation can be mitigated by aligning the cavity along the radial direction or further optimization of the shuttling function. These results stimulate the research of fast shuttling of a chain of tens of ions as a unit cell of logical qubit with heralded entanglement [28] and highrates entanglement of quantum processors across large distances.}{{'page_num': 4, 'start_char': 2, 'end_char': 1385, 'success': True, 'similarity': 0.9884309472161966}}</source_annotations><source_annotations>{FIG. 4. |↓⟩↔|↑⟩carrier excitation of nine-ion chain before and after shuttling. The horizontal axis is the global 729 nm beam probe time, and the vertical axis is the average ion excitation on the |↑⟩state. Error bars denote one standard deviation of the quantum projection noise. (a) Rabi oscillations of the sideband-cooled ions (red dots). The red line is a numerical simulation of thermal distribution with ¯nth = 4.0 ± 3.0. (b) Rabi oscillation after the transport at half speed of the transport function in Fig. 2(a). The blue line is a numerical simulation with with ¯nth = 4, ¯nα = 50 ± 5. (c) Rabi oscillation after the transport at full speed. The green line is a numerical simulation with with ¯nth = 4, ¯nα = 110 ± 5.}{{'page_num': 3, 'start_char': 14, 'end_char': 743, 'success': True, 'similarity': 1.0}}</source_annotations><source_annotations>{processing capability beside their natural interface with light at convenient wavelengths for quantum frequency conversion (QFC) [23, 24], and the possibility of longlived storage of entanglement [25, 26]. On the other hand, implementing a multiplexed light-matter interface with these systems is technically challenging. Towards overcoming this problem, a few multiplexing schemes have already been proposed for ion and atom-based quantum processors [27–30]. The only reported experimental work, we are aware of, is the demonstration of multiplexing using a static three-ion chain [15]. In view of the recent advances of the quantum CCD architecture [31–33], a complementary approach to multiplexing is the process of ion-transport through a specific spatial location with maximized photon coupling efficiency.}{{'page_num': 0, 'start_char': 3422, 'end_char': 4232, 'success': True, 'similarity': 0.9876543209876543}}</source_annotations><source_annotations>{frequencies of 1.9 MHz to allow fast transport of the ion chain close to the speed of the COM mode frequency. The programmed and the measured waveform show a negligible latency effect from the filters (Fig. 2(a)). The forward shuttling function has eight steps, during each of which a different ion is placed in the focus of the addressing beam for 1.7 µs with the beam turned on simultaneously. After completing this sequence, we move the entire ion chain back to the original position in 35 µs using the same function form in one step. The voltage ramping function on the two endcaps V1,2(t) is in the form of a sigmoid-like polynomial function such that the first and second order derivative at the beginning and the end of the transport vanish [38].}{{'page_num': 2, 'start_char': 1461, 'end_char': 2213, 'success': True, 'similarity': 0.9946808510638298}}</source_annotations><source_annotations>{FIG. 1. Schematics of multiplexed ion-photon interface. (a) A nine-ion chain is confined in an RF Paul trap. Controlling DC endcap voltages allows for ion transport. A beam of 397 nm and 866 nm light illuminating all ions is used for Doppler cooling. An objective collects the 397 nm single photons and guides them to a 50/50 beamsplitter, followed by a photomultiplier tube on each exit port for photon detection. An 866 nm beam individual addressing beam counter-propagates with the single photons. (b), (c) Excitation scheme and pulse sequence for the 397 nm single-photon generation. First, a global 397 nm beam prepares the ions to the 32D3/2 state. Then, the 866 nm addressing beam (resonance with 32D3/2 ↔42P1/2) is stroboscopically switched on during the transport to extract photons from the target ions.}{{'page_num': 1, 'start_char': 358, 'end_char': 1171, 'success': True, 'similarity': 0.985239852398524}}</source_annotations>
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
    print("Sources annotations extracted:", source_annotations.items())
    print(type(source_annotations["High-rate remote entanglement between photon and matter-based qubits is essential for distributed quantum information processing. A key technique to increase the modest entangling rates of existing long-distance quantum networking approaches is multiplexing. Here, we demonstrate a temporally multiplexed ion-photon interface via rapid transport of a chain of nine calcium ions across 74 µm within 86 µs. The non-classical nature of the multiplexed photons is verified by measuring the second-order correlation function with an average value of g(2)(0) = 0.060(13). This indicates low crosstalk between the multiplexed modes, and can be reduced to negligible level once fiber coupling of single photons is incorporated. In addition, we characterize the motional degree-of-freedom of the ion crystal after transport and find that it is coherently excited to as much as ¯nα ≈110 for the center-of-mass mode. Our proof-of-principle implementation paves the way for large-scale quantum networking with trapped ions, but highlights some challenges that must be overcome."]))
    print(source_annotations["High-rate remote entanglement between photon and matter-based qubits is essential for distributed quantum information processing. A key technique to increase the modest entangling rates of existing long-distance quantum networking approaches is multiplexing. Here, we demonstrate a temporally multiplexed ion-photon interface via rapid transport of a chain of nine calcium ions across 74 µm within 86 µs. The non-classical nature of the multiplexed photons is verified by measuring the second-order correlation function with an average value of g(2)(0) = 0.060(13). This indicates low crosstalk between the multiplexed modes, and can be reduced to negligible level once fiber coupling of single photons is incorporated. In addition, we characterize the motional degree-of-freedom of the ion crystal after transport and find that it is coherently excited to as much as ¯nα ≈110 for the center-of-mass mode. Our proof-of-principle implementation paves the way for large-scale quantum networking with trapped ions, but highlights some challenges that must be overcome."]['start_char'])
    print("Sources annotations extracted (first item):", next(iter(source_annotations.items())))

    source_objects = []
    for index, (key, value) in enumerate(refined_source_pages.items()):
        source_objects.append({
            "index": index,
            "referenceString": key,
            "page": value,
            "refinedIndex": refined_source_index.get(key, {}),
            "sourceAnnotation": {
                "pageNum": source_annotations.get(key, {}).get("page_num", 0),
                "startChar": source_annotations.get(key, {}).get("start_char", 0),
                "endChar": source_annotations.get(key, {}).get("end_char", 0),
                "success": source_annotations.get(key, {}).get("success", True),
                "similarity": source_annotations.get(key, {}).get("similarity", 0.0)
            }
        })
    
    # Test case 4: Empty message
    test_message_4 = """
<thinking>Processing documents ...



**📙 Loading documents done ...**



**🔍 Loading RAG embeddings ...**

**🔍 Loading RAG embeddings ...**

**🔍 Loading RAG embeddings ...**

**🔍 Loading RAG embeddings ...**

**🔍 Loading RAG embeddings ...**

**🔍 Loading RAG embeddings ...**

**🔍 Loading RAG embeddings ...**

**🔍 RAG embeddings ready ...**</thinking>

**🧠 Loading response ...**



# **📚 Loading PDF content with distributed word budget ...**



**📚 PDF content loading complete ...**

<response>Summary

The document presents a lightweight and real-time human fall detection system designed to address the increasing need for in-home medical care for the elderly, given the global rise in the aged population and the shortage of caretakers [<1>][<8>]. The proposed system leverages pose estimation, specifically using the 'Movenet Thunder' model, to extract human joint key-points from video input, enabling accurate fall detection without the need for high computing power [<1>][<3>][<5>]. This approach allows the system to operate efficiently on low-computing devices such as standard laptops, desktops, or mobile phones, processing video at over 30 frames per second in real time [<2>][<3>].

A key advantage of the system is its ability to perform all computations locally, ensuring that no personal image or video data is transmitted to the cloud, thereby preserving user privacy; only the fall detection output is sent to caretaker centers for necessary medical intervention [<2>][<3>]. The system was evaluated using two datasets, including a newly introduced GMDCSA dataset, and demonstrated high sensitivity values (0.9375 for GMDCSA and 0.9167 for URFD), indicating robust performance despite its low computational requirements [<2>][<3>][<5>].

The document also highlights the broader context of fall detection technologies, noting that vision-based systems are particularly suitable for elderly care as they do not require wearable sensors, which can be inconvenient and require frequent charging [<8>]. The proposed method stands out among state-of-the-art techniques by balancing accuracy, speed, privacy, and accessibility, making it a practical solution for real-world deployment in home environments [<1>][<3>][<5>].</response><appendix>

**🔍 Retrieving sources ...**

<source>{Abstract. The elderly population is increasing rapidly around the world. There are no enough caretakers for them. Use of AI-based in-home medical care systems is gaining momentum due to this. Human fall detection is one of the most important tasks of medical care system for the aged people. Human fall is a common problem among elderly people. Detection of a fall and providing medical help as early as possible is very important to reduce any further complexity. The chances of death and other medical complications can be reduced by detecting and providing medical help as early as possible after the fall. There are many state-ofthe-art fall detection techniques available these days, but the majority of them need very high computing power. In this paper, we proposed a lightweight and fast human fall detection system using pose estimation. We used ‘Movenet’ for human joins key-points extraction. Our proposed method can work in real-time on any low-computing device with any basic camera.}{0.0}</source><source>{Real Time ‘Movenet’ processes the video with 30+ FPS [9] (real-time) in the majority of current low computing devices like mobile phones, laptops, and desktops. So the proposed system can work in real-time on these devices. We tested our work on an average computing laptop with inbuilt webcam. Lightweight The proposed system does not required very high computing power and can work on any normal laptop/desktop or mobile device. Local Computation All computation can be processed locally. There is no personal data (images/frames) transfer from edge [10] to the cloud and vice versa. Only the output (fall) is sent to the caretaker center for necessary medical help. In this way, our system also preserves the privacy of the subject. GMDCSA Dataset A new fall detection dataset named GMDCSA was introduced.}{0.9279093518853188}</source><source>{In this paper, we proposed a lightweight and fast human fall detection system using ‘Movenet Thunder’ pose estimation. Our proposed system is very fast and requires very low computing power. It can run easily in real-time on any low-computing device like mobile, laptop, desktop, etc. All computation is done locally, so it also preserves the privacy of the subject. The metrics are also good enough considering the low computing requirement of the system. The proposed}{0.9898746674880385}</source><source>{Snidaro, Deep neural networks for real-time remote fall detection, in: International Conference on Pattern Recognition, Springer, 2021, pp. 188–201. 14. G. V. Leite, G. P. da Silva, H. Pedrini, Three-stream convolutional neural network for human fall detection, in: Deep Learning Applications, Volume 2, Springer, 2021, pp. 49–80. 15. Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, Realtime multi-person 2d pose estimation using part affinity fields, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 7291–7299. 16. T. Chen, Z. Ding, B. Li, Elderly fall detection based on improved yolov5s network, IEEE Access 10 (2022) 91273–91282. 17. Ultralytics, Yolov5, https://github.com/ultralytics/yolov5, [Accessed 14-Jan2023]. 18. W. Liu, X. Liu, Y. Hu, J. Shi, X. Chen, J. Zhao, S. Wang, Q. Hu, Fall detection for shipboard seafarers based on optimized blazepose and lstm, Sensors 22 (14) (2022) 5449. 19. D. R. Beddiar, M. Oussalah, B.}{0.9553488343954086}</source><source>{Keywords: Fall Detection, Pose Estimation, GMDCSA, Movenet, Lightweight Fall Detection, Real-time Fall Detection}{0.7393143773078918}</source><source>{8. R. Bajpai, D. Joshi, Movenet: A deep neural network for joint profile prediction across variable walking speeds and slopes, IEEE Transactions on Instrumentation and Measurement 70 (2021) 1–11. 9. MoveNet: Ultra fast and accurate pose detection model. — TensorFlow Hub — tensorflow.org, https://www.tensorflow.org/hub/tutorials/movenet, [Accessed 21Oct-2022]. 10. A. Sufian, E. Alam, A. Ghosh, F. Sultana, D. De, M. Dong, Deep learning in computer vision through mobile edge computing for iot, in: Mobile Edge Computing, Springer, 2021, pp. 443–471. 11. U. Asif, S. Von Cavallar, J. Tang, S. Harrer, Sshfd: Single shot human fall detection with occluded joints resilience, arXiv preprint arXiv:2004.00797 (2020). 12. Z. Chen, Y. Wang, W. Yang, Video based fall detection using human poses, in: CCF Conference on Big Data, Springer, 2022, pp. 283–296. 13. A. Apicella, L.}{0.9629828110337257}</source><source>{Nini, Fall detection using body geometry and human pose estimation in video sequences, Journal of Visual Communication and Image Representation 82 (2022) 103407. 20. M. Amsaprabhaa, et al., Multimodal spatiotemporal skeletal kinematic gait feature fusion for vision-based fall detection, Expert Systems with Applications 212 (2023) 118681. 21. B. Kwolek, M. Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer methods and programs in biomedicine 117 (3) (2014) 489–501. 22. M. Kocabas, S. Karagoz, E. Akbas, Multiposenet: Fast multi-person pose estimation using pose residual network, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 417–433.}{0.9820577725768089}</source><source>{Human fall is one of the major reasons for hospitalization in elder people around the world [1]. Detection of human falls is very vital so that medical help can be provided as early as possible. Human fall detection can be done using wearable, ambient, or vision sensors [2]. Vision-based fall detection system is more suitable, especially for elder people [3]. There is no need to attach the vision sensor to the body like wearable sensors. Wearable sensors need to be charged frequently whereas vision sensors can work on a direct home power supply. Human fall detection is one of the useful application of computer vision [4], [5]. In this paper, we have proposed a lightweight human fall detection system using pose}{1.0}</source><source>{1. E. Alam, A. Sufian, P. Dutta, M. Leo, Vision-based human fall detection systems using deep learning: A review, Computers in Biology and Medicine (2022) 105626. 2. Z. Wang, V. Ramamoorthy, U. Gal, A. Guez, Possible life saver: A review on human fall detection technology, Robotics 9 (3) (2020) 55. 3. J. Guti´errez, V. Rodr´ıguez, S. Martin, Comprehensive review of vision-based fall detection systems, Sensors 21 (3) (2021) 947. 4. E. Alam, A. Sufian, A. K. Das, A. Bhattacharya, M. F. Ali, M. H. Rahman, Leveraging deep learning for computer vision: A review, in: 2021 22nd International Arab Conference on Information Technology (ACIT), IEEE, 2021, pp. 1–8. 5. X. Wang, J. Ellul, G. Azzopardi, Elderly fall detection systems: A literature survey, Frontiers in Robotics and AI 7 (2020) 71. 6. T. L. Munea, Y. Z. Jembre, H. T. Weldegebriel, L. Chen, C. Huang, C.}{0.9328219667077065}</source><source_page>{Abstract. The elderly population is increasing rapidly around the world. There are no enough caretakers for them. Use of AI-based in-home medical care systems is gaining momentum due to this. Human fall detection is one of the most important tasks of medical care system for the aged people. Human fall is a common problem among elderly people. Detection of a fall and providing medical help as early as possible is very important to reduce any further complexity. The chances of death and other medical complications can be reduced by detecting and providing medical help as early as possible after the fall. There are many state-ofthe-art fall detection techniques available these days, but the majority of them need very high computing power. In this paper, we proposed a lightweight and fast human fall detection system using pose estimation. We used ‘Movenet’ for human joins key-points extraction. Our proposed method can work in real-time on any low-computing device with any basic camera.}{0}</source_page><source_page>{Real Time ‘Movenet’ processes the video with 30+ FPS [9] (real-time) in the majority of current low computing devices like mobile phones, laptops, and desktops. So the proposed system can work in real-time on these devices. We tested our work on an average computing laptop with inbuilt webcam. Lightweight The proposed system does not required very high computing power and can work on any normal laptop/desktop or mobile device. Local Computation All computation can be processed locally. There is no personal data (images/frames) transfer from edge [10] to the cloud and vice versa. Only the output (fall) is sent to the caretaker center for necessary medical help. In this way, our system also preserves the privacy of the subject. GMDCSA Dataset A new fall detection dataset named GMDCSA was introduced.}{1}</source_page><source_page>{In this paper, we proposed a lightweight and fast human fall detection system using ‘Movenet Thunder’ pose estimation. Our proposed system is very fast and requires very low computing power. It can run easily in real-time on any low-computing device like mobile, laptop, desktop, etc. All computation is done locally, so it also preserves the privacy of the subject. The metrics are also good enough considering the low computing requirement of the system. The proposed}{7}</source_page><source_page>{Snidaro, Deep neural networks for real-time remote fall detection, in: International Conference on Pattern Recognition, Springer, 2021, pp. 188–201. 14. G. V. Leite, G. P. da Silva, H. Pedrini, Three-stream convolutional neural network for human fall detection, in: Deep Learning Applications, Volume 2, Springer, 2021, pp. 49–80. 15. Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, Realtime multi-person 2d pose estimation using part affinity fields, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 7291–7299. 16. T. Chen, Z. Ding, B. Li, Elderly fall detection based on improved yolov5s network, IEEE Access 10 (2022) 91273–91282. 17. Ultralytics, Yolov5, https://github.com/ultralytics/yolov5, [Accessed 14-Jan2023]. 18. W. Liu, X. Liu, Y. Hu, J. Shi, X. Chen, J. Zhao, S. Wang, Q. Hu, Fall detection for shipboard seafarers based on optimized blazepose and lstm, Sensors 22 (14) (2022) 5449. 19. D. R. Beddiar, M. Oussalah, B.}{9}</source_page><source_page>{Keywords: Fall Detection, Pose Estimation, GMDCSA, Movenet, Lightweight Fall Detection, Real-time Fall Detection}{0}</source_page><source_page>{8. R. Bajpai, D. Joshi, Movenet: A deep neural network for joint profile prediction across variable walking speeds and slopes, IEEE Transactions on Instrumentation and Measurement 70 (2021) 1–11. 9. MoveNet: Ultra fast and accurate pose detection model. — TensorFlow Hub — tensorflow.org, https://www.tensorflow.org/hub/tutorials/movenet, [Accessed 21Oct-2022]. 10. A. Sufian, E. Alam, A. Ghosh, F. Sultana, D. De, M. Dong, Deep learning in computer vision through mobile edge computing for iot, in: Mobile Edge Computing, Springer, 2021, pp. 443–471. 11. U. Asif, S. Von Cavallar, J. Tang, S. Harrer, Sshfd: Single shot human fall detection with occluded joints resilience, arXiv preprint arXiv:2004.00797 (2020). 12. Z. Chen, Y. Wang, W. Yang, Video based fall detection using human poses, in: CCF Conference on Big Data, Springer, 2022, pp. 283–296. 13. A. Apicella, L.}{9}</source_page><source_page>{Nini, Fall detection using body geometry and human pose estimation in video sequences, Journal of Visual Communication and Image Representation 82 (2022) 103407. 20. M. Amsaprabhaa, et al., Multimodal spatiotemporal skeletal kinematic gait feature fusion for vision-based fall detection, Expert Systems with Applications 212 (2023) 118681. 21. B. Kwolek, M. Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer methods and programs in biomedicine 117 (3) (2014) 489–501. 22. M. Kocabas, S. Karagoz, E. Akbas, Multiposenet: Fast multi-person pose estimation using pose residual network, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 417–433.}{9}</source_page><source_page>{Human fall is one of the major reasons for hospitalization in elder people around the world [1]. Detection of human falls is very vital so that medical help can be provided as early as possible. Human fall detection can be done using wearable, ambient, or vision sensors [2]. Vision-based fall detection system is more suitable, especially for elder people [3]. There is no need to attach the vision sensor to the body like wearable sensors. Wearable sensors need to be charged frequently whereas vision sensors can work on a direct home power supply. Human fall detection is one of the useful application of computer vision [4], [5]. In this paper, we have proposed a lightweight human fall detection system using pose}{0}</source_page><source_page>{1. E. Alam, A. Sufian, P. Dutta, M. Leo, Vision-based human fall detection systems using deep learning: A review, Computers in Biology and Medicine (2022) 105626. 2. Z. Wang, V. Ramamoorthy, U. Gal, A. Guez, Possible life saver: A review on human fall detection technology, Robotics 9 (3) (2020) 55. 3. J. Guti´errez, V. Rodr´ıguez, S. Martin, Comprehensive review of vision-based fall detection systems, Sensors 21 (3) (2021) 947. 4. E. Alam, A. Sufian, A. K. Das, A. Bhattacharya, M. F. Ali, M. H. Rahman, Leveraging deep learning for computer vision: A review, in: 2021 22nd International Arab Conference on Information Technology (ACIT), IEEE, 2021, pp. 1–8. 5. X. Wang, J. Ellul, G. Azzopardi, Elderly fall detection systems: A literature survey, Frontiers in Robotics and AI 7 (2020) 71. 6. T. L. Munea, Y. Z. Jembre, H. T. Weldegebriel, L. Chen, C. Huang, C.}{8}</source_page><refined_source_page>{Abstract. The elderly population is increasing rapidly around the world. There are no enough caretakers for them. Use of AI-based in-home medical care systems is gaining momentum due to this. Human fall detection is one of the most important tasks of medical care system for the aged people. Human fall is a common problem among elderly people. Detection of a fall and providing medical help as early as possible is very important to reduce any further complexity. The chances of death and other medical complications can be reduced by detecting and providing medical help as early as possible after the fall. There are many state-ofthe-art fall detection techniques available these days, but the majority of them need very high computing power. In this paper, we proposed a lightweight and fast human fall detection system using pose estimation. We used ‘Movenet’ for human joins key-points extraction. Our proposed method can work in real-time on any low-computing device with any basic camera.}{1}</refined_source_page><refined_source_page>{Real Time ‘Movenet’ processes the video with 30+ FPS [9] (real-time) in the majority of current low computing devices like mobile phones, laptops, and desktops. So the proposed system can work in real-time on these devices. We tested our work on an average computing laptop with inbuilt webcam. Lightweight The proposed system does not required very high computing power and can work on any normal laptop/desktop or mobile device. Local Computation All computation can be processed locally. There is no personal data (images/frames) transfer from edge [10] to the cloud and vice versa. Only the output (fall) is sent to the caretaker center for necessary medical help. In this way, our system also preserves the privacy of the subject. GMDCSA Dataset A new fall detection dataset named GMDCSA was introduced.}{2}</refined_source_page><refined_source_page>{In this paper, we proposed a lightweight and fast human fall detection system using ‘Movenet Thunder’ pose estimation. Our proposed system is very fast and requires very low computing power. It can run easily in real-time on any low-computing device like mobile, laptop, desktop, etc. All computation is done locally, so it also preserves the privacy of the subject. The metrics are also good enough considering the low computing requirement of the system. The proposed}{8}</refined_source_page><refined_source_page>{Snidaro, Deep neural networks for real-time remote fall detection, in: International Conference on Pattern Recognition, Springer, 2021, pp. 188–201. 14. G. V. Leite, G. P. da Silva, H. Pedrini, Three-stream convolutional neural network for human fall detection, in: Deep Learning Applications, Volume 2, Springer, 2021, pp. 49–80. 15. Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, Realtime multi-person 2d pose estimation using part affinity fields, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 7291–7299. 16. T. Chen, Z. Ding, B. Li, Elderly fall detection based on improved yolov5s network, IEEE Access 10 (2022) 91273–91282. 17. Ultralytics, Yolov5, https://github.com/ultralytics/yolov5, [Accessed 14-Jan2023]. 18. W. Liu, X. Liu, Y. Hu, J. Shi, X. Chen, J. Zhao, S. Wang, Q. Hu, Fall detection for shipboard seafarers based on optimized blazepose and lstm, Sensors 22 (14) (2022) 5449. 19. D. R. Beddiar, M. Oussalah, B.}{10}</refined_source_page><refined_source_page>{Keywords: Fall Detection, Pose Estimation, GMDCSA, Movenet, Lightweight Fall Detection, Real-time Fall Detection}{1}</refined_source_page><refined_source_page>{8. R. Bajpai, D. Joshi, Movenet: A deep neural network for joint profile prediction across variable walking speeds and slopes, IEEE Transactions on Instrumentation and Measurement 70 (2021) 1–11. 9. MoveNet: Ultra fast and accurate pose detection model. — TensorFlow Hub — tensorflow.org, https://www.tensorflow.org/hub/tutorials/movenet, [Accessed 21Oct-2022]. 10. A. Sufian, E. Alam, A. Ghosh, F. Sultana, D. De, M. Dong, Deep learning in computer vision through mobile edge computing for iot, in: Mobile Edge Computing, Springer, 2021, pp. 443–471. 11. U. Asif, S. Von Cavallar, J. Tang, S. Harrer, Sshfd: Single shot human fall detection with occluded joints resilience, arXiv preprint arXiv:2004.00797 (2020). 12. Z. Chen, Y. Wang, W. Yang, Video based fall detection using human poses, in: CCF Conference on Big Data, Springer, 2022, pp. 283–296. 13. A. Apicella, L.}{10}</refined_source_page><refined_source_page>{Nini, Fall detection using body geometry and human pose estimation in video sequences, Journal of Visual Communication and Image Representation 82 (2022) 103407. 20. M. Amsaprabhaa, et al., Multimodal spatiotemporal skeletal kinematic gait feature fusion for vision-based fall detection, Expert Systems with Applications 212 (2023) 118681. 21. B. Kwolek, M. Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer methods and programs in biomedicine 117 (3) (2014) 489–501. 22. M. Kocabas, S. Karagoz, E. Akbas, Multiposenet: Fast multi-person pose estimation using pose residual network, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 417–433.}{10}</refined_source_page><refined_source_page>{Human fall is one of the major reasons for hospitalization in elder people around the world [1]. Detection of human falls is very vital so that medical help can be provided as early as possible. Human fall detection can be done using wearable, ambient, or vision sensors [2]. Vision-based fall detection system is more suitable, especially for elder people [3]. There is no need to attach the vision sensor to the body like wearable sensors. Wearable sensors need to be charged frequently whereas vision sensors can work on a direct home power supply. Human fall detection is one of the useful application of computer vision [4], [5]. In this paper, we have proposed a lightweight human fall detection system using pose}{1}</refined_source_page><refined_source_page>{1. E. Alam, A. Sufian, P. Dutta, M. Leo, Vision-based human fall detection systems using deep learning: A review, Computers in Biology and Medicine (2022) 105626. 2. Z. Wang, V. Ramamoorthy, U. Gal, A. Guez, Possible life saver: A review on human fall detection technology, Robotics 9 (3) (2020) 55. 3. J. Guti´errez, V. Rodr´ıguez, S. Martin, Comprehensive review of vision-based fall detection systems, Sensors 21 (3) (2021) 947. 4. E. Alam, A. Sufian, A. K. Das, A. Bhattacharya, M. F. Ali, M. H. Rahman, Leveraging deep learning for computer vision: A review, in: 2021 22nd International Arab Conference on Information Technology (ACIT), IEEE, 2021, pp. 1–8. 5. X. Wang, J. Ellul, G. Azzopardi, Elderly fall detection systems: A literature survey, Frontiers in Robotics and AI 7 (2020) 71. 6. T. L. Munea, Y. Z. Jembre, H. T. Weldegebriel, L. Chen, C. Huang, C.}{9}</refined_source_page><refined_source_index>{Abstract. The elderly population is increasing rapidly around the world. There are no enough caretakers for them. Use of AI-based in-home medical care systems is gaining momentum due to this. Human fall detection is one of the most important tasks of medical care system for the aged people. Human fall is a common problem among elderly people. Detection of a fall and providing medical help as early as possible is very important to reduce any further complexity. The chances of death and other medical complications can be reduced by detecting and providing medical help as early as possible after the fall. There are many state-ofthe-art fall detection techniques available these days, but the majority of them need very high computing power. In this paper, we proposed a lightweight and fast human fall detection system using pose estimation. We used ‘Movenet’ for human joins key-points extraction. Our proposed method can work in real-time on any low-computing device with any basic camera.}{2}</refined_source_index><refined_source_index>{Real Time ‘Movenet’ processes the video with 30+ FPS [9] (real-time) in the majority of current low computing devices like mobile phones, laptops, and desktops. So the proposed system can work in real-time on these devices. We tested our work on an average computing laptop with inbuilt webcam. Lightweight The proposed system does not required very high computing power and can work on any normal laptop/desktop or mobile device. Local Computation All computation can be processed locally. There is no personal data (images/frames) transfer from edge [10] to the cloud and vice versa. Only the output (fall) is sent to the caretaker center for necessary medical help. In this way, our system also preserves the privacy of the subject. GMDCSA Dataset A new fall detection dataset named GMDCSA was introduced.}{2}</refined_source_index><refined_source_index>{In this paper, we proposed a lightweight and fast human fall detection system using ‘Movenet Thunder’ pose estimation. Our proposed system is very fast and requires very low computing power. It can run easily in real-time on any low-computing device like mobile, laptop, desktop, etc. All computation is done locally, so it also preserves the privacy of the subject. The metrics are also good enough considering the low computing requirement of the system. The proposed}{2}</refined_source_index><refined_source_index>{Snidaro, Deep neural networks for real-time remote fall detection, in: International Conference on Pattern Recognition, Springer, 2021, pp. 188–201. 14. G. V. Leite, G. P. da Silva, H. Pedrini, Three-stream convolutional neural network for human fall detection, in: Deep Learning Applications, Volume 2, Springer, 2021, pp. 49–80. 15. Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, Realtime multi-person 2d pose estimation using part affinity fields, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 7291–7299. 16. T. Chen, Z. Ding, B. Li, Elderly fall detection based on improved yolov5s network, IEEE Access 10 (2022) 91273–91282. 17. Ultralytics, Yolov5, https://github.com/ultralytics/yolov5, [Accessed 14-Jan2023]. 18. W. Liu, X. Liu, Y. Hu, J. Shi, X. Chen, J. Zhao, S. Wang, Q. Hu, Fall detection for shipboard seafarers based on optimized blazepose and lstm, Sensors 22 (14) (2022) 5449. 19. D. R. Beddiar, M. Oussalah, B.}{2}</refined_source_index><refined_source_index>{Keywords: Fall Detection, Pose Estimation, GMDCSA, Movenet, Lightweight Fall Detection, Real-time Fall Detection}{2}</refined_source_index><refined_source_index>{8. R. Bajpai, D. Joshi, Movenet: A deep neural network for joint profile prediction across variable walking speeds and slopes, IEEE Transactions on Instrumentation and Measurement 70 (2021) 1–11. 9. MoveNet: Ultra fast and accurate pose detection model. — TensorFlow Hub — tensorflow.org, https://www.tensorflow.org/hub/tutorials/movenet, [Accessed 21Oct-2022]. 10. A. Sufian, E. Alam, A. Ghosh, F. Sultana, D. De, M. Dong, Deep learning in computer vision through mobile edge computing for iot, in: Mobile Edge Computing, Springer, 2021, pp. 443–471. 11. U. Asif, S. Von Cavallar, J. Tang, S. Harrer, Sshfd: Single shot human fall detection with occluded joints resilience, arXiv preprint arXiv:2004.00797 (2020). 12. Z. Chen, Y. Wang, W. Yang, Video based fall detection using human poses, in: CCF Conference on Big Data, Springer, 2022, pp. 283–296. 13. A. Apicella, L.}{2}</refined_source_index><refined_source_index>{Nini, Fall detection using body geometry and human pose estimation in video sequences, Journal of Visual Communication and Image Representation 82 (2022) 103407. 20. M. Amsaprabhaa, et al., Multimodal spatiotemporal skeletal kinematic gait feature fusion for vision-based fall detection, Expert Systems with Applications 212 (2023) 118681. 21. B. Kwolek, M. Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer methods and programs in biomedicine 117 (3) (2014) 489–501. 22. M. Kocabas, S. Karagoz, E. Akbas, Multiposenet: Fast multi-person pose estimation using pose residual network, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 417–433.}{2}</refined_source_index><refined_source_index>{Human fall is one of the major reasons for hospitalization in elder people around the world [1]. Detection of human falls is very vital so that medical help can be provided as early as possible. Human fall detection can be done using wearable, ambient, or vision sensors [2]. Vision-based fall detection system is more suitable, especially for elder people [3]. There is no need to attach the vision sensor to the body like wearable sensors. Wearable sensors need to be charged frequently whereas vision sensors can work on a direct home power supply. Human fall detection is one of the useful application of computer vision [4], [5]. In this paper, we have proposed a lightweight human fall detection system using pose}{2}</refined_source_index><refined_source_index>{1. E. Alam, A. Sufian, P. Dutta, M. Leo, Vision-based human fall detection systems using deep learning: A review, Computers in Biology and Medicine (2022) 105626. 2. Z. Wang, V. Ramamoorthy, U. Gal, A. Guez, Possible life saver: A review on human fall detection technology, Robotics 9 (3) (2020) 55. 3. J. Guti´errez, V. Rodr´ıguez, S. Martin, Comprehensive review of vision-based fall detection systems, Sensors 21 (3) (2021) 947. 4. E. Alam, A. Sufian, A. K. Das, A. Bhattacharya, M. F. Ali, M. H. Rahman, Leveraging deep learning for computer vision: A review, in: 2021 22nd International Arab Conference on Information Technology (ACIT), IEEE, 2021, pp. 1–8. 5. X. Wang, J. Ellul, G. Azzopardi, Elderly fall detection systems: A literature survey, Frontiers in Robotics and AI 7 (2020) 71. 6. T. L. Munea, Y. Z. Jembre, H. T. Weldegebriel, L. Chen, C. Huang, C.}{2}</refined_source_index><source_annotations>{Abstract. The elderly population is increasing rapidly around the world. There are no enough caretakers for them. Use of AI-based in-home medical care systems is gaining momentum due to this. Human fall detection is one of the most important tasks of medical care system for the aged people. Human fall is a common problem among elderly people. Detection of a fall and providing medical help as early as possible is very important to reduce any further complexity. The chances of death and other medical complications can be reduced by detecting and providing medical help as early as possible after the fall. There are many state-ofthe-art fall detection techniques available these days, but the majority of them need very high computing power. In this paper, we proposed a lightweight and fast human fall detection system using pose estimation. We used ‘Movenet’ for human joins key-points extraction. Our proposed method can work in real-time on any low-computing device with any basic camera.}{{'page_num': 0, 'start_char': 467, 'end_char': 1463, 'success': True, 'similarity': 0.9939759036144579}}</source_annotations><source_annotations>{Real Time ‘Movenet’ processes the video with 30+ FPS [9] (real-time) in the majority of current low computing devices like mobile phones, laptops, and desktops. So the proposed system can work in real-time on these devices. We tested our work on an average computing laptop with inbuilt webcam. Lightweight The proposed system does not required very high computing power and can work on any normal laptop/desktop or mobile device. Local Computation All computation can be processed locally. There is no personal data (images/frames) transfer from edge [10] to the cloud and vice versa. Only the output (fall) is sent to the caretaker center for necessary medical help. In this way, our system also preserves the privacy of the subject. GMDCSA Dataset A new fall detection dataset named GMDCSA was introduced.}{{'page_num': 1, 'start_char': 188, 'end_char': 994, 'success': True, 'similarity': 0.9925558312655087}}</source_annotations><source_annotations>{In this paper, we proposed a lightweight and fast human fall detection system using ‘Movenet Thunder’ pose estimation. Our proposed system is very fast and requires very low computing power. It can run easily in real-time on any low-computing device like mobile, laptop, desktop, etc. All computation is done locally, so it also preserves the privacy of the subject. The metrics are also good enough considering the low computing requirement of the system. The proposed}{{'page_num': 7, 'start_char': 1042, 'end_char': 1511, 'success': True, 'similarity': 1.0}}</source_annotations><source_annotations>{Snidaro, Deep neural networks for real-time remote fall detection, in: International Conference on Pattern Recognition, Springer, 2021, pp. 188–201. 14. G. V. Leite, G. P. da Silva, H. Pedrini, Three-stream convolutional neural network for human fall detection, in: Deep Learning Applications, Volume 2, Springer, 2021, pp. 49–80. 15. Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, Realtime multi-person 2d pose estimation using part affinity fields, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 7291–7299. 16. T. Chen, Z. Ding, B. Li, Elderly fall detection based on improved yolov5s network, IEEE Access 10 (2022) 91273–91282. 17. Ultralytics, Yolov5, https://github.com/ultralytics/yolov5, [Accessed 14-Jan2023]. 18. W. Liu, X. Liu, Y. Hu, J. Shi, X. Chen, J. Zhao, S. Wang, Q. Hu, Fall detection for shipboard seafarers based on optimized blazepose and lstm, Sensors 22 (14) (2022) 5449. 19. D. R. Beddiar, M. Oussalah, B.}{{'page_num': 9, 'start_char': 898, 'end_char': 1865, 'success': True, 'similarity': 0.9979317476732161}}</source_annotations><source_annotations>{Keywords: Fall Detection, Pose Estimation, GMDCSA, Movenet, Lightweight Fall Detection, Real-time Fall Detection}{{'page_num': 0, 'start_char': 1806, 'end_char': 1918, 'success': True, 'similarity': 1.0}}</source_annotations><source_annotations>{8. R. Bajpai, D. Joshi, Movenet: A deep neural network for joint profile prediction across variable walking speeds and slopes, IEEE Transactions on Instrumentation and Measurement 70 (2021) 1–11. 9. MoveNet: Ultra fast and accurate pose detection model. — TensorFlow Hub — tensorflow.org, https://www.tensorflow.org/hub/tutorials/movenet, [Accessed 21Oct-2022]. 10. A. Sufian, E. Alam, A. Ghosh, F. Sultana, D. De, M. Dong, Deep learning in computer vision through mobile edge computing for iot, in: Mobile Edge Computing, Springer, 2021, pp. 443–471. 11. U. Asif, S. Von Cavallar, J. Tang, S. Harrer, Sshfd: Single shot human fall detection with occluded joints resilience, arXiv preprint arXiv:2004.00797 (2020). 12. Z. Chen, Y. Wang, W. Yang, Video based fall detection using human poses, in: CCF Conference on Big Data, Springer, 2022, pp. 283–296. 13. A. Apicella, L.}{{'page_num': 9, 'start_char': 21, 'end_char': 893, 'success': True, 'similarity': 0.9954128440366973}}</source_annotations><source_annotations>{Nini, Fall detection using body geometry and human pose estimation in video sequences, Journal of Visual Communication and Image Representation 82 (2022) 103407. 20. M. Amsaprabhaa, et al., Multimodal spatiotemporal skeletal kinematic gait feature fusion for vision-based fall detection, Expert Systems with Applications 212 (2023) 118681. 21. B. Kwolek, M. Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer methods and programs in biomedicine 117 (3) (2014) 489–501. 22. M. Kocabas, S. Karagoz, E. Akbas, Multiposenet: Fast multi-person pose estimation using pose residual network, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 417–433.}{{'page_num': 9, 'start_char': 1868, 'end_char': 2593, 'success': True, 'similarity': 0.9972413793103448}}</source_annotations><source_annotations>{Human fall is one of the major reasons for hospitalization in elder people around the world [1]. Detection of human falls is very vital so that medical help can be provided as early as possible. Human fall detection can be done using wearable, ambient, or vision sensors [2]. Vision-based fall detection system is more suitable, especially for elder people [3]. There is no need to attach the vision sensor to the body like wearable sensors. Wearable sensors need to be charged frequently whereas vision sensors can work on a direct home power supply. Human fall detection is one of the useful application of computer vision [4], [5]. In this paper, we have proposed a lightweight human fall detection system using pose}{{'page_num': 0, 'start_char': 1934, 'end_char': 2648, 'success': True, 'similarity': 1.0}}</source_annotations><source_annotations>{1. E. Alam, A. Sufian, P. Dutta, M. Leo, Vision-based human fall detection systems using deep learning: A review, Computers in Biology and Medicine (2022) 105626. 2. Z. Wang, V. Ramamoorthy, U. Gal, A. Guez, Possible life saver: A review on human fall detection technology, Robotics 9 (3) (2020) 55. 3. J. Guti´errez, V. Rodr´ıguez, S. Martin, Comprehensive review of vision-based fall detection systems, Sensors 21 (3) (2021) 947. 4. E. Alam, A. Sufian, A. K. Das, A. Bhattacharya, M. F. Ali, M. H. Rahman, Leveraging deep learning for computer vision: A review, in: 2021 22nd International Arab Conference on Information Technology (ACIT), IEEE, 2021, pp. 1–8. 5. X. Wang, J. Ellul, G. Azzopardi, Elderly fall detection systems: A literature survey, Frontiers in Robotics and AI 7 (2020) 71. 6. T. L. Munea, Y. Z. Jembre, H. T. Weldegebriel, L. Chen, C. Huang, C.}{{'page_num': 8, 'start_char': 601, 'end_char': 1466, 'success': True, 'similarity': 1.0}}</source_annotations>
"""
    
    print("\nTest Case 4:")
    # answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_4)
    # print("Thinking content extracted:", thinking[:100] + "..." if thinking else "")
    # print("Answer content extracted:", answer[:100] + "..." if answer else "")
    # print("Sources extracted:", len(sources), "items")
    # print("Source Pages extracted:", len(source_pages), "items")
    # print("Source Annotations extracted:", len(source_annotations), "items")
    # print("Refined Source Pages extracted:", len(refined_source_pages), "items")
    # print("Refined Source Index extracted:", len(refined_source_index), "items")
    # print("Follow-up Questions extracted:", follow_up_questions)
    answer, sources, source_pages, source_annotations, refined_source_pages, refined_source_index, follow_up_questions, thinking = extract_answer_content(test_message_4)
    print("Thinking content extracted:", thinking)
    print("Answer content extracted:", answer)
    print("Sources extracted:", sources)
    print("Source Pages extracted:", source_pages)
    print("Refined Source Pages extracted:", refined_source_pages)
    print("Refined Source Index extracted:", refined_source_index.values())
    print("Follow-up Questions extracted:", follow_up_questions)
    print("Sources annotations extracted:", source_annotations.items())

    # Add the format check for sources_annotations
    source_objects = []
    for index, (key, value) in enumerate(refined_source_pages.items()):
        source_objects.append({
            "index": index,
            "referenceString": key,
            "page": value,
            "refinedIndex": refined_source_index.get(key, {}),
            "sourceAnnotation": {
                "pageNum": source_annotations.get(key, {}).get("page_num", 0),
                "startChar": source_annotations.get(key, {}).get("start_char", 0),
                "endChar": source_annotations.get(key, {}).get("end_char", 0),
                "success": source_annotations.get(key, {}).get("success", True),
                "similarity": source_annotations.get(key, {}).get("similarity", 0.0)
            }
        })

if __name__ == "__main__":
    test_extract_answer_content()