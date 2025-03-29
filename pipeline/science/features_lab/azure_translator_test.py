import requests, uuid, json
import os
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

def azure_translate_content(
    text: str,
    source_language: str = "en",
    target_languages: List[str] = ["zh"]
) -> Dict[str, Any]:
    """
    Translates text using the Azure Translator API.
    
    Args:
        text: The text to translate
        source_language: The source language code (default: "en" for English)
        target_languages: List of target language codes (default: ["zh"] for Chinese)
        
    Returns:
        Dict[str, Any]: Translation response from Azure
    
    Raises:
        ValueError: If Azure Translator credentials are not properly configured
        requests.RequestException: If the API request fails
    """

    load_dotenv()

    key = os.getenv("AZURE_TRANSLATOR_KEY")
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
    location = os.getenv("AZURE_TRANSLATOR_LOCATION")
    
    if not key or not endpoint or not location:
        raise ValueError("Azure Translator credentials not properly configured. Check your environment variables.")
    
    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': source_language,
        'to': target_languages
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text': text
    }]

    try:
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        request.raise_for_status()  # Raise exception for HTTP errors
        return request.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Translation request failed: {e}")

# Example usage
if __name__ == "__main__":
    sample_text = """<response>
The significance of the growing literature on efficiently computable but biased estimates of the softmax function lies in its critical role for advancing large-scale machine learning systems, particularly in the context of modern Large Language Models (LLMs) and their computational demands. Here's a structured breakdown:

### 1. Softmax Function and Its Role
The softmax function is a cornerstone of classification tasks in machine learning, converting raw model outputs (logits) into probability distributions. For example, in LLMs, it enables token prediction by assigning probabilities to the next word in a sequence. However, its computational cost scales with the number of classes (e.g., vocabulary size), becoming prohibitive for models with billions of parameters or massive vocabularies.

2. Need for Efficiency
As models grow larger (e.g., GPT-4, LLaMA), exact softmax computation becomes a bottleneck:

Memory and Speed: For LLMs with vocabularies of 100k+ tokens, exact softmax requires storing and computing probabilities for all tokens, which is infeasible in real-time applications.
Energy Costs: Training and inference on such models demand significant computational resources. Efficient approximations reduce energy consumption and hardware requirements.
3. Biased Estimates: Trade-offs and Implications
Biased estimation methods (e.g., sampled softmax, noise contrastive estimation) approximate the softmax by subsampling a subset of tokens or using hashing techniques. While these methods reduce computational costs, they introduce bias:

Performance Impact: Biases might distort probability distributions, leading to suboptimal predictions (e.g., favoring frequent tokens over rare ones).
Emergent Capabilities: As noted in the paper, LLMs are breaking benchmarks at an accelerating rate. Biased approximations could either hinder or unpredictably alter these capabilities, especially in tasks requiring nuanced reasoning or rare token usage.
4. Literature Contributions
Recent work addresses this trade-off by:

Theoretical Analysis: Quantifying bias-variance trade-offs to guide approximation choices.
Adaptive Methods: Dynamically adjusting approximation fidelity based on context (e.g., prioritizing critical tokens).
Benchmarking: Evaluating how approximations affect downstream tasks, aligning with the paper's emphasis on rethinking benchmarks for evolving AI systems.
5. Broader Context in AI
The literature reflects a broader trend in AI: balancing scalability with reliability. For instance:

Tool-Augmented LLMs: Systems like Auto-GPT (Yang et al., 2023) rely on efficient softmax approximations to interact with external tools in real time.
Benchmark Saturation: As models surpass human performance on static benchmarks (e.g., GLUE, MMLU), efficient approximations enable exploration of harder tasks (e.g., professional-level reasoning) while managing computational costs.
Conclusion
This literature is pivotal for enabling scalable, sustainable AI systems without sacrificing critical capabilities. By rigorously analyzing biased approximations, researchers aim to unlock efficient computation while preserving model integrity—a necessity as LLMs evolve into general-purpose systems with ever-expanding requirements.</response><appendix>"""
    
    response = azure_translate_content(sample_text)
    print(response[0]["translations"][0]["text"])