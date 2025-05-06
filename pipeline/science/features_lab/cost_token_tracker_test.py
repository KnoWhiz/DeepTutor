from langchain_openai import AzureChatOpenAI
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from typing import Any, Iterator, Tuple, TypedDict
import time

from dotenv import load_dotenv
import os
import json
import pathlib
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

class CostInfo(TypedDict):
    """Type definition for cost information returned by track_cost_and_stream."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    provider: str
    model: str
    time_elapsed: float
    cost_source: str  # Indicates whether cost came from callback or was calculated from config

def track_cost_and_stream(user_input: str, chain: Any) -> Tuple[Iterator[str], CostInfo]:
    """
    Tracks token usage and cost for a langchain chain and returns a streaming generator.
    
    Args:
        user_input: The user input to the chain
        chain: A langchain chain that supports streaming
        
    Returns:
        Tuple containing:
        - A generator that yields chunks from the chain's response
        - A dictionary with cost information including tokens, cost details, and time elapsed

        "callback" - Cost was directly obtained from the LLM provider's callback
        "config_calculation" - Cost was calculated using token counts and rates from the configuration file
        "config_not_found" - Provider/model information was not found in the config file
        "config_error" - An error occurred while trying to access the config file
        "unknown" - Default value if no cost calculation was performed
    """
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _initialise_cost_info() -> CostInfo:
        """Create a fresh cost-info dictionary with default values."""
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "provider": "",
            "model": "",
            "time_elapsed": 0.0,
            "cost_source": "unknown",
        }

    def _detect_provider(local_llm: Any) -> Tuple[str, str]:
        """Return provider and model identifiers for a given LLM instance."""
        if isinstance(local_llm, AzureChatOpenAI):
            return "azure", local_llm.deployment_name
        if isinstance(local_llm, ChatSambaNovaCloud):
            return "sambanova", local_llm.model
        return "unknown", str(type(local_llm))

    def _compute_cost_from_config(local_cost_info: CostInfo) -> None:
        """Populate *local_cost_info* with cost estimated from the JSON config file."""
        try:
            config_path = pathlib.Path(__file__).parents[2] / "science" / "pipeline" / "cost_config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                cost_config = json.load(f)

            provider = local_cost_info["provider"]
            model = local_cost_info["model"]

            model_config = (
                cost_config.get("providers", {})
                .get(provider, {})
                .get("models", {})
                .get(model)
            )

            if model_config is None:
                local_cost_info["cost_source"] = "config_not_found"
                return

            input_cost = model_config["input_cost_per_token"]
            output_cost = model_config["output_cost_per_token"]

            local_cost_info["total_cost"] = (
                local_cost_info["input_tokens"] * input_cost
                + local_cost_info["output_tokens"] * output_cost
            )
            local_cost_info["cost_source"] = "config_calculation"
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            local_cost_info["cost_source"] = "config_error"

    # ------------------------------------------------------------------
    # Main streaming generator
    # ------------------------------------------------------------------

    cost_info: CostInfo = _initialise_cost_info()

    # Detect provider/model once and cache it.
    detected_llm = chain.steps[-2] if hasattr(chain, "steps") and len(chain.steps) > 1 else None
    cost_info["provider"], cost_info["model"] = _detect_provider(detected_llm)

    def _stream() -> Iterator[str]:
        """Wrapper around *chain.stream* that records cost and timing information."""
        start_time = time.time()
        with get_openai_callback() as cb:
            try:
                for chunk in chain.stream({"user_input": user_input}):
                    yield chunk
            finally:
                # Always capture timing & cost, even on error/early exit
                cost_info["time_elapsed"] = time.time() - start_time

                # Populate metrics from callback
                cost_info["input_tokens"] = cb.prompt_tokens
                cost_info["output_tokens"] = cb.completion_tokens
                cost_info["total_tokens"] = cb.total_tokens
                cost_info["total_cost"] = cb.total_cost

                if cb.total_cost > 0:
                    cost_info["cost_source"] = "callback"
                elif cost_info["provider"] != "unknown":
                    _compute_cost_from_config(cost_info)

    # ------------------------------------------------------------------
    return _stream(), cost_info

# Example usage
if __name__ == "__main__":
    # Prepare the four model instances we want to exercise.
    llm_instances = [
        (
            "gpt-4o",
            AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                openai_api_version="2024-08-01-preview",
                deployment_name="gpt-4o",
                temperature=0,
                streaming=True,
                model_kwargs={"stream_options": {"include_usage": True}},
            ),
        ),
        (
            "gpt-4o-mini",
            AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                openai_api_version="2024-07-01-preview",
                azure_deployment="gpt-4o-mini",
                temperature=0,
                streaming=True,
                model_kwargs={"stream_options": {"include_usage": True}},
            ),
        ),
        (
            "Meta-Llama-3.3-70B-Instruct",
            ChatSambaNovaCloud(
                model="Meta-Llama-3.3-70B-Instruct",
                api_key=os.getenv("SAMBANOVA_API_KEY"),
                base_url="https://api.sambanova.ai/v1",
                max_tokens=1024,
                temperature=0,
                top_p=0.1,
                streaming=True,
                model_kwargs={"stream_options": {"include_usage": True}},
            ),
        ),
        (
            "DeepSeek-R1",
            ChatSambaNovaCloud(
                model="DeepSeek-R1",
                api_key=os.getenv("SAMBANOVA_API_KEY"),
                base_url="https://preview.snova.ai/v1",
                max_tokens=1024,
                temperature=0,
                top_p=0.1,
                streaming=True,
                model_kwargs={"stream_options": {"include_usage": True}},
            ),
        ),
        (
            "DeepSeek-R1-Distill-Llama-70B",
            ChatSambaNovaCloud(
                model="DeepSeek-R1-Distill-Llama-70B",
                api_key=os.getenv("SAMBANOVA_API_KEY"),
                base_url="https://api.sambanova.ai/v1",
                max_tokens=1024,
                temperature=0,
                top_p=0.1,
                streaming=True,
                model_kwargs={"stream_options": {"include_usage": True}},
            ),
        ),
        (
            "o3-mini",
            AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
                api_version="2024-12-01-preview",
                deployment_name="o3-mini",
                model_kwargs={"stream_options": {"include_usage": True}, "max_completion_tokens": 100000},
                streaming=True,
            ),
        ),
        (
            "o4-mini",
            AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
                api_version="2024-12-01-preview",
                deployment_name="o4-mini",
                model_kwargs={"stream_options": {"include_usage": True}, "max_completion_tokens": 100000},
                streaming=True,
            ),
        ),
        (
            "gpt-4.1",
            AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
                api_version="2024-12-01-preview",
                deployment_name="gpt-4.1",
                model_kwargs={"stream_options": {"include_usage": True}},
                streaming=True,
            ),
        ),
        (
            "gpt-4.1-mini",
            AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
                api_version="2024-12-01-preview",
                deployment_name="gpt-4.1-mini",
                model_kwargs={"stream_options": {"include_usage": True}},
                streaming=True,
            ),
        ),
    ]

    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("""what is {user_input}? explain the answer in great detail""")

    for model_name, llm in llm_instances[:]:
        print("=" * 80)
        print(f"Testing model: {model_name}")
        chain = prompt | llm | parser

        stream_generator, cost_info = track_cost_and_stream(user_input="1+1=?", chain=chain)

        print("Streaming response:")
        for chunk in stream_generator:
            print(chunk, end="", flush=True)

        print("\n\nCost Information:")
        print(json.dumps(cost_info, indent=2))
        print(f"Time elapsed: {cost_info['time_elapsed']:.4f} seconds\n")