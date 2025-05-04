from langchain_openai import AzureChatOpenAI
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.tracers import ConsoleCallbackHandler
from typing import Dict, Any, Generator, Tuple, Iterator, TypedDict, Union, cast
import time

from dotenv import load_dotenv
import os
import json
import pathlib
from functools import wraps
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
    # Dictionary to store cost information
    cost_info: CostInfo = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "provider": "",
        "model": "",
        "time_elapsed": 0.0,
        "cost_source": "unknown"  # Initialize as unknown
    }
    
    # Get the LLM from the chain to determine provider and model
    llm = chain.steps[-2] if hasattr(chain, "steps") and len(chain.steps) > 1 else None
    
    # Determine current model and provider
    if llm is not None:
        if isinstance(llm, AzureChatOpenAI):
            cost_info["provider"] = "azure"
            cost_info["model"] = llm.deployment_name
        elif isinstance(llm, ChatSambaNovaCloud):
            cost_info["provider"] = "sambanova"
            cost_info["model"] = llm.model
        else:
            cost_info["provider"] = "unknown"
            cost_info["model"] = str(type(llm))
    
    class CostTrackingStreamWrapper:
        """A wrapper class to handle streaming with cost tracking."""
        
        def __init__(self, chain: Any):
            self.chain = chain
            self.callback = get_openai_callback()
            self.chunks = []
            self.start_time = 0.0
            self.end_time = 0.0
        
        def __iter__(self) -> Iterator[str]:
            with self.callback as cb:
                try:
                    # Record start time
                    self.start_time = time.time()
                    
                    response = self.chain.stream({"user_input": user_input})
                    for chunk in response:
                        self.chunks.append(chunk)
                        yield chunk
                    
                    # Record end time
                    self.end_time = time.time()
                    
                    # Calculate time elapsed
                    cost_info["time_elapsed"] = self.end_time - self.start_time

                    # Update cost information
                    cost_info["input_tokens"] = cb.prompt_tokens
                    cost_info["output_tokens"] = cb.completion_tokens
                    cost_info["total_tokens"] = cb.total_tokens
                    cost_info["total_cost"] = cb.total_cost

                    logging.info(f"Callback cost info: {cost_info}")
                    
                    # Check if the callback provided a non-zero cost
                    if cb.total_cost > 0:
                        cost_info["cost_source"] = "callback"
                    # Check if total cost is 0 and compute from config
                    elif cost_info["provider"] != "unknown":
                        self._compute_cost_from_config()
                except Exception as e:
                    # Still record end time even if there's an error
                    self.end_time = time.time()
                    cost_info["time_elapsed"] = self.end_time - self.start_time
                    print(f"Error in streaming: {str(e)}")
        
        def _compute_cost_from_config(self) -> None:
            """Compute cost from configuration file if available."""
            try:
                config_path = pathlib.Path(__file__).parents[2] / "science" / "pipeline" / "cost_config.json"
                with open(config_path, "r") as f:
                    cost_config = json.load(f)
                
                provider = cost_info["provider"]
                model = cost_info["model"]
                
                # Retrieve model-specific cost information
                if provider in cost_config["providers"] and model in cost_config["providers"][provider]["models"]:
                    model_config = cost_config["providers"][provider]["models"][model]
                    input_cost = model_config["input_cost_per_token"]
                    output_cost = model_config["output_cost_per_token"]
                    
                    # Calculate estimated cost based on actual token usage
                    estimated_cost = (cost_info["input_tokens"] * input_cost) + (cost_info["output_tokens"] * output_cost)
                    cost_info["total_cost"] = estimated_cost
                    cost_info["cost_source"] = "config_calculation"
                else:
                    cost_info["cost_source"] = "config_not_found"
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                cost_info["cost_source"] = "config_error"
                print(f"Error retrieving cost configuration: {str(e)}")
    
    # Return the generator wrapper and cost info
    return CostTrackingStreamWrapper(chain), cost_info

# Example usage
if __name__ == "__main__":
    # Set up a simple chain for testing
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
        openai_api_version="2024-08-01-preview",
        deployment_name="gpt-4o",
        temperature=0,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
        openai_api_version="2024-07-01-preview",
        azure_deployment="gpt-4o-mini",
        temperature=0,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        api_key=os.getenv("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
        max_tokens=1024,
        temperature=0,
        top_p=0.1,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
        api_version="2024-12-01-preview",
        deployment_name="o3-mini",
        model_kwargs={"stream_options": {"include_usage": True}, "max_completion_tokens": 100000},
        streaming=True
    )

    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("""what is {user_input}? explain the answer in great detail""")
    chain = prompt | llm | parser
    
    # Use our new function to get a streaming generator and cost info
    stream_generator, cost_info = track_cost_and_stream(user_input="1+1=?", chain=chain)
    
    print("Streaming response:")
    for chunk in stream_generator:
        print(chunk, end="", flush=True)
    
    print("\n\nCost Information:")
    print(json.dumps(cost_info, indent=2))
    print(f"\nTime elapsed: {cost_info['time_elapsed']:.4f} seconds")