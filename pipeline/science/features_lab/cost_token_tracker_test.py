from langchain_openai import AzureChatOpenAI
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.tracers import ConsoleCallbackHandler

from dotenv import load_dotenv
import os
import json
import pathlib
load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
    openai_api_version="2024-07-01-preview",
    azure_deployment="gpt-4o-mini",
    temperature=0,
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}}
)

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
    openai_api_version="2024-08-01-preview",
    azure_deployment="gpt-4o",
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

parser = StrOutputParser()
# error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
prompt = "what is 2 + 2 = ? explain the answer in great detail"
prompt = ChatPromptTemplate.from_template(prompt)
chain = prompt | llm | parser

# Using the get_openai_callback context manager to track token usage and cost
with get_openai_callback() as cb:
    response = chain.stream({})
    print(type(response))
    # print(response)
    for chunk in response:
        print(chunk, end="", flush=True)
    
    # Print token usage and cost information
    print("\nToken Usage Statistics:")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost:.10f}")
    
    # Check if total cost is 0 and provide information from the cost config file
    if cb.total_cost == 0:
        # Determine current model and provider
        if isinstance(llm, AzureChatOpenAI):
            provider = "azure"
            model = llm.azure_deployment
        elif isinstance(llm, ChatSambaNovaCloud):
            provider = "sambanova"
            model = llm.model
        else:
            provider = "unknown"
            model = "unknown"
        
        # Get the cost config
        config_path = pathlib.Path(__file__).parents[2] / "science" / "pipeline" / "cost_config.json"
        try:
            with open(config_path, "r") as f:
                cost_config = json.load(f)
            
            # Retrieve model-specific cost information
            if provider in cost_config["providers"] and model in cost_config["providers"][provider]["models"]:
                model_config = cost_config["providers"][provider]["models"][model]
                input_cost = model_config["input_cost_per_token"]
                output_cost = model_config["output_cost_per_token"]
                
                # Calculate estimated cost based on actual token usage
                estimated_cost = (cb.prompt_tokens * input_cost) + (cb.completion_tokens * output_cost)
                
                print("\nCost calculation from config:")
                print(f"Provider: {provider}")
                print(f"Model: {model}")
                print(f"Input cost per token: ${input_cost}")
                print(f"Output cost per token: ${output_cost}")
                print(f"Estimated cost: ${estimated_cost:.10f}")
            else:
                print(f"\nNo cost information found for {provider}/{model} in config file")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"\nError retrieving cost configuration: {str(e)}")