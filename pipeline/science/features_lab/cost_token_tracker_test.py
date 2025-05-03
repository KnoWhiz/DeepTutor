from langchain_openai import AzureChatOpenAI
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.tracers import ConsoleCallbackHandler

from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatSambaNovaCloud(
    model="Meta-Llama-3.3-70B-Instruct",
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
    max_tokens=1024,
    temperature=0,
    top_p=0.1,
    streaming=True
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

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
    openai_api_version="2024-08-01-preview",
    azure_deployment="gpt-4o",
    temperature=0,
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