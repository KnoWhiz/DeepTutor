from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
    openai_api_version='2024-07-01-preview',
    azure_deployment='gpt-4o-mini',
    temperature=0,
    streaming=False,
)

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
    openai_api_version='2024-06-01',
    azure_deployment='gpt-4o',
    temperature=0,
    streaming=False,
)

parser = StrOutputParser()
error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
prompt = "what is 2 + 2 = ? explain the answer"
prompt = ChatPromptTemplate.from_template(prompt)
chain = prompt | llm | error_parser
response = chain.invoke({})
print(response)