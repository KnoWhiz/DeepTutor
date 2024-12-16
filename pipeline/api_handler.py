import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings


def create_env_file(root_path):
    api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
    env_content = \
        f"""
GRAPHRAG_API_KEY={api_key}
"""
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    with open(os.path.join(root_path, '.env'), 'w') as env_file:
        env_file.write(env_content)


class ApiHandler:
    def __init__(self, para):
        load_dotenv(para['openai_key_dir'])
        self.para = para
        self.api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
        self.azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
        # self.openai_api_key = str(os.getenv("OPENAI_API_KEY"))
        self.models = self.load_models()
        self.embedding_models = self.load_embedding_models()


    def get_models(self, api_key, endpoint, api_version, deployment_name, temperature, host='azure'):
        if host == 'openai':
            return ChatOpenAI(
                streaming=False,
                api_key=api_key,
                model_name=deployment_name,
                temperature=temperature,
            )
        elif host == 'azure':
            return AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version=api_version,
                azure_deployment=deployment_name,
                temperature=temperature,
            )
        elif host == 'sambanova':
            return ChatOpenAI(
                model=deployment_name,
                api_key=api_key,
                base_url=endpoint,
                streaming=False,
            )
        

    def load_models(self):
        llm_basic = self.get_models(api_key=self.api_key, endpoint=self.azure_endpoint, api_version='2024-07-01-preview', deployment_name='gpt-4o-mini', temperature=self.para['temperature'], host='azure')
        llm_advance = self.get_models(api_key=self.api_key, endpoint=self.azure_endpoint, api_version='2024-06-01', deployment_name='gpt-4o', temperature=self.para['temperature'], host='azure')
        llm_creative = self.get_models(api_key=self.api_key, endpoint=self.azure_endpoint, api_version='2024-06-01', deployment_name='gpt-4o', temperature=self.para['creative_temperature'], host='azure')

        models = {
            'basic': {'instance': llm_basic, 'context_window': 128000},
            'advance': {'instance': llm_advance, 'context_window': 128000},
            'creative': {'instance': llm_creative, 'context_window': 128000},
        }
        return models


    def load_embedding_models(self):
        embedding_model = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-large",
            model="text-embedding-3-large",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_EMBEDDINGS'),
            openai_api_key =os.getenv('OPENAI_API_KEY_EMBEDDINGS'),
            openai_api_type="azure",
            chunk_size=1)
        
        models = {
            'default': {'instance': embedding_model},
        }
        return models