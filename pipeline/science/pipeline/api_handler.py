import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek


def create_env_file(GraphRAG_embedding_folder):
    api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
    env_content = \
        f"""
GRAPHRAG_API_KEY={api_key}
"""
    if not os.path.exists(GraphRAG_embedding_folder):
        os.makedirs(GraphRAG_embedding_folder)
    with open(os.path.join(GraphRAG_embedding_folder, '.env'), 'w') as env_file:
        env_file.write(env_content)


class ApiHandler:
    def __init__(self, para):
        load_dotenv(para['openai_key_dir'])
        self.para = para
        self.api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
        self.azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
        # self.openai_api_key = str(os.getenv("OPENAI_API_KEY"))
        self.deepseek_api_key = str(os.getenv("DEEPSEEK_API_KEY"))
        self.models = self.load_models()
        self.embedding_models = self.load_embedding_models()


    def get_models(self, api_key, temperature=0, deployment_name=None, endpoint=None, api_version=None, host='azure'):
        """
        Get language model instances based on the specified host platform.

        Args:
            api_key (str): API key for authentication
            endpoint (str): API endpoint URL
            api_version (str): API version for Azure
            deployment_name (str): Model deployment name/identifier
            temperature (float): Temperature parameter for model responses
            host (str): Host platform ('azure', 'openai', 'sambanova', or 'deepseek')

        Returns:
            Language model instance configured for the specified platform
        """
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
        # elif host == 'deepseek':
        #     return ChatDeepSeek(
        #         model="deepseek-chat",
        #         temperature=0,
        #         max_tokens=None,
        #         timeout=None,
        #         # max_retries=2,
        #     )


    def load_models(self):
        llm_basic = self.get_models(api_key=self.api_key,
                                    temperature=self.para['temperature'],
                                    deployment_name='gpt-4o-mini',
                                    endpoint=self.azure_endpoint,
                                    api_version='2024-07-01-preview',
                                    host='azure')
        llm_advance = self.get_models(api_key=self.api_key,
                                      temperature=self.para['temperature'],
                                      deployment_name='gpt-4o',
                                      endpoint=self.azure_endpoint,
                                      api_version='2024-06-01',
                                      host='azure')
        llm_creative = self.get_models(api_key=self.api_key,
                                      temperature=self.para['creative_temperature'],
                                      deployment_name='gpt-4o',
                                      endpoint=self.azure_endpoint,
                                      api_version='2024-06-01',
                                      host='azure')
        # llm_deepseek = self.get_models(api_key=self.deepseek_api_key,
        #                                temperature=self.para['temperature'],
        #                                host='deepseek')

        if self.para['llm_source'] == 'azure' or self.para['llm_source'] == 'openai':
            models = {
                'basic': {'instance': llm_basic, 'context_window': 128000},
                'advance': {'instance': llm_advance, 'context_window': 128000},
                'creative': {'instance': llm_creative, 'context_window': 128000},
            }
        # elif self.para['llm_source'] == 'deepseek':
        #     models = {
        #         'basic': {'instance': llm_deepseek, 'context_window': 65536},
        #         'advance': {'instance': llm_deepseek, 'context_window': 65536},
        #         'creative': {'instance': llm_deepseek, 'context_window': 65536},
        #     }
        return models


    def load_embedding_models(self):
        embedding_model = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-large",
            model="text-embedding-3-large",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_EMBEDDINGS'),
            openai_api_key =os.getenv('OPENAI_API_KEY_EMBEDDINGS'),
            openai_api_type="azure",
            chunk_size=1000)
        
        lite_embedding_model = AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_EMBEDDINGS'),
            openai_api_key =os.getenv('OPENAI_API_KEY_EMBEDDINGS'),
            openai_api_type="azure",
            chunk_size=1000)

        models = {
            'default': {'instance': embedding_model},
            'lite': {'instance': lite_embedding_model},
        }
        return models