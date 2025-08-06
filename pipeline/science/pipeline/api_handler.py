import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.prompts import ChatPromptTemplate
from .claude_proxy_wrapper import create_claude_proxy_model

import logging
logger = logging.getLogger("tutorpipeline.science.api_handler")
load_dotenv()


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
    def __init__(self, para, stream=False):
        load_dotenv(para['openai_key_dir'])
        self.para = para
        self.para['stream'] = stream
        self.api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
        self.azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.azure_endpoint_backup = str(os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"))
        self.azure_api_key_backup = str(os.getenv("AZURE_OPENAI_API_KEY_BACKUP"))
        # self.openai_api_key = str(os.getenv("OPENAI_API_KEY"))
        # self.deepseek_api_key = str(os.getenv("DEEPSEEK_API_KEY"))
        self.sambanova_api_key = str(os.getenv("SAMBANOVA_API_KEY"))
        
        # Claude proxy configuration
        self.claude_proxy_url = str(os.getenv("CLAUDE_PROXY_URL", "http://localhost:8082"))
        self.claude_proxy_key = str(os.getenv("CLAUDE_PROXY_KEY", "your-key"))
        
        self.models = self.load_models()
        self.embedding_models = self.load_embedding_models()


    def get_models(self, api_key, temperature=0, deployment_name=None, endpoint=None, api_version=None, host='azure', stream=False):
        """
        Get language model instances based on the specified host platform.

        Args:
            api_key (str): API key for authentication
            endpoint (str): API endpoint URL
            api_version (str): API version for Azure
            deployment_name (str): Model deployment name/identifier
            temperature (float): Temperature parameter for model responses
            host (str): Host platform ('azure', 'openai', 'sambanova', 'claude_proxy', or 'deepseek')

        Returns:
            Language model instance configured for the specified platform
        """
        if host == 'openai':
            return ChatOpenAI(
                streaming=stream,
                api_key=api_key,
                model_name=deployment_name,
                temperature=temperature,
                max_tokens=20000,
                model_kwargs={"stream_options": {"include_usage": True}} if stream else {}
            )
        elif host == 'azure':
            return AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version=api_version,
                azure_deployment=deployment_name,
                temperature=temperature,
                streaming=stream,
                max_tokens=20000,
                model_kwargs={"stream_options": {"include_usage": True}} if stream else {}
            )
        elif host == 'sambanova':
            return ChatSambaNovaCloud(
                model=deployment_name,
                api_key=self.sambanova_api_key,
                base_url="https://api.sambanova.ai/v1",
                max_tokens=3000,
                temperature=temperature,
                top_p=0.1,
                streaming=stream,
                model_kwargs={"stream_options": {"include_usage": True}} if stream else {}
            )
        elif host == 'claude_proxy':
            # Map deployment_name to Claude model
            claude_model_map = {
                'gpt-4.1': 'claude-3-5-opus-20241022',
                'gpt-4.1-mini': 'claude-3-5-sonnet-20241022',
                'o3-mini': 'claude-3-5-haiku-20241022',
                'default': 'claude-3-5-sonnet-20241022'
            }
            claude_model = claude_model_map.get(deployment_name, claude_model_map['default'])
            
            return create_claude_proxy_model(
                base_url=self.claude_proxy_url,
                api_key=self.claude_proxy_key,
                model_name=claude_model,
                temperature=temperature,
                max_tokens=20000,
                streaming=stream
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
        llm_basic = self.get_models(api_key=self.azure_api_key_backup,
                                    temperature=self.para['temperature'],
                                    deployment_name='gpt-4.1-mini',
                                    endpoint=self.azure_endpoint_backup,
                                    api_version='2024-12-01-preview',
                                    host='azure',
                                    stream=self.para['stream'])
        llm_advance = self.get_models(api_key=self.azure_api_key_backup,
                                      temperature=self.para['temperature'],
                                      deployment_name='gpt-4.1',
                                      endpoint=self.azure_endpoint_backup,
                                      api_version='2024-12-01-preview',
                                      host='azure',
                                      stream=self.para['stream'])
        llm_creative = self.get_models(api_key=self.azure_api_key_backup,
                                      temperature=self.para['creative_temperature'],
                                      deployment_name='gpt-4.1',
                                      endpoint=self.azure_endpoint_backup,
                                      api_version='2024-12-01-preview',
                                      host='azure',
                                      stream=self.para['stream'])
        # llm_deepseek = self.get_models(api_key=self.deepseek_api_key,
        #                                temperature=self.para['temperature'],
        #                                host='deepseek')
        llm_llama = self.get_models(api_key=self.sambanova_api_key,
                                    temperature=self.para['temperature'],
                                    deployment_name='Meta-Llama-3.3-70B-Instruct',
                                    host='sambanova',
                                    stream=self.para['stream'])

        if self.para['llm_source'] == 'azure' or self.para['llm_source'] == 'openai':
            models = {
                'basic': {'instance': llm_basic, 'context_window': 128000},
                'advanced': {'instance': llm_advance, 'context_window': 128000},
                'creative': {'instance': llm_creative, 'context_window': 128000},
                'backup': {'instance': llm_llama, 'context_window': 128000},
            }
        elif self.para['llm_source'] == 'sambanova':
            models = {
                'basic': {'instance': llm_llama, 'context_window': 128000},
                'advanced': {'instance': llm_llama, 'context_window': 128000},
                'creative': {'instance': llm_llama, 'context_window': 128000},
                'backup': {'instance': llm_basic, 'context_window': 128000},
            }
        elif self.para['llm_source'] == 'claude_proxy':
            # Create Claude proxy models
            llm_claude_basic = self.get_models(api_key=self.claude_proxy_key,
                                              temperature=self.para['temperature'],
                                              deployment_name='gpt-4.1-mini',
                                              host='claude_proxy',
                                              stream=self.para['stream'])
            llm_claude_advance = self.get_models(api_key=self.claude_proxy_key,
                                                temperature=self.para['temperature'],
                                                deployment_name='gpt-4.1',
                                                host='claude_proxy',
                                                stream=self.para['stream'])
            llm_claude_creative = self.get_models(api_key=self.claude_proxy_key,
                                                 temperature=self.para['creative_temperature'],
                                                 deployment_name='gpt-4.1',
                                                 host='claude_proxy',
                                                 stream=self.para['stream'])
            
            models = {
                'basic': {'instance': llm_claude_basic, 'context_window': 128000},
                'advanced': {'instance': llm_claude_advance, 'context_window': 128000},
                'creative': {'instance': llm_claude_creative, 'context_window': 128000},
                'backup': {'instance': llm_basic, 'context_window': 128000},  # Fallback to Azure
            }
        # elif self.para['llm_source'] == 'deepseek':
        #     models = {
        #         'basic': {'instance': llm_deepseek, 'context_window': 65536},
        #         'advanced': {'instance': llm_deepseek, 'context_window': 65536},
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
            chunk_size=2000)
        
        lite_embedding_model = AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_EMBEDDINGS'),
            openai_api_key =os.getenv('OPENAI_API_KEY_EMBEDDINGS'),
            openai_api_type="azure",
            chunk_size=2000)
        
        small_embedding_model = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            model="text-embedding-3-small",
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_EMBEDDINGS'),
            openai_api_key =os.getenv('OPENAI_API_KEY_EMBEDDINGS'),
            openai_api_type="azure",
            chunk_size=2000)

        models = {
            'default': {'instance': embedding_model},
            'lite': {'instance': lite_embedding_model},
            'small': {'instance': small_embedding_model},
        }
        return models