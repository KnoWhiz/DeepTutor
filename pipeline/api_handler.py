import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI

class ApiHandler:
    def __init__(self, para):
        load_dotenv(para['openai_key_dir'])
        self.para = para
        self.api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
        self.azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.api_key_backup = str(os.getenv("AZURE_OPENAI_API_KEY_BACKUP"))
        self.azure_endpoint_backup = str(os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"))
        self.openai_api_key = str(os.getenv("OPENAI_API_KEY"))

        # # TEST
        # self.sambanova_api_key = str(os.getenv("SAMBANOVA_API_KEY"))

        self.models = self.load_models()

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
        # llm_stable = self.get_models(api_key=self.api_key, endpoint=self.azure_endpoint, api_version='2024-06-01', deployment_name='gpt-35-turbo', temperature=0, host='azure')
        llm_stable = self.get_models(api_key=self.api_key_backup, endpoint=self.azure_endpoint_backup, api_version='2024-08-01-preview', deployment_name='gpt-4', temperature=self.para['temperature'], host='azure')
        llm_basic_backup_1 = self.get_models(api_key=self.api_key_backup, endpoint=self.azure_endpoint_backup, api_version='2024-07-01-preview', deployment_name='gpt-4o-mini', temperature=self.para['temperature'], host='azure')
        llm_basic_backup_2 = self.get_models(api_key=self.openai_api_key, endpoint=self.azure_endpoint, api_version='2024-07-01-preview', deployment_name='gpt-4o-mini', temperature=self.para['temperature'], host='openai')

        # # TEST
        # llm_basic = self.get_models(api_key=self.sambanova_api_key, endpoint="https://api.sambanova.ai/v1", api_version='2024-06-01', deployment_name='Meta-Llama-3.1-405B-Instruct', temperature=self.para['temperature'], host='sambanova')
        
        models = {
            'basic': {'instance': llm_basic, 'context_window': 128000},
            'advance': {'instance': llm_advance, 'context_window': 128000},
            'creative': {'instance': llm_creative, 'context_window': 128000},
            'stable': {'instance': llm_stable, 'context_window': 128000},
            'basic_backup_1': {'instance': llm_basic_backup_1, 'context_window': 128000},
            'basic_backup_2': {'instance': llm_basic_backup_2, 'context_window': 128000},
        }
        return models
