import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

import logging
logger = logging.getLogger("autogen_test.py")

async def main() -> None:
    agent = AssistantAgent("assistant", OpenAIChatCompletionClient(model="gpt-4o"))
    logger.info(await agent.run(task="Say 'Hello World!'"))

asyncio.run(main())