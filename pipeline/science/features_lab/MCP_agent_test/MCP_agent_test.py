#!/usr/bin/env python3
"""
Light-weight CLI chatbot that connects to a Composio-hosted Notion MCP
server and answers questions with live workspace context.
"""
import asyncio, os, json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()
MODEL          = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MCP_URL        = os.getenv("NOTION_MCP_URL")
COMPOSIO_TOKEN = os.getenv("COMPOSIO_API_KEY")

if not MCP_URL:
    raise ValueError("⚠️  NOTION_MCP_URL is missing in .env")

async def main():
    headers = {"Authorization": f"Bearer {COMPOSIO_TOKEN}"} if COMPOSIO_TOKEN else {}
    # 1️⃣  open a single-server MCP client (streamable HTTP is what Composio serves)
    client = MultiServerMCPClient({
        "notion": {
            "transport": "streamable_http",
            "url": MCP_URL,
            "headers": headers,
        }
    })
    # 2️⃣  introspect the server → get its tool list
    tools = await client.get_tools()
    # 3️⃣  plug the tools into a ReAct agent powered by OpenAI
    llm   = ChatOpenAI(model=MODEL, temperature=0)
    agent = create_react_agent(llm, tools)

    print("🟢  Notion chatbot ready – type a question (or 'quit'):")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        result  = await agent.ainvoke({"messages": user})
        message = result["messages"][-1].content if isinstance(result, dict) else result
        print(f"\nAssistant: {message}")

if __name__ == "__main__":
    asyncio.run(main())