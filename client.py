import asyncio
import sys
import os
import sqlite3
import json

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from main import run_pipeline
from memory import MemoryStore, new_session_id

load_dotenv()  # load environment variables from .env


MODEL = "gemini-3.6-flash"
llm = ChatGoogleGenerativeAI(model=MODEL)

workspace_path = os.path.abspath("./workspace")
os.makedirs(workspace_path, exist_ok=True)

memory = MemoryStore(db_path=os.path.join(workspace_path, "memory.db"))

def get_server_params(args_list: list[str]) -> StdioServerParameters:
    """Configures the MCP server subprocess for both local scripts and CLI commands.
    
    Examples:
        - Local Python file: ["python", "server.py"]
        - Node package:     ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
        - Python CLI tool:  ["uvx", "mcp-server-fetch"]
    """
    if not args_list:
        raise ValueError("Must provide a server command or script path.")

    command = args_list[0]
    args = args_list[1:]

    passthrough_keys = [
        "TAVILY_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    server_env = {
        key: os.environ[key] for key in passthrough_keys if key in os.environ
    }

    if command.endswith(".py"):
        return StdioServerParameters(command="python", args=[command] + args, env=server_env)
    elif command.endswith(".js"):
        return StdioServerParameters(command="node", args=[command] + args, env=server_env)
 
    # Handle direct CLI commands (npx, uvx, docker, python -m, etc.)
    return StdioServerParameters(command=command, args=args, env=server_env)

async def chat_loop(client: Client, mcp_tools: list) -> None:
    """Run an interactive chat loop"""
    session_id = new_session_id()
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")
 
    while True:
        try:
            query = (await asyncio.to_thread(input, "\nQuery: ")).strip()
        except EOFError:
            break
 
        if query.lower() == 'quit':
            break
 
        try:
            response = await process_query(client, mcp_tools, query, session_id)
            
            print("\n" + response)
        except Exception as e:
            print(f"\nError: {e}")

async def process_query(client: Client, mcp_tools: list, query: str, session_id: str) -> str:
    context = memory.build_context(session_id, query, top_k=3, turns=6)
    result = await run_pipeline(
        user_task=query,
        client=client,
        mcp_tools=mcp_tools,
        context=context,
    )
    if isinstance(result, dict):
        plan = result.get("plan", [])
        research_results = result.get("research_results", [])
        summary = result.get("summary", "") or result.get("output", "")
        tool_calls_log = result.get("tool_calls_log") or [t.name for t in mcp_tools]
        output = summary or str(result)
    else:
        plan = []
        research_results = []
        summary = str(result)
        tool_calls_log = [t.name for t in mcp_tools] if mcp_tools else []
        output = summary
 
    # --- STORE: short-term turn + long-term note ---
    memory.add_short_term(session_id, "user", query)
    memory.add_short_term(session_id, "assistant", output)
 
    memory.add_long_term(
        task=query,
        plan=plan,
        research_results=research_results,
        summary=summary,
        tool_calls_log=tool_calls_log,
    )
 
    return output
 

async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <command_or_script> [args...]")
        print("Example: python client.py npx -y @modelcontextprotocol/server-filesystem ./workspace")
        sys.exit(1)

    params = get_server_params(sys.argv[1:])

    try:
        async with Client(stdio_client(params)) as client:
            tool_list = await client.list_tools()
            print("\nConnected to MCP server! Available tools:")
            for t in tool_list.tools:
                print(f"- {t.name}")
 
            await chat_loop(client, tool_list.tools)
    finally:
        memory.close()


if __name__ == "__main__":
    asyncio.run(main())