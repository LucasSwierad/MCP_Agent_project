import asyncio
import sys
import os

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from main import run_pipeline

load_dotenv()  # load environment variables from .env


MODEL = "gemini-3.6-flash"
llm = ChatGoogleGenerativeAI(model=MODEL)

workspace_path = os.path.abspath("./workspace")
os.makedirs(workspace_path, exist_ok=True)

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

    # Handle local python/node file paths passed as a single argument
    if command.endswith(".py"):
        return StdioServerParameters(command="python", args=[command] + args)
    elif command.endswith(".js"):
        return StdioServerParameters(command="node", args=[command] + args)

    # Handle direct CLI commands (npx, uvx, docker, python -m, etc.)
    return StdioServerParameters(command=command, args=args)

# async def process_query(client: Client, query: str) -> str:
#     """Process a query using Claude and available tools"""
#     messages = [
#         {
#             "role": "user",
#             "content": query
#         }
#     ]

#     tool_list = await client.list_tools()
#     available_tools = [{
#         "name": tool.name,
#         "description": tool.description,
#         "input_schema": tool.input_schema
#     } for tool in tool_list.tools]

#     # Initial Claude API call
#     response = llm.messages.create(
#         model=MODEL,
#         max_tokens=1000,
#         messages=messages,
#         tools=available_tools
#     )

#     # Process response and handle tool calls
#     final_text = []
#     tool_results = []

#     for content in response.content:
#         if content.type == 'text':
#             final_text.append(content.text)
#         elif content.type == 'tool_use':
#             tool_name = content.name
#             tool_args = content.input

#             # Execute tool call
#             result = await client.call_tool(tool_name, tool_args)
#             final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

#             tool_results.append({
#                 "type": "tool_result",
#                 "tool_use_id": content.id,
#                 "content": "\n".join(
#                     block.text
#                     for block in result.content
#                     if isinstance(block, TextContent)
#                 ),
#                 "is_error": result.is_error
#             })

#     if tool_results:
#         messages.append({"role": "assistant", "content": response.content})
#         messages.append({"role": "user", "content": tool_results})

#         # Get next response from Claude
#         response = llm.messages.create(
#             model=MODEL,
#             max_tokens=1000,
#             messages=messages,
#             tools=available_tools
#         )

#         for content in response.content:
#             if content.type == 'text':
#                 final_text.append(content.text)

#     return "\n".join(final_text)


async def chat_loop(client: Client, mcp_tools: list) -> None:
    """Run an interactive chat loop"""
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
            response = await process_query(client, mcp_tools, query)
            print("\n" + response)
        except Exception as e:
            print(f"\nError: {e}")

async def process_query(client: Client, mcp_tools: list, query: str) -> str:
    output = await run_pipeline(user_task=query, client=client, mcp_tools=mcp_tools)
    return output

async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <command_or_script> [args...]")
        print("Example: python client.py npx -y @modelcontextprotocol/server-filesystem ./workspace")
        sys.exit(1)

    params = get_server_params(sys.argv[1:])

    async with Client(stdio_client(params)) as client:
        tool_list = await client.list_tools()
        print(tool_list.tools[0].__class__.model_fields.keys())
        print("\nConnected to MCP server! Available tools:")
        for t in tool_list.tools:
            print(f"- {t.name}")
        
        await chat_loop(client, tool_list.tools)


if __name__ == "__main__":
    asyncio.run(main())