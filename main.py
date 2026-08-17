# Step 1: Define tools and model
 
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
load_dotenv()

model = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai",
    temperature=0
)

# Step 2: Define state

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field

class PipelineState(TypedDict):
    task: str
    plan: list[str]
    research_results: Annotated[list[str], operator.add]
    output: str

class PlannerSchema(BaseModel):
    sub_tasks: list[str] = Field(description="List of 3 to 4 sub-tasks or research questions.")

class ResearcherSchema(BaseModel):
    sub_tasks: list[str] = Field(description="List of 3 to 4 research findings.")

class SummarizerSchema(BaseModel):
    summary: str = Field(description="2-4 sentence summary of the research findings. Include a 2 bullet point list of the key takeaways.")

class ToolChoice(BaseModel):
    tool_name: str = Field(description="Exact name of the MCP tool to call.")
    tool_args: dict[str, Any] = Field(description="Arguments to pass to the tool, matching its input schema.")
    reasoning: str = Field(description="One sentence on why this tool answers the sub-task.")

# Step 3: Define model node
from langchain.messages import SystemMessage


def planner_call(state: PipelineState):
    """LLM breaks down the task into a plan of action"""

    task = state["task"]

    planner_model = model.with_structured_output(PlannerSchema)

    result = planner_model.invoke([
        SystemMessage(content="You are a project planner. Break down the user's request into actionable sub-tasks."),
        HumanMessage(content=f"Task: {task}")
    ])

    return {
        "plan": result.sub_tasks}

# def researcher(state: PipelineState):
#     """LLM researches each sub-task and returns results"""

#     plan = state["plan"]

#     researcher_model = model.with_structured_output(ResearcherSchema)

#     result = researcher_model.invoke([
#         SystemMessage(content="You are a researcher. Research each sub-task and return your findings."),
#         HumanMessage(content=f"Plan: {plan}")
#     ])

#     return {
#         "research_results": result.sub_tasks}

# def summarizer(state: PipelineState):
#     """LLM summarizes the research results"""

#     research_results = state["research_results"]

#     summarizer_model = model.with_structured_output(SummarizerSchema)

#     result = summarizer_model.invoke([
#         SystemMessage(content="You are a summarizer. Summarize the research findings."),
#         HumanMessage(content=f"Research Results: {research_results}")
#     ])

#     return {
#         "output": result.summary,
#     }


def make_researcher_node(client, mcp_tools: list):
    # Build a compact text description of available tools once, since
    # it doesn't change between sub-tasks in a single run.
    tools_description = "\n".join(
        f"- {t.name}: {t.description} | input schema: {t.input_schema}"
        for t in mcp_tools
    )
 
    async def researcher(state: PipelineState):
        """Calls a real MCP tool for each sub-task in the plan."""
        plan = state.get("plan", [])
        print(f"\n--- [RESEARCHER NODE] Processing {len(plan)} sub-tasks via MCP ---")
 
        chooser_model = model.with_structured_output(ToolChoice)
        findings = []
 
        for sub_task in plan:
            try:
                # Step A: ask the LLM which tool + args best address
                # this sub-task, given the real available tools.
                choice = chooser_model.invoke([
                    SystemMessage(content=(
                        "You select which tool to call to research a sub-task. "
                        f"Available tools:\n{tools_description}"
                    )),
                    HumanMessage(content=f"Sub-task: {sub_task}")
                ])
 
                print(f"  -> sub-task: {sub_task}")
                print(f"     chose tool: {choice.tool_name}({choice.tool_args})")
 
                # Step B: actually call the MCP tool for real. This is
                # the direct client.call_tool() path — no bind_tools(),
                # no LangChain Tool wrapping.
                result = await client.call_tool(choice.tool_name, choice.tool_args)
 
                # MCP results come back as a CallToolResult with a
                # `content` list (usually TextContent blocks). Flatten
                # that into a plain string for the pipeline state.
                text_parts = [
                    block.text for block in result.content
                    if hasattr(block, "text")
                ]
                observation = "\n".join(text_parts) if text_parts else str(result.content)
 
                findings.append(f"[{sub_task}] -> {observation}")
 
            except Exception as e:
                print(f"     ERROR calling tool for '{sub_task}': {e}")
                findings.append(f"[{sub_task}] -> ERROR: {e}")
 
        return {"research_results": findings}
 
    return researcher


def summarizer(state: PipelineState):
    """LLM summarizes the real research results."""
    research_results = state.get("research_results", [])
    print("\n--- [SUMMARIZER NODE] Synthesizing final summary ---")
 
    summarizer_model = model.with_structured_output(SummarizerSchema)
    result = summarizer_model.invoke([
        SystemMessage(content="You are a summarizer. Summarize the research findings clearly and concisely."),
        HumanMessage(content=f"Research Results:\n{research_results}")
    ])
 
    return {"output": result.summary}

# Step 5: Define logic to determine whether to end

from typing import Literal
from langgraph.graph import StateGraph, START, END


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: PipelineState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Step 6: Build agent

# Build workflow

from langgraph.graph import StateGraph, START, END
 
 
def build_agent(client, mcp_tools: list):
    researcher_node = make_researcher_node(client, mcp_tools)
 
    agent_builder = StateGraph(PipelineState)
    agent_builder.add_node("planner_call", planner_call)
    agent_builder.add_node("researcher", researcher_node)
    agent_builder.add_node("summarizer", summarizer)
 
    agent_builder.add_edge(START, "planner_call")
    agent_builder.add_edge("planner_call", "researcher")
    agent_builder.add_edge("researcher", "summarizer")
    agent_builder.add_edge("summarizer", END)
 
    return agent_builder.compile()


async def run_pipeline(user_task: str, client, mcp_tools: list) -> str:
    agent = build_agent(client, mcp_tools)

    try:
        graph_bytes = agent.get_graph(xray=True).draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(graph_bytes)
    except Exception as e:
        print(f"(Skipping graph image — rendering failed: {e})")
 
    initial_state = {"task": user_task}
    result = await agent.ainvoke(initial_state)
 
    print("\nGENERATED PLAN:", result["plan"])
    return result["output"]

