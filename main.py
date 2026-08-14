# Step 1: Define tools and model

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState

load_dotenv()

model = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai",
    temperature=0
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

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
    llm_calls: int

class PlannerSchema(BaseModel):
    sub_tasks: list[str] = Field(description="List of 3 to 4 sub-tasks or research questions.")

class ResearcherSchema(BaseModel):
    sub_tasks: list[str] = Field(description="List of 3 to 4 research findings.")

class SummarizerSchema(BaseModel):
    summary: str = Field(description="2-4 sentence summary of the research findings. Include a 2 bullet point list of the key takeaways.")


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

def researcher(state: PipelineState):
    """Researcher Node (Stubbed): Simulates gathering research for each sub-task."""
    plan = state.get("plan", [])
    
    print(f"\n--- [RESEARCHER NODE] Processing {len(plan)} sub-tasks ---")
    
    # Fake research findings matching the plan topics
    fake_findings = [
        "Finding 1: AsyncIO operates on a single-threaded event loop using non-blocking I/O tasks.",
        "Finding 2: Multiprocessing spawns separate OS processes with dedicated Python interpreters, bypassing the GIL.",
        "Finding 3: Use AsyncIO for network/disk bound tasks; use Multiprocessing for heavy CPU computation."
    ]

    return {
        "research_results": fake_findings
    }


def summarizer(state: PipelineState):
    """Summarizer Node (Stubbed): Simulates synthesizing research results into a final output."""
    research_results = state.get("research_results", [])
    
    print("\n--- [SUMMARIZER NODE] Synthesizing final summary ---")
    
    # Fake synthesized summary
    fake_summary = (
        "## Summary of Python Concurrency\n\n"
        "1. **AsyncIO**: Best for I/O-bound tasks (web requests, database calls). Runs concurrently on a single thread.\n"
        "2. **Multiprocessing**: Best for CPU-bound tasks (math, data processing). Runs in parallel across multiple CPU cores.\n\n"
        "**Rule of Thumb:** If your code spends most of its time waiting, use AsyncIO. If it spends time computing, use Multiprocessing."
    )

    return {
        "output": fake_summary
    }


# Step 4: Define tool node

from langchain.messages import ToolMessage


def tool_node(state: PipelineState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

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
agent_builder = StateGraph(PipelineState)

# Add nodes
agent_builder.add_node("planner_call", planner_call)
agent_builder.add_node("researcher", researcher)
agent_builder.add_node("summarizer", summarizer)

# Add edges to connect nodes
agent_builder.add_edge(START, "planner_call")
agent_builder.add_edge("planner_call","researcher")
agent_builder.add_edge("researcher","summarizer")
# Fix: Connect summarizer to END
agent_builder.add_edge("summarizer", END)

# Compile the agent
agent = agent_builder.compile()

# Print the ASCII graph directly in the console
print(agent.get_graph(xray=True).draw_ascii())

# Save the graph locally
graph_bytes = agent.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_bytes)

print("Graph saved to graph.png!\n")

# Invoke
from langchain_core.messages import HumanMessage

result = agent.invoke({
    "task": "Explain the difference between AsyncIO and Multiprocessing in Python with code examples."
})

print("GENERATED PLAN:", result["plan"])
print("\nFINAL SUMMARY:\n", result["output"])