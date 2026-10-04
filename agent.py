import argparse
import sys
from typing import TypedDict, Annotated, Literal

from typing_extensions import NotRequired

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import WRITE_OK_PREFIX, is_read_failure, tools, write_report


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    tool_rounds: NotRequired[int]


MAX_TOOL_ROUNDS = 8

ResearchStatus = Literal["written", "ready", "unsourced"]

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a research agent. When given a topic:
1. Search the web for relevant information
2. Read the content of 1-2 of the most relevant URLs
3. Write a comprehensive markdown report using write_report

Always call write_report as your final action. Be thorough but concise.

You must follow these rules:
	Rules:
	- If a tool returns SEARCH_EMPTY or SEARCH_ERROR, retry with a different query (max 3 attempts).
	- Never call write_report unless you have read at least one URL successfully.
	- If you cannot find sources, say so instead of writing a report.

"""


def research_status(messages) -> ResearchStatus:
    wrote = False
    sourced = False
    for message in messages:
        name = getattr(message, "name", None)
        content = str(getattr(message, "content", ""))
        if name == "write_report" and content.startswith(WRITE_OK_PREFIX):
            wrote = True
        if name == "read_url" and content and not is_read_failure(content):
            sourced = True
    if wrote:
        return "written"
    if sourced:
        return "ready"
    return "unsourced"


def _source_text(messages) -> str:
    chunks = []
    for message in messages:
        if getattr(message, "name", None) != "read_url":
            continue
        content = str(getattr(message, "content", ""))
        if content and not is_read_failure(content):
            chunks.append(content)
    return "\n\n".join(chunks)


def call_model(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _batch_size(messages) -> int:
    for message in reversed(messages):
        calls = getattr(message, "tool_calls", None)
        if calls:
            return len(calls)
    return 0


def _model_text(messages) -> str:
    last = messages[-1]
    if isinstance(last, AIMessage):
        return str(last.content).strip()
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content).strip()
    return ""


def finalize_report(state: AgentState):
    model_text = _model_text(state["messages"])
    sources = _source_text(state["messages"])
    if model_text and sources and sources not in model_text:
        content = f"{model_text}\n\n## Sources\n\n{sources}"
    else:
        content = model_text or sources
    result = write_report.invoke({"topic": state["topic"], "content": content})
    return {
        "messages": [
            ToolMessage(content=result, name="write_report", tool_call_id="finalize")
        ]
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        if state.get("tool_rounds", 0) + len(last.tool_calls) > MAX_TOOL_ROUNDS:
            if research_status(state["messages"]) == "ready":
                return "finalize"
            return END
        return "tools"
    if research_status(state["messages"]) == "ready":
        return "finalize"
    return END


def count_tools(state: AgentState):
    return {"tool_rounds": state.get("tool_rounds", 0) + _batch_size(state["messages"])}


def after_tools(state: AgentState) -> str:
    status = research_status(state["messages"])
    if status == "written":
        return END
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        if status == "ready":
            return "finalize"
        return END
    return "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_node("count", count_tools)
graph.add_node("finalize", finalize_report)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "finalize": "finalize", END: END}
)
graph.add_edge("tools", "count")
graph.add_conditional_edges(
    "count", after_tools, {"agent": "agent", "finalize": "finalize", END: END}
)
graph.add_edge("finalize", END)

app = graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument("topic", help="Topic to research")
    args = parser.parse_args()

    print(f"\n🔍 Researching: {args.topic}\n")

    initial_state = {
        "topic": args.topic,
        "messages": [
            HumanMessage(content=f"Research this topic and write a report: {args.topic}")
        ],
    }

    wrote_path = None
    for step in app.stream(initial_state, stream_mode="updates"):
        for node_name, data in step.items():
            print(f"📦 [Node: {node_name}]")
            if "messages" not in data:
                continue
            last_msg = data["messages"][-1]
            if node_name == "agent":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool in last_msg.tool_calls:
                        print(f"   🤖 Model called tool: '{tool['name']}'")
                else:
                    print("   🤖 Model output text content.")
            elif node_name in ("tools", "finalize"):
                content = str(last_msg.content)
                print(f"   🔧 Tool executed and returned content summary: {content[:100]}...")
                if content.startswith(WRITE_OK_PREFIX):
                    wrote_path = content.removeprefix(WRITE_OK_PREFIX)

    if wrote_path:
        print(f"\n✅ Done. Report saved to {wrote_path}\n")
        return
    print("\nNo report written. The agent found no usable sources.\n")
    sys.exit(2)


if __name__ == "__main__":
    main()
