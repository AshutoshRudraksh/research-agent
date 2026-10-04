import argparse
import sys
from typing import TypedDict, Annotated, Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import READ_FAIL_PREFIX, WRITE_OK_PREFIX, tools, write_report


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str

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
        if name == "read_url" and content and not content.startswith(READ_FAIL_PREFIX):
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
        if content and not content.startswith(READ_FAIL_PREFIX):
            chunks.append(content)
    return "\n\n".join(chunks)


def call_model(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def finalize_report(state: AgentState):
    last = state["messages"][-1]
    content = str(getattr(last, "content", "")).strip() or _source_text(state["messages"])
    result = write_report.invoke({"topic": state["topic"], "content": content})
    return {
        "messages": [
            ToolMessage(content=result, name="write_report", tool_call_id="finalize")
        ]
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if research_status(state["messages"]) == "ready":
        return "finalize"
    return END


def after_tools(state: AgentState) -> str:
    if research_status(state["messages"]) == "written":
        return END
    return "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_node("finalize", finalize_report)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "finalize": "finalize", END: END}
)
graph.add_conditional_edges(
    "tools", after_tools, {"agent": "agent", END: END}
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
