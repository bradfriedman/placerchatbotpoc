# Define the agent state
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_last_human_message(state: AgentState) -> BaseMessage:
    messages = state["messages"]
    last_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    return last_human_message
