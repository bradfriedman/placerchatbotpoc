import asyncio
import uuid
from typing import List, Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

import streamlit as st

from tools.learn import LearnTool

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

learn_tool = LearnTool(index_name="articles", namespace="kb", model_name="gpt-3.5-turbo", temperature=0, top_k=6)


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def greet_user():
    """Respond to a greeting from the user with another friendly greeting."""
    print("ENTERING GREET USER")
    return ["hello"]


tools = [
    learn_tool.as_structured_tool(),
    StructuredTool.from_function(coroutine=greet_user, name="GreetUser",
                                 description="Respond to a greeting from the user with another friendly greeting.",),
]

reasoning_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, streaming=True).bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["end", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
async def call_model(state: AgentState):
    messages = state['messages']
    response = await reasoning_model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Create the LangGraph agent
def create_workflow():
    # Create the graph
    checkpointer = MemorySaver()
    graph = create_react_agent(reasoning_model, tools, checkpointer=checkpointer)
    return graph


async def run_graph(initial_messages: List[BaseMessage], message_placeholder, thread_id: str):
    workflow = create_workflow()
    full_response = ""
    contextualized_question = ""
    debug_output = ""

    agent_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    async for event in workflow.astream_events({"messages": initial_messages},
                                               agent_config, version="v1"):
        kind = event["event"]
        metadata = event.get("metadata", {})
        tags = event.get("tags", [])
        langgraph_node = metadata.get("langgraph_node", "")

        debug_output += f"Event: {kind}, Node: {langgraph_node}, Tags: {tags}"

        if kind == "on_chat_model_stream":
            # Determine the current step based on tags
            # TODO: THIS IS SO BRITTLE AND BAD, THERE MUST BE A BETTER WAY
            current_step = "UNKNOWN"
            if "seq:step:2" in tags:
                current_step = "contextualize"
            elif "seq:step:3" in tags:
                current_step = "generate"
            debug_output += f", Current Step: {current_step}"

            print("Event = ", event)
            content = event["data"]["chunk"].content
            if content:
                # full_response += content
                # message_placeholder.markdown(full_response + "▌")
                if current_step == "contextualize":
                    debug_output += f"Contextualized question content: {content}\n"
                    contextualized_question += content
                elif current_step == "generate":
                    debug_output += f"RAG tool content: {content}\n"
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

        debug_output += '\n'

    print("Debug output:")
    print(debug_output)
    message_placeholder.markdown(full_response)
    return full_response


# Streamlit UI
async def streamlit_ui():
    st.title("Placer Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Your query here..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Prepare the initial state
            initial_messages = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages
            ]

            # Debug information
            # st.write("Debug - Initial State:")

            # Run the agent
            try:
                full_response = (
                    await run_graph(initial_messages, message_placeholder, st.session_state.session_id))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = ""

            message_placeholder.markdown(full_response)

            # Debug information after response
            # st.write("Debug - Final Response:")
            # st.write(f"Response length: {len(full_response)}")
            # st.write(f"Contextualized question: {contextualized_question}")

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    asyncio.run(streamlit_ui())
