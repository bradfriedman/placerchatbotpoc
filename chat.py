import asyncio
import os
import uuid
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

import streamlit as st

from agent import PlacerAgent

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


async def process_query(initial_messages: List[BaseMessage], message_placeholder):
    # Initialize the LangGraph agent responsible for orchestrating the answer to the user's query
    placer_agent = PlacerAgent(st.session_state.session_id)

    full_response = ""
    debug_output = ""
    async for event in placer_agent.astream_events(initial_messages):
        kind = event["event"]
        metadata = event.get("metadata", {})
        tags = event.get("tags", [])
        langgraph_node = metadata.get("langgraph_node", "")

        debug_output += f"Event: {kind}, Node: {langgraph_node}, Tags: {tags}"

        if kind == "on_chat_model_stream":
            print("Event = ", event)
            content = event["data"]["chunk"].content
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

    message_placeholder.markdown(full_response)
    return full_response


# Streamlit UI
async def streamlit_ui():
    st.title("Placer Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Your query here..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Prepare the initial state
            initial_messages = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages
            ]

            # Run the agent
            try:
                full_response = await process_query(initial_messages, message_placeholder)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = ""

            message_placeholder.markdown(full_response)

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    asyncio.run(streamlit_ui())
