import asyncio
import os
import uuid
from typing import Dict, List, Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_debug
from langgraph.checkpoint import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.prebuilt import create_react_agent, ToolNode
from pydantic import BaseModel

import streamlit as st

set_debug(True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Setup functions
def load_system_prompt():
    with open(os.path.join('prompts', 'chatbot_system_prompt.dat')) as f:
        return f.read()


def setup_vector_store(index_name: str, namespace: str, embeddings_model_name: str = "text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=embeddings_model_name)
    return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)


async def greet_user():
    """Respond to a greeting from the user with another friendly greeting."""
    print("ENTERING GREET USER")
    return ["hello"]


# Create the conversational RAG chain
def create_rag_chain(index_name: str, namespace: str, model_name: str = "gpt-4o", temperature: int = 0):
    llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True)
    system_prompt = load_system_prompt()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    vector_store = setup_vector_store(index_name, namespace)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


async def rag_tool(question: str):
    """Use this for complex queries requiring information retrieval."""
    # question = state["messages"][-1].content
    # history = state["messages"][:-1]
    rag_chain = create_rag_chain("articles", "kb")
    response = await rag_chain.ainvoke({"input": question, "chat_history": []})
    return {"messages": [response]}
    #return AgentState(messages=state["messages"] + [AIMessage(content=response["answer"])])


tools = [
    StructuredTool.from_function(coroutine=rag_tool, name="RAGTool",
                                 description="Use this for queries requiring information retrieval about Placer.ai."),
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
                full_response += content
                message_placeholder.markdown(full_response + "▌")
                # if current_step == "contextualize":
                #     debug_output += f"Contextualized question content: {content}\n"
                #     contextualized_question += content
                # elif current_step == "generate":
                #     debug_output += f"RAG tool content: {content}\n"
                #     full_response += content
                #     message_placeholder.markdown(full_response + "▌")
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
