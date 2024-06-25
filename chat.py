import asyncio
import os
import uuid
from typing import List, Annotated, TypedDict, Literal, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name="articles", embedding=embeddings, namespace="kb")
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_knowledge_base_entries",
    "Search and return information about Placer, its offerings, and its industry research."
)


def get_last_human_message(state: AgentState) -> BaseMessage:
    messages = state["messages"]
    last_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    return last_human_message


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = get_last_human_message(state).content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score  # noqa

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]
    agent_model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    agent_model = agent_model.bind_tools([retriever_tool])
    response = agent_model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state: AgentState):
    """
    Transform the query to produce a better question.

    Args:
        state (AgentState): The current state of the agent

    Returns:
        dict: The updated state with rephrased question
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = get_last_human_message(state).content

    msg = [
        HumanMessage(
            content=f""" \n
Look at the input and try to reason about the underlying semantic intent. Here is the initial question:
\n ------- \n
{question}
\n ------- \n
Formulate an improved question: """,
        )
    ]

    # Grader
    grader_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)
    response = grader_model.invoke(msg)
    return {"messages": [response]}


def generate(state: AgentState):
    """Generate a response to the user's query.

    Args:
        state (AgentState): The current state of the agent

    Returns:
        dict: The updated state with rephrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = get_last_human_message(state).content
    last_message = messages[-1]

    docs = last_message.content

    # Extract chat history
    chat_history = []
    for msg in messages[:-1]:  # Excluding the last message, which is the most recent query
        if isinstance(msg, HumanMessage):
            chat_history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            chat_history.append(f"AI: {msg.content}")
    chat_history_str = "\n".join(chat_history)

    # Prompt
    with open(os.path.join('prompts', 'chatbot_system_prompt.dat')) as f:
        prompt_str = f.read()
    prompt = PromptTemplate(
        template=prompt_str,
        input_variables=["context", "question"],
    )

    # LLM
    llm = ChatOpenAI(model='gpt-4o', temperature=0, streaming=True)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question, "chat_history": chat_history_str})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
# workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.set_entry_point("agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    }
)

workflow.add_edge("retrieve", "generate")

workflow.add_edge("generate", END)
# workflow.add_edge("rewrite", "agent")

# Compile the workflow
graph: CompiledGraph = workflow.compile(checkpointer=MemorySaver())

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.session_id}}


async def run_graph(initial_messages: List[BaseMessage], message_placeholder):
    full_response = ""
    debug_output = ""
    print("ENTERING RUN GRAPH 2")
    async for event in graph.astream_events({"messages": initial_messages}, config, version="v1"):
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


# async def run_graph(initial_messages: List[BaseMessage], message_placeholder, thread_id: str):
#     workflow = create_workflow()
#     full_response = ""
#     contextualized_question = ""
#     debug_output = ""
#
#     agent_config = {
#         "configurable": {
#             "thread_id": thread_id,
#         }
#     }
#
#     async for event in workflow.astream_events({"messages": initial_messages},
#                                                agent_config, version="v1"):
#         kind = event["event"]
#         metadata = event.get("metadata", {})
#         tags = event.get("tags", [])
#         langgraph_node = metadata.get("langgraph_node", "")
#
#         debug_output += f"Event: {kind}, Node: {langgraph_node}, Tags: {tags}"
#
#         if kind == "on_chat_model_stream":
#             # Determine the current step based on tags
#             # TODO: THIS IS SO BRITTLE AND BAD, THERE MUST BE A BETTER WAY
#             current_step = "UNKNOWN"
#             if "seq:step:2" in tags:
#                 current_step = "contextualize"
#             elif "seq:step:3" in tags:
#                 current_step = "generate"
#             debug_output += f", Current Step: {current_step}"
#
#             print("Event = ", event)
#             content = event["data"]["chunk"].content
#             if content:
#                 # full_response += content
#                 # message_placeholder.markdown(full_response + "▌")
#                 if current_step == "contextualize":
#                     debug_output += f"Contextualized question content: {content}\n"
#                     contextualized_question += content
#                 elif current_step == "generate":
#                     debug_output += f"RAG tool content: {content}\n"
#                     full_response += content
#                     message_placeholder.markdown(full_response + "▌")
#         elif kind == "on_tool_start":
#             print("--")
#             print(
#                 f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
#             )
#         elif kind == "on_tool_end":
#             print(f"Done tool: {event['name']}")
#             print(f"Tool output was: {event['data'].get('output')}")
#             print("--")
#
#         debug_output += '\n'
#
#     print("Debug output:")
#     print(debug_output)
#     message_placeholder.markdown(full_response)
#     return full_response


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
                    #await run_graph(initial_messages, message_placeholder, st.session_state.session_id))
                    await run_graph(initial_messages, message_placeholder))
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
    #streamlit_ui()
    asyncio.run(streamlit_ui())
