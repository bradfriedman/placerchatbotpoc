import os
import uuid

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langsmith import Client as LangsmithClient

from pinecone.grpc import PineconeGRPC, GRPCIndex

import streamlit as st

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

set_debug(True)

langsmith_client = LangsmithClient()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


def get_session_history(i: str) -> BaseChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store = {}

    if i not in st.session_state.store:
        st.session_state.store[i] = ChatMessageHistory()

    return st.session_state.store[i]


with open(os.path.join('prompts', 'chatbot_system_prompt.dat')) as f:
    system_prompt = f.read()

index_name = "articles"
namespace = "elevio_articles"

pc = PineconeGRPC(api_key=PINECONE_API_KEY)
index: GRPCIndex = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
response_chunks = []


def get_answer_from_response_stream(s):
    for chunk in s:
        if 'answer' in chunk:
            content = chunk['answer']
            response_chunks.append(content)
            yield content


def streamlit_entrypoint():
    st.title("Placer Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the user input state if it's not already in the session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    chat_history_container = st.container()
    for message in st.session_state.messages:
        chat_history_message = chat_history_container.chat_message(message["role"])
        chat_history_message.markdown(message["content"])

    input_container = st.container()

    query = input_container.chat_input("Your query here...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        chat_history_message = chat_history_container.chat_message("user")
        chat_history_message.markdown(query)

        query_text = query

        assistant_message_container = chat_history_container.chat_message("assistant")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True, callbacks=[])
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        retriever = vector_store.as_retriever()
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        with st.spinner("Generating response..."):
            response_stream = conversational_rag_chain.stream(
                    {"input": query_text},
                    config={"configurable": {
                        "session_id": session_id
                    }},
            )

            assistant_message_container.write_stream(get_answer_from_response_stream(response_stream))

            new_assistant_message = {"role": "assistant", "content": ''.join(response_chunks)}
            st.session_state.messages.append(new_assistant_message)
            response_chunks.clear()


if __name__ == "__main__":
    streamlit_entrypoint()
