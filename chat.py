import os
import uuid
from typing import Optional, Iterator, Callable

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import Output
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import streamlit as st

set_debug(True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


class PlacerKnowledgeBaseTool:
    def __init__(self, model_name: str = "gpt-4o", temperature: int = 0, index_name: str = "articles",
                 namespace: str = "kb", messages_history_func: Callable[[str], ChatMessageHistory] = None):
        self.messages_history_func = messages_history_func
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True)
        self.system_prompt = self.load_system_prompt()
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

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

        self.vector_store = self.setup_vector_store(index_name, namespace)

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 6})
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)

        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.messages_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    @staticmethod
    def load_system_prompt():
        with open(os.path.join('prompts', 'chatbot_system_prompt.dat')) as f:
            return f.read()

    @staticmethod
    def setup_vector_store(index_name: str, namespace: str,
                           embeddings_model_name: str = "text-embedding-3-small") -> PineconeVectorStore:
        embeddings = OpenAIEmbeddings(model=embeddings_model_name)
        return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)

    def invoke_stream_with_history(self, query: str, session_id: str) -> Iterator[Output]:
        response_stream = self.conversational_rag_chain.stream(
            {"input": query},
            config={"configurable": {
                "session_id": session_id,
            }},
        )
        return response_stream


class PlacerKnowledgeBaseAgent:
    def __init__(self):
        self.index_name = "articles"
        self.namespace = "kb"
        self.response_chunks = []
        self.session_id = self.setup_session()

    @staticmethod
    def setup_session():
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id

    @staticmethod
    def get_messages_history(i: str) -> ChatMessageHistory:
        if "store" not in st.session_state:
            st.session_state.store = {}

        if i not in st.session_state.store:
            st.session_state.store[i] = ChatMessageHistory()

        return st.session_state.store[i]

    def get_answer_from_response_stream(self, s):
        for chunk in s:
            if 'answer' in chunk:
                content = chunk['answer']
                self.response_chunks.append(content)
                yield content

    def streamlit_entrypoint(self):
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
            with st.spinner("Generating response..."):
                kb_tool = PlacerKnowledgeBaseTool(messages_history_func=self.get_messages_history)

                st.session_state.messages.append({"role": "user", "content": query})
                chat_history_message = chat_history_container.chat_message("user")
                chat_history_message.markdown(query)

                assistant_message_container = chat_history_container.chat_message("assistant")

                # Invoke the tool to get LLM response
                response_stream = kb_tool.invoke_stream_with_history(query, self.session_id)

                # Stream response to UI
                assistant_message_container.write_stream(
                    self.get_answer_from_response_stream(response_stream))

                # Finalize the assistant response in the UI chat history container
                new_assistant_message = {"role": "assistant", "content": ''.join(self.response_chunks)}
                st.session_state.messages.append(new_assistant_message)
                self.response_chunks.clear()


if __name__ == "__main__":
    agent = PlacerKnowledgeBaseAgent()
    agent.streamlit_entrypoint()
