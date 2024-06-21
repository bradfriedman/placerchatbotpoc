import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain


class LearnTool:
    def __init__(self, index_name: str, namespace: str, model_name: str, temperature: int = 0, top_k: int = 6,
                 embeddings_model_name: str = "text-embedding-3-small"):
        self.index_name = index_name
        self.namespace = namespace
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.embeddings_model_name = embeddings_model_name
        self.vector_store = self.setup_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self.system_prompt = self.load_system_prompt()

    @staticmethod
    def load_system_prompt():
        with open(os.path.join('prompts', 'chatbot_system_prompt.dat')) as f:
            return f.read()

    def setup_vector_store(self):
        embeddings = OpenAIEmbeddings(model=self.embeddings_model_name)
        return PineconeVectorStore(index_name=self.index_name, embedding=embeddings, namespace=self.namespace)

    # Create the conversational RAG chain
    def create_rag_chain(self):
        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, streaming=True)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
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

        history_aware_retriever = create_history_aware_retriever(llm, self.retriever, contextualize_q_prompt)

        return create_retrieval_chain(history_aware_retriever, qa_chain)

    async def rag_tool_async(self, question: str):
        """Use this for complex queries requiring information retrieval."""
        rag_chain = self.create_rag_chain()
        response = await rag_chain.ainvoke({"input": question, "chat_history": []})
        return {"messages": [response]}

    def as_structured_tool(self):
        return StructuredTool.from_function(
            name="LearnTool",
            description="A tool for answering questions about Placer.ai, its offerings, and its industry research.",
            coroutine=self.rag_tool_async,
        )