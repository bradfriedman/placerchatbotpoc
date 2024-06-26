import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from utils.prompts import LEARN_TOOL_SYSTEM_PROMPT
from utils.state import AgentState, get_last_human_message


class LearnRetriever:
    def __init__(self, embeddings: str = "text-embedding-3-small", index_name: str = "articles",
                 namespace: str = "kb", top_k: int = 6, llm_model: str = "gpt-4o"):
        self.embeddings = OpenAIEmbeddings(model=embeddings)
        self.vector_store = PineconeVectorStore(index_name=index_name, embedding=self.embeddings, namespace=namespace)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_knowledge_base_entries",
            "Search and return information about Placer, its offerings, and its industry research."
        )
        self.llm_model = llm_model

    def generate(self, state: AgentState, temperature: int = 0):
        """Generate a response to the user's query.

        Args:
            state (AgentState): The current state of the agent
            temperature (int): The temperature (randomness) of the LLM output (0-1)

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
        prompt = PromptTemplate(
            template=LEARN_TOOL_SYSTEM_PROMPT,
            input_variables=["context", "question"],
        )

        # LLM
        llm = ChatOpenAI(model=self.llm_model, temperature=temperature, streaming=True)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question, "chat_history": chat_history_str})
        return {"messages": [response]}