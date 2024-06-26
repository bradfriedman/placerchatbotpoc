from typing import Literal, List

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.pydantic_v1 import BaseModel, Field

from tools.learn import LearnRetriever
from utils.prompts import GRADE_RELEVANCE_PROMPT, REWRITE_QUERY_PROMPT
from utils.state import AgentState, get_last_human_message


class RelevanceGradeModel(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score ('yes' or 'no')")


class PlacerAgent:
    def __init__(self, session_id: str):
        # Session ID will determine conversational memory thread
        self.session_id = session_id

        # Define a new graph
        self.workflow = StateGraph(AgentState)

        self.learn_retriever = LearnRetriever()

        ## Define the nodes we will cycle between

        # The starting point, the agent that decides the next step(s) to take
        self.workflow.add_node("agent", self.agent)

        # The Learn Tool node for answering questions about Placer, its offerings, or its industry research
        retrieve = ToolNode([self.learn_retriever.retriever_tool])
        self.workflow.add_node("retrieve", retrieve)

        # TODO: Currently not using rewrite, consider removing
        # workflow.add_node("rewrite", rewrite)

        # The node that generates a natural-language response using the given query, context, and chat history
        self.workflow.add_node("generate", self.learn_retriever.generate)

        # Declaring the entry point to be the agent node
        self.workflow.set_entry_point("agent")

        # Decide whether to retrieve context documents for RAG
        self.workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            }
        )

        # After we retrieve context, we generate an answer with an enriched prompt
        self.workflow.add_edge("retrieve", "generate")

        # After we generate the response, we're done!
        self.workflow.add_edge("generate", END)

        # TODO: Not currently using rewrite, consider removing
        # workflow.add_edge("rewrite", "agent")

        # Compile the workflow
        self.graph: CompiledGraph = self.workflow.compile(checkpointer=MemorySaver())

        # Define the tools that are available to our agent
        self.tools = [self.learn_retriever.retriever_tool]

    def agent(self, state: AgentState):
        print("---CALL AGENT---")
        messages = state["messages"]
        agent_model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
        agent_model = agent_model.bind_tools(self.tools, tool_choice="any")
        response = agent_model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    @staticmethod
    def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (AgentState): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """
        print("---CHECK RELEVANCE---")

        # LLM
        model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(RelevanceGradeModel)

        # Prompt
        prompt = PromptTemplate(
            template=GRADE_RELEVANCE_PROMPT,
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

    @staticmethod
    def rewrite_query(state: AgentState) -> dict[str, list[BaseMessage]]:
        """
        Transform the query to produce a better question.

        Args:
            state (AgentState): The current state of the agent

        Returns:
            dict: The updated state with rephrased question
        """
        print("---TRANSFORM QUERY---")
        question = get_last_human_message(state).content

        msg = [HumanMessage(REWRITE_QUERY_PROMPT.format(question=question))]

        # Grader
        grader_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)
        response = grader_model.invoke(msg)
        return {"messages": [response]}

    async def astream_events(self, initial_messages: List[BaseMessage]):
        """
        Stream graph events asynchronously.

        Args:
            initial_messages (List[BaseMessage]): The messages leading up to this part of the conversation

        Yields:
            StreamEvent: The next event in the conversation (such as a token being received or a tool starting)
        """
        graph_config = {"configurable": {"thread_id": self.session_id}}

        async for event in self.graph.astream_events({"messages": initial_messages}, graph_config, version="v1"):
            yield event
