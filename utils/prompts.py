# Load Grade Relevance prompt
with open("prompts/grade_relevance_prompt.dat", "r") as f:
    GRADE_RELEVANCE_PROMPT = f.read()

# Load Rewrite Query prompt
with open("prompts/rewrite_query_prompt.dat", "r") as f:
    REWRITE_QUERY_PROMPT = f.read()

# Load Learn Tool RAG prompt
with open("prompts/chatbot_system_prompt.dat", "r") as f:
    LEARN_TOOL_SYSTEM_PROMPT = f.read()
