import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools.retriever import create_retriever_tool

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# CV retriever from Pinecone namespace
cv_vectorstore = PineconeVectorStore(
    index_name="langchain-test",
    embedding=embeddings,
    namespace="cv"
)

retriever = cv_vectorstore.as_retriever(search_kwargs={"k": 3})

cv_tool = create_retriever_tool(
    retriever,
    name="cv_search",
    description="Search the candidate's CV for skills, experience, and background information."
)

search_tool = TavilySearch(max_results=3)

tools = [search_tool, cv_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a career coach assistant. 
    You MUST follow these steps in order, no exceptions:
    STEP 1: Call cv_search with query "skills experience projects" to read the candidate's CV first
    STEP 2: Call tavily_search to find current AI/ML internship postings
    STEP 3: Compare the CV skills to the job requirements and give specific feedback:
        - List skills the candidate already has that match
        - List specific missing skills with the exact job that needs them
        - Give 3 concrete things to add to the CV immediately
    Do NOT skip step 1. Do NOT give generic advice.
    """
)

response = agent.invoke({
    "input": "Search my CV and find current AI/ML internship postings, then tell me what skills I have, what I am missing, and what I should add to my CV."
})

print(response["messages"][-1].content)