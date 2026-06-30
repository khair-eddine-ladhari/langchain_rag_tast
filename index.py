import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pip install langchain-tavily

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of india?"),
        ("ai", "New Delhi"),
        ("human", "{question}"),
    ]
)

question = "what is the capital of france?"

llm_chain = template | llm

response = llm_chain.invoke({"question": question})
print(response.content)



loader = TextLoader("document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)


print(f"Number of chunks: {len(chunks)}")
print(chunks[0])




pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "langchain-test"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 outputs 384 dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vectorstore = PineconeVectorStore.from_documents(
    chunks, embeddings, index_name=index_name
)

results = vectorstore.similarity_search("What is RAG used for?", k=3)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print("---")


context = "\n\n".join([doc.page_content for doc in results])

rag_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the following context:\n\n{context}"),
    ("human", "{question}")
])

rag_chain = rag_template | llm

response = rag_chain.invoke({"context": context, "question": "What is RAG used for?"})
print(response.content)