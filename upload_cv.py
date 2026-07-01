import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load CV
loader = TextLoader("cv.txt")
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"CV chunks: {len(chunks)}")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Upload to existing index, CV namespace
vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name="langchain-test",
    namespace="cv"
)

print("CV uploaded successfully")