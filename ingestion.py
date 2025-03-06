import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

MODEL_NAME = "models/text-embedding-004"
INDEX_NAME = "medium-blogs-embeddings-index"

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
        f"{os.environ.get('ABSOLUTE_FILE_PATH')}/mediumblog1.txt"
    )
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODEL_NAME
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=INDEX_NAME
    )
    print("finish")
