# Create up-to-date database
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

openai_api_key = '2c8668618f3f41a5845cdfd3d72ff8b3'

CHROMA_PATH = "data/medical_database_chroma"
DATA_PATH = "data/medical_sources/"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    # loader = DirectoryLoader(DATA_PATH, glob="*.md")
    # loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    # loader = PyPDFLoader(DATA_PATH)
    loader = PyPDFDirectoryLoader(DATA_PATH)

    documents = loader.load()
    # print(documents)
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    # embedding_function = OpenAIEmbeddings()
    embedding_function = AzureOpenAIEmbeddings( 
        openai_api_key=openai_api_key,
        azure_endpoint="https://cuhk-aiot-gpt4.openai.azure.com/",
        azure_deployment = "text-embedding-ada-002"
        )    
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()