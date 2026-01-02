# Generate vector database for MedDM baseline (RAG for guideline trees retrieval)
from langchain_community.document_loaders import DirectoryLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import argparse

openai_api_key = "2c8668618f3f41a5845cdfd3d72ff8b3"

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=1000)
parser.add_argument("--chunk_overlap", type=int, default=100)
parser.add_argument("--mode", type=str, default='txt', help='txt or pdf')
args = parser.parse_args()

# # txt files, already transformed as tree format.
# CHROMA_PATH = "database/chroma_guideline_tree"
# DATA_PATH = "data/Guideline_trees/"  

# raw guideline pdf files.
CHROMA_PATH = "data/guideline_trees/vector_databases"
DATA_PATH = "data/guideline_trees/txt_guidelines"    

def main(mode):
    generate_data_store(mode)

def generate_data_store(mode):
    documents = load_documents(mode)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents(mode):
    print('Guideline tree formats: ',mode)
    if mode == 'txt':
        loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    if mode == 'pdf':
        loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    # print(documents)
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
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
    mode = args.mode
    main(mode)