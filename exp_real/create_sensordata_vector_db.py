# Create vector database of sensor data
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import warnings
warnings.filterwarnings('ignore')

# Set Azure OpenAI key
openai_api_key="2c8668618f3f41a5845cdfd3d72ff8b3"

# Path of sensor data and vector database
user_name = 'cj_t2'
CHROMA_PATH = "data/vector_databases/"+user_name
DATA_PATH = "data/sensor_data/"+user_name+".csv"

# Size of chunking in vector database
SIZE = 100000 

# Code for vector database generation
def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = CSVLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
    main()