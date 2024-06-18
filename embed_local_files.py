import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone.grpc import PineconeGRPC, GRPCIndex

from dotenv import load_dotenv
load_dotenv()


def create_loader(file):
    name, extension = os.path.splitext(file)
    with open(file, 'rb') as f:
        if extension == '.html':
            from langchain.document_loaders import UnstructuredHTMLLoader
            print(f'load {file}...')
            loader = UnstructuredHTMLLoader(f)
        elif extension == '.txt':
            from langchain.document_loaders import TextLoader
            print(f'load {file}...')
            loader = TextLoader(f)
        elif extension == '.pdf':
            from langchain_community.document_loaders import PyMuPDFLoader
            print(f'load {file}...')
            loader = PyMuPDFLoader(file)
        elif extension == '.docx':
            from langchain.document_loaders import Docx2txtLoader
            print(f'load {file}...')
            loader = Docx2txtLoader(f)
        else:
            print('The document format is not supported!')
            return None

    return loader

def main():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    index_name = "articles"
    namespace = "whitepapers"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pc = PineconeGRPC(api_key=pinecone_api_key)
    index: GRPCIndex = pc.Index(index_name)

    # Iterate through all files insider ./data (recursively) and create a document loader for each one
    loaders = []
    for root, dirs, files in os.walk('./data'):
        for file in files:
            filepath = os.path.join(root, file)
            new_loader = create_loader(filepath)
            if new_loader:
                loaders.append(new_loader)

    # Load all the documents
    all_documents = []
    for loader in loaders:
        all_documents.extend(loader.load())

    # Embed and upsert the documents
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        all_documents,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )








if __name__ == "__main__":
    main()