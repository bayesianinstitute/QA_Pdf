import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from pdfminer.high_level import extract_text

def read_pdf_text(dir_path):
    """
    Reads text from all PDF files in the given directory.
    """
    raw_text = ''
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(dir_path, file_name)
            text = extract_text(file_path)
            if text:
                raw_text += text
    return raw_text

def split_text(raw_text):
    """
    Splits the given text into smaller chunks for information retrieval.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

def create_similarity_index(texts):
    """
    Creates a similarity search index for the given texts using embeddings from OpenAI.
    """
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts_batched(texts, embeddings, batch_size=100)
    return docsearch

def get_answer(query, docsearch, chain):
    """
    Gets the answer to the given query using the given similarity search index and language model.
    """
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    return answer

def main():
    # Set up paths and directories
    pdf_folder_path = os.path.abspath('./whitePaperDoc')
    dir_path = '/home/developer/faijan/Generative AI/pdfQuestion/whitePaperDoc/'

    # Check if the directory exists
    if not os.path.isdir(pdf_folder_path):
        print(f"Error: Directory {pdf_folder_path} not found")
        exit()

    # Read text from PDF files
    raw_text = read_pdf_text(dir_path)

    # Split text into smaller chunks
    texts = split_text(raw_text)

    # Create similarity search index
    docsearch = create_similarity_index(texts)

    # Load language model for question answering
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Prompt user for questions and get answers
    while True:
        # Get user query
        query = input("Enter your question (or 'exit' to quit): ")
        print("")
        if query.lower() == "exit":
            break

        # Get answer and print it
        answer = get_answer(query, docsearch, chain)
        print("Solution : ",answer)
        print("")


if __name__ == "__main__":
    main()
