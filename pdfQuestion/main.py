import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
import streamlit_authenticator as stauth


import yaml
from yaml.loader import SafeLoader



load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def get_answer(querys,db,chains):
    doc=db.similarity_search(querys)
    return chains.run(input_documents=doc,question=querys)

def main():
    st.title("Ask Anything About Blockchain and CryptoCurrency")

    embedding=OpenAIEmbeddings()
    new_db=FAISS.load_local("PdfEmbeedingdb",embedding)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Prompt user for questions and get answers
    query = st.text_input("Enter your question:")
    if query:
        if query.lower() == "exit":
            st.stop()

        # Get answer and print it
        answer = get_answer(query, new_db, chain)
        st.write("Question:", query)
        st.write("Solution:", answer)

if __name__ == "__main__" :
    main()