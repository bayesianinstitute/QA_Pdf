import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from streamlit_option_menu import option_menu



import yaml
from yaml.loader import SafeLoader


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def get_answer(querys,db,chains):
    doc=db.similarity_search(querys)
    return chains.run(input_documents=doc,question=querys)

def login_page():
    st.title("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Verify user credentials
        if username == "user" and password == "password":
            st.success("Logged in as user!")
        else:
            st.error("Invalid username or password.")
        
def main():
    st.set_page_config(page_title="Blockchain and Cryptocurrency QA", page_icon=":guardsman:", layout="wide")

    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload PDF","User Login"],
        icons=["house", 'upload','box-arrow-in-right'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal")
    if selected == "User Login" :
        login_page()
        
    if selected == "Home":
        st.title("Ask Anything About Blockchain and Cryptocurrency")

        embedding=OpenAIEmbeddings()
        db=FAISS.load_local("PdfEmbeedingdb",embedding)
        # db2=FAISS.load_local("NewDB",embedding)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # Prompt user for questions and get answers
        query = st.text_input("Enter your question:",key="QA")
        if query:
            if query.lower() == "exit":
                st.stop()

            # Get answer and print it
            answer = get_answer(query, db, chain)
            st.write("Question:", query)
            st.write("Solution:", answer)

    elif selected == "Upload PDF":
        st.title("Upload PDFs FOR QA")

        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            embedding=OpenAIEmbeddings()
            text_splitter = CharacterTextSplitter(
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200,
                length_function = len,
            )
            raw_text = ""
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            raw_text += text

            texts = text_splitter.split_text(raw_text)
            db = FAISS.from_texts(texts, embedding)
            db.save_local("userdb")
            # st.success("Embeddings created successfully!")
            
            chain = load_qa_chain(OpenAI(), chain_type="stuff")

            # Prompt user for questions and get answers
            query = st.text_input("Enter your question:",key="Upload")
            if query:
                if query.lower() == "exit":
                    st.stop()

                # Get answer and print it
                answer = get_answer(query, db, chain)
                st.write("Question:", query)
                st.write("Solution:", answer)

if __name__ == '__main__':
    main()
