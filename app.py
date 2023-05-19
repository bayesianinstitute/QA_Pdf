import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import InvalidRequestError
from streamlit_chat import message
from langchain.callbacks import get_openai_callback


def ask_question(question, qa, chat_history):
    try:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        st.write(f"-> **Question**: {question}")
        st.write(f"**Answer**: {result['answer']}")
    except InvalidRequestError:
        st.write("Try another chain type as the token size is larger for this chain type")


def display_chat_history(chat_history):
    
    for i, (question, answer) in enumerate(chat_history):
        st.info(f"Question {i + 1}: {question}")
        st.success(f"Answer {i + 1}: {answer}")
        st.write("----------")


def load_embeddings(api_key):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("PdfEmbeedingdb", embedding)
    return embedding, db


def process_uploaded_files(uploaded_files,api_key):
    embedding, db = load_embeddings(api_key)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
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

    db1 = FAISS.load_local("PdfEmbeedingdb", embedding)
    db.merge_from(db1)
    db.save_local("PdfEmbeedingdb")

    return db



def ui():

    expander = st.sidebar.expander("**Difference between Conversational Chain Types**")

    # Inside the expander, create the columns and buttons
    with expander:
        col1, col2, col3 = st.columns(3)
        button_width = 200  # Adjust the width of the buttons as needed
        
        with col1:
            if st.button("**Stuff**", key="button1",help="Click to get info"):
                st.sidebar.markdown("It uses all of the text from the documents in the prompt. It may cause rate-limiting errors if it exceeds the token limit.")
                if st.sidebar.button("Close",help="click to close info"):
                    st.sidebar.empty()
                    
        with col2:
            if st.button("**Map Reduce**", key="button2",help="Click to get info"):
                st.sidebar.markdown("It separates texts into batches and feeds each batch with the question to the language model separately. The final answer is based on the answers from each batch.")
                if st.sidebar.button("Close",help="click to close info"):
                    st.sidebar.empty()
                    
        with col3:
            if st.button("**Refine**", key="button3",help="Click to get info"):
                st.sidebar.markdown("It separates texts into batches and feeds the first batch to the language model. Then it feeds the answer and the second batch to the language model and refines the answer by going through all the batches.")
                if st.sidebar.button("Close",help="click to close info"):
                    st.sidebar.empty()

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["history"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.clear()



def main():

    st.title("🤷Ask Your Blockchain Expert 🕵️‍♂️")

    with st.sidebar.expander("API Key Input"):
        with st.form(key="api_form"):
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button and api_key:
                # Perform actions using the API key
                st.success("API key submitted:")



    if api_key:
        ui()
        embedding, db = load_embeddings(api_key)
        

        with st.sidebar.expander("File Uploader") :

            uploaded_files = st.file_uploader(
                "Choose PDF files", type=["pdf"], accept_multiple_files=True
            )

        chain_type = st.sidebar.select_slider(
            "**Select Conversational Chain Type**",
            options=["stuff", "map_reduce", "refine"],
            key="chain_type",
        )
        
        if uploaded_files:
            db = process_uploaded_files(uploaded_files,api_key)

        retriever = db.as_retriever()
        model = OpenAI(temperature=0, openai_api_key=api_key)

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about Blockchain 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
                )
        # st.session_state.me
        st.sidebar.button("New Chat", on_click = new_chat, type='primary')

        

        chain = ConversationalRetrievalChain.from_llm(
            llm=model, memory=st.session_state.memory, retriever=retriever, chain_type=chain_type
        )

            #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        def conversational_chat(query):
            try:
        
                with get_openai_callback() as cb:
                    result = chain({"question": query, "chat_history": st.session_state['history']})
                    st.session_state['history'].append((query, result["answer"]))
                    # expander = st.sidebar.expander("Token Details")

                    # if expander.button("Show token"):
                    #     expander.write(f"Total Tokens: {cb.total_tokens}")
                    #     expander.write(f"Prompt Tokens: {cb.prompt_tokens}")
                    #     expander.write(f"Completion Tokens: {cb.completion_tokens}")
                    #     expander.write(f"Total Cost (USD): ${cb.total_cost}")
                                
                    return result["answer"]
                
            
            except InvalidRequestError:
                st.write("Try another chain type as the token size is larger for this chain type")
        

        # Allow to download as well
        download_str = []
        with container:
            
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Type Your Query (:", key='input')
                submit_button = st.form_submit_button(label='Send',type='primary')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)

                
                    
                
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                
                
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
                            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
                            download_str.append(st.session_state["past"][i])
                            download_str.append(st.session_state["generated"][i])
                
                
    
            
            
            # download_str = '\n'.join(download_str)
            # if download_str:
            #         st.sidebar.download_button('Download Conversion',download_str)

            with st.sidebar.expander("**View Chat History**"):
                display_chat_history(st.session_state.history)

            if st.session_state.history:   
                if st.button("Clear-all",help="Clear all chat"):
                    st.session_state.history=[]


            # Allow the user to clear all stored conversation sessions
        # if st.button("Download Chat History"):
        #     # Create a string representation of the chat history
        #     chat_history_str = "\n".join([f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n----------" for i, (q, a) in enumerate(st.session_state.memory)])

        #     # Save the chat history to a text file
        #     with open("QA_chat_history.txt", "w") as file:
        #         file.write(chat_history_str)

        #     st.success("Chat history downloaded successfully!")

            


    else :
        st.sidebar.warning("Please Enter Api Key")    




if __name__ == '__main__':
    main()
