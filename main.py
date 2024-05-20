import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFium2Loader
from data import set_pdf

# page title
st.title(":orange[Pdf Q&A and Summary RAG Chatbot with gpt-3.5-turbo]")
st.subheader("Use the sidebar menu to select different functions to analyse your pdf.")

sidebar_api_msg = st.sidebar.empty()
api_msg = st.empty()
logout_btn = st.sidebar.empty()


def addapi():
    openai_api_key = sidebar_api_msg.text_input("openai api key", key="OPENAI_API_KEY", type="password")
    os.environ['OPENAI_API_KEY'] = openai_api_key
    load_pdf()

def logout():
    del os.environ['OPENAI_API_KEY']
    if os.environ.get('FILE'):
        del os.environ['FILE']
    
def load_pdf():
    if os.environ.get('OPENAI_API_KEY'):
        if not os.environ.get('FILE'):
            # streamlit file uploader
            uploaded_file = st.file_uploader('Choose a pdf file', type='pdf')
            if uploaded_file is not None:
                try:
                    # save the file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                
                    if not os.path.exists(tmp_file_path):
                        raise ValueError(f"The file does not exist in this path: {tmp_file_path} ")

                    # make sure that the file is pdf
                    if not tmp_file_path.lower().endswith('.pdf'):
                        raise ValueError("The file must be a .pdf file")

                    # loading the pdf file using PyPDFium2Loader
                    file_loader = PyPDFium2Loader(tmp_file_path)
                    pdf = file_loader.load()

                    if pdf:
                        set_pdf(pdf)
                        st.success('Success!')
                        os.environ['FILE'] = tmp_file_path
            
                except Exception as e:
                    st.error(f"Error in reading the file: {e}")


if not os.environ.get('OPENAI_API_KEY'):
    addapi()

if not os.environ.get('OPENAI_API_KEY'):
    api_msg.info("Please add your OpenAI API key to continue.")

else:
    sidebar_api_msg.info("API key was entered. You can now select the pdf file to analyse.")
    btn = logout_btn.button("Log out")
    if btn:
        logout()
        api_msg.info("Please add your OpenAI API key to continue.")
        logout_btn.empty()
        addapi()


