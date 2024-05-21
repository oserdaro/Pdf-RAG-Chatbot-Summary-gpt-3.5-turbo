import os
import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import textwrap
import data
import warnings
warnings.filterwarnings('ignore')

#main header
html_temp = """
<div style="background-color:orange;padding:10px">
<h3 style="color:white;text-align:center;">Pdf File Q&A Chatbot RAG Chatbot with gpt-3.5-turbo</h3>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


if not os.environ.get('OPENAI_API_KEY'):
    st.write("Please add your OpenAI API key to continue.")
    st.stop()

elif not data.pdf:
    st.write("No pdf file was loaded!")
    st.stop()


openai.api_key = os.environ['OPENAI_API_KEY']

def init_model(pdf):
    global index, loaded_index, chain
   
    # document splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
 
    text_splitter.split_documents(pdf)

    # word embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

    # vectorstores
    index = FAISS.from_documents(documents=pdf,
                                embedding=embeddings,
                                )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")


def get_query(query, k=5):
    matching_results = index.similarity_search(query, k=k)
    return matching_results


def get_answer(query):
    doc_search = get_query(query)
    response = chain.invoke(input={"input_documents": doc_search, "question": query})["output_text"]
    wrapped_text = textwrap.fill(response, width=100)

    return wrapped_text


st.write("----")
user_query = st.text_input("Write your question:")
btn = st.button("Get answer")

if btn:
    if not user_query:
        st.write("No user query was entered!")
        exit()

    init_model(data.pdf)

    result = get_answer(user_query)
    st.subheader("Response : ")
    st.text(result)
