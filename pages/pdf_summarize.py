import os
import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
import textwrap
import pathlib
import data
import warnings;
warnings.filterwarnings('ignore')

#main header
html_temp = """
<div style="background-color:orange;padding:10px">
<h3 style="color:white;text-align:center;">Pdf Summary RAG Chatbot with gpt-3.5-turbo</h3>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


if not os.environ['OPENAI_API_KEY']:
    st.write("Please add your OpenAI API key to continue.")
    st.stop()

elif not data.pdf:
    st.write("No pdf file was loaded!")
    st.stop()

openai.api_key = os.environ['OPENAI_API_KEY']
  

def pdf_card():   
    file = pathlib.Path(os.environ["FILE"])
    ext = file.suffix
    st.write("File type:", ext)
    pgs = len(data.pdf)
    st.write("Pages:", pgs)


def page_summary(pdf, selfrom, selto):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0125',max_tokens=1024)
    chain = load_summarize_chain(llm, chain_type='stuff', verbose=False)
    output_summary = chain.invoke(pdf[selfrom:selto])['output_text']
    wrapped_text = textwrap.fill(output_summary, width=100)

    return wrapped_text


def short_summary(pdf):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0125',max_tokens=1024)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_documents(pdf)
    output_summary = chain.invoke(chunks)["output_text"]
    wrapped_text = textwrap.fill(output_summary, width=100)

    return wrapped_text


def detailed_summary(pdf):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0125',max_tokens=1024)
    # prompt for combined summaries
    final_combined_prompt='''
    Provide a final summary of the entire text with at least 1000 words.
    Add a Generic  Title,
    Start the precise summary with an introduction and provide the
    summary in bullet points for the text.
    text: '{text}'
    summary:
    '''
    final_combined_prompt_template = PromptTemplate(input_variables=['text'], template=final_combined_prompt)
    chain = load_summarize_chain(llm=llm, chain_type='map_reduce', combine_prompt=final_combined_prompt_template)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_documents(pdf)
    output_summary = chain.invoke(chunks)["output_text"]
    wrapped_text = textwrap.fill(output_summary, replace_whitespace=False, width=200 )

    return wrapped_text

pgs = 1

st.write("----")
file = pathlib.Path(os.environ["FILE"])
ext = file.suffix
st.write("File type:", ext)
pgs = len(data.pdf)
st.write("Pages:", pgs)
st.write("----")

type = st.radio("Please select the summary type:", 
                ["Summary of Certain Pages","Short Summary of the Entire Document","Detailed Summary of the Entire Document"])

if type == "Summary of Certain Pages":
    selfrom = st.selectbox("From:",list(range(1,pgs+1)))-1
    selto = st.selectbox("To:",list(range(selfrom+2, pgs+1)))
       
btn = st.button("Get summary")

if btn:
    if type == "Summary of Certain Pages":
        st.subheader("Response : ")
        st.text(page_summary(data.pdf, selfrom, selto))
    elif type == "Short Summary of the Entire Document":
        st.subheader("Response : ")
        st.text(short_summary(data.pdf))
    else: 
        type == "Detailed Summary of the Entire Document"
        st.subheader("Response : ")
        st.text(detailed_summary(data.pdf))
