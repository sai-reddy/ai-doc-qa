import streamlit as st

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

import os
from io import StringIO

# App framework
def init_streamlit(uploaded_doc, prompt):
    st.title('🦜️🔗 DOC QA GPT')
    uploaded_doc = st.file_uploader("Upload doc to be read by AI", 
        type=['pdf'])
    prompt = st.text_input('Type your question here')
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def init_pinecone():
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )

def index_document(doc):
    index_name = 'ai-doc-qa'
    loader = UnstructuredPDFLoader(doc)
    data = loader.load()

    # Split into smaller ∆documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # Create Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])  

    return Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

def app():
    uploaded_doc, prompt, docsearch, file = None, None, None, None
    init_streamlit(uploaded_doc, prompt)
    init_pinecone()

    if uploaded_doc is not None:
        with open(os.path.join("data", uploaded_doc.name), "wb") as f:
            f.write(uploaded_doc.getvalue())
        file= "./data/" + uploaded_doc.name
        st.write(file)
        docsearch = index_document(file)
        # Create llm and chain to answer questions from docsearch
        llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
        chain = load_qa_chain(llm, chain_type="stuff")
        st.write("AI Processed document, now ask questions")
    else:
        st.write('Please upload document and ask questions')

    if prompt:
        query = prompt
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.write('Answer:' + response)

def cleanup():
    pass

if __name__ == '__main__':    
    try:
        app()
    finally:
        cleanup()