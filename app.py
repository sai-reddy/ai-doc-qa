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

CHUNK_SIZE = 2000
PINECONE_INDEX_NAME = 'ai-doc-qa'

# App framework
def init_streamlit():
    st.title('🦜️🔗 DOC QA GPT')
    st.file_uploader("Upload doc to be read by AI", 
        type=['pdf'], on_change=upload_and_index, key='current_doc')
    st.session_state['prompt'] = st.text_input('Type your question here')
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

def upload_and_index():
    with st.spinner('Wait for AI to process document'):
        uploaded_doc = st.session_state['current_doc']
        with open(os.path.join("data", uploaded_doc.name), "wb") as f:
            f.write(uploaded_doc.getvalue())
            file= "./data/" + uploaded_doc.name
            st.write(file)
        index_document(file)
    st.success("AI Processed document, now ask questions")

def index_document(doc):
    loader = UnstructuredPDFLoader(doc)
    data = loader.load()

    # Split into smallest docs possible
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # If vector count is nearing free limits delete index and recreate it
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():        
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, metric='cosine')

    # Create Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])  
    st.session_state['pinecone_index'] = Pinecone.from_texts([t.page_content for t in texts],
                                                              embeddings, index_name=PINECONE_INDEX_NAME)

def app():
    init_streamlit()
    init_pinecone()
    
    # Create llm and chain to answer questions from pinecone index
    llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
    chain = load_qa_chain(llm, chain_type="stuff")

    if st.session_state['prompt']:
        query = st.session_state['prompt']
        docs = st.session_state['pinecone_index'].similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.write('Answer:' + response)

def cleanup():
    pass

if __name__ == '__main__':    
    try:
        app()
    finally:
        cleanup()