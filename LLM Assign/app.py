import os
from PyPDF2 import PdfReader
from dotted_dict import DottedDict
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd

# Set up HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set Google API Key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyBBb1byoq4oVGIis8cqsISidFzdB5DBUVc'

# Streamlit page configuration
st.set_page_config(
    page_title="TEAM 1 LLM",
    page_icon='tredence-squareLogo-1650044669923.webp',
    layout="wide",
)

# Load the CSV file into a DataFrame
csv_path = "processed_documents.csv"
df = pd.read_csv(csv_path)

# Recreate Document objects from the DataFrame
documents = []
for index, row in df.iterrows():
    doc = Document(
        page_content=row["content"],
        metadata={"pdf_name": row["pdf_name"]}
    )
    documents.append(doc)

# Cache the FAISS retriever
@st.cache_resource
def ret():
    return FAISS.from_documents(documents, embeddings).as_retriever(search_kwargs={"k": 10})

retriever = ret()

# Prompt template for the QA model
PROMPT_TEMPLATE = """
Go through the context and answer the given question strictly based on context.
Context: {context}
Question: {question}
Answer:
"""

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Set up the retrieval QA chain
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)},
                                    retriever=retriever,
                                    return_source_documents=True)

# Streamlit app layout
st.title('LLM Assignment')
st.markdown("## Welcome to the Story Q&A Application!")
st.markdown("Ask questions about the stories and get precise answers based on the content.")

st.markdown("---")

# Input form
with st.form(key='question_form'):
    question = st.text_input("Enter your question here:")
    submit_button = st.form_submit_button(label='Get Answer')

# Display results
if submit_button:
    if question:
        with st.spinner('Generating answer...'):
            result = chain({'query': str(question)})
            st.success('Answer generated!')
            st.write("### Answer")
            st.write(result['result'])
            st.markdown("#### Source Documents")
            for doc in result['source_documents']:
                st.write(f"- {doc.metadata['pdf_name']}")
        st.markdown("---")
    else:
        st.error("Please enter a question to get an answer.")

# Footer
st.markdown("---")
st.markdown("Powered by OpenAI and Google Gemini AI.")
