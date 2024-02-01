import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import urllib
import warnings
from pathlib import Path as p
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
warnings.filterwarnings("ignore")

# Set up Streamlit title and sidebar
st.title("Customer Support Chatbot")
st.sidebar.header("User Input")

# Load Google Generative AI API key
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Load Google Generative AI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Prompt template
prompt_template = f"""Answer the question as precisely as possible using the provided context.
                    If the answer is not contained in the context, say "answer not available in context"\n\n
                    Context:\n {{context}}?\n
                    Question:\n {{question}} \n
                    Answer:
                  """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize vector_index outside the button click condition
vector_index = None

# Handle PDF upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Uploading and processing the PDF..."):
        # Create the 'temp' directory if it doesn't exist
        temp_directory = "./temp"
        os.makedirs(temp_directory, exist_ok=True)

        # Save the uploaded PDF to a temporary file
        temp_file_path = os.path.join(temp_directory, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Show loading circle for processing PDF
        st.spinner("Processing the PDF...")

        # Initialize PyPDFLoader with the temporary file path
        pdf_loader = PyPDFLoader(temp_file_path)
        pages = pdf_loader.load_and_split()
        context = "\n".join(str(p.page_content) for p in pages)

        # Remove the temporary file after processing
        # os.remove(temp_file_path)

        # Convert text to vector database
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        texts = text_splitter.split_text(context)

        # Convert text to vector database
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

# Get user input for the question
question = st.sidebar.text_input("Enter your question:")

# Define function to ask a question
def ask_question(que):
    # Get relevant data from vector database
    docs = vector_index.get_relevant_documents(que)

    # Initialize model
    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    # Show loading circle for processing question
    with st.spinner("Processing the question..."):
        stuff_answer = stuff_chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return stuff_answer['output_text']

# Check if the question is asked
if st.sidebar.button("Ask Question"):
    if not question:
        st.warning("Please enter a question.")
    else:
        # Ask the question and display the answer
        with st.spinner("Answering the question..."):
            answer = ask_question(question)
        st.subheader("Answer:")
        st.write(answer)
