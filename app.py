import streamlit as st  
from langchain.prompts import PromptTemplate  # ✅ Add this import  
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama  
from langchain.chains import RetrievalQA  
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Generate embeddings  
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# Streamlit file uploader  
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")  

if uploaded_file:  
    with open("temp.pdf", "wb") as f:  
        f.write(uploaded_file.getvalue())  

    # Load PDF text  
    loader = PDFPlumberLoader("temp.pdf")  
    docs = loader.load()

    # Split text into semantic chunks  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  
    documents = text_splitter.split_documents(docs)


    vector_store = FAISS.from_documents(documents, embeddings)
    # Connect retriever  
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})  

    # LLM  
    llm = Ollama(model="gemma:2b")


    # Prompt Template  
    prompt = """  
    1. Use ONLY the context below.  
    2. If unsure, say "I don’t know".  
    3. Keep answers under 7 sentences.  

    Context: {context}  

    Question: {question}  

    Answer:  
    """  
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # RAG pipeline  
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Streamlit UI  
    user_input = st.text_input("Ask your PDF a question:")  

    if user_input:  
        with st.spinner("Thinking..."):  
            response = qa.invoke({"query": user_input})  # ✅ Use `.invoke()`  
            st.write(response["result"])  
