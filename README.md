# "Ask your PDF" Streamlit ChatBot
# RAG Application with Llama-Index, HuggingFace embeddings and Groq Llama3

This repository contains an example of a Retrieval Augmented Generation (RAG) application built using llama-index, HF embeddings, and Groq Llama3. The RAG system combines retrieval-based methods with language models to generate coherent and contextually relevant responses based on uploaded PDF

## Components

1. **llama-index**:
   - llama-index provides the core functionality for handling language models, prompts, and text processing.
   - We use the Llama3 LLM (Large Language Model) from llama-index for text generation.

2. **Chroma**:
   - Chroma is used as the vector store for document embeddings.
   - It organizes and indexes documents based on high-dimensional vectors.

3. **Groq Llama3**:
   - Groq Llama3 is integrated for querying and retrieving relevant documents.
   - It combines Groq queries with Llama3 embeddings to fetch contextually relevant information from PDF.


## Usage

1. **Installation**:
   - Install the required Python packages using `pip install -r requirements.txt`.

2. **Configuration**:
   - Set up your Groq API key and other necessary credentials.

3. **Run the RAG System**:
   - Initialize the RAG system with llama-index and Groq Llama3 on Streamlit App
   - Provide your PDF and retrieve contextually relevant information from it with RAGBOT.

## To run App
```python
streamlit run main.py

