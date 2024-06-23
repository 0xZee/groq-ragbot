import streamlit as st
import tempfile
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.cohere import CohereEmbedding
# from llama_index.embeddings.gemini import GeminiEmbedding


# SETTING : LLM, EMBEDDINGS
def set_llm_embed(api_key):
    # Settings.system_prompt="You are an expert on human-computer interaction and your job is to answer questions based of the context provided. Assume that all questions are related to the context. Keep your answers based on facts from this knowledge and respond providing low-level details. Do not hallucinate facts or make up answers."
    Settings.llm = Groq(model="llama3-8b-8192",
                        temperature=0.3, api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5")


### INDEX ###
# LOAD AND INDEX DATA UPDATED
@st.cache_resource(show_spinner=False)
def get_index(user_file):
    """Chunk the PDF & store it in persistant Vector Store."""
    if user_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, user_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(user_file.getvalue())

        loader = SimpleDirectoryReader(input_files=[temp_file_path])
        documents = loader.load_data()
        # storage_context = StorageContext.from_defaults()

        index = VectorStoreIndex.from_documents(
            documents,
            # storage_context=storage_context,
        )
        st.info(f"PDF loaded into vector store in {len(documents)} documents")
        return index
    return None


### CHAT ENGINES ###

# Set ChatEngine : Simple Chat Mode
def set_simple_chat_engine(api_key):
    llm_groq = Groq(model="llama3-8b-8192", temperature=0.3, api_key=api_key)
    chat_engine = SimpleChatEngine.from_defaults(
        llm=llm_groq,
        memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        system_prompt=("""
      You are a kind and talkative chatbot, able to have normal interactions, as well as providing low-level detailed responses.\n
      \nInstruction: Use the previous chat history, to interact and help the user.
      """),
        verbose=True)
    return chat_engine


# Set ChatEngine : Condense Chat Mode
def set_condense_chatengine(api_key, index):
    llm_groq = Groq(model="llama3-8b-8192", temperature=0.2, api_key=api_key)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm_groq,
        memory=memory,
        context_prompt=("""
            You are a kind and talkative chatbot, able to have normal interactions, as well as provide low-level detailed responses.
            Here are the relevant documents for the context:\n"
            {context_str}
            \nInstruction: Use the previous chat history, or the context above, to interact and help the user. If you don't know, juste respond so, do not make up answers.
            """),
        verbose=True,
    )
    return chat_engine
