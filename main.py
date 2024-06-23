import streamlit as st
from zeetools import get_index, set_condense_chatengine, set_llm_embed

# page config
st.set_page_config(page_title="Groq RAG ChatBot", page_icon=":space_invader:",
                   layout="centered", initial_sidebar_state="auto", menu_items=None)

# sidebar
with st.sidebar:
    "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/streamlit/llm-examples)"
    st.subheader(
        "🤖 :orange-background[0xZee] Groq :red-background[RAG] ChatBot", divider="orange")
    # GROQ API Control
    st.subheader("🔐 GROQ INFERENCE API", divider="grey")
    # if ("GROQ_API" in st.secrets and st.secrets['GROQ_API'].startswith('gsk_')):
    if ("GROQ_API" in st.secrets):
        st.success('✅ GROQ API Key')
        api_key = st.secrets['GROQ_API']
    else:
        api_key = st.text_input('Enter your Groq API Key', type='password')
        # if not (api_key.startswith('gsk_') and len(api_key) == 56):
        if not (len(api_key) == 56):
            st.warning("Enter a Valid GROQ API key")
        else:
            st.success('✅ GROQ API Key')


# main page
st.subheader(
    "💬 :orange[GROQ] :orange-background[RAG] :orange[CHATBOT] :books:", divider='orange')

# File Control and indexing
st.subheader("📚 RAG DOCUMENTS", divider="grey")
user_file = st.file_uploader("Upload a PDF file :", type="pdf")
# sample_file = st.radio("Or Select a sample file to chat with :", ["data/WEF_Global_Risks_Report_2024.pdf", "data/llm.pdf"], captions = ["World Economic Forum Report 2024", "Arxiv Large Language Model"] , index=None)

if user_file and api_key:
    set_llm_embed(api_key=api_key)
    with st.spinner("Loading and Indexing Data.."):
        try:
            # article = user_file.read()
            # index = load_index_data(user_file, api_key=api_key)
            index = get_index(user_file)
        except Exception as e:
            st.error(f"error in indexing user file : {e}")
    st.success(f"User File {user_file.name} Loaded and Indexed")

    # Set chat engin
    if "chat_engine" not in st.session_state:
        try:
            st.session_state["chat_engine"] = set_condense_chatengine(
                api_key=api_key, index=index)
        except Exception as e:
            st.error(f"error occurred in setting chat engine : {e}")

    # Initialize the chat messages history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant",
                "content": "Ask me a question or tell me what comes in your mind !"}
        ]
    # Display the prior chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat session controls
    with st.sidebar:
        st.subheader("⚙️ CHAT SESSION PARAM.", divider="grey")
        if st.button("Clear Chat Session", use_container_width=True, type="primary"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": ":sparkles: Hi, I'm here to help, How can I assist you today ? :star:"}]
        if st.button("Clear Chat Memory", use_container_width=True, type="secondary"):
            st.session_state["chat_engine"].reset()

    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Your question"):
        # add_to_message_history("user", prompt)
        st.session_state["messages"].append(
            {"role": "user", "content": str(prompt)})

        # Display the new question immediately after it is entered
        with st.chat_message("user"):
            st.write(prompt)
        # If last message is not from assistant, generate a new response
        # if st.session_state["messages"][-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = st.session_state["chat_engine"].stream_chat(prompt)
            response_str = ""
            response_container = st.empty()
            for token in response.response_gen:
                response_str += token
                response_container.markdown(response_str)
            # st.write(response.source)
            # add_to_message_history("assistant", response.response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": str(response.response)})

        # Save the state of the generator
        st.session_state["response_gen"] = response.response_gen
