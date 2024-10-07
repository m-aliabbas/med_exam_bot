import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
logger = get_logger('Langchain-Chatbot')

#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):
        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        # for msg in st.session_state["messages"]:
        #     st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_all_messages():
    for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])


# def enable_chat_history(*decorator_args):
#     def decorator(func):
#         if os.environ.get("OPENAI_API_KEY"):
#             # To clear chat history after switching chatbot
#             current_page = func.__qualname__
#             if "current_page" not in st.session_state:
#                 st.session_state["current_page"] = current_page
#             if st.session_state["current_page"] != current_page:
#                 try:
#                     st.cache_resource.clear()
#                     del st.session_state["current_page"]
#                     del st.session_state["messages"]
#                 except:
#                     pass

#             # To show chat history on UI
#             if "messages" not in st.session_state:
#                 st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
#             for msg in st.session_state["messages"]:
#                 st.chat_message(msg["role"]).write(msg["content"])

#         def execute(*args, **kwargs):
#             # Pass the decorator arguments (if needed) to the function
#             func(*args, **kwargs, *decorator_args)
#         return execute
#     return decorator

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
    

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
        )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key

def configure_llm():
    available_llms = ["gpt-4o","gpt-4o-mini"]
    selected_llm = available_llms[0]
    llm = ChatOpenAI(model_name=selected_llm, temperature=0, streaming=True, api_key=os.environ["OPENAI_API_KEY"])
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}. Here is chat history ignore it {history}",
        ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    return chain

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
