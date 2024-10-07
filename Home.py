import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from utils import utils
from utils.streaming_handler import StreamHandler
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from graph_agent.exam_agent import ExamBotStateMachine
import pandas as pd
load_dotenv()
st.set_page_config(page_title="Obstetrics and gynaecology Bot", page_icon="⭐")
st.header('Obstetrics and gynaecology Bot')
st.write('A way to practice medical exam. This is a Mockup UI for inital version')


class ContextChatbot:
    def __init__(self):
        utils.sync_st_session()
        if 'llm_chain' not in st.session_state:
            st.session_state.llm_chain = ExamBotStateMachine(data_source='./Data/dummy_data.csv')

    def setup_chain(_self):
        return st.session_state.llm_chain
     
    @utils.enable_chat_history
    def chat_tab(self):
        # Tab 1 for chat interaction
        self.chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        utils.display_all_messages()
        if user_query:
            utils.display_msg(user_query, 'user')
            response = self.chain.forward(user_input=user_query)
            utils.display_msg(response, 'assistant')

    def random_text_tab(self):
    # Tab 2 for random text or other 
        try:
            if len(self.chain.state['evaluation']) > 0:
                questions = self.chain.state['questions']
                answers = self.chain.state['answers']
                evaluation = [result.status for result in self.chain.state['evaluation']]
                
                # Create a DataFrame
                data = {
                    'Questions': questions,
                    'Answers': answers,
                    'Evaluation': evaluation
                }
                df = pd.DataFrame(data)
                
                # Display the DataFrame as a table in Streamlit
                st.dataframe(df)
            
            else:
                st.write('Nothing to show for evaluation')
                
        except Exception as e:
            st.write(f"Error: {e}")

    def main(self):
        # Create two tabs
        tab1, tab2 = st.tabs(["Chat", "Evalution"])

        # Chat functionality in tab1
        with tab1:
            self.chat_tab()

        # Random text or other content in tab2
        with tab2:
            self.random_text_tab()


if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
