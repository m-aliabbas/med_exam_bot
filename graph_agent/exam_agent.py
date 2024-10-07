from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from typing import Literal
from langchain.memory import ChatMessageHistory
import pandas as pd
load_dotenv()

llm = ChatOpenAI(model='gpt-4o',temperature=0)

class ExamBotStateMachine:
    def __init__(self,data_source = '') -> None:
        self.state = {}
        self.construct_state()
        self.data_source = data_source

    def construct_state(self):
        self.state  = {
            'topic':[],
            'data': '',
            'scenario':'',
            'questions':[],
            'answers': [],
            'chat_hisotry': ChatMessageHistory(),
            'evaluation':[]
        }

    def get_random_row(self,df, category_column, specific_value):
        """
        Returns a random row from the DataFrame based on the specific value of a category column.
        """
        filtered_df = df[df[category_column] == specific_value]
        if filtered_df.empty:
            return None
        return filtered_df.sample(n=1,random_state=None).iloc[0]

    def get_docs(self,df_path='', topics=[], category_column='cetagory'):
        """
        Combines text from random rows selected from the DataFrame based on topics.
        Each topic retrieves one random row from the DataFrame with the given category.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the data
            topics (list): List of categories to fetch random rows
            category_column (str): The column in the DataFrame to filter rows by categories
        
        Returns:
            str: Combined text from selected rows or a message if no data is found for a topic.
        """
        df_path = self.data_source
        text_combined = []

        df = pd.read_csv(df_path)

        topics = list(df['cetagory'].unique())[:3]
        self.add_topic(topics)
        for topic in topics:
            text_combined.append('================')  # Separator for each section
            row_data = self.get_random_row(df, category_column, topic)
            
            if row_data is None:
                text_combined.append(f"No data found for topic: {topic}")
            else:
                # Extracting only the string value without additional metadata
                title = row_data['title'] if isinstance(row_data['title'], str) else row_data['title'].item()
                cleaned_text = row_data['cleaned_text'] if isinstance(row_data['cleaned_text'], str) else row_data['cleaned_text'].item()
                
                text_combined.append(title)
                text_combined.append(cleaned_text)
        
        return '\n'.join(text_combined)
    def forward(self,user_input):
        # if len(self.state['answers']) >= 3 and len(self.state['answers']) >= 3:
        #     self.evaluation_generator()
        #     return 'The END'
        resp = ''
        self.state['chat_hisotry'].add_user_message(user_input)
        if self.state['scenario'] == '' or self.state['scenario'] == None:
            self.get_data()
            self.scenario_generator()
            resp = self.state['scenario']
        elif len(self.state['questions'])  == 0:
            resp = self.question_generator()
        else:
            next_node = self.router_node(user_input)
            print(user_input,next_node)
            if next_node == 'QUESTION_NODE':
                self.state['answers'].append(user_input)
                if  len(self.state['answers']) >= 10:
                    self.evaluation_generator()
                    return 'The END'
                resp = self.question_generator()
            elif  next_node == 'CLARIFICATION_NODE':
                last_question = self.state['questions'][-1]
                resp = self.clarification_node(last_question,user_input)
            else:
                resp = self.general_conversation_agent()
        
        self.state['chat_hisotry'].add_ai_message(resp)
        return resp
    def add_topic(self,topic):
        self.state['topic'] = topic
    
    def get_data(self):
        """
        A function to generate data;
        """
        # prompt = ChatPromptTemplate.from_messages(
        # [
        #     (
        #         "system",
        #         "You are a highly specialized data generator agent. You receive a topic from the user and generate a comprehensive, informative description of the topic, "
        #         "ensuring accuracy and depth. The description should be fact-based, logically structured, and written in a professional tone."
        #     ),
        #     ("human", "Please generate a detailed description for the topic: {topic}.")
        # ]
        #  )

        # chain = prompt | llm
        # ai_msg = chain.invoke({"topic":self.state['topic']})
        self.state['data'] = self.get_docs()
        return self.state['data'] 
    

    def scenario_generator(self):
        """
        An Experience Scenario maker that design scenario for interview.
        """
        class Scenario(BaseModel):
            scenario: str = Field(description="The generated scenario for paticular topic")
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional Medical examiner and interviewer tasked with generating a realistic and concise scenario for an interview or exam. "
                "The scenario should provide a real-world situation based on the given topic and data. It should be clear, contextual, and no more than 3 sentences. "
                "The scenario will be returned in JSON format with the key 'scenario'."
            ),
            ("human", "Please generate a real-world scenario for the topic {topic} using the provided data: {data}. Return the scenario in JSON format.")
        ]
         )

        parser = PydanticOutputParser(pydantic_object=Scenario) 
        chain = prompt | llm | parser
        ai_msg = chain.invoke({"topic":self.state['topic'],'data':self.state['data']})
        self.state['scenario'] = ai_msg.scenario
        return ai_msg.scenario + "\n If you want to start the exam please respond with start exam."

    
    def question_generator(self):
        """
        You are an examiner and you need evaluate a candiate 's knowledge of a particular subject.
        You will ask questions based on given scenario, and data. 
        Also you will have access to previous asked questions. There might be previous asked questions.
        You will not answer question your self. You also will not help user in question.
        """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional Medical examiner conducting an evaluation on a specific topics. These are realted to medical office practice, Obstetrics and Gynecology.  "
                    "The questions should progressively assess the depth of the candidate's understanding, ranging from basic to advanced levels. "
                    "You will generate only one question at a time. Not more then one questions"
                    "The question will be a scenario based. But be concise."
                    "Ensure that the questions do not provide answers or hints. Return only the question text without any preamble or additional information."
                    "You will generate question based on topics list, data given, and previous questions if any"
                    "Approximately 30% of the questions on the test will be in the area of Obstetrics, 30% in Gynecology, 30% in Office Practice and Women’s Health, and 10% in Cross Content. The approximate percentage of questions in subcategories is shown below."
                ),
                ("human", "Generate a question for the exam on the topics {topic}, considering the provided scenario: {data} and previous questions: {previous_questions}.")
            ]
        )

        chain = prompt | llm 
        ai_msg = chain.invoke({"topic":self.state['topic'],'data':self.state['data'],
                               'previous_questions':self.state['questions']})
        self.state['questions'].append(ai_msg.content)
        return ai_msg.content
        
    def router_node(self,user_input):
        """
        You are a router node that need to decide  which path to take.
        """
        
        # from typing import Literal
        class PathObj(BaseModel):
            next_node: Literal['QUESTION_NODE', 'LLM_NODE', 'CLARIFICATION_NODE'] = Field(
                description="The next step the state machine should follow based on the user's input. "
                            "Valid options are: 'QUESTION_NODE' if user answer to this question and now we give him next question, 'CLARIFICATION_NODE' if user asking for help in question, "
                            "or 'LLM_NODE' for handling general conversation.",
                example="QUESTION_NODE"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a routing node responsible for determining the next logical step based on user input. "
                    "You will return next_node as QUESTION_NODE if User answered to given question or It is not asking for clarification or general chat. "
                    "If user say I don't know or I quit go to next question"
                    "If user is asking for clarification  or general chat, return CLARIFICATION_NODE or LLM_NODE respectively."
                    "You MUST return a valid JSON object with a key 'next_node' and value being one of ['QUESTION_NODE', 'LLM_NODE', 'CLARIFICATION_NODE']. "
                    "Always ensure the output is in the format"
                ),
                ("human", "Given the user input: {input} on  previous question: {question}, which of the following nodes should be selected: ['QUESTION_NODE', 'LLM_NODE', 'CLARIFICATION_NODE']?")
            ]
        )

        parser = PydanticOutputParser(pydantic_object=PathObj)
        chain = prompt | llm | parser

        # Safeguard against invalid output by wrapping in a try-except block
        try:
            given_question = self.state['questions'][-1] if len(self.state['questions']) > 0 else []
            ai_msg = chain.invoke({"input": user_input, 'question': given_question})
            return ai_msg.next_node
        except Exception as e:
            print(f"Error in parsing router node output: {e}")
            return 'LLM_NODE'  # Default fallback if parsing fails
    
    def clarification_node(self,restate_input,user_response):
        """
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional examiner who provides clarifications when candidates do not understand the question. "
                    "You will rephrase or restate the question in simpler, more understandable language without providing any hints or answers."
                ),
                ("human", "Rephrase the question: {restate_input}. The user responded with: {user_response}.")
            ]
        )

        chain = prompt | llm 
        ai_msg = chain.invoke({"restate_input":restate_input,'user_response':user_response})
        return ai_msg.content
    
    def general_conversation_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an ExamBot conducting an examination. When engaging in general conversation, guide the user back to the exam. "
                    "You may provide friendly or neutral responses, but avoid answering exam-related questions. "
                    "Ensure the output is returned as a JSON object with the key 'text_msg'."
                ),
                MessagesPlaceholder(variable_name="chat_history")
            ]
        )


        chain = prompt | llm 
        ai_msg = chain.invoke({'chat_history':self.state['chat_history'].messages})
        return ai_msg.content
    
    def evaluation_generator(self):
        """
        An Experience Evaluation agent that evaluates Question and Answers.
        """

        class Evaluation(BaseModel):
            status: str = Field(
                description="Indicates whether the user's answer is correct or incorrect. The value must be either 'correct' or 'incorrect'.",
                example="correct"
            )
            score: int = Field(
                description="A score between 1 and 10 based on how accurate the user's answer is. "
                            "Higher scores indicate closer matches to the expected answer.",
                ge=1,
                le=10,
                example=8
            )
            feedback: str = Field(
                description="Constructive feedback for the user, explaining the strengths and weaknesses of their answer. "
                            "Suggestions for improvement can also be included here.",
                example="Good understanding of the topic, but missed key points regarding budget management."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional Medical exam evaluator. Based on the provided scenario, data, question, and user's answer, "
                    "you will assess the quality and correctness of the response. "
                    "You are an experienced board examiner for the American Board of Obstetrics and Gynecology. You evaluate the clinical knowledge obstetrician gynecologists"
                    "  Generally, you are testing the ability of the candidate to:"
                    "1. develop and diagnose, including the necessary clinical, laboratory, and diagnostic"
                    "procedures;"
                    "2. select and apply proper treatment under elective and emergency conditions;"
                    "3. prevent, recognize, and manage complications; and"
                    "4. plan and direct follow-up and continuing care"
                    "You will assign a score from 1 to 10 based on how close the answer is to the correct response, and you will provide constructive feedback. "
                    "Indicate if the answer is 'correct' or 'incorrect'. "
                    "Respond strictly in a valid JSON format with keys: 'status', 'score', and 'feedback'."
                ),
                ("human", "Evaluate the user's response: {answer} to the question: {question}, based on the scenario: {scenario} and data: {data}.")
            ]
        )

        parser = PydanticOutputParser(pydantic_object=Evaluation)

        # Define the LLM chain
        chain = prompt | llm | parser

        questions = self.state['questions']
        answers = self.state['answers']


        questions = questions[:len(answers)]

        pairs_list = [{"scenario": self.state['scenario'], 'data': self.state['data'], 'answer': answers[i], 'question': questions[i]} for i in range(len(questions))]

        try:
            # Run the chain in batch mode for multiple Q&A pairs
            ai_msg = chain.batch(pairs_list)
            self.state['evaluation'] = ai_msg
            return ai_msg

        except Exception as e:
            # Check if JSON parsing failed and handle invalid outputs
            print(f"Error in parsing JSON output: {e}")
            return {"error": "Invalid output received from model. Ensure the output adheres to the required JSON format."}
        
    

if __name__  == "__main__":
    exam_bot = ExamBotStateMachine(data_source='./Data/dummy_data.csv')
    result = ''
    while True:
        if 'END' in result:
            print('Found END Breaking')
            break
        user_resp = input('Please say something: ')
        result = exam_bot.forward(user_resp)
        print(result)

