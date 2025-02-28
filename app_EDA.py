import streamlit as st
import os
import pandas as pd

from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv,find_dotenv
from langchain_groq import ChatGroq




load_dotenv(find_dotenv())

os.environ['GROQ_API_KEY'] = os.getenv('groq_api_key')



st.set_page_config(page_title="TailorTalk Assistant for CSV Data", page_icon="📊", layout="wide")


#welcoming message
st.title("Hello! 👋 i am your TailorTalk's Assistant 🤖 ")

with st.sidebar:
    st.write('🚀 *CSV RAG Agent: Intelligent Data Retrieval from Your CSV Files!*')
    st.caption("**Empower your CSV files with AI-driven Retrieval-Augmented Generation (RAG). Ask questions, get insights, and make data-driven decisions effortlessly!**")
    st.divider()




    st.caption("Made with ❤️ by [TailorTalk](https://tailortalk.ai/)") 



#intilise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("let's get started",on_click=clicked,args=(1,))

if st.session_state.clicked[1]:
    user_csv_file = st.file_uploader("Upload your CSV file", type="csv")
    if user_csv_file is not None:
        user_csv_file.seek(0)
        df = pd.read_csv(user_csv_file,low_memory=False)


        #llm
        # llm=OpenAI(temperature=0,model="gpt-4o")
        llm =ChatGroq(model="llama-3.3-70b-versatile",temperature=0)




        pandas_agent = create_pandas_dataframe_agent(llm,df,verbose=True,allow_dangerous_code=True)


        @st.cache_data()
        def functon_agent():
             st.write("**Data Overview**")
             st.write("The First row data looks like this")
             st.write(df.head())
             coloumns_name = pandas_agent.run("list out all the columns name")
             if isinstance(coloumns_name, list):
                 coloumns_name = ", ".join(coloumns_name)  # Convert list to comma-separated string

             st.header("Columns name you are interested to know ")
             st.subheader("Type the columns name and get visualize data on that")
             st.text(coloumns_name)


        @st.cache_data()
        def function_question_variable(user_question_variable):
            user_question = user_question_variable.capitalize()
            st.line_chart(df,y=[user_question])
            summary_statistics= pandas_agent.run(f"Give me a summary of the statistics of {user_question}")
            st.text(summary_statistics)
            return
        
        @st.cache_data
        def function_question_dataframe(user_question_dataframe):
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

             
        functon_agent()



        st.header("Variable")
        st.subheader("What variable are you interested?")
        user_question_variable = st.text_input('Enter the column name you are interested in:')
        if user_question_variable is not None and user_question_variable !="":

            function_question_variable(user_question_variable)

            st.subheader('Further Study')
        if user_question_variable:
            user_question_dataframe = st.text_input("is there anything else you would like to know about data")
            if user_question_variable is not None and user_question_variable not in ("","no","No"):
                function_question_dataframe(user_question_dataframe)
            if user_question_dataframe is ("no","No"):
                st.write("")