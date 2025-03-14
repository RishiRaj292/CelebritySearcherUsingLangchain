##INTEGRATE OUR CODE WITHE OPENAI API KEY
import os
import streamlit as st
from constants import openai_key
from langchain_community.llms import OpenAI


os.environ["OPENAI_API_KEY"] = openai_key

#streamlit framework-

st.title('Langchain Demonstration With OPENAI API')
input_text = st.text_input("Search")

llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))

#this is a test update
#this is another update 
#updated