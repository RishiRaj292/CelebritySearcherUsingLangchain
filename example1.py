##INTEGRATE OUR CODE WITHE OPENAI API KEY
import os
import streamlit as st
from constants import openai_key
from langchain_community.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
#for memory-
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

os.environ["OPENAI_API_KEY"] = openai_key

#streamlit framework-

st.title('Celebrity Search Using Langchain')
input_text = st.text_input("Search")

# Dictionary to store session histories
session_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

#Prompt Template-1
first_input_prompt=PromptTemplate(input_variables=['name'],template="Tell me about celebrity {name}")

##1st part of chain of LLM or we can say the first chain of LLM-
llm = OpenAI(temperature=0.8)

# Memory integration using new approach
llm_with_memory = RunnableWithMessageHistory(llm, get_session_history)

chain=LLMChain(llm=llm_with_memory,prompt=first_input_prompt,verbose=True,output_key='person')#the use of this output_key
#the use of this output_key is to LABEL this output and send it as a speicific input to the next prompt/prompt tempalte we want

#Prompt Template-2
second_input_prompt=PromptTemplate(input_variables=['name'],template="When was {name} born")

chain2=LLMChain(llm=llm_with_memory,prompt=second_input_prompt,verbose=True,output_key='dob')#we put label 'dob' on output of 2nd prompt

#Prompt Template-3
third_input_prompt=PromptTemplate(input_variables=['dob'],template="Give 5 things that happened in the world around {dob}" )
chain3=LLMChain(llm=llm_with_memory,prompt=third_input_prompt,verbose=True,output_key='description')

# parent_chain=SimpleSequentialChain(chain=[chain,chain2],verbose=True)
# The following SequentialChain usage does not propagate the config correctly,
# so we'll manually chain the calls below.
# parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'], verbose=True)

if input_text:
    session_id = "user_session"  # Replace with a unique identifier as needed
    # st.write(llm(input_text))

    #next format for when SimpleSequentialgame are we-
    # st.write(parent_chain.run(input_text)
    
    # Manual sequential chaining with config propagation:
    person_output = chain.invoke({'name': input_text}, config={"configurable": {"session_id": session_id}})
    dob_output = chain2.invoke({'name': input_text}, config={"configurable": {"session_id": session_id}})
    description_output = chain3.invoke({'dob': dob_output['dob']}, config={"configurable": {"session_id": session_id}})
    
    response = {
        'person': person_output['person'],
        'dob': dob_output['dob'],
        'description': description_output['description']
    }
    st.write(response)
    
    with st.expander('Person Name'): 
        st.info(get_session_history(session_id).messages)

    with st.expander('Major Events'): 
        st.info(get_session_history(session_id).messages)
