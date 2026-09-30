import streamlit as st
from dotenv import load_dotenv
import os 
from google import genai
#LOAD ENVIRONMENT VARIABLES
load_dotenv()
#get api key
api_key = os.getenv("GEMINI_AI_KEY")

#create gemini client
client = genai.Client(api_key=api_key)

#page configuration
st.set_page_config(
    page_title="Gemini AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

#title
st.title("** Gemini AI Chatbot")

st.write("Ask Gemini anything!")

#use prompt
prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Explain Artificial Intelligence in simple words..."
)
 #generate button
if st.button("Generate Response"):
    if prompt:
        with st.spinner("Gemini is thinking...."):
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",contents=prompt
            )
        st.success("Response generated!")   
        st.write(response.text) 
        

