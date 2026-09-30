import streamlit as st
import ollama 

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="*"
)
st.title("MY AI CHATBOT")
st.caption("Powered by ollama + Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
prompt = st.chat_input("Type your message...")    

if prompt:
    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message("user"):
        st.write(prompt)
    response = ollama.chat(
        model="llama3.2",
        messages=st.session_state.messages
    )   

    answer = response["message"]["content"]

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })

    with st.chat_message("assistant"):
        st.write(answer)

