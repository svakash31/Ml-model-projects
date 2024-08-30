from dotenv import load_dotenv
load_dotenv()  # Load environment variables

import streamlit as st
import os
import google.generativeai as genai

# Configure the Gemini API with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get responses tailored for career guidance
def get_gemini_response(question):
    # Prompt template for career guidance
    prompt = f"""
    You are an AI expert in career counseling. A user has asked for career guidance with the following question:
    "{question}"
    
    Please provide insightful, helpful advice considering the user's potential career paths, skill development, and opportunities for growth. Focus on being supportive and practical.
    """
    response = chat.send_message(prompt, stream=True)
    return response

# Initialize the Streamlit app
st.set_page_config(page_title="Career Guidance Chatbot")

st.header("Gemini Career Guidance Chatbot")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Ask your career-related question:", key="input")
submit = st.button("Get Advice")

if submit and user_input:
    response = get_gemini_response(user_input)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", user_input))
    st.subheader("Response:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

st.subheader("Chat History:")

for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
