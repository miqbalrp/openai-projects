import streamlit as st
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

st.title('Statistics GPT Assistant')

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

# Steps
# 1. Begin button
# 2. Ask for the analysis objective.
# 3, Confirm the objective

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.button('Begin', on_click=set_state, args=[1])

if st.session_state.stage == 1:
    if "chat" not in st.session_state:
        st.session_state.chat = []
        initial_question = "Please tell me your analysis objective!"
        st.session_state.chat.append({"role": "assistant", "content": initial_question})

    # Display the history of chat
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Display the chat from prompt
    if prompt := st.chat_input("What is up?"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_content = """
        You are a statistics expert to help the user on giving the suggestion of what method that best to used best on the objective of the analysis.
        User will input the objective and the dataset, you will suggest the statistics method.
        Don't make assumption on the method, ask for clarification if user input is ambiguous. 
        Make sure the suggested method is clear, for example if it's t-test, we should know is it an independent or pair-wise, one-tail or two tail, etc.
        """
        user_objective = prompt

        messages = []
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_objective})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_suggested_method",
                    "description": "Determine the best statistics method based on objective from the user and additional context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "description": "The suggested method."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "The explanation why the suggested method is suitable with the analysis objective."
                            }
                            },
                        "required": ["method", "explanation"]
                    }
                }
            },
        ]

        chat_completion = chat_completion_request(messages, tools=tools, tool_choice="auto")
        assistant_message = chat_completion.choices[0].message
        suggested_method_result, tool_message = check_suggested_method(assistant_message)
        messages.append(assistant_message)
        if tool_message:
            messages.append(tool_message)