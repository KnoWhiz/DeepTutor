import streamlit as st
from pipeline.chat_history_manager import save_chat_history


# Function to display the pdf
def previous_page():
    if st.session_state.current_page > 1:
        st.session_state.current_page -= 1


# Function to display the pdf
def next_page():
    if st.session_state.current_page < st.session_state.total_pages:
        st.session_state.current_page += 1


# Function to close the pdf
def close_pdf():
    st.session_state.show_pdf = False


# Function to open the pdf
def file_changed():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# Function to handle chat content
def chat_content():
    st.session_state.chat_history.append(
        {"role": "user", "content": st.session_state.user_input}
    )
    # Save chat history after each new message
    save_chat_history(st.session_state.session_id, st.session_state.chat_history)


# Function to handle follow-up question clicks
def handle_follow_up_click(question: str):
    st.session_state.next_question = question
    
    # Create a temporary chat history for context-specific follow-up questions
    temp_chat_history = []
    
    # Find the last assistant message that generated this follow-up question
    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        msg = st.session_state.chat_history[i]
        if msg["role"] == "assistant" and "follow_up_questions" in msg:
            if question in msg["follow_up_questions"]:
                # Include the context: previous user question and assistant's response
                if i > 0 and st.session_state.chat_history[i-1]["role"] == "user":
                    temp_chat_history.append(st.session_state.chat_history[i-1])  # Previous user question
                temp_chat_history.append(msg)  # Assistant's response
                break
    
    # Store the temporary chat history in session state for the agent to use
    st.session_state.temp_chat_history = temp_chat_history
    
    # Add the new question to the full chat history
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )
    # Save chat history after follow-up question
    save_chat_history(st.session_state.session_id, st.session_state.chat_history)
