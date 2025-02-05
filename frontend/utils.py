import streamlit as st


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


# Function to handle follow-up question clicks
def handle_follow_up_click(question: str):
    st.session_state.next_question = question
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )
