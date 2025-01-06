import os
import boto3
import streamlit as st
from dotenv import load_dotenv


# Cognito User Pool Details
load_dotenv(".env")
USER_POOL_ID = str(os.getenv("USER_POOL_ID"))
CLIENT_ID = str(os.getenv("CLIENT_ID"))


# Initialize Cognito client
cognito_client = boto3.client("cognito-idp", region_name="us-west-2")


def sign_up(username, password, email):
    try:
        response = cognito_client.sign_up(
            ClientId=CLIENT_ID,
            Username=username,
            Password=password,
            UserAttributes=[
                {"Name": "email", "Value": email},
            ],
        )
        return response
    except Exception as e:
        st.error(f"Error: {e}")


def sign_in(username, password):
    try:
        response = cognito_client.initiate_auth(
            ClientId=CLIENT_ID,
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={
                "USERNAME": username,
                "PASSWORD": password,
            },
        )
        return response["AuthenticationResult"]["IdToken"]
    except Exception as e:
        st.error(f"Error: {e}")


# Streamlit App
def show_auth():
    columns = st.columns([1, 3, 1])
    with columns[1]:
        with st.container(border=True):
            if not st.session_state['isAuth']:
                st.title("KnoWhiz Tutor")
                auth_option = st.radio("Select Option", ["Sign Up", "Sign In"])
                if auth_option == "Sign Up":
                    #username = st.text_input("Username")
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    if st.button("Sign Up"):
                        response = sign_up(email, password, email)
                        st.success("Sign-up successful! Please check and confirm with your email.")

                elif auth_option == "Sign In":
                    username = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    if st.button("Sign In"):
                        token = sign_in(username, password)
                        if type(token)==str:
                            # st.success(f"Signed in! Token: {token}")
                            st.session_state['isAuth'] = True
                            st.rerun()
                        else:
                            st.error("Sign in failed, please try again")
