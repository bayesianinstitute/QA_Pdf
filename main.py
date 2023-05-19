import subprocess
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def execute_script(script_path):
    subprocess.run(['streamlit', 'run', script_path])

def login_page():
    st.title("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Verify user credentials
        if username == "admin" and password == "admin_password":
            st.success("Logged in as admin!")
            return True, "admin"
        elif username == "user" and password == "user_password":
            st.success("Logged in as user!")
            return True, "user"
        else:
            st.error("Invalid username or password.")
    return False, None


def main():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status:
        if username == "admin":
            execute_script('app.py')

                    
        if username == "user":
            execute_script('app.py')
            

if __name__ == '__main__':
    main()