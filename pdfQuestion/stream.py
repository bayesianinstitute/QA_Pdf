from streamlit_option_menu import option_menu
import streamlit as st
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader


# Initialize the session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
# --- USER AUTHENTICATION --

hashed_passwords = stauth.Hasher(['abc', 'def']).generate()


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


if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    st.info(f"Welcome !{name}")
    
    # authenticator.logout('Logout', 'main', key='unique_key')
    

    selected = option_menu(
        menu_title=None,
        options=["Home", "Menu", "Application"],
        icons=["house", "envelope", "app"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal")

    if selected == "Home":
        st.write("Home")
    elif selected == "Menu":
        st.write("Menu")
    elif selected == "Application":
        st.write("Application")
# else:
#     st.error("Authentication failed. Please try again.")

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')