import streamlit as st

def get_user_email():
    # Extract headers using Streamlit's context
    headers = st.context.headers
    
    # The header names may vary; check the specific headers set by Azure App Service
    user_email = headers.get('X-MS-CLIENT-PRINCIPAL-NAME')
    user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID')
    
    return user_email, user_id

st.title("User Email")

# Get the user's email from the headers
if st.button("Start"):
    user_email, user_id = get_user_email()
    
    if user_email or user_id:
        st.write(f"User's email: {user_email}")
        st.write(f"User's ID: {user_id}")
    else:
        st.write("No user email found")
