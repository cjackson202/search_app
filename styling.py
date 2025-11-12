import streamlit as st


def global_page_style2():  
    st.set_page_config(layout="wide")  
    with open('style.css') as f:
        css = f.read()
    # Note: Using unsafe_allow_html for CSS only - ensure style.css is controlled and sanitized
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)