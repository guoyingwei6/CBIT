import streamlit as st
from modules.common import show_footer



st.set_page_config(page_title="About",page_icon="💻",layout="centered",initial_sidebar_state="expanded")
st.title('About this tool')

st.write('This tool is designed to help you identify different breeds of cattle.')


st.header('Contact')
st.write('If you have any questions, please contact us at: yingwei.guo@foxmail.com')
if __name__ == '__main__':
    show_footer()