import streamlit as st
from modules.common import show_footer, load_css

st.set_page_config(page_title="Home",page_icon="🏠",layout="centered",initial_sidebar_state="expanded")

# 使用Markdown和style标签来设置不同颜色
st.markdown("""
<h1 style='font-size: 42px;'>
    <span style='color: red;'>C</span>attle
    <span style='color: green;'> B</span>reed
    <span style='color: blue;'> I</span>dentification
    <span style='color: orange;'> T</span>ool
    (CBIT)
</h1>
""", unsafe_allow_html=True)


st.write('This tool is designed to help you identify different breeds of cattle.')


st.markdown('![image.png](https://picbed.guoyingwei.top/2024/08/202408051048682.png)')



if __name__ == '__main__':
    load_css()
    show_footer()