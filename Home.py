import streamlit as st
from modules.common import show_footer, load_css

st.set_page_config(page_title="Home",page_icon="🏠",layout="centered",initial_sidebar_state="expanded")

# 使用Markdown和style标签来设置不同颜色
st.markdown("""
<h1 style='font-size: 42px;'>
    CBIT | 
    <span style='color: red;'>C</span>attle
    <span style='color: green;'> B</span>reed
    <span style='color: blue;'> I</span>dentification
    <span style='color: orange;'> T</span>ool
    
</h1>
""", unsafe_allow_html=True)


st.markdown('''
            Cattle is an important livestock that provides meat, milk, and other products to humans.
            Identifying cattle breeds is essential for breeding, management, and conservation.  

            Here, we provide **CBIT** to help you identify different cattle breeds and the genomic breed content based on genotypic data.
            For more information on **breed identification** and **GBC analysis** functions, please refer to the usage instructions of each tool.

            There is a total of **2913 samples** in our dataset, including **49 breeds** from Asia and Europe.
            The details of the sample information are as follows. 
            ''')
            


st.markdown('![image.png](https://picbed.guoyingwei.top/2024/08/202408051048682.png)')



if __name__ == '__main__':
    load_css()
    show_footer()