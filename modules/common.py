import streamlit as st


def show_footer():
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        <p>All rights reserved. ©Institute of Animal Science (IAS) of Chinese Academy of Agricultural Sciences (CAAS). Contact us at <a href="mailto:yingwei.guo@foxmail.com">yingwei.guo@foxmail.com</a></p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)