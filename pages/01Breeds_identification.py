import streamlit as st
import pandas as pd


st.title('Breed Identification analysis')
st.header('Breed Identification analysis')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader('DataFrame')
    st.write(df)
    st.subheader('Descriptive Statistics')
    st.write(df.describe())
else:
    st.info('Upload a CSV file')


