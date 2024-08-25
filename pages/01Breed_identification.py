import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.svm import SVC
from modules.common import show_footer, load_css


###todo:
### 1. st.tab开发两个版本的文件上传，一个是array，一个是plink格式
### 2. 添加2000模型

# 设置页面配置
st.set_page_config(page_title="Breed Identification", page_icon="🐂", layout="centered", initial_sidebar_state="expanded")

@st.cache_data(ttl=3600)
def load_breed_codes():
    """从文件中读取品种代码，传入字典中。"""
    df = pd.read_csv('attachments/breed_code.csv')
    code_breed_dict = pd.Series(df.Breed.values, index=df.Code).to_dict()
    return code_breed_dict

@st.cache_resource(ttl=3600)
def load_model():
    """加载模型。"""
    clf = joblib.load('attachments/SVC_500_best.pkl')
    return clf

def breed_classifier(genotype_array):
    """品种分类函数。"""
    clf = load_model()
    prediction = clf.predict(genotype_array)
    breed_code_dict = load_breed_codes()
    breed_prediction = [breed_code_dict[code] for code in prediction]
    return breed_prediction

def page_frame():
    st.title('Breed Identification Tool')
    st.write('This tool is designed to help you identify different breeds of cattle.')
    st.write('Please upload a genotype file to begin analysis.')
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            gt_df = pd.read_csv(uploaded_file, sep=' ', header=None)
            gt_array = gt_df.to_numpy()
            st.session_state.gt_array = gt_array
            st.session_state.uploaded_file_name = uploaded_file.name
            st.info('Genotype file has been uploaded.')
        except Exception as e:
            st.error(f'Invalid file. Please upload a valid genotype file. Error: {e}')
    
    if st.button('Analyze'):
        if 'gt_array' in st.session_state:
            result = breed_classifier(st.session_state.gt_array)
            st.session_state.result = result  # Saving the result to session state
            st.subheader('Analysis Results')
            st.write(result)
        else:
            st.error("No genotype data to analyze. Please upload a file.")

if __name__ == '__main__':
    load_css()
    page_frame()
    show_footer()