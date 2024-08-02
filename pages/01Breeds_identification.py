import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.svm import SVC

###todo:
### 1. st.tab开发两个版本的文件上传，一个是array，一个是plink格式
### 2. st.session_state保存用户上传的文件


# 从文件中读取品种代码，传入字典中
def load_breed_codes():
    # 从CSV文件中读取数据
    df = pd.read_csv('attachments/breed_code.csv')
    # 转换DataFrame为字典，其中第二列是键，第一列是值
    code_breed_dict = pd.Series(df.Breed.values, index=df.Code).to_dict()
    return code_breed_dict


# 加载模型
def load_model():
    # 加载模型
    clf = joblib.load('attachments/SVC_500_best.pkl')
    return clf


# 定义品种分类函数
def breed_classifier(genotype_array):
    clf = load_model()
    prediction = clf.predict(genotype_array)
    breed_code_dict = load_breed_codes()
    breed_prediction = [breed_code_dict[code] for code in prediction]
    return breed_prediction


def page_frame():
    st.title('Breed Identification Tool')
    st.write('This tool is designed to help you identify different breeds of cattle.')
    st.write('Please upload a genotype file to begin analysis.')
    
    # 文件上传，读取文件为df
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            gt_df = pd.read_csv(uploaded_file, sep=' ', header=None)
            gt_array = gt_df.to_numpy()
            st.info('Genotype file have been uploaded.')
        except:
            st.error('Invalid file. Please upload a valid genotype file.')
    else:
        st.info('Upload you genotype file to begin analysis.')
    
    # 点击按钮，调用品种分类函数，显示结果
    # 按钮总是显示，并且检查它是否被点击
    if st.button('Analyze'):
        # 只有当按钮被点击时，才调用品种分类函数
        result = breed_classifier(gt_array)
        st.subheader('Analysis Results')
        st.write(result)
    else:
        # 可以在这里添加一些默认显示，或者仅保持显示按钮
        st.write("Click the button above to perform breed assignment.")



# 主应用逻辑，调用开始分析的函数
if __name__ == '__main__':
    page_frame()
    load_breed_codes()
    load_model()