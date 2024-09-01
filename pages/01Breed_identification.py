import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.impute import SimpleImputer
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
def load_model_fast():
    """加载模型。"""
    clf = joblib.load('attachments/SVC_500_best.pkl')
    return clf

def load_model_accurate():
    """加载模型。"""
    clf = joblib.load('attachments/SVC_2000_best.pkl')
    return clf

def breed_classifier(genotype_array, model='fast'):
    """品种分类函数。"""
    if model == 'fast':
        clf = load_model_fast()
    elif model == 'accurate':
        clf = load_model_accurate()
    prediction = clf.predict(genotype_array)
    breed_code_dict = load_breed_codes()
    breed_prediction = [breed_code_dict[code] for code in prediction]
    return breed_prediction

def page_frame():
    st.title('Breed Identifier')
    st.info('''
            ## Introduction
            This tool is designed to help you identify different breeds of cattle.
            To determine the best machine learning classification model, we collected 49 breeds and 2913 samples mentioned on the home page.
            
            After comparing the performance of different models, 
            we found the workflow using RF as feature selector and SVM as classifier has the best performance.
            For more detailed information on the accuracy of different models and factors influencing the accuracy, please refer to our paper.  
            
            Here, we provide the classification models with 500 and 2000 SNPs, respectively. 
            You can choose the model according to your data and expectations.

            ''')
    
    st.warning('''
            ## Usage

            **1. Preparing the genotype file.**
            - The genotype file must contain and only contain the SNPs we specify. 
            The locations of them based on **ARS-UCD2.0** are available here 
            for **[fast model](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/svc_500.map)**
            and **[accurate model](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/svc_2000.map)**. 
            - Recoded by **0, 1, and 2**, representing the genotypes AA, AB, and BB, respectively.
            - **One individual per line** and **one SNP per column**.
            - The **first column** should be the **sample name**, which is used as index and displayed in the results.
            - **Space or tab-separated** text file.
            - **Missing values (NA)** do not affect the analysis, but affect the accuracy. 
               We still highly recommend performing **imputation** with BEAGLE before analysis if your data contains missing values.
            - If you don't have a genotype file now or want to see the details of the file format,
            you can download the example file here 
            for **[fast model](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/genotypes_for_Breed_identifier500.txt)**
            and **[accurate model](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/genotypes_for_Breed_identifier2000.txt)**.

            **2. Uploading the genotype file.**
            - Click the 'Choose a file' button on the page and select your genotype file.

            **3. Select the model to use for analysis.**
            - There are two models available: '**fast**' and '**accurate**'.
            - The 'fast' model uses 500 SNPs, while the 'accurate' model uses 2000 SNPs.
            - The 'fast' model is recommended for quick analysis, while the 'accurate' model provides more accurate results.
               
            **4. Click the 'Analyze' button to predict the breed.**
            - You can use the demo file mentioned above to test the tool and see the output details.
            ''')
    st.success('''## Analysis''')

 
    uploaded_file = st.file_uploader("Please upload a genotype file to begin analysis")
    model_choice = st.selectbox('Choose the model to use for analysis:', ['fast', 'accurate'], index=0)  # 默认选择'fast'
    if uploaded_file is not None:
        try:
            gt_df = pd.read_csv(uploaded_file, sep='\s+', header=None)
            sample_names = gt_df.iloc[:, 0]  # 提取样本名
            gt_array = gt_df.iloc[:, 1:].to_numpy()  # 提取基因型数据
            # 创建 SimpleImputer 对象，强制将缺失值填充为 0
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
            # 使用 fit_transform 方法填充缺失值
            gt_array_imputed = imputer.fit_transform(gt_array)
            st.session_state.gt_array = gt_array_imputed
            st.session_state.sample_names = sample_names.tolist()  # Store sample names as a list
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.model_choice = model_choice
            st.info('Genotype file has been uploaded.')
        except Exception as e:
            st.error(f'Invalid file. Please upload a valid genotype file. Error: {e}')
    
    if st.button('Analyze'):
        if 'gt_array_imputed' in st.session_state and 'model_choice' in st.session_state:
            result = breed_classifier(st.session_state.gt_array, model=st.session_state.model_choice)
            # 样本名和预测结果合并
            combined_results = list(zip(st.session_state.sample_names, result))
            st.session_state.result = combined_results
            st.subheader('Analysis Results')
            for sample, breed in combined_results:
                st.write(f"{sample}: {breed}")
        else:
            st.error("No genotype data to analyze. Please upload a file and select a model.")


if __name__ == '__main__':
    load_css()
    page_frame()
    show_footer()