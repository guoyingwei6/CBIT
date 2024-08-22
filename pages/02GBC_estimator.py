import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib

# 设置页面配置
st.set_page_config(page_title="GBC estimator", page_icon="🐂", layout="centered", initial_sidebar_state="expanded")


@st.cache_data(ttl=3600)
def load_AF():
    """从文件中读取等位基因频率"""
    allele_freqs = joblib.load('attachments/AF_for_gbc.pkl')
    return allele_freqs


def GBC_estimator(genotypes, confidence=0.05):
    """根据等位基因频率和基因型数据，拟合线性模型估计各品种对待测个体的血统比例"""

    # 读取等位基因频率数据
    allele_freqs=load_AF()

    # 找到两个数据集中CHR:POS的交集
    common_snps = allele_freqs.index.intersection(genotypes.index)

    # 根据交集筛选allele_freqs和genotypes中的相应行
    filtered_allele_freqs = allele_freqs.loc[common_snps]
    filtered_genotypes = genotypes.loc[common_snps]

    # 转换为矩阵形式以便计算
    allele_freqs_matrix = filtered_allele_freqs.values

    # 初始化字典来存储每个个体的贡献
    contributions_dict = {}

    # 循环处理每个个体
    for i in range(filtered_genotypes.shape[1]):
        genotype = filtered_genotypes.iloc[:, i].values  # 获取第i个个体的基因型数据
        
        # 构建线性模型，无截距
        X = allele_freqs_matrix
        model = sm.OLS(genotype, X)
        results = model.fit()

        # 提取系数（b向量）
        coefficients = results.params

        # 将负系数转换为0
        coefficients[coefficients < 0] = 0

        # 计算每个品种的遗传贡献比例
        contributions = coefficients / sum(coefficients)
        
        # 将contributions中小于0.02的系数转换为0
        contributions[contributions < confidence] = 0

        # 再次计算每个品种的遗传贡献比例
        contributions = contributions / sum(contributions)

        # 存储到字典
        contributions_dict[filtered_genotypes.columns[i]] = contributions

    # 使用字典创建DataFrame，索引设置为品种名称
    individual_contributions_rounded = pd.DataFrame(contributions_dict, index=filtered_allele_freqs.columns).round(4)

    return individual_contributions_rounded

def upload_gt():
    """从文件中读取基因型数据"""
    uploaded_file = st.file_uploader("Choose a file")
    confidence = st.number_input('Set the confidence threshold', min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.02f")
    st.caption("Adjust the confidence threshold to filter out minor contributions. Values should be between 0 and 1.")

    if st.button('Analyze'):
        if uploaded_file is not None:
            try:
                gt = pd.read_table(uploaded_file, sep=' ', header=0, index_col='CHR:POS')
                st.session_state['gt'] = gt
                result = GBC_estimator(st.session_state['gt'], confidence)
                st.subheader('Analysis Results')
                st.write(result)
            except Exception as e:
                st.error(f'Invalid file. Please upload a valid genotype file. Error: {e}')
        else:
            st.error("No genotype data to analyze. Please upload a file.")


if __name__ == '__main__':
    upload_gt()
