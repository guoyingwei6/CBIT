import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from modules.common import show_footer, load_css

# 设置页面配置
st.set_page_config(page_title="GBC estimator", page_icon="🐂", layout="centered", initial_sidebar_state="expanded")
st.title('GBC estimator')
st.markdown("""
            ## Introduction
            This tool is designed to help you estimate the genomic breed content (GBC) in a mixed breed cattle population.  
            
            We estimate the GBC using a linear model based on the genotype data:
            $$
            y = Fb + e
            $$
            where $y$ is the genotype vector ($M \\times 1$) of all $M$ SNPs of the individual to be estimated,
            and the SNP genotypes are represented by $0$ (AA), $1$ (AB), and $2$ (BB) respectively. 
            $F$ is the allele frequency matrix with $M \\times T$, 
            where $T$ is the number of breeds in the reference popution.
            The regression coefficient vector $b$ ($T \\times 1$) is the GBC of each breed to the individual to be estimated.
            $e$ is the error term.  
            
            Then, we solve the linear model using **ordinary least squares (OLS) regression**, where $\hat{b} = (F^{\prime} F)^{-1}F^{\prime}y$.  
            
            Finally, we normalize the regression coefficients to sum to 1 and filter out minor contributions based on a confidence threshold.
            
            ## Usuage
            **1. Upload the genotype file.**
            - A genotype file is needed with **one individual per column and one SNP per line with header and index**. 
            The first column should be the SNP ID (CHR:POS) and the first row should be the sample ID.            
            - The file should be in the format of a **space or tab-separated** text file.
            - More accurate results depend on more SNPs. We recommend using a file with **at least 1000 SNPs**, and **50,000 SNPs** above are highly recommended.
            - If you don't have a genotype file now or want to see the details of the file format, 
            you can download the example file [here](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/genotype_examples_for_GBC_extimator.txt).
            
            **2. Set the confidence threshold to filter out minor contributions.**
            - The minor contributions will be filtered out based on the confidence threshold you set. 
            - A larger threshold will exclude the interference from irrelevant breeds, 
            but there is also a risk of overestimating the true contributions of some breeds. Smaller thresholds have the opposite effect. 
            - By experience, a threshold between **0.02** (using about 200,000 SNPs) and **0.1** (5,000 SNPs below) is appropriate. 
            We recommend a threshold of **0.05** by default, it can be changed according to your data and expectations.

            **3. Click the 'Analyze' button to estimate the GBC.**
            - The analysis will take a few seconds to complete, depending on the size of the genotype file. 
            - Based on prior exprience, a file with 100 samples and 200,000 SNPs will take about 150 seconds (**one sample every 1.5 seconds**).
            - Smaller sample size and SNPs dataset will take less time.  

            **4. The results will be displayed as a table, showing the GBC of each breed for each individual.**
            - You can save the results as a CSV file by click the download button in the upper right corner.
            
            ## Analysis
            Please upload a genotype file to begin the analysis.""")

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
                gt = pd.read_table(uploaded_file, sep='\s+', header=0, index_col='CHR:POS')
                st.session_state['gt'] = gt
                result = GBC_estimator(st.session_state['gt'], confidence)
                st.subheader('Analysis Results')
                st.write(result)
            except Exception as e:
                st.error(f'Invalid file. Please upload a valid genotype file. Error: {e}')
        else:
            st.error("No genotype data to analyze. Please upload a file.")


if __name__ == '__main__':
    load_css()
    upload_gt()
    show_footer()
