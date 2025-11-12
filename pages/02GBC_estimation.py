import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from modules.common import show_footer, load_css, add_spacing
import io

# 设置页面配置
st.set_page_config(
    page_title="GBC Estimator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义样式
load_css()

# 页面标题
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1>📈 Genomic Breed Composition (GBC) Estimator</h1>
    <p style='font-size: 18px; color: #64748b;'>
        Estimate breed composition in crossbred cattle populations
    </p>
</div>
""", unsafe_allow_html=True)

add_spacing(20)

# 使用标签页组织内容
tab1, tab2, tab3 = st.tabs(["📖 Introduction", "📋 Usage Guide", "🔬 Analysis"])

with tab1:
    add_spacing(20)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### About This Tool

        The GBC Estimator helps you determine the **genomic breed composition** in mixed or crossbred
        cattle populations using advanced statistical modeling.

        #### Statistical Model

        We use a linear regression model based on genotype data:
        """)

        st.latex(r"y = Fb + e")

        st.markdown("""
        Where:
        - $y$: Genotype vector ($M \\times 1$) of all $M$ SNPs for the individual
        - $F$: Allele frequency matrix ($M \\times T$), where $T$ = number of breeds
        - $b$: Regression coefficients ($T \\times 1$), representing GBC for each breed
        - $e$: Error term

        #### Methodology

        1. **Solve using OLS regression**: $\\hat{b} = (F^{\\prime} F)^{-1}F^{\\prime}y$
        2. **Normalize** coefficients to sum to 1
        3. **Filter** minor contributions based on confidence threshold
        """)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 20px; border-radius: 15px; margin-top: 20px;'>
            <h3 style='color: white; margin-top: 0;'>Key Features</h3>
            <ul style='margin-bottom: 0;'>
                <li>OLS regression model</li>
                <li>Handles missing data</li>
                <li>Adjustable thresholds</li>
                <li>49 reference breeds</li>
                <li>High accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    add_spacing(20)

    st.markdown("#### Performance Expectations")

    perf_data = pd.DataFrame({
        'Dataset Size': ['Small (10 samples, 5K SNPs)', 'Medium (50 samples, 50K SNPs)', 'Large (100 samples, 200K SNPs)'],
        'Processing Time': ['~5 seconds', '~30 seconds', '~150 seconds'],
        'Time per Sample': ['~0.5s', '~0.6s', '~1.5s']
    })
    st.dataframe(perf_data, hide_index=True, use_container_width=True)

with tab2:
    add_spacing(20)

    st.markdown("### Step-by-Step Guide")

    # 步骤 1
    with st.expander("📁 **Step 1: Prepare Your Genotype File**", expanded=True):
        st.markdown("""
        Your genotype file must meet specific format requirements:

        **File Format:**
        - Encoded as **0, 1, and 2** (genotypes AA, AB, BB)
        - **One SNP per row**, **one individual per column**
        - **First column**: SNP ID in format `CHR:POS` (based on **ARS-UCD2.0**)
        - **First row**: Sample IDs
        - **Space or tab-separated** text file

        **SNP Requirements:**
        - **Minimum**: 1,000 SNPs (basic analysis)
        - **Recommended**: 50,000+ SNPs (high accuracy)
        - More SNPs = More accurate results

        **Missing Values:**
        - Automatically handled (dropped from analysis)
        - ⚠️ Excessive missing values reduce accuracy
        - 💡 We recommend **BEAGLE imputation** before analysis

        **Example File:**
        - [Download Example File](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/genotypes_for_GBC_extimator.txt)
        - [Download Example Results](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/GBC_results.csv)
        """)

    # 步骤 2
    with st.expander("🎚️ **Step 2: Set Confidence Threshold**"):
        st.markdown("""
        The confidence threshold filters out minor breed contributions:

        **Threshold Guidelines:**

        | SNP Count | Recommended Threshold | Description |
        |-----------|----------------------|-------------|
        | < 5,000 | 0.10 | Higher threshold for fewer SNPs |
        | 5,000 - 50,000 | 0.05 | **Default (balanced)** |
        | 50,000 - 200,000 | 0.02 | Lower threshold for many SNPs |

        **Trade-offs:**
        - **Higher threshold (0.10)**: Excludes noise, may overestimate main breeds
        - **Lower threshold (0.02)**: Captures minor breeds, may include noise

        💡 **Default 0.05** works well for most datasets
        """)

    # 步骤 3-4
    with st.expander("🔍 **Step 3: Run Analysis**"):
        st.markdown("""
        1. Upload your prepared genotype file
        2. Set the confidence threshold
        3. Click **'🔍 Analyze'** button
        4. Wait for processing (time depends on file size)
        """)

    with st.expander("📊 **Step 4: View & Download Results**"):
        st.markdown("""
        Results show breed composition for each individual:
        - View detailed composition table
        - Each column represents an individual
        - Each row shows a breed's contribution (0-1)
        - Download results as CSV file
        """)

with tab3:
    add_spacing(20)
    st.markdown("### 🔬 Run Your Analysis")
    st.markdown("Upload your genotype file and configure parameters to estimate genomic breed composition.")

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
        # 将contributions中小于cutoff的系数转换为0
        contributions[contributions < confidence] = 0
        # 再次计算每个品种的遗传贡献比例
        contributions = contributions / sum(contributions)
        # 存储到字典
        contributions_dict[filtered_genotypes.columns[i]] = contributions
    # 使用字典创建DataFrame，索引设置为品种名称
    individual_contributions_rounded = pd.DataFrame(contributions_dict, index=filtered_allele_freqs.columns).round(4)

    return individual_contributions_rounded

def upload_gt():
    """从文件中读取基因型数据并进行分析"""

    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📤 Upload Genotype File")
        uploaded_file = st.file_uploader(
            "Choose your genotype file",
            type=['txt', 'csv'],
            help="Upload a space or tab-separated file with SNP format: CHR:POS in first column, sample IDs in first row"
        )

    with col2:
        st.markdown("#### 🎚️ Confidence Threshold")
        confidence = st.number_input(
            'Set threshold:',
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.02f",
            help="Filter out breeds with contribution below this threshold"
        )

        # 根据阈值给出建议
        if confidence <= 0.02:
            st.info("🔍 **Low threshold**\n\nCaptures minor breeds")
        elif confidence >= 0.10:
            st.info("🎯 **High threshold**\n\nMain breeds only")
        else:
            st.info("⚖️ **Balanced**\n\nGood for most cases")

    add_spacing(20)

    # 显示文件信息
    if uploaded_file is not None:
        try:
            with st.spinner('📊 Loading genotype file...'):
                # 先读取文件预览
                gt = pd.read_table(uploaded_file, sep='\s+', header=0, index_col='CHR:POS').dropna()
                st.session_state['gt'] = gt
                st.session_state['confidence'] = confidence
                st.session_state['uploaded_file_name'] = uploaded_file.name

            # 显示文件统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File Name", uploaded_file.name)
            with col2:
                st.metric("🧬 SNPs", gt.shape[0])
            with col3:
                st.metric("👥 Samples", gt.shape[1])

            st.success('✅ Genotype file loaded successfully!')

            # SNP数量建议
            if gt.shape[0] < 1000:
                st.warning("⚠️ Your file has fewer than 1,000 SNPs. Results may be less accurate. Consider using more SNPs.")
            elif gt.shape[0] >= 50000:
                st.info("🎯 Excellent! Your file has 50,000+ SNPs, which will provide highly accurate results.")

        except Exception as e:
            st.error(f'❌ Invalid file format. Please check your file format.\n\n**Error details:** {e}')
            st.info("💡 Make sure:\n- First column is SNP ID (CHR:POS)\n- First row is sample IDs\n- File is space or tab-separated")

    add_spacing(20)

    # 分析按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button('🔍 Analyze', use_container_width=True, type="primary")

    if analyze_button:
        if 'gt' in st.session_state:
            # 估算处理时间
            estimated_time = len(st.session_state['gt'].columns) * 1.5
            time_display = f"{estimated_time:.1f} seconds" if estimated_time < 60 else f"{estimated_time/60:.1f} minutes"

            st.info(f"⏱️ Estimated processing time: {time_display}")

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text('🔬 Analyzing genomic breed composition...')
                progress_bar.progress(30)

                result = GBC_estimator(
                    st.session_state['gt'],
                    st.session_state['confidence']
                )

                progress_bar.progress(100)
                status_text.text('✅ Analysis complete!')

                st.session_state['result'] = result

                add_spacing(20)

                # 显示结果
                st.markdown("### 📊 Analysis Results")

                # 显示结果表格
                st.markdown("#### Breed Composition Table")
                st.markdown("Each column represents a sample, each row represents a breed. Values show the proportion (0-1) of each breed's contribution.")

                st.dataframe(
                    result.style.format("{:.4f}").background_gradient(cmap='Blues'),
                    use_container_width=True,
                    height=min(600, (len(result) + 1) * 35 + 3)
                )

                add_spacing(20)

                # 统计信息
                st.markdown("#### 📈 Summary Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # 每个样本中检测到的品种数量
                    breeds_per_sample = (result > 0).sum(axis=0)
                    st.metric("🐄 Avg Breeds per Sample", f"{breeds_per_sample.mean():.1f}")

                with col2:
                    # 检测到的总品种数
                    total_breeds_detected = (result > 0).any(axis=1).sum()
                    st.metric("🌍 Total Breeds Detected", total_breeds_detected)

                with col3:
                    st.metric("📊 Samples Analyzed", len(result.columns))

                # 下载按钮
                add_spacing(20)

                csv_buffer = io.StringIO()
                result.to_csv(csv_buffer)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"GBC_results_{st.session_state['uploaded_file_name'].split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f'❌ Analysis failed. Please check your data.\n\n**Error details:** {e}')

        else:
            st.error("⚠️ No genotype data to analyze. Please upload a file first.")


if __name__ == '__main__':
    upload_gt()
    show_footer()
