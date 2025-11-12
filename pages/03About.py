import streamlit as st
from modules.common import show_footer, load_css, add_spacing, create_feature_card

st.set_page_config(
    page_title="About CBIT",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义样式
load_css()

# 页面标题
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1>💻 About CBIT</h1>
    <p style='font-size: 18px; color: #64748b;'>
        Learn more about our cattle breed identification platform
    </p>
</div>
""", unsafe_allow_html=True)

add_spacing(30)

def main():
    # 项目简介
    st.markdown("## 🎯 Project Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **CBIT** (Cattle Breed Identification Tool) is a comprehensive genomic analysis platform designed
        to help researchers, breeders, and agricultural professionals identify cattle breeds and estimate
        genomic breed composition with high accuracy.

        ### Mission

        Our mission is to provide accessible, accurate, and efficient tools for cattle breed analysis,
        supporting breeding programs, conservation efforts, and agricultural research worldwide.

        ### Technology

        CBIT leverages advanced machine learning algorithms and statistical models:
        - **Random Forest** for feature selection
        - **Support Vector Machine** for classification
        - **Ordinary Least Squares** regression for breed composition estimation

        ### Data

        Our reference dataset includes:
        - **2,913 samples** from diverse cattle populations
        - **49 breeds** from Asia and Europe
        - High-quality genotypic data for accurate analysis
        """)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 25px; border-radius: 15px;'>
            <h3 style='color: white; margin-top: 0;'>📊 Platform Stats</h3>
            <hr style='border-color: rgba(255,255,255,0.3); margin: 15px 0;'>
            <p style='font-size: 16px; margin: 10px 0;'>
                <strong>2,913</strong> Reference Samples<br>
                <strong>49</strong> Cattle Breeds<br>
                <strong>2</strong> Analysis Tools<br>
                <strong>100+</strong> SNPs (Fast Model)<br>
                <strong>1,000+</strong> SNPs (Accurate Model)
            </p>
        </div>
        """, unsafe_allow_html=True)

    add_spacing(40)

    # 功能特点
    st.markdown("## ✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        create_feature_card(
            "🎯",
            "High Accuracy",
            "State-of-the-art machine learning models trained on comprehensive datasets "
            "ensure accurate breed identification and composition estimation."
        )

        create_feature_card(
            "⚡",
            "Fast Processing",
            "Optimized algorithms provide quick results, with the fast model analyzing "
            "samples in seconds and even large datasets completing within minutes."
        )

    with col2:
        create_feature_card(
            "🔒",
            "Data Privacy",
            "Your uploaded data is processed securely and cached temporarily (1 hour) "
            "for your convenience, then automatically removed."
        )

        create_feature_card(
            "📊",
            "Comprehensive Results",
            "Detailed analysis results with visualizations, statistics, and downloadable "
            "reports in CSV format for further analysis."
        )

    add_spacing(40)

    # 使用流程
    st.markdown("## 📖 How to Use CBIT")

    tab1, tab2 = st.tabs(["🐂 Breed Identification", "📈 GBC Estimation"])

    with tab1:
        add_spacing(20)
        st.markdown("""
        ### Breed Identification Workflow

        **Step 1: Prepare Your Data**
        - Format your genotype data (0/1/2 encoding)
        - Ensure one individual per row, one SNP per column
        - Include sample names in the first column

        **Step 2: Upload & Configure**
        - Navigate to the Breed Identification page
        - Upload your genotype file
        - Select a model (Fast: 100 SNPs, Accurate: 1000 SNPs)

        **Step 3: Analyze**
        - Click the "Analyze" button
        - Wait for processing to complete

        **Step 4: Review Results**
        - View breed predictions for each sample
        - Check summary statistics
        - Download results as CSV

        **Important Notes:**
        - Missing values are automatically imputed
        - Results are cached for 1 hour
        - BEAGLE imputation recommended for better accuracy
        """)

    with tab2:
        add_spacing(20)
        st.markdown("""
        ### GBC Estimation Workflow

        **Step 1: Prepare Your Data**
        - Format file: CHR:POS in first column, sample IDs in first row
        - One SNP per row, one individual per column
        - Minimum 1,000 SNPs (50,000+ recommended)

        **Step 2: Upload & Configure**
        - Navigate to the GBC Estimation page
        - Upload your genotype file
        - Set confidence threshold (default: 0.05)

        **Step 3: Analyze**
        - Click the "Analyze" button
        - Monitor progress (processing time depends on file size)

        **Step 4: Review Results**
        - View breed composition table
        - Analyze summary statistics
        - Download detailed results

        **Threshold Guidelines:**
        - **0.02**: Captures minor breeds (for 50K+ SNPs)
        - **0.05**: Balanced approach (default)
        - **0.10**: Main breeds only (for <5K SNPs)
        """)

    add_spacing(40)

    # 技术支持和联系方式
    st.markdown("## 📞 Contact & Support")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); border-left: 5px solid #667eea;'>
            <h3 style='color: #667eea; margin-top: 0;'>📧 Get in Touch</h3>
            <p style='font-size: 16px; line-height: 1.8;'>
                If you have questions, suggestions, or need technical support,
                please don't hesitate to contact us:
            </p>
            <p style='font-size: 18px; margin-top: 20px;'>
                <strong>Email:</strong><br>
                <a href="mailto:yingwei.guo@foxmail.com" style='color: #667eea; text-decoration: none;'>
                    yingwei.guo@foxmail.com
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); border-left: 5px solid #764ba2;'>
            <h3 style='color: #764ba2; margin-top: 0;'>🏛️ Institution</h3>
            <p style='font-size: 16px; line-height: 1.8;'>
                <strong>Institute of Animal Science (IAS)</strong><br>
                Chinese Academy of Agricultural Sciences (CAAS)
            </p>
            <p style='font-size: 14px; color: #64748b; margin-top: 20px;'>
                We are committed to advancing agricultural research and
                providing innovative tools for the livestock industry.
            </p>
        </div>
        """, unsafe_allow_html=True)

    add_spacing(30)

    # 致谢和版权
    st.markdown("## 🙏 Acknowledgments")

    st.markdown("""
    <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;'>
        <p style='font-size: 16px; line-height: 1.8; margin: 0;'>
            Thank you for using CBIT! We appreciate your feedback and suggestions,
            which help us continuously improve this tool. If you use CBIT in your research,
            please consider citing our work.
        </p>
    </div>
    """, unsafe_allow_html=True)

    add_spacing(20)

    # 版本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; border-radius: 10px;'>
            <h4 style='margin: 0; color: white;'>Version</h4>
            <p style='font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>1.0</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; border-radius: 10px;'>
            <h4 style='margin: 0; color: white;'>Release Year</h4>
            <p style='font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>2024</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; border-radius: 10px;'>
            <h4 style='margin: 0; color: white;'>Status</h4>
            <p style='font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>Active</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
    show_footer()

