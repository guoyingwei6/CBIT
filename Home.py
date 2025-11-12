import streamlit as st
from modules.common import show_footer, load_css, create_feature_card, create_stat_card, add_spacing

st.set_page_config(
    page_title="CBIT - Cattle Breed Identification Tool",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义样式
load_css()

# 欢迎横幅
st.markdown("""
<div style='text-align: center; padding: 40px 0 20px 0;'>
    <h1 style='font-size: 56px; margin-bottom: 10px;'>
        🐄 CBIT
    </h1>
    <h2 style='font-size: 32px; font-weight: 400; color: #64748b; margin-top: 0;'>
        <span style='color: #ef4444;'>C</span>attle
        <span style='color: #22c55e;'>B</span>reed
        <span style='color: #3b82f6;'>I</span>dentification
        <span style='color: #f59e0b;'>T</span>ool
    </h2>
    <p style='font-size: 18px; color: #64748b; margin-top: 20px; max-width: 800px; margin-left: auto; margin-right: auto;'>
        A comprehensive genomic analysis platform for cattle breed identification and genomic breed composition estimation
    </p>
</div>
""", unsafe_allow_html=True)

add_spacing(30)

# 关键统计数据
col1, col2, col3 = st.columns(3)
with col1:
    create_stat_card("2913", "Samples in Dataset")
with col2:
    create_stat_card("49", "Cattle Breeds")
with col3:
    create_stat_card("2", "Analysis Tools")

add_spacing(40)

# 项目介绍
st.markdown("## 📖 About CBIT")
st.markdown("""
<div style='font-size: 16px; line-height: 1.8; color: #475569;'>
Cattle is an important livestock that provides meat, milk, and other products to humans.
Identifying cattle breeds is essential for breeding, management, and conservation.

<strong>CBIT</strong> provides advanced genomic analysis tools to help researchers and breeders:
<ul>
    <li>🎯 <strong>Identify cattle breeds</strong> with high accuracy using machine learning</li>
    <li>📊 <strong>Estimate genomic breed composition (GBC)</strong> in crossbred populations</li>
    <li>🌏 <strong>Analyze diverse populations</strong> covering 49 breeds from Asia and Europe</li>
</ul>
</div>
""", unsafe_allow_html=True)

add_spacing(30)

# 功能卡片
st.markdown("## 🚀 Our Tools")
col1, col2 = st.columns(2)

with col1:
    create_feature_card(
        "🐂",
        "Breed Identification",
        "Identify cattle breeds using state-of-the-art machine learning models. "
        "Choose between fast (100 SNPs) and accurate (1000 SNPs) models based on your needs."
    )

with col2:
    create_feature_card(
        "📈",
        "GBC Estimation",
        "Estimate genomic breed composition in crossbred cattle populations using "
        "ordinary least squares regression on allele frequencies."
    )

add_spacing(30)

# 数据集信息
st.markdown("## 📊 Dataset Information")
st.markdown("""
<div style='font-size: 16px; line-height: 1.8; color: #475569;'>
Our reference dataset contains <strong>2913 samples</strong> from <strong>49 breeds</strong> across Asia and Europe,
providing comprehensive coverage for accurate breed identification and GBC estimation.
</div>
""", unsafe_allow_html=True)

add_spacing(20)

# 显示样本信息图片
st.image(
    'https://picbed.guoyingwei.top/2024/08/202408051048682.png',
    caption='Sample distribution across different cattle breeds',
    use_container_width=True
)

add_spacing(30)

# 快速开始指南
st.markdown("## 🎯 Quick Start")

quick_start_col1, quick_start_col2 = st.columns(2)

with quick_start_col1:
    st.markdown("""
    ### For Breed Identification:
    1. 📁 Prepare your genotype file (0/1/2 coded)
    2. 📤 Upload the file to the Breed Identification page
    3. ⚙️ Select a model (fast or accurate)
    4. 🔍 Click Analyze to get results
    """)

with quick_start_col2:
    st.markdown("""
    ### For GBC Estimation:
    1. 📁 Prepare your genotype file (SNP format)
    2. 📤 Upload the file to the GBC Estimation page
    3. 🎚️ Set the confidence threshold
    4. 📊 View detailed breed composition results
    """)

add_spacing(20)

# 导航提示
st.info("👈 Use the sidebar to navigate to different tools and learn more about CBIT!")

if __name__ == '__main__':
    show_footer()