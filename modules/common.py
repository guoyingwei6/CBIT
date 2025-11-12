import streamlit as st


def show_footer():
    """显示页脚信息"""
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 15px;
        font-size: 13px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    .footer a {
        color: #ffd700;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <p>© 2024 Institute of Animal Science (IAS), Chinese Academy of Agricultural Sciences (CAAS) |
        Contact: <a href="mailto:yingwei.guo@foxmail.com">yingwei.guo@foxmail.com</a></p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

def load_css():
    """加载自定义CSS样式"""
    st.markdown("""
        <style>
        /* 全局字体和基础样式 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* 主标题样式 */
        h1 {
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* 副标题样式 */
        h2, h3 {
            color: #1e293b;
            font-weight: 600;
        }

        /* 卡片样式 */
        .feature-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            margin: 15px 0;
            border-left: 5px solid #667eea;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        }

        .feature-card h3 {
            color: #667eea;
            margin-top: 0;
        }

        /* 信息框样式优化 */
        .stAlert {
            border-radius: 10px;
            border-left: 5px solid;
        }

        /* 按钮样式 */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
        }

        /* 文件上传器样式 */
        .uploadedFile {
            border-radius: 10px;
            border: 2px dashed #667eea;
        }

        /* 数据表格样式 */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }

        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }

        /* 分隔线样式 */
        hr {
            margin: 30px 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }

        /* 提示框图标样式 */
        .stAlert > div {
            padding: 15px;
        }

        /* 进度条样式 */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }

        /* 统计卡片 */
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .stat-card h2 {
            color: white;
            margin: 0;
            font-size: 2.5em;
        }

        .stat-card p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }

        /* 移除底部空白，为footer留空间 */
        .main .block-container {
            padding-bottom: 100px;
        }
        </style>
        """, unsafe_allow_html=True)

def create_feature_card(icon, title, description, link=None):
    """创建功能卡片"""
    card_html = f"""
    <div class="feature-card">
        <h3>{icon} {title}</h3>
        <p>{description}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_stat_card(value, label):
    """创建统计卡片"""
    stat_html = f"""
    <div class="stat-card">
        <h2>{value}</h2>
        <p>{label}</p>
    </div>
    """
    st.markdown(stat_html, unsafe_allow_html=True)

def add_spacing(height=20):
    """添加垂直间距"""
    st.markdown(f'<div style="margin: {height}px 0;"></div>', unsafe_allow_html=True)