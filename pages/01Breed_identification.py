import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.impute import SimpleImputer
from modules.common import show_footer, load_css, add_spacing
import io


# 设置页面配置
st.set_page_config(
    page_title="Breed Identification",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义样式
load_css()

def page_frame():
    # 页面标题
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>🐂 Breed Identification</h1>
        <p style='font-size: 18px; color: #64748b;'>
            Identify cattle breeds using machine learning with high accuracy
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

            The Breed Identification tool helps you identify different breeds of cattle using advanced
            machine learning algorithms trained on a comprehensive dataset of **49 breeds** and **2913 samples**.

            #### Model Performance

            After extensive comparison of different machine learning approaches, we developed a workflow using:
            - 🎯 **Random Forest (RF)** for feature selection
            - 🤖 **Support Vector Machine (SVM)** for classification

            This combination provides the best performance for cattle breed identification.

            #### Available Models

            We offer two models to suit different needs:
            """)

        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 20px; border-radius: 15px; margin-top: 20px;'>
                <h3 style='color: white; margin-top: 0;'>Key Features</h3>
                <ul style='margin-bottom: 0;'>
                    <li>49 cattle breeds</li>
                    <li>2913 training samples</li>
                    <li>RF + SVM algorithm</li>
                    <li>High accuracy</li>
                    <li>Fast processing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # 模型比较表格
        st.markdown("#### Model Comparison")
        model_comparison = pd.DataFrame({
            'Model': ['Fast Model', 'Accurate Model'],
            'SNPs Used': ['100', '1000'],
            'Speed': ['⚡ Very Fast', '🐢 Moderate'],
            'Accuracy': ['✓ Good', '✓✓ Excellent'],
            'Recommended For': ['Quick screening', 'Detailed analysis']
        })
        st.dataframe(model_comparison, hide_index=True, use_container_width=True)

        st.info("📄 For detailed information on model accuracy and influencing factors, please refer to our paper.")

    with tab2:
        add_spacing(20)

        # 创建步骤指南
        st.markdown("### Step-by-Step Guide")

        # 步骤 1
        with st.expander("📁 **Step 1: Prepare Your Genotype File**", expanded=True):
            st.markdown("""
            Your genotype file must meet the following requirements:

            **File Format:**
            - Encoded as **0, 1, and 2** (representing genotypes AA, AB, and BB)
            - **One individual per row**, **one SNP per column**
            - **First column**: Sample name (used as identifier in results)
            - **Space or tab-separated** text file

            **SNP Requirements:**
            - Must contain **only** the SNPs we specify
            - SNP locations based on **ARS-UCD2.0** genome assembly

            **Download SNP Maps:**
            - [Fast Model SNP Map](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/map_for_Breed_identifier_fast_model.txt) (100 SNPs)
            - [Accurate Model SNP Map](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/map_for_Breed_identifier_accurate_model.txt) (1000 SNPs)

            **Missing Values:**
            - Missing values (NA) are imputed to 0 automatically
            - ⚠️ Many missing values may reduce accuracy
            - 💡 We recommend using **BEAGLE** for imputation before analysis
            """)

            st.markdown("**Example Files:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("- [Fast Model Example](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/genotypes_for_Breed_identifier_fast_model.txt)")
            with col2:
                st.markdown("- [Accurate Model Example](https://raw.githubusercontent.com/guoyingwei6/CBIT/develop/attachments/genotypes_for_Breed_identifier_accurate_model.txt)")

        # 步骤 2-4
        with st.expander("📤 **Step 2: Upload Your File**"):
            st.markdown("""
            - Click the **'Browse files'** button in the Analysis tab
            - Select your prepared genotype file
            - Wait for the upload confirmation message
            """)

        with st.expander("⚙️ **Step 3: Select Model**"):
            st.markdown("""
            Choose between two models:
            - **Fast Model**: 100 SNPs, quick results, good accuracy
            - **Accurate Model**: 1000 SNPs, best accuracy, slightly slower
            """)

        with st.expander("🔍 **Step 4: Analyze & Download Results**"):
            st.markdown("""
            1. Click the **'🔍 Analyze'** button
            2. Wait for the analysis to complete
            3. View results showing breed predictions for each sample
            4. Download results as CSV file using the download button
            """)

    with tab3:
        add_spacing(20)
        st.markdown("### 🔬 Run Your Analysis")
        st.markdown("Upload your genotype file and select a model to begin breed identification.")

@st.cache_data(ttl=3600)
def load_breed_codes():
    """从文件中读取品种代码，传入字典中。"""
    df = pd.read_csv('attachments/breed_code.csv')
    code_breed_dict = pd.Series(df.Breed.values, index=df.Code).to_dict()
    return code_breed_dict

@st.cache_resource(ttl=3600)
def load_model_fast():
    """加载模型。"""
    clf = joblib.load('attachments/Breed_identifier_fast_model.pkl')
    return clf

@st.cache_resource(ttl=3600)
def load_model_accurate():
    """加载模型。"""
    clf = joblib.load('attachments/Breed_identifier_accurate_model.pkl')
    return clf

def breed_classifier(genotype_array, model='accurate'):
    """品种分类函数。"""
    if model == 'fast':
        clf = load_model_fast()
    elif model == 'accurate':
        clf = load_model_accurate()
    prediction = clf.predict(genotype_array)
    breed_code_dict = load_breed_codes()
    breed_prediction = [breed_code_dict[code] for code in prediction]
    return breed_prediction



def analysis():
    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📤 Upload Genotype File")
        uploaded_file = st.file_uploader(
            "Choose your genotype file",
            type=['txt', 'csv'],
            help="Upload a space or tab-separated file with genotype data"
        )

    with col2:
        st.markdown("#### ⚙️ Model Selection")
        model_choice = st.selectbox(
            'Select Model:',
            ['accurate', 'fast'],
            index=0,
            help="Accurate model uses 1000 SNPs, Fast model uses 100 SNPs"
        )

        # 显示模型信息
        if model_choice == 'accurate':
            st.info("🎯 **Accurate Model**\n\n1000 SNPs\n\nBest accuracy")
        else:
            st.info("⚡ **Fast Model**\n\n100 SNPs\n\nQuick results")

    add_spacing(20)

    if uploaded_file is not None:
        try:
            with st.spinner('📊 Processing genotype file...'):
                gt_df = pd.read_csv(uploaded_file, sep='\s+', header=None)
                sample_names = gt_df.iloc[:, 0]  # 提取样本名
                gt_array = gt_df.iloc[:, 1:].to_numpy()  # 提取基因型数据

                # 创建 SimpleImputer 对象，强制将缺失值填充为 0
                imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
                # 使用 fit_transform 方法填充缺失值
                gt_array_imputed = imputer.fit_transform(gt_array)

                st.session_state.gt_array_imputed = gt_array_imputed
                st.session_state.sample_names = sample_names.tolist()
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.model_choice = model_choice

            # 显示文件信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File Name", uploaded_file.name)
            with col2:
                st.metric("👥 Samples", len(sample_names))
            with col3:
                st.metric("🧬 SNPs", gt_array.shape[1])

            st.success('✅ Genotype file uploaded successfully!')

        except Exception as e:
            st.error(f'❌ Invalid file format. Please upload a valid genotype file.\n\n**Error details:** {e}')

    add_spacing(20)

    # 分析按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button('🔍 Analyze', use_container_width=True, type="primary")

    if analyze_button:
        if 'gt_array_imputed' in st.session_state and 'model_choice' in st.session_state:
            with st.spinner('🔬 Analyzing breed composition...'):
                result = breed_classifier(st.session_state.gt_array_imputed, model=st.session_state.model_choice)

                # 样本名和预测结果合并
                combined_results = list(zip(st.session_state.sample_names, result))
                st.session_state.result = combined_results

            add_spacing(20)

            # 显示结果
            st.markdown("### 📊 Analysis Results")

            # 创建结果 DataFrame
            results_df = pd.DataFrame(combined_results, columns=['Sample ID', 'Predicted Breed'])

            # 显示结果表格
            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True,
                height=min(400, (len(results_df) + 1) * 35 + 3)
            )

            # 统计信息
            st.markdown("#### 📈 Summary Statistics")
            breed_counts = results_df['Predicted Breed'].value_counts()

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Breed Distribution:**")
                for breed, count in breed_counts.items():
                    percentage = (count / len(results_df)) * 100
                    st.write(f"- {breed}: {count} samples ({percentage:.1f}%)")

            with col2:
                st.metric("🧬 Total Samples", len(results_df))
                st.metric("🐄 Unique Breeds", len(breed_counts))

            # 下载按钮
            add_spacing(20)
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name=f"breed_identification_results_{st.session_state.uploaded_file_name.split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.error("⚠️ No genotype data to analyze. Please upload a file first.")


if __name__ == '__main__':
    page_frame()
    show_footer()
    analysis()