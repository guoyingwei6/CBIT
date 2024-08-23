import streamlit as st
from modules.common import show_footer

st.set_page_config(page_title="About",page_icon="💻",layout="centered",initial_sidebar_state="expanded")
st.title('About')

st.write('This tool is designed to help you identify different breeds of cattle.')


st.header('Contact')
st.write('If you have any questions, please contact us at: yingwei.guo@foxmail.com')

def main():


    st.markdown("""
    ## Introduction
    This tool is designed to help users identify different cattle breeds through genotypic data. 
    It utilizes a machine learning model to predict breeds based on uploaded genotype files.
    
    ## Step 1: Prepare Your Data
    - Prepare your genotypic data ensuring it meets the format requirements (e.g., space-separated text files).
    - Each line should represent the genotype data for an individual.
    
    ## Step 2: Upload Your File
    - Visit the tool's webpage.
    - Click the “Choose a file” button on the page.
    - In the file selector dialog that appears, locate and select your genotype file.
    
    ## Step 3: Perform Analysis
    - After uploading the file, click the “Analyze” button to start the analysis.
    - The system will process your file and display the breed identification results.
    
    ## Important Notes
    - Once uploaded, your data will be cached for one hour. Even if you refresh the page or revisit, 
      the previous analysis results will still be visible unless a new file is uploaded.
    - Ensure your file is in the correct format, otherwise, the system will indicate that the file is invalid.
    
    ## Interpreting Results
    - The analysis results will be displayed under “Analysis Results,” where you can see the predicted breed for each individual.
    - Results are derived using a pre-trained machine learning model that employs the Support Vector Machine (SVM) algorithm.
    
    ## Technical Support
    - If you encounter any technical issues while using the tool, you can contact the developer through the provided contact methods (email, phone, etc.).
    
    ## Conclusion
    We hope this tool makes it easy for you to identify cattle breeds. Thank you for using it, and we look forward to your feedback to help us improve this tool.
    """, unsafe_allow_html=False)

if __name__ == '__main__':
    main()
    show_footer()

