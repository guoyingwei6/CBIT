import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
import sys

def load_breed_codes():
    """从文件中读取品种代码，传入字典中。"""
    df = pd.read_csv('attachments/breed_code.csv')
    code_breed_dict = pd.Series(df.Breed.values, index=df.Code).to_dict()
    return code_breed_dict

def load_model():
    """加载模型。"""
    clf = joblib.load('attachments/SVC_500_best.pkl')
    return clf

def breed_classifier(genotype_array):
    """品种分类函数。"""
    clf = load_model()
    prediction = clf.predict(genotype_array)
    breed_code_dict = load_breed_codes()
    breed_prediction = [breed_code_dict[code] for code in prediction]
    return breed_prediction

if __name__ == '__main__':
    '''python3 Breed_identifier.py attachments/genotypes_for_Breed_identifier.txt
    The argument is the genotype file.
    You can find the results in the attachments folder named breed_results.csv.'''
    genotype_file = sys.argv[1] # 命令行第一个参数是基因型文件
    gt_df = pd.read_csv(genotype_file, sep='\s+', header=None)
    gt_array = gt_df.to_numpy()
    result = breed_classifier(gt_array)
    pd.DataFrame(result).to_csv('attachments/breed_results.csv', index=False)