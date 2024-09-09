import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import sys

def load_breed_codes():
    """从文件中读取品种代码，传入字典中。"""
    df = pd.read_csv('attachments/breed_code.csv')
    code_breed_dict = pd.Series(df.Breed.values, index=df.Code).to_dict()
    return code_breed_dict

def load_model_fast():
    """加载模型。"""
    clf = joblib.load('attachments/Breed_identifier_fast_model.pkl')
    return clf

def load_model_accurate():
    """加载模型。"""
    clf = joblib.load('attachments/Breed_identifier_accurate_model.pkl')
    return clf

def breed_classifier(genotype_array, model='accurate'):
    """品种分类加预测概率函数。"""
    if model == 'fast':
        clf = load_model_fast()
    elif model == 'accurate':
        clf = load_model_accurate()
    predictions = clf.predict(genotype_array)
    probs = clf.predict_proba(genotype_array)
    max_probs = np.max(probs, axis=1)  # 获取最大概率
    breed_code_dict = load_breed_codes()
    breed_predictions = [breed_code_dict[code] for code in predictions]
    return list(zip(breed_predictions, max_probs))  # 返回标签和最大概率的元组列表



def analysis():
    genotype_file = sys.argv[1] # 命令行第一个参数是基因型文件
    model = sys.argv[2]  # 命令行第二个参数是模型选择
    

    gt_df = pd.read_csv(genotype_file, sep='\s+', header=None)
    sample_names = gt_df.iloc[:, 0]  # 提取样本名
    gt_array = gt_df.iloc[:, 1:].to_numpy()  # 提取基因型数据
    # 创建 SimpleImputer 对象，强制将缺失值填充为 0
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    # 使用 fit_transform 方法填充缺失值
    gt_array_imputed = imputer.fit_transform(gt_array)
    results = breed_classifier(gt_array_imputed, model=model)
    results_df = pd.DataFrame(results, columns=['Breed', 'Probability'])
    results_df.insert(0, 'Sample', sample_names)  # 将样本名插入到结果DataFrame的第一列
    print(results_df.to_string(index=False))  # 输出结果DataFrame，不显示索引

if __name__ == '__main__':
    '''python3 modules/Breed_identifier.py attachments/genotypes_for_Breed_identifier_accurate_model.txt accurate'''
    analysis()