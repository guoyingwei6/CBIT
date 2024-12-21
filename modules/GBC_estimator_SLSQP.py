import pandas as pd
import numpy as np
from scipy.optimize import minimize

def read_allele_freqs(freq_file):
    """
    读取等位基因频率文件，'CHR:POS'为索引，后面列是各群体的等位基因频率
    """
    allele_freqs = pd.read_table(freq_file, sep='\s+', header=0, index_col='CHR:POS').dropna()
    allele_freqs_matrix = allele_freqs.values
    return allele_freqs, allele_freqs_matrix

def read_genotypes(gt_file):
    """
    读取个体基因型文件，'CHR:POS'为索引，其余为个体列
    """
    genotypes = pd.read_table(gt_file, sep='\s+', header=0).dropna()
    genotypes.set_index('CHR:POS', inplace=True)
    return genotypes

def negative_log_likelihood(w, allele_freqs_matrix, genotypes):
    """
    计算平均负对数似然
    w: 祖先成分比例向量，长度为T
    allele_freqs_matrix: (num_snps, T)
    genotypes: (num_snps,)
    """
    # 确保w是numpy数组
    w = np.array(w)
    
    # 计算加权等位基因频率 f_k = sum_j w_j * x_jk
    f = allele_freqs_matrix.dot(w)  # (num_snps,)
    
    # 增加数值稳定性，避免f=0或f=1
    epsilon = 1e-8
    f = np.clip(f, epsilon, 1 - epsilon)
    
    # 基因型
    g = genotypes  # (num_snps,) 的基因型0,1,2
    
    # 计算对数概率
    # g=0: lnP = 2 * ln(1 - f)
    # g=1: lnP = ln(2) + ln(f) + ln(1 - f)
    # g=2: lnP = 2 * ln(f)
    
    lnP = np.zeros_like(g, dtype=float)
    mask0 = (g == 0)
    mask1 = (g == 1)
    mask2 = (g == 2)
    
    lnP[mask0] = 2 * np.log(1 - f[mask0])
    lnP[mask1] = np.log(2) + np.log(f[mask1]) + np.log(1 - f[mask1])
    lnP[mask2] = 2 * np.log(f[mask2])
    
    # 检查是否有NaN或inf
    if np.isnan(lnP).any() or np.isinf(lnP).any():
        print(f"Encountered NaN or inf: w={w}")
        return np.inf
    
    # 计算平均负对数似然
    ll = np.sum(lnP)
    avg_nll = -ll / len(g)
    return avg_nll

def estimate_admixture(allele_freqs_matrix, genotypes, T):
    """
    估计祖先成分比例向量w
    使用SLSQP方法，设置约束：w_j >= 0，sum(w_j) = 1
    """
    # 初始值为均匀分布
    init_w = np.full(T, 1.0 / T)
    
    # 约束条件
    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    
    # 边界条件
    bounds = [(0, 1) for _ in range(T)]
    
    # 优化
    res = minimize(negative_log_likelihood, init_w,
                   args=(allele_freqs_matrix, genotypes),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints,
                   options={'disp': False, 'maxiter': 1000})
    
    if res.success:
        return res.x
    else:
        print(f"Optimization failed: {res.message}")
        return None

def main():
    # 文件路径（请根据实际情况修改）
    freq_file = 'WGS.merge_MP.frq.fmt'  # 等位基因频率文件
    gt_file = 'HN.merge.FinalSNP.rename_chrs.prune.raw.trans.fmt' # 基因型文件
    
    # 读取等位基因频率数据
    allele_freqs, allele_freqs_matrix = read_allele_freqs(freq_file)
    T = allele_freqs_matrix.shape[1]  # 群体数目
    
    # 读取基因型数据
    genotypes = read_genotypes(gt_file)
    
    # 对齐SNP
    common_snps = allele_freqs.index.intersection(genotypes.index)
    filtered_allele_freqs = allele_freqs.loc[common_snps]
    filtered_genotypes = genotypes.loc[common_snps]
    
    # 数据清洗：检查缺失值
    if filtered_allele_freqs.isnull().values.any():
        print("等位基因频率数据中存在缺失值。已删除含缺失值的SNP。")
        filtered_allele_freqs.dropna(inplace=True)
        filtered_genotypes = filtered_genotypes.loc[filtered_allele_freqs.index]
    
    if filtered_genotypes.isnull().values.any():
        print("基因型数据中存在缺失值。已删除含缺失值的SNP。")
        filtered_genotypes.dropna(inplace=True)
        filtered_allele_freqs = filtered_allele_freqs.loc[filtered_genotypes.index]
    
    # 确保等位基因频率在[0,1]范围内
    if ((filtered_allele_freqs < 0).any().any() or (filtered_allele_freqs > 1).any().any()):
        print("等位基因频率数据中存在不在[0,1]范围内的值。已裁剪到[0,1]范围。")
        filtered_allele_freqs = filtered_allele_freqs.clip(lower=0, upper=1)
    
    
    # 获取样本列表
    samples = filtered_genotypes.columns
    
    contributions_dict = {}
    
    # 对每个样本估计其祖先成分
    for sample_name in samples:
        g = filtered_genotypes[sample_name].values
        # 检查基因型是否为0,1,2
        if not np.all(np.isin(g, [0,1,2])):
            print(f"Sample {sample_name} contains invalid genotypes. Skipping.")
            contributions_dict[sample_name] = [None] * T
            continue
        w = estimate_admixture(filtered_allele_freqs.values, g, T)
        contributions_dict[sample_name] = w
    
    # 准备将结果保存为 CSV
    # 列名为 "SM" 加上参考群体名称
    col_names = ["SM"] + list(allele_freqs.columns)
    
    results_list = []
    for sample_name, w in contributions_dict.items():
        if w is not None:
            # w是长度为T的数组，每个元素是对应群体的祖先比例
            results_list.append([sample_name] + list(w))
        else:
            # 优化失败用None表示
            none_list = [None]*T
            results_list.append([sample_name] + none_list)
    
    df_results = pd.DataFrame(results_list, columns=col_names)
    output_csv = "WGS.SLSQP.csv"
    df_results.to_csv(output_csv, index=False)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
