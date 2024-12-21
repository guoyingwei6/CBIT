import pandas as pd
import numpy as np
from scipy.optimize import minimize

def read_allele_freqs(freq_file):
    # 读取等位基因频率文件，第一列为'CHR:POS'，其他列为各群体等位基因频率
    # index_col='CHR:POS' 将SNP标识设为行索引
    allele_freqs = pd.read_table(freq_file, sep='\s+', header=0, index_col='CHR:POS').dropna()
    # allele_freqs_matrix: shape = (num_snps, T)
    allele_freqs_matrix = allele_freqs.values
    return allele_freqs, allele_freqs_matrix

def read_genotypes(gt_file):
    # 读取个体基因型文件，一列为'CHR:POS'，后面为各个体列
    genotypes = pd.read_table(gt_file, sep='\s+', header=0).dropna()
    genotypes.set_index('CHR:POS', inplace=True)
    # genotypes_matrix: shape = (num_snps, num_samples)
    # columns为样本名，rows为SNP
    return genotypes

def negative_log_likelihood(w_scalar, allele_freqs_matrix, genotypes):
    w1 = w_scalar[0]
    if w1 < 0 or w1 > 1:
        return np.inf
    w2 = 1 - w1
    w = np.array([w1, w2])
    
    f = allele_freqs_matrix.dot(w)
    epsilon = 1e-8  # 增大epsilon
    f = np.clip(f, epsilon, 1 - epsilon)

    g = genotypes
    mask0 = (g == 0)
    mask1 = (g == 1)
    mask2 = (g == 2)

    lnP = np.zeros_like(g, dtype=float)
    lnP[mask0] = 2 * np.log(1 - f[mask0])
    lnP[mask1] = np.log(2) + np.log(f[mask1]) + np.log(1 - f[mask1])
    lnP[mask2] = 2 * np.log(f[mask2])

    # 检查lnP是否有NaN
    if np.isnan(lnP).any():
        return np.inf  # 避免NaN导致优化崩溃

    ll = np.sum(lnP)
    return -ll

def estimate_admixture(allele_freqs_matrix, genotypes):
    # 假设T=2，使用单参数w1表示w=[w1, 1-w1]
    bounds = [(0,1)]
    res = minimize(negative_log_likelihood, [0.5],
                   args=(allele_freqs_matrix, genotypes),
                   method='L-BFGS-B', bounds=bounds, options={'disp': False})
    if res.success:
        w1 = res.x[0]
        w2 = 1 - w1
        return np.array([w1, w2])
    else:
        return None

def main():
    # 文件路径
    freq_file = 'WGS.merge_MP.frq.fmt'  # 等位基因频率文件
    gt_file = 'HN.merge.FinalSNP.rename_chrs.prune.raw.trans.fmt' # 基因型文件
    
    
    # 读取等位基因频率数据
    allele_freqs, allele_freqs_matrix = read_allele_freqs(freq_file)
    T = allele_freqs_matrix.shape[1]  # 群体数目，这里假设为2，请确保数据中为2列

    # 读取基因型数据
    genotypes = read_genotypes(gt_file)
    
    # 对齐SNP
    common_snps = allele_freqs.index.intersection(genotypes.index)
    filtered_allele_freqs = allele_freqs.loc[common_snps]
    filtered_genotypes = genotypes.loc[common_snps]
    
    # 获取样本列表
    samples = filtered_genotypes.columns
    
    contributions_dict = {}

    # 对每个样本估计其祖先成分
    for sample_name in samples:
        g = filtered_genotypes[sample_name].values
        w = estimate_admixture(filtered_allele_freqs.values, g)
        contributions_dict[sample_name] = w

    # 准备将结果保存为 CSV
    # 创建一个列表，用于存储最终结果的数据行
    results_list = []

    for sample_name, w in contributions_dict.items():
        if w is not None:
            # w 是一个包含两个祖先成分比例的数组，如 w = [w_M, w_P]
            w_m = w[0]
            w_p = w[1]
            results_list.append([sample_name, w_m, w_p])
        else:
            # 如果优化失败，可以选择跳过或给出默认值
            # 这里假设失败的结果不写入文件，也可以写入 NaN
            results_list.append([sample_name, None, None])
            pass

    # 将结果转为 DataFrame
    df_results = pd.DataFrame(results_list, columns=["SM", "M", "P"])

    # 保存为 CSV 文件
    output_csv = "WGS_BFGS.csv"
    df_results.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
