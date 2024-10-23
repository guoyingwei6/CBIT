import numpy as np
from scipy.optimize import minimize
import argparse
import joblib
import pandas as pd
from scipy.optimize import check_grad


def load_AF():
    """从文件中读取等位基因频率"""
    allele_freqs = joblib.load('attachments/AF_for_gbc.pkl')
    return allele_freqs

def common_snps(gt_file):
    allele_freqs = load_AF()
    genotypes = pd.read_table(gt_file, sep='\s+', header=0, index_col='CHR:POS').dropna()
    common_snps = allele_freqs.index.intersection(genotypes.index)
    filtered_allele_freqs = allele_freqs.loc[common_snps]
    filtered_genotypes = genotypes.loc[common_snps]
    allele_freqs_matrix = filtered_allele_freqs.values
    genotypes_matrix = filtered_genotypes.values
    return allele_freqs_matrix, genotypes_matrix, filtered_allele_freqs.columns.tolist(), filtered_genotypes.columns.tolist()

def check_genotypes(genotypes_vector):
    valid_genotypes = [0, 1, 2]
    if not np.all(np.isin(genotypes_vector, valid_genotypes)):
        print("Warning: genotypes_vector contains invalid values.")
        genotypes_vector = np.where(np.isin(genotypes_vector, valid_genotypes), genotypes_vector, np.nan)
    return genotypes_vector

def ancestryCLL(x, allele_freqs_matrix, genotypes_vector):
    valid_mask = np.isin(genotypes_vector, [0, 1, 2])
    valid_genotypes = genotypes_vector[valid_mask]
    valid_allele_freqs = allele_freqs_matrix[valid_mask]
    WAFSUM = np.dot(valid_allele_freqs, x)
    WAFSUM_c = 1 - WAFSUM
    epsilon = 1e-15
    WAFSUM = np.clip(WAFSUM, epsilon, 1 - epsilon)
    WAFSUM_c = np.clip(WAFSUM_c, epsilon, 1 - epsilon)
    probs = np.zeros_like(valid_genotypes, dtype=np.float64)
    mask0 = (valid_genotypes == 0)
    mask1 = (valid_genotypes == 1)
    mask2 = (valid_genotypes == 2)
    probs[mask0] = WAFSUM[mask0] ** 2
    probs[mask1] = 2 * WAFSUM[mask1] * WAFSUM_c[mask1]
    probs[mask2] = WAFSUM_c[mask2] ** 2
    probs = np.clip(probs, epsilon, None)
    log_likelihood = np.sum(np.log(probs))
    return -log_likelihood

def ancestryCLL_grad(x, allele_freqs_matrix, genotypes_vector):
    valid_mask = np.isin(genotypes_vector, [0, 1, 2])
    valid_genotypes = genotypes_vector[valid_mask]
    valid_allele_freqs = allele_freqs_matrix[valid_mask]
    WAFSUM = np.dot(valid_allele_freqs, x)
    WAFSUM_c = 1 - WAFSUM
    epsilon = 1e-15
    WAFSUM = np.clip(WAFSUM, epsilon, 1 - epsilon)
    WAFSUM_c = np.clip(WAFSUM_c, epsilon, 1 - epsilon)
    p_jk = valid_allele_freqs
    mask0 = (valid_genotypes == 0)
    mask1 = (valid_genotypes == 1)
    mask2 = (valid_genotypes == 2)
    grad_matrix = np.zeros_like(p_jk)
    if np.any(mask0):
        grad_matrix[mask0] = (2 * p_jk[mask0]) / WAFSUM[mask0][:, np.newaxis]
    if np.any(mask1):
        grad_matrix[mask1] = p_jk[mask1] * (1 - 2 * WAFSUM[mask1])[:, np.newaxis] / (WAFSUM[mask1] * WAFSUM_c[mask1])[:, np.newaxis]
    if np.any(mask2):
        grad_matrix[mask2] = (-2 * p_jk[mask2]) / WAFSUM_c[mask2][:, np.newaxis]
    grad = np.sum(grad_matrix, axis=0)
    return -grad

def main(args):
    allele_freqs_matrix, genotypes_matrix, poplabels, sample_names = common_snps(args.gt_file)
    POPS = len(poplabels)
    results_dict = {}
    for idx, sample_name in enumerate(sample_names):
        #init_vector = np.zeros(POPS)
        #init_vector[0] = 1.0
        init_vector = np.ones(POPS) / POPS
        bounds = [(0, 1) for _ in range(POPS)]
        current_genotypes = genotypes_matrix[:, idx]
        current_genotypes = check_genotypes(current_genotypes)
        # 梯度校验
        grad_error = check_grad(
            lambda x: ancestryCLL(x, allele_freqs_matrix, current_genotypes),
            lambda x: ancestryCLL_grad(x, allele_freqs_matrix, current_genotypes),
            init_vector
        )
        print(f"Gradient check error for {sample_name}: {grad_error}")
        # 约束条件
        def constraint(x):
            return np.sum(x) - 1
        constraints = {'type': 'eq', 'fun': constraint}
        # 进行优化
        result = minimize(
            fun=ancestryCLL,
            x0=init_vector,
            args=(allele_freqs_matrix, current_genotypes),
            jac=ancestryCLL_grad,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'disp': False}
        )
        results_dict[sample_name] = result.x
    ancestry_results = pd.DataFrame(results_dict, index=poplabels).round(4)
    ancestry_results.to_csv('attachments/ancestry_results.csv')
    return ancestry_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ancestry estimation using SLSQP optimization.")
    parser.add_argument("--gt_file", type=str, required=True, help="Genotype file.")
    args = parser.parse_args()
    main(args)