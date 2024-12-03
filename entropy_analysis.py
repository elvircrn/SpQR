import torch
import os
import sys

if __name__ == '__main__':
    base_path = sys.argv[1]
    print('tensor;nnz;sparsity;mean;variance;outlier_threshold')
    for outlier_threshold in os.listdir(base_path):
        for t_name in os.listdir(os.path.join(base_path, outlier_threshold, '0')):
            tensor_path = os.path.join(base_path, outlier_threshold, '0', t_name)
            # print(f'tensor_path = {tensor_path}')
            if os.path.isfile(tensor_path):
                t = torch.load(tensor_path, map_location='cpu')
                W = t['quant_weights']
                m = W.shape[0]
                n = W.shape[1]
                values, counts = torch.unique(W, return_counts=True)
                nnz = t["outliers_matrix"].to_sparse_csr().values().shape[0]
                counts = counts.float() / counts.float().sum()
                # print(f'Tensor {t_name} stats\nnnz = {nnz}\n:counts = {counts}\nmean = {torch.mean(counts)}\nvariance = {torch.var(counts)}')
                if 'q_proj' in tensor_path:
                    print(f'{os.path.basename(tensor_path)};{nnz};{1 - nnz / (m * n)};{torch.mean(counts):.4f};{torch.var(counts):.4f};{outlier_threshold}')