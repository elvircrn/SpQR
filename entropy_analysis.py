import torch
import os
import sys

if __name__ == '__main__':
    base_path = sys.argv[1]
    for p in os.listdir(base_path):
        for t in os.listdir(os.path.join(base_path, p)):
            tensor_path = os.path.join(base_path, p, t)
            print(f'tensor_path = {tensor_path}')
            if os.path.isfile(tensor_path):
                t = torch.load(tensor_path, map_location='cpu')
                print(t.keys())
                W = t['quant_weights']
                m = W.shape[0]
                n = W.shape[1]
                values, counts = torch.unique(W, return_counts=True)
                nnz = t["outliers_matrix"].to_sparse_csr().values().shape[0]
                counts = counts.float() / counts.float().sum()
                print(f'Tensor {t} stats\nnnz = {nnz}\n:counts = {counts}\nmean = {torch.mean(counts)}\nvariance = {torch.var(counts)}')
                break
