import torch
import os
import sys

import heapq

class Node:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    heap = [Node(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_huffman_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = current_code
    else:
        if node.left:
            generate_huffman_codes(node.left, current_code + "0", codes)
        if node.right:
            generate_huffman_codes(node.right, current_code + "1", codes)
    return codes

def huffman_from_frequencies(frequencies):
    tree = build_huffman_tree(frequencies)
    huffman_table = generate_huffman_codes(tree)

    return huffman_table


def estimate_compression_rate(freq, sequence):
    freq = {i: f for i, f in enumerate(freq)}

    return huffman_from_frequencies(freq)['compression_ratio']


def flatten_tensor(W):
    """
    @return: Utility function: flattens the input tensor.
    """
    if torch.is_tensor(W):
        return W.flatten()
    else:
        return torch.cat(W).flatten()


if __name__ == '__main__':
    base_path = sys.argv[1]
    first = True

    outlier_thresholds = os.listdir(base_path)
    outlier_thresholds = sorted(outlier_thresholds)

    for outlier_threshold in outlier_thresholds:
        for t_name in os.listdir(os.path.join(base_path, outlier_threshold, '0')):
            tensor_path = os.path.join(base_path, outlier_threshold, '0', t_name)
            # print(f'tensor_path = {tensor_path}')
            if os.path.isfile(tensor_path):
                t = torch.load(tensor_path, map_location='cpu')
                W = t['quant_weights']
                m = W.shape[0]
                n = W.shape[1]
                W = flatten_tensor(W)
                values, counts = torch.unique(W, return_counts=True)
                nnz = t["outliers_matrix"].to_sparse_csr().values().shape[0]

                if first:
                    first = False
                    print('tensor;nnz;sparsity;mean;variance;outlier_threshold;compression_rate')
                counts = counts.float() / counts.float().sum()
                # print(f'Tensor {t_name} stats\nnnz = {nnz}\n:counts = {counts}\nmean = {torch.mean(counts)}\nvariance = {torch.var(counts)}')
                if 'q_proj' in tensor_path:
                    print(
                        f'{os.path.basename(tensor_path)};{nnz};{1 - nnz / (m * n)};{torch.mean(counts):.4f};{torch.var(counts):.4f};{outlier_threshold};{estimate_compression_rate(counts, W.int())}')
