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


def flatten_tensor(W):
    if torch.is_tensor(W):
        return W.flatten()
    return torch.cat(W).flatten()


def calculate_compression_ratio(input_sequence, code_table, input_bits_per_symbol):
    total_initial_bits = len(input_sequence) * input_bits_per_symbol
    total_compressed_bits = sum(len(code_table[symbol]) for symbol in input_sequence.tolist())
    compression_ratio = total_initial_bits / total_compressed_bits if total_compressed_bits > 0 else float("inf")
    return {
        "initial_bits": total_initial_bits,
        "compressed_bits": total_compressed_bits,
        "compression_ratio": compression_ratio,
    }


def estimate_compression_rate(counts, sequence):
    return torch.tensor([4, 4, 3, 2, 2, 3, 4, 4], dtype=torch.float).dot(counts).sum() / 3


if __name__ == "__main__":
    base_path = sys.argv[1]
    first = True

    outlier_thresholds = sorted(os.listdir(base_path))

    for matrix in [
                    'down_proj',
                    'gate_proj',
                    'up_proj',
                    'k_proj',
                    'o_proj',
                    'q_proj',
                    'v_proj'
    ]:
        for outlier_threshold in outlier_thresholds:
            outlier_path = os.path.join(base_path, outlier_threshold, "0")
            if not os.path.isdir(outlier_path):
                continue
            for t_name in os.listdir(outlier_path):
                tensor_path = os.path.join(outlier_path, t_name)
                if os.path.isfile(tensor_path):
                    t = torch.load(tensor_path, map_location="cpu")
                    W = t["quant_weights"]
                    m, n = W.shape[0], W.shape[1]
                    W = flatten_tensor(W)
                    values, freq = torch.unique(W, return_counts=True)
                    nnz = t["outliers_matrix"].to_sparse_csr().values().shape[0]

                    if first:
                        first = False
                        print("tensor;nnz;sparsity;mean;variance;outlier_threshold;compression_rate")
                    freq = freq.float() / freq.sum()
                    if matrix in tensor_path:
                        print(
                            f"{os.path.basename(tensor_path)};{nnz};{1 - nnz / (m * n):.6f};"
                            f"{torch.mean(freq):.4f};{torch.var(freq):.4f};{outlier_threshold};"
                            f"{estimate_compression_rate(freq, W.int())}"
                        )
