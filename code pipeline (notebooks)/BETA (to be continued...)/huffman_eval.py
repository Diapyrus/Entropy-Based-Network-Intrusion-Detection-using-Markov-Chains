import polars as pl
import pandas as pd
import heapq
from collections import Counter
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import numpy as np

class HuffmanNode:
    """A node in the Huffman Tree."""
    def __init__(self, char: Any, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # Required for heapq to compare nodes
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies: Dict[Any, int]) -> Optional[HuffmanNode]:
    """
    Builds a Huffman Tree from a dictionary of symbol frequencies.
    """
    if not frequencies:
        return None
        
    heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Create a merged node
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node: Optional[HuffmanNode], current_code: str = "", codes: Dict[Any, str] = None) -> Dict[Any, str]:
    """
    Traverses the Huffman tree to generate binary codes for each symbol.
    """
    if codes is None:
        codes = {}
        
    if node is None:
        return codes

    # If it's a leaf node, assign the code
    if node.char is not None:
        codes[node.char] = current_code if current_code else "0"
        return codes

    build_huffman_codes(node.left, current_code + "0", codes)
    build_huffman_codes(node.right, current_code + "1", codes)

    return codes

def encode_sequence(sequence: List[Any]) -> Tuple[str, Dict[Any, str]]:
    """
    Encodes a sequence of symbols using Huffman Coding.
    
    Returns:
        A tuple of (encoded_binary_string, huffman_code_dictionary)
    """
    if not sequence:
        return "", {}
        
    frequencies = Counter(sequence)
    root = build_huffman_tree(frequencies)
    codes = build_huffman_codes(root)
    
    encoded_string = "".join(codes[sym] for sym in sequence)
    return encoded_string, codes

def extract_time_window(df: pl.DataFrame, label: str, duration_minutes: int = 5) -> List[Any]:
    """
    Extracts a chronological window of data for a specific label.
    """
    # Filter by label
    df_filtered = df.filter(pl.col("Label") == label)
    
    if df_filtered.is_empty():
        return []
        
    # Ensure it's sorted by time
    df_filtered = df_filtered.sort("Timestamp")
    
    # Get the start time
    start_time = df_filtered["Timestamp"][0]
    end_time = start_time + pd.Timedelta(minutes=duration_minutes)
    
    # Extract the window
    window_df = df_filtered.filter(pl.col("Timestamp") <= end_time)
    
    return window_df["Tot_Fwd_Pkts_sym"].to_list()

def evaluate_huffman_compression(parquet_path: str) -> pd.DataFrame:
    """
    Isolates Benign and Attack windows, applies Huffman coding, and compares the compression.
    
    Returns:
        A Pandas DataFrame with the comparison metrics.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    # Read the dataset and parse timestamps
    df = pl.read_parquet(path).with_columns(
        pl.col("Timestamp").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S")
    )
    
    # Extract 5-minute windows
    benign_seq = extract_time_window(df, label="Benign", duration_minutes=5)
    attack_seq = extract_time_window(df, label="DDoS", duration_minutes=5)
    
    # Fallback if specific attack label differs
    if not attack_seq:
        # Get the most common attack label
        attack_labels = df.filter(pl.col("Label") != "Benign")["Label"].unique().to_list()
        if attack_labels:
            attack_seq = extract_time_window(df, label=attack_labels[0], duration_minutes=5)
            
    results = []
    
    for seq_name, sequence in [("Benign Traffic (5 min)", benign_seq), ("Attack Traffic (5 min)", attack_seq)]:
        if not sequence:
            continue
            
        # Standard uncompressed size assumes 8 bins -> 3 bits per symbol
        # Or more dynamically: ceil(log2(unique_symbols))
        unique_symbols = len(set(sequence))
        bits_per_symbol_uncompressed = max(1, int(np.ceil(np.log2(unique_symbols))))
        original_size_bits = len(sequence) * bits_per_symbol_uncompressed
        
        encoded_str, _ = encode_sequence(sequence)
        compressed_size_bits = len(encoded_str)
        
        avg_bits_per_symbol = compressed_size_bits / len(sequence) if len(sequence) > 0 else 0
        compression_ratio = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else 0
        
        results.append({
            "Traffic Type": seq_name,
            "Sequence Length (Flows)": len(sequence),
            "Original Size (Bits)": original_size_bits,
            "Compressed Size (Bits)": compressed_size_bits,
            "Compression Ratio": round(compression_ratio, 2),
            "Avg Bits/Symbol": round(avg_bits_per_symbol, 3)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parquet_file = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed.parquet"
    if Path(parquet_file).exists():
        print("Evaluating Huffman Compression...")
        df_comp = evaluate_huffman_compression(parquet_file)
        print("\nCompression Results:")
        print(df_comp)
    else:
        print(f"File not found: {parquet_file}")
