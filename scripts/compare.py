import pandas as pd
import numpy as np

def compare():
    print("Loading preprocessed.csv...")
    df1 = pd.read_csv('preprocessed.csv')
    print("Loading MCU_Inference/train_preprocessed_snv.csv...")
    df2 = pd.read_csv('MCU_Inference/train_preprocessed_snv.csv')
    
    cols1 = [c for c in df1.columns if c.startswith('ch.')]
    cols2 = [c for c in df2.columns if c.startswith('ch.')]
    
    v1 = df1[cols1].values
    v2 = df2[cols2].values
    
    print(f"Shape 1: {v1.shape}, Shape 2: {v2.shape}")
    
    # If shapes differ, we slice to the min
    min_rows = min(v1.shape[0], v2.shape[0])
    min_cols = min(v1.shape[1], v2.shape[1])
    
    v1 = v1[:min_rows, :min_cols]
    v2 = v2[:min_rows, :min_cols]
    
    match = np.allclose(v1, v2, atol=1e-3)
    print(f"Exact Match (atol=1e-3): {match}")
    
    if not match:
        diff = np.abs(v1 - v2)
        print(f"Max diff: {np.max(diff)}")
        print(f"Mean diff: {np.mean(diff)}")

if __name__ == "__main__":
    compare()
