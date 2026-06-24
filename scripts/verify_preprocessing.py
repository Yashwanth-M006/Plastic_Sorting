import pandas as pd
import numpy as np
import scipy.ndimage

def median_3tap(arr):
    # vectorized 3-element sorting network
    # median(a,b,c) = max(min(a,b), min(max(a,b),c))
    out = np.empty_like(arr)
    # Handle edges by duplication
    # out[0] = median(arr[0], arr[0], arr[1]) = arr[0] if arr[0] between arr[0] and arr[1]... wait, median(x, x, y) = x
    a = np.pad(arr, (1, 1), mode='edge')[:-2]
    b = np.pad(arr, (1, 1), mode='edge')[1:-1]
    c = np.pad(arr, (1, 1), mode='edge')[2:]
    out = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
    return out

def sg_5tap(arr):
    # 5-tap SG kernel: [-22, 88, 124, 88, -22] / 256
    kernel = np.array([-22, 88, 124, 88, -22], dtype=np.float64) / 256.0
    return scipy.ndimage.convolve1d(arr, kernel, mode='nearest')

def snv(arr):
    # subtract mean, divide by population std
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)
    if std == 0:
        std = 1.0 # prevent div by zero
    return (arr - mean) / std

def main():
    print("Loading data...")
    train_df = pd.read_csv('SpectrumData_2021Y-trainSet.csv')
    dark_df = pd.read_csv('SpectrumData_2021Y-darkcurrent.csv')

    # Calculate mean dark current per channel
    dark_mean = dark_df.mean(axis=0).values

    # Generate C header for dark current
    print("Generating C header for dark current...")
    with open('dark_current_mean.h', 'w') as f:
        f.write("#ifndef DARK_CURRENT_MEAN_H\n")
        f.write("#define DARK_CURRENT_MEAN_H\n\n")
        f.write(f"#define NUM_CHANNELS {len(dark_mean)}\n\n")
        f.write("const float g_dark_current_mean[NUM_CHANNELS] = {\n")
        for i in range(0, len(dark_mean), 8):
            f.write("    " + ", ".join([f"{val:.4f}f" for val in dark_mean[i:i+8]]) + ",\n")
        f.write("};\n\n")
        f.write("#endif // DARK_CURRENT_MEAN_H\n")

    print("Applying preprocessing pipeline...")
    # Extract just the channel data (assuming 224 channels, based on previous shape check)
    channels = [c for c in train_df.columns if c.startswith('ch.')]
    
    # We will process row by row to simulate what the MCU does
    preprocessed_data = []
    
    # Process only the first 1000 rows to save time, or all if preferred. Let's do all.
    # Actually, vectorized is faster. Let's do it vectorized for python speed but logically identical.
    raw_signals = train_df[channels].values
    
    # 1. Subtract Dark Current
    subtracted = raw_signals - dark_mean
    
    # 2. Median Filter & 3. SG Filter & 4. SNV
    # Using a loop for exact identical reproduction per spectrum
    out_signals = np.empty_like(subtracted)
    for i in range(subtracted.shape[0]):
        # Median
        med = median_3tap(subtracted[i])
        # SG
        sg = sg_5tap(med)
        # SNV
        out_signals[i] = snv(sg)
        
        if i % 10000 == 0:
            print(f"Processed {i} / {subtracted.shape[0]} spectra")

    print("Saving preprocessed.csv...")
    # Put it back into dataframe
    out_df = train_df.copy()
    out_df[channels] = out_signals
    out_df.to_csv('preprocessed.csv', index=False)
    
    print("Done!")

if __name__ == '__main__':
    main()
