# validate_selftest.py
# Run on PC with your trained model to get expected MCU predictions
# Compare output against MCU LED blinks

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CSV_PATH   = "train_preprocessed_snv.csv"
MODEL_PATH = "nir_cnn_backbone (1).pt"   # your saved PyTorch model
N_PER_CLASS = 2

df = pd.read_csv(CSV_PATH)
y_col  = "class_id"
x_cols = [c for c in df.columns if c != y_col]

picked = (df.groupby(y_col, group_keys=False)
            .apply(lambda g: g.sample(2, random_state=0),
                   include_groups=False))
picked.insert(0, y_col, df.loc[picked.index, y_col].values)
picked = picked.reset_index(drop=True)

X = torch.tensor(picked[x_cols].values, dtype=torch.float32).unsqueeze(1)
y = picked[y_col].values

model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()
with torch.no_grad():
    logits = model(X)
    preds = logits.argmax(dim=1).numpy() + 1   # 0-indexed -> 1-indexed

print("Sample  True  Pred  Match")
for i, (t, p) in enumerate(zip(y, preds)):
    print(f"  {i+1:2d}     {t}     {p}    {'OK' if t==p else 'WRONG'}")
print(f"\nExpected accuracy: {np.mean(y==preds)*100:.1f}%")
print("\nMCU should blink these predicted classes:")
print([int(p) for p in preds])