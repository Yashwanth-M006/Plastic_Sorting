"""
final_verify.py - Final e2e verification with new matched SVM
"""
import torch, torch.nn as nn, numpy as np, joblib, re

V6 = r"E:\CSIR\medley\nir_v6"
H  = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_selftest_data.h"

pt = torch.load(f"{V6}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd = pt["cnn_state"]
new_pipe = joblib.load(r"E:\CSIR\medley\MCU_Inference\nir_svm_pipeline_new.pkl")
svm, scaler = new_pipe["svm"], new_pipe["scaler"]

class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv1d(1,  64,  25, 8, 12, bias=False)
        self.b1 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(64, 128, 5,  1, 2,  bias=False)
        self.b4 = nn.BatchNorm1d(128)
        self.p  = nn.MaxPool1d(2, 2)
        self.c7 = nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        self.b8 = nn.BatchNorm1d(128)
        self.g  = nn.GELU()
    def forward(self, x):
        x = self.g(self.b1(self.c0(x)))
        x = self.g(self.b4(self.c3(x)))
        x = self.p(x)
        x = self.g(self.b8(self.c7(x)))
        return x.mean(-1)

net = B()
net.c0.weight.data = sd["backbone.0.weight"]
net.b1.weight.data = sd["backbone.1.weight"]
net.b1.bias.data   = sd["backbone.1.bias"]
net.b1.running_mean = sd["backbone.1.running_mean"].clone()
net.b1.running_var  = sd["backbone.1.running_var"].clone()
net.c3.weight.data = sd["backbone.3.weight"]
net.b4.weight.data = sd["backbone.4.weight"]
net.b4.bias.data   = sd["backbone.4.bias"]
net.b4.running_mean = sd["backbone.4.running_mean"].clone()
net.b4.running_var  = sd["backbone.4.running_var"].clone()
net.c7.weight.data = sd["backbone.7.weight"]
net.b8.weight.data = sd["backbone.8.weight"]
net.b8.bias.data   = sd["backbone.8.bias"]
net.b8.running_mean = sd["backbone.8.running_mean"].clone()
net.b8.running_var  = sd["backbone.8.running_var"].clone()
net.eval()

with open(H) as f:
    txt = f.read()

labels = list(map(int, re.findall(r"g_selftest_labels\[10\]\s*=\s*\{([^}]+)\}", txt)[0].split(",")))
blks = re.split(r"/\* sample \d+ [^*]*\*/", txt)
samples = []
for blk in blks[1:]:
    nums = re.findall(r"[-+]?\d+\.\d+e?[+-]?\d*f", blk)
    vals = [float(n.rstrip("f")) for n in nums]
    if len(vals) >= 224:
        samples.append(vals[:224])

X = np.array(samples[:10], dtype=np.float32)

print("=== FINAL VERIFICATION (selftest + new matched SVM) ===")
correct = 0
with torch.no_grad():
    for i in range(len(X)):
        x  = torch.tensor(X[i]).unsqueeze(0).unsqueeze(0)
        ft = net(x).numpy()[0]
        fs = scaler.transform(ft.reshape(1, -1))[0]
        sc = svm.coef_ @ fs + svm.intercept_
        p  = int(svm.classes_[np.argmax(sc)])
        ok = (p == labels[i])
        if ok:
            correct += 1
        status = "OK" if ok else "WRONG"
        print(f"  Sample {i}  true={labels[i]}  pred={p}  {status}  scores={np.round(sc, 2)}")

print(f"\nAccuracy: {correct}/{len(X)}  ({100*correct/len(X):.0f}%)")
print()
print("New SVM intercept (bias):", svm.intercept_)
print("Expected: now well-separated, not dominated by one class")
