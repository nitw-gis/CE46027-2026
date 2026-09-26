import time
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Load dataset
df = pd.read_csv('Cars.csv')

# Create binary target (1 if MPG >= 35, else 0)
df['Target'] = (df['MPG'] >= 35).astype(int)

X = df[['HP', 'VOL', 'SP', 'WT']]
y = df['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. t-SNE Plot
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Low MPG (0)', alpha=0.8, c='#1f77b4', edgecolors='k')
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='High MPG (1)', alpha=0.8, c='#ff7f0e', edgecolors='k')
plt.title('2D t-SNE Projection of Feature Space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Class')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('tsne_plot.png', dpi=300)
plt.show()

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (Polynomial)': SVC(kernel='poly', degree=3, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'DT (Shallow, d=2)': DecisionTreeClassifier(max_depth=2, random_state=42),
    'DT (Deep, d=8)': DecisionTreeClassifier(max_depth=8, random_state=42),
    'DT (min_split=5)': DecisionTreeClassifier(min_samples_split=5, random_state=42)
}

results = []

for name, model in models.items():
    tracemalloc.start()
    t0 = time.perf_counter()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        'Model': name,
        'Acc': f"{acc:.2f}",
        'Prec': f"{prec:.2f}",
        'Rec': f"{rec:.2f}",
        'F1': f"{f1:.2f}",
        'Time': f"{(t1-t0)*1000:.2f}",
        'CM': f"[{tn}, {fp}, {fn}, {tp}]"
    })

# 4. Clean aligned table printing without any dashed lines
print(f"{'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Time(ms)':<8} | {'CM [TN,FP,FN,TP]':<16}")
for r in results:
    m = r['Model']
    a = r['Acc']
    p = r['Prec']
    rc = r['Rec']
    f = r['F1']
    tm = r['Time']
    cm_str = r['CM']
    print(f"{m:<20} | {a:<6} | {p:<6} | {rc:<6} | {f:<6} | {tm:<8} | {cm_str:<16}")