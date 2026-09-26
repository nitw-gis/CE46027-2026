import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv('Cars.csv')

# Create binary classification target based on median MPG
df['Target'] = (df['MPG'] >= 35).astype(int)

# Separate features and target
X = df[['HP', 'VOL', 'SP', 'WT']]
y = df['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Perform t-SNE dimensional reduction
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Low MPG (0)', alpha=0.8, c='#1f77b4', edgecolors='k')
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='High MPG (1)', alpha=0.8, c='#ff7f0e', edgecolors='k')
plt.title('2D t-SNE Feature Space Visualization', fontsize=12, fontweight='bold')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Class')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300)
plt.show()

