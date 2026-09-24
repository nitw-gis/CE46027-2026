import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load data
df = pd.read_csv(r"D:\ASWATHI P\class work\WineQT.csv")

# Drop Id column and separate features & target
X = df.drop(["quality", "Id"], axis=1)
y = df["quality"]

print("Features shape:", X.shape)
print("Target classes:", sorted(y.unique()))
print(df.head())

# Scale features
X_scaled = StandardScaler().fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="tab10", s=60, alpha=0.8)
plt.title("2D t-SNE Visualization of Wine Quality Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Quality")
plt.tight_layout()
plt.savefig("tsne_plot.png", dpi=300)
plt.show()

print("t-SNE plot saved as tsne_plot.png")