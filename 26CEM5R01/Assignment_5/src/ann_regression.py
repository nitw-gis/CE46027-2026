import pandas as pd
import os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import glob

# --- Auto-detect regression file ---
# data folder me jo bhi regression csv hai usko lega
files = glob.glob("data/*regression*.csv") + glob.glob("data/*Regression*.csv") + glob.glob("data/*.csv")
print("Data files found:", files)
# Agar tera file ka naam alag hai to niche path change kar dena
csv_path = files[0] if files else "data/regression_data.csv"
# EXAMPLE: agar tera file data/housing.csv hai to: csv_path = "data/housing.csv"
df = pd.read_csv(csv_path)

# Last column ko target maan raha hu, agar alag hai to yahan change kar
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

layers_range = range(1, 11)
activations = ['relu', 'tanh', 'logistic']
lrs = [0.1, 0.01, 0.001]

os.makedirs("plots/ann_regression", exist_ok=True)
os.makedirs("output", exist_ok=True)

results = []
for lr in lrs:
    plt.figure(figsize=(10,6))
    for act in activations:
        mse_list = []
        for L in layers_range:
            print(f"ANN Reg: LR={lr} Act={act} Layer={L}")
            mlp = MLPRegressor(hidden_layer_sizes=(100,)*L, activation=act,
                               learning_rate_init=lr, max_iter=500, random_state=42)
            mlp.fit(X_train_s, y_train)
            mse = mean_squared_error(y_test, mlp.predict(X_test_s))
            mse_list.append(mse)
            results.append([L, act, lr, mse])
        plt.plot(list(layers_range), mse_list, marker='o', label=act)
    
    plt.title(f"ANN Regression - MSE vs Layers (1-10) LR={lr}")
    plt.xlabel("Layers (1-10)"); plt.ylabel("MSE (lower is better)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"plots/ann_regression/mse_LR_{lr}.png")
    plt.close()
    print(f" -> Saved mse_LR_{lr}.png")

pd.DataFrame(results, columns=['layers','activation','lr','mse']).to_csv("output/ann_regression_results.csv", index=False)
print("DONE ANN Regression - 3 graphs ban gaye plots/ann_regression me")