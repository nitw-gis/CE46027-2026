import pandas as pd
import os, warnings, glob
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Sahi regression file
csv_files = glob.glob("data/smart_city*stress*.csv")
csv_path = csv_files[0] if csv_files else "data/smart_city_traffic_stress_data.csv"
print(f"Using: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Shape: {df.shape}, Columns: {list(df.columns)[:5]}")

# Last column target hai, agar target ka naam 'stress_level' hai to neeche change kar sakta hai
# Example: X = df.drop('stress_level', axis=1)
X = df.iloc[:, :-1].select_dtypes(include=['number']).values
y = df.iloc[:, -1].values
# Agar y string hai toh encode kar de
if y.dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

layers_range = range(1, 11)
activations = ['relu', 'tanh', 'logistic']
lrs = [0.1, 0.01, 0.001]

os.makedirs("plots/ann_regression", exist_ok=True)

results = []
for lr in lrs:
    plt.figure(figsize=(10,6))
    for act in activations:
        mse_list = []
        for L in layers_range:
            print(f"ANN Reg: LR={lr} Act={act} Layer={L}")
            mlp = MLPRegressor(hidden_layer_sizes=(64,)*L, activation=act,
                               learning_rate_init=lr, max_iter=400, random_state=42)
            mlp.fit(X_train_s, y_train)
            mse = mean_squared_error(y_test, mlp.predict(X_test_s))
            mse_list.append(mse)
            results.append([L, act, lr, mse])
        plt.plot(list(layers_range), mse_list, marker='o', label=act)
    plt.title(f"ANN Regression - MSE vs Layers LR={lr}")
    plt.xlabel("Layers"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True)
    plt.savefig(f"plots/ann_regression/mse_LR_{lr}.png")
    plt.close()

pd.DataFrame(results, columns=['layers','activation','lr','mse']).to_csv("output/ann_regression_results.csv", index=False)
print("DONE ANN Regression - 3 graphs ban gaye")