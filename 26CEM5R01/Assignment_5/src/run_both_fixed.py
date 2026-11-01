import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd, warnings, glob
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_ANN = os.path.join(BASE_DIR, "plots", "ann_regression")
PLOT_CNN = os.path.join(BASE_DIR, "plots", "cnn_regression")
os.makedirs(PLOT_ANN, exist_ok=True)
os.makedirs(PLOT_CNN, exist_ok=True)

# Data
csv_files = glob.glob(os.path.join(BASE_DIR, "data", "*stress*.csv"))
if not csv_files:
    csv_files = glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))
csv_path = csv_files[1] if len(csv_files)>1 else csv_files[0]
print(f"Using: {csv_path}")

df = pd.read_csv(csv_path)
X = df.iloc[:, :-1].select_dtypes(include=['number']).values
y = df.iloc[:, -1].values
if y.dtype == object:
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
n_features = X_train_s.shape[1]
X_train_cnn = X_train_s.reshape(-1, n_features, 1)
X_test_cnn = X_test_s.reshape(-1, n_features, 1)

layers_range = range(1, 11)
activations_ann = ['relu', 'tanh', 'logistic']
activations_cnn = {'relu': 'relu', 'tanh': 'tanh', 'logistic': 'sigmoid'}
lrs = [0.1, 0.01, 0.001]

# ANN
print("\n=== ANN REGRESSION ===")
for lr in lrs:
    lr_tag = str(lr).replace('.', '_')
    plt.figure(figsize=(10,6))
    for act in activations_ann:
        mse_list = []
        for L in layers_range:
            print(f"ANN LR={lr} {act} L={L}/10")
            mlp = MLPRegressor(hidden_layer_sizes=(64,)*L, activation=act,
                               learning_rate_init=lr, max_iter=300, random_state=42)
            mlp.fit(X_train_s, y_train)
            mse_list.append(mean_squared_error(y_test, mlp.predict(X_test_s)))
        plt.plot(list(layers_range), mse_list, marker='o', label=act)
    plt.title(f"ANN Regression MSE vs Layers LR={lr}")
    plt.xlabel("Layers"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)
    save_path = os.path.join(PLOT_ANN, f"mse_LR_{lr_tag}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

# CNN
print("\n=== CNN REGRESSION ===")
for lr in lrs:
    lr_tag = str(lr).replace('.', '_')
    plt.figure(figsize=(10,6))
    for act_name, act_tf in activations_cnn.items():
        mse_list = []
        for L in layers_range:
            print(f"CNN LR={lr} {act_name} L={L}/10")
            model = Sequential()
            model.add(Conv1D(8, 2, activation=act_tf, input_shape=(n_features,1)))
            for _ in range(L-1):
                model.add(Conv1D(8, 2, activation=act_tf, padding='same'))
            model.add(Flatten()); model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, verbose=0)
            pred = model.predict(X_test_cnn, verbose=0).flatten()
            mse_list.append(mean_squared_error(y_test, pred))
        plt.plot(list(layers_range), mse_list, marker='o', label=act_name)
    plt.title(f"CNN Regression MSE vs Layers LR={lr}")
    plt.xlabel("Layers"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)
    save_path = os.path.join(PLOT_CNN, f"mse_LR_{lr_tag}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

print("\nALL DONE! 6 graphs ready")