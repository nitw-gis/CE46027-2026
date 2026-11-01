import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd, warnings, glob
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

csv_files = glob.glob("data/smart_city*stress*.csv")
csv_path = csv_files[0] if csv_files else "data/smart_city_traffic_stress_data.csv"
print(f"Using: {csv_path}")

df = pd.read_csv(csv_path)
X = df.iloc[:, :-1].select_dtypes(include=['number']).values
y = df.iloc[:, -1].values
if y.dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

n_features = X_train_s.shape[1]
X_train_cnn = X_train_s.reshape(-1, n_features, 1)
X_test_cnn = X_test_s.reshape(-1, n_features, 1)

layers_range = range(1, 11)
activations_map = {'relu': 'relu', 'tanh': 'tanh', 'logistic': 'sigmoid'}
lrs = [0.1, 0.01, 0.001]

os.makedirs("plots/cnn_regression", exist_ok=True)
results = []

for lr in lrs:
    plt.figure(figsize=(10,6))
    for act_name, act_tf in activations_map.items():
        mse_list = []
        for L in layers_range:
            print(f"CNN Reg: LR={lr} {act_name} Layer={L}/10")
            model = Sequential()
            model.add(Conv1D(8, 2, activation=act_tf, input_shape=(n_features,1)))
            for _ in range(L-1):
                model.add(Conv1D(8, 2, activation=act_tf, padding='same'))
            model.add(Flatten())
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, verbose=0)
            pred = model.predict(X_test_cnn, verbose=0).flatten()
            mse = mean_squared_error(y_test, pred)
            mse_list.append(mse)
            results.append([L, act_name, lr, mse])
        plt.plot(list(layers_range), mse_list, marker='o', label=act_name)
    plt.title(f"CNN Regression - MSE vs Layers LR={lr}")
    plt.xlabel("Layers"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True)
    plt.savefig(f"plots/cnn_regression/mse_LR_{lr}.png")
    plt.close()

pd.DataFrame(results, columns=['layers','activation','lr','mse']).to_csv("output/cnn_regression_results.csv", index=False)
print("DONE CNN Regression")