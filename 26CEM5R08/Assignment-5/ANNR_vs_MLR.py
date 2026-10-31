
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load dataset
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()

y = df["Crop_Yield_ton_per_hectare"]
X = pd.get_dummies(df.drop(columns=[y.name]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. MLR
mlr = LinearRegression().fit(X_train, y_train)
mlr_pred = mlr.predict(X_test)

# 3. KNNR
knnr = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
knnr_pred = knnr.predict(X_test)

# 4. ANN training and 18 experiments
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

activations = ["relu", "tanh", "elu"]
learning_rates = [0.1, 0.01, 0.001]
architectures = [(128, 64), (64, 32)]

results = []

for act in activations:
    for lr in learning_rates:
        for layers in architectures:

            tf.keras.backend.clear_session()

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(layers[0], activation=act),
                tf.keras.layers.Dense(layers[1], activation=act),
                tf.keras.layers.Dense(1)
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="mse"
            )

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10,
                    restore_best_weights=True
                )],
                verbose=0
            )

            pred = model.predict(X_val, verbose=0).flatten()
            rmse = np.sqrt(mean_squared_error(y_val, pred))

            results.append({
                "Activation": act,
                "Learning_Rate": lr,
                "Architecture": layers,
                "RMSE": rmse,
                "Epochs": len(history.history["loss"])
            })

# 5. Find best ANN
results = pd.DataFrame(results)
results.to_csv("ANN_Experiments.csv", index=False)

best = results.loc[results["RMSE"].idxmin()]

tf.keras.backend.clear_session()

ann = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(best["Architecture"][0], activation=best["Activation"]),
    tf.keras.layers.Dense(best["Architecture"][1], activation=best["Activation"]),
    tf.keras.layers.Dense(1)
])

ann.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=best["Learning_Rate"]
    ),
    loss="mse"
)

ann.fit(X_train, y_train, epochs=int(best["Epochs"]),
        batch_size=32, verbose=0)

ann_pred = ann.predict(X_test, verbose=0).flatten()

# 6. Plot 1: Activation function comparison
activation_rmse = results.groupby("Activation")["RMSE"].mean()

plt.figure(figsize=(7, 5))
activation_rmse.plot(kind="bar")
plt.title("Effect of Activation Functions on ANN")
plt.xlabel("Activation Function")
plt.ylabel("Average Validation RMSE")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 7. Plot 2: MLR, KNNR and ANN comparison
models = {
    "MLR": mlr_pred,
    "KNNR": knnr_pred,
    "ANN": ann_pred
}

comparison = pd.DataFrame({
    name: {
        "R2": r2_score(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred)
    }
    for name, pred in models.items()
}).T

print("\nMODEL COMPARISON")
print(comparison.round(4))
print("\nBEST ANN CONFIGURATION")
print(best)

comparison.to_csv("Model_Comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

comparison["R2"].plot(kind="bar", ax=axes[0], title="R2 Comparison")
comparison["RMSE"].plot(kind="bar", ax=axes[1], title="RMSE Comparison")

for ax in axes:
    ax.set_xlabel("Models")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y")

plt.tight_layout()
plt.show()
