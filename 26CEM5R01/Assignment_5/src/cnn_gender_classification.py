import pandas as pd
import os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("data/gender_classification_v7.csv")
X = df.drop("gender", axis=1).values
y = LabelEncoder().fit_transform(df["gender"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_cnn = X_train_s.reshape(-1, 7, 1)
X_test_cnn = X_test_s.reshape(-1, 7, 1)

layers_range = range(1, 11)
activations_map = {'relu': 'relu', 'tanh': 'tanh', 'logistic': 'sigmoid'}
lrs = [0.1, 0.01, 0.001]

os.makedirs("plots/cnn_gender_classification", exist_ok=True)
os.makedirs("output", exist_ok=True)

all_results = []

for lr in lrs:
    plt.figure(figsize=(10,6))
    for act_name, act_tf in activations_map.items():
        acc_list = []
        for L in layers_range:
            print(f"CNN Gender: LR={lr} Act={act_name} Layer={L}")
            model = Sequential()
            model.add(Conv1D(16, 2, activation=act_tf, input_shape=(7,1)))
            for _ in range(L-1):
                model.add(Conv1D(16, 2, activation=act_tf, padding='same'))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_cnn, y_train, epochs=15, batch_size=32, verbose=0)
            
            y_pred = (model.predict(X_test_cnn, verbose=0) > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)
            all_results.append([L, act_name, lr, acc])
        plt.plot(list(layers_range), acc_list, marker='o', label=act_name)
    plt.title(f"CNN Gender - Accuracy vs Layers (1-10) LR={lr}")
    plt.xlabel("Layers (1-10)"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)
    plt.savefig(f"plots/cnn_gender_classification/acc_LR_{lr}.png")
    plt.close()

pd.DataFrame(all_results, columns=['layers','activation','lr','accuracy']).to_csv("output/cnn_gender_results.csv", index=False)
print("DONE CNN Gender - 3 graphs ban gaye plots/cnn_gender_classification me")