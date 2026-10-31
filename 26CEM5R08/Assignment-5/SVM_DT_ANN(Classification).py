import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load data & create binary target (MPG >= 35)
df = pd.read_csv('Cars.csv')
df['Target'] = (df['MPG'] >= 35).astype(int)

X = df[['HP', 'VOL', 'SP', 'WT']]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 2. Tune ANN across activations and learning rates
activations = ['relu', 'tanh', 'elu']
learning_rates = [0.1, 0.01, 0.001]
results_grid = []

for act in activations:
    for lr in learning_rates:
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation=act),
            tf.keras.layers.Dense(16, activation=act),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                            epochs=50, batch_size=16, verbose=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        
        val_preds = (model.predict(X_val, verbose=0).flatten() >= 0.5).astype(int)
        results_grid.append({'Activation': act, 'LR': lr, 'Acc': accuracy_score(y_val, val_preds), 'Epochs': len(history.history['loss'])})

best = pd.DataFrame(results_grid).sort_values(by='Acc', ascending=False).iloc[0]

# Train Best ANN on full training set
tf.keras.backend.clear_session()
best_ann = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation=best['Activation']),
    tf.keras.layers.Dense(16, activation=best['Activation']),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
best_ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best['LR']), loss='binary_crossentropy')

t0 = time.perf_counter()
best_ann.fit(X_train, y_train, epochs=int(best['Epochs']), batch_size=16, verbose=0)
ann_preds = (best_ann.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
ann_time = (time.perf_counter() - t0) * 1000

# 3. Define all models to compare (SVMs & Decision Trees variants + Tuned ANN)
models = {
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (Polynomial)': SVC(kernel='poly', degree=3, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'DT (Shallow, d=2)': DecisionTreeClassifier(max_depth=2, random_state=42),
    'DT (Deep, d=8)': DecisionTreeClassifier(max_depth=8, random_state=42),
    'DT (min_split=5)': DecisionTreeClassifier(min_samples_split=5, random_state=42)
}

results = []

# Add Tuned ANN results
cm_ann = confusion_matrix(y_test, ann_preds)
tn, fp, fn, tp = cm_ann.ravel()
results.append({
    'Model': 'Tuned ANN',
    'Acc': f"{accuracy_score(y_test, ann_preds):.2f}",
    'Prec': f"{precision_score(y_test, ann_preds, zero_division=0):.2f}",
    'Rec': f"{recall_score(y_test, ann_preds, zero_division=0):.2f}",
    'F1': f"{f1_score(y_test, ann_preds, zero_division=0):.2f}",
    'Time': f"{ann_time:.2f}",
    'CM': f"[{tn}, {fp}, {fn}, {tp}]"
})

for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    t1 = time.perf_counter()
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        'Model': name,
        'Acc': f"{accuracy_score(y_test, preds):.2f}",
        'Prec': f"{precision_score(y_test, preds, zero_division=0):.2f}",
        'Rec': f"{recall_score(y_test, preds, zero_division=0):.2f}",
        'F1': f"{f1_score(y_test, preds, zero_division=0):.2f}",
        'Time': f"{(t1-t0)*1000:.2f}",
        'CM': f"[{tn}, {fp}, {fn}, {tp}]"
    })

# 4. Print Clean Formatted Table
print(f"\n{'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Time(ms)':<8} | {'CM [TN,FP,FN,TP]':<16}")
print("-" * 80)
for r in results:
    print(f"{r['Model']:<20} | {r['Acc']:<6} | {r['Prec']:<6} | {r['Rec']:<6} | {r['F1']:<6} | {r['Time']:<8} | {r['CM']:<16}")