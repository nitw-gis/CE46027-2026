import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/gender_classification_v7.csv")
X = df.drop("gender", axis=1)
y = LabelEncoder().fit_transform(df["gender"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

os.makedirs("output/Gender_ANN_SVM_DT_Comparison", exist_ok=True)

models = {
  "ANN_10layer": MLPClassifier(hidden_layer_sizes=(100,)*10, activation='relu', max_iter=400, random_state=42),
  "SVM_rbf": SVC(kernel='rbf'),
  "SVM_linear": SVC(kernel='linear'),
  "DT_depth10": DecisionTreeClassifier(max_depth=10, random_state=42)
}

results=[]
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    results.append([name, acc, f1])
    print(f"{name}: Acc={acc:.4f} F1={f1:.4f}")
    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(name); plt.savefig(f"output/Gender_ANN_SVM_DT_Comparison/cm_{name}.png"); plt.close()

pd.DataFrame(results, columns=['model','accuracy','f1']).to_csv("output/Gender_ANN_SVM_DT_Comparison/comparison_results.csv", index=False)