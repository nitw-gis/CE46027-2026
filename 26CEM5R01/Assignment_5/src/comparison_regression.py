import pandas as pd, numpy as np, os, time, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("output/MLR_KNNR_ANNR_Comparison/plots", exist_ok=True)

print("Loading data...")
df = pd.read_csv("data/smart_city_traffic_stress_dataset.csv").sample(n=5000, random_state=42)
df_enc = pd.get_dummies(df, drop_first=True)
X = df_enc.drop(columns=["stress_index"]).values.astype('float32')
y = df_enc["stress_index"].values.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler(); X_train_s=scaler.fit_transform(X_train); X_test_s=scaler.transform(X_test)

results=[]
# 1. MLR
print("Training MLR...")
start=time.time(); mlr=LinearRegression(); mlr.fit(X_train_s,y_train); pred=mlr.predict(X_test_s); t=time.time()-start
results.append(["MLR", np.sqrt(mean_squared_error(y_test,pred)), mean_absolute_error(y_test,pred), r2_score(y_test,pred), t])

# 2. KNNR
print("Training KNNR...")
start=time.time(); knn=KNeighborsRegressor(n_neighbors=5); knn.fit(X_train_s,y_train); pred=knn.predict(X_test_s); t=time.time()-start
results.append(["KNNR", np.sqrt(mean_squared_error(y_test,pred)), mean_absolute_error(y_test,pred), r2_score(y_test,pred), t])

# 3. ANNR (Best ANN from before - 2 layer, relu, lr 0.001)
print("Training ANNR...")
in_dim=X.shape[1]
class ANN(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x): return self.net(x)
m=ANN(); opt=torch.optim.Adam(m.parameters(),lr=0.001); crit=nn.MSELoss()
ds=TensorDataset(torch.from_numpy(X_train_s),torch.from_numpy(y_train).reshape(-1,1)); loader=DataLoader(ds,batch_size=64,shuffle=True)
start=time.time()
for _ in range(10):
    for xb,yb in loader: opt.zero_grad(); loss=crit(m(xb),yb); loss.backward(); opt.step()
t=time.time()-start
with torch.no_grad(): pred=m(torch.from_numpy(X_test_s)).numpy().flatten()
results.append(["ANNR", np.sqrt(mean_squared_error(y_test,pred)), mean_absolute_error(y_test,pred), r2_score(y_test,pred), t])

df_res=pd.DataFrame(results, columns=["Model","RMSE","MAE","R2","Time"])
print(df_res)
df_res.to_csv("output/MLR_KNNR_ANNR_Comparison/three_model_comparison.csv", index=False)
df_res.sort_values("R2",ascending=False).to_csv("output/MLR_KNNR_ANNR_Comparison/three_model_performance_ranking.csv", index=False)

# Plots
for metric in ["RMSE","MAE","R2"]:
    plt.figure(); plt.bar(df_res["Model"], df_res[metric]); plt.title(f"{metric} Comparison"); plt.ylabel(metric); plt.grid(True,axis='y')
    plt.savefig(f"output/MLR_KNNR_ANNR_Comparison/plots/0{['RMSE','MAE','R2'].index(metric)+1}_{metric.lower()}_comparison.png"); plt.close()

print("DONE Regression Comparison!")