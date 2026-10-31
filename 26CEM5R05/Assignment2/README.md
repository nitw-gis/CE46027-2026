# CE46027 - Assignment 2 - Gold XAUUSD Price Prediction
## MLR vs KNNR Comparison (Fixed - No Leakage Version)

**Student ID: 26CEM5R05**

### Problem Statement
Predict Gold (XAUUSD) Close Price using regression models.

### Initial Issue: R2 = 1.0
Initially we got R2 = 1.0 for both models due to **Target Leakage**:
- Used same-day `High`, `Low`, `Open` to predict same-day `Close`
- By definition, `Low <= Close <= High`, so model learns this inequality and gets perfect score

### Fix Applied
**1. Removed Leakage Columns:**
- Dropped same-day OHLC
- Created lag features: `Close_lag1`, `Close_lag2`, `Open_lag1`, `High_lag1`
- Created `MA5`, `MA10` from previous closes only

**2. Time Series Split:**
- Chronological 80-20 split instead of random shuffle

**3. Regularization:**
- Used Ridge Regression (alpha=1.0)

### Results (After Fix)
| Model | R2 Score | Test Time |
|-------|----------|-----------|
| MLR (Ridge, No Leakage) | **0.938688** | 0.000329s |
| KNNR (K=3) | **0.802281** | 0.002131s |

**Conclusion:** MLR outperforms KNNR for Gold. R2=0.93 is realistic, not 1.0.

