
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "output"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL PERFORMANCE RESULTS
# ============================================================

# Results obtained from Assignment 2 (MLR and KNNR)
# and Assignment 5 (ANN Regression).

results = {
    "Model": [
        "MLR",
        "KNNR (K=5)",
        "ANN"
    ],

    "RMSE": [
        2.020552,
        2.925932,
        2.0463
    ],

    "MAE": [
        1.611121,
        2.360400,
        1.6364
    ],

    "R2": [
        0.988983,
        0.976898,
        0.9887
    ]
}

df = pd.DataFrame(results)


# ============================================================
# 3. SAVE CONSOLIDATED RESULTS
# ============================================================

comparison_file = OUTPUT_DIR / "three_model_comparison.csv"

df.to_csv(
    comparison_file,
    index=False
)

print("\n")
print("=" * 60)
print("THREE-MODEL REGRESSION PERFORMANCE COMPARISON")
print("=" * 60)

print(df.to_string(index=False))

print("\nComparison results saved successfully.")
print("File:", comparison_file)


# ============================================================
# 4. FUNCTION TO GENERATE COMPARISON PLOTS
# ============================================================

def create_comparison_plot(
    metric,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        df["Model"],
        df[metric]
    )

    plt.title(
        title,
        fontsize=14
    )

    plt.xlabel(
        "Regression Model",
        fontsize=11
    )

    plt.ylabel(
        ylabel,
        fontsize=11
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.bar_label(
        bars,
        fmt="%.4f",
        padding=4
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Generated:", filename)


# ============================================================
# 5. RMSE COMPARISON
# ============================================================

create_comparison_plot(
    metric="RMSE",
    title="RMSE Comparison of Regression Models",
    ylabel="Root Mean Square Error (RMSE)",
    filename="01_rmse_comparison.png"
)


# ============================================================
# 6. MAE COMPARISON
# ============================================================

create_comparison_plot(
    metric="MAE",
    title="MAE Comparison of Regression Models",
    ylabel="Mean Absolute Error (MAE)",
    filename="02_mae_comparison.png"
)


# ============================================================
# 7. R2 COMPARISON
# ============================================================

create_comparison_plot(
    metric="R2",
    title="R² Comparison of Regression Models",
    ylabel="Coefficient of Determination (R²)",
    filename="03_r2_comparison.png"
)


# ============================================================
# 8. IDENTIFY BEST MODEL FOR EACH METRIC
# ============================================================

print("\n")
print("=" * 60)
print("BEST MODEL BY EVALUATION METRIC")
print("=" * 60)

# Lowest RMSE
best_rmse = df.loc[
    df["RMSE"].idxmin()
]

print("\nBest model based on RMSE:")
print("Model:", best_rmse["Model"])
print("RMSE:", best_rmse["RMSE"])


# Lowest MAE
best_mae = df.loc[
    df["MAE"].idxmin()
]

print("\nBest model based on MAE:")
print("Model:", best_mae["Model"])
print("MAE:", best_mae["MAE"])


# Highest R2
best_r2 = df.loc[
    df["R2"].idxmax()
]

print("\nBest model based on R2:")
print("Model:", best_r2["Model"])
print("R2:", best_r2["R2"])


# ============================================================
# 9. OVERALL PERFORMANCE RANKING
# ============================================================

# Rank models separately for each metric.
# Lower RMSE and MAE are better.
# Higher R2 is better.

df["RMSE_Rank"] = df["RMSE"].rank(
    ascending=True,
    method="min"
)

df["MAE_Rank"] = df["MAE"].rank(
    ascending=True,
    method="min"
)

df["R2_Rank"] = df["R2"].rank(
    ascending=False,
    method="min"
)

df["Average_Rank"] = df[
    [
        "RMSE_Rank",
        "MAE_Rank",
        "R2_Rank"
    ]
].mean(axis=1)

df["Overall_Rank"] = df["Average_Rank"].rank(
    ascending=True,
    method="min"
).astype(int)

ranking_df = df.sort_values(
    by="Overall_Rank"
).reset_index(drop=True)

ranking_df.to_csv(
    OUTPUT_DIR / "three_model_performance_ranking.csv",
    index=False
)

print("\n")
print("=" * 60)
print("OVERALL MODEL PERFORMANCE RANKING")
print("=" * 60)

print(
    ranking_df[
        [
            "Model",
            "RMSE_Rank",
            "MAE_Rank",
            "R2_Rank",
            "Average_Rank",
            "Overall_Rank"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. FINAL OUTPUT SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("REGRESSION MODEL COMPARISON COMPLETED")
print("=" * 60)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nPlot folder:")
print(PLOT_DIR)

print("\nGenerated CSV files:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print("-", file.name)

print("\nGenerated plot files:")

for file in sorted(PLOT_DIR.glob("*.png")):
    print("-", file.name)

print("\nTotal CSV files:", len(list(OUTPUT_DIR.glob("*.csv"))))
print("Total plots:", len(list(PLOT_DIR.glob("*.png"))))

print("\nAll tasks completed successfully.")
