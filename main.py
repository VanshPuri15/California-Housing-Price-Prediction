
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_FILE = "housing.csv"
MODEL_FILE = "model.pkl"

TARGET = "median_house_value"

print("\n===================================")
print(" CALIFORNIA HOUSING PRICE PREDICTION")
print("===================================\n")


# ==========================================
# 2. LOAD DATASET
# ==========================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. "
        "Place housing.csv in the project folder."
    )

housing = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Dataset shape:", housing.shape)

print("\nFirst five records:")
print(housing.head())


# ==========================================
# 3. STRATIFIED TRAIN-TEST SPLIT
# ==========================================

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(
    housing, housing["income_cat"]
):
    strat_train_set = housing.loc[train_index].copy()
    strat_test_set = housing.loc[test_index].copy()

# Remove temporary income category
for dataset in (strat_train_set, strat_test_set):
    dataset.drop("income_cat", axis=1, inplace=True)


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X_train = strat_train_set.drop(TARGET, axis=1)
y_train = strat_train_set[TARGET].copy()

X_test = strat_test_set.drop(TARGET, axis=1)
y_test = strat_test_set[TARGET].copy()

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================

num_features = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

cat_features = X_train.select_dtypes(
    exclude=[np.number]
).columns.tolist()


# ==========================================
# 6. DATA PREPROCESSING
# ==========================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])


# ==========================================
# 7. CREATE RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


# ==========================================
# 8. TRAIN THE MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\nPredictions generated successfully!")


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

rmse = np.sqrt(mean_squared_error(y_test, predictions))

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\n===================================")
print("         MODEL EVALUATION")
print("===================================")

print(f"RMSE: ${rmse:,.2f}")
print(f"MAE:  ${mae:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"R2 Percentage: {r2 * 100:.2f}%")

print("===================================")


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(model, MODEL_FILE)

print(f"\nTrained model saved as {MODEL_FILE}")


# ==========================================
# 12. SAVE PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Predicted Price": predictions
})

predictions_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")


# ==========================================
# 13. ACTUAL VS PREDICTED COMPARISON
# ==========================================

results = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})

results["Difference"] = (
    results["Actual Price"] - results["Predicted Price"]
)

results.to_csv("model_results.csv", index=False)

print("Comparison saved to model_results.csv")

print("\nSample Actual vs Predicted Prices:")
print(results.head(10))


# ==========================================
# 14. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")

plt.title("Actual vs Predicted House Prices")

min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--"
)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()


# ==========================================
# 15. PREDICTION DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.hist(
    predictions,
    bins=50,
    edgecolor="black"
)

plt.xlabel("Predicted House Price")
plt.ylabel("Frequency")

plt.title("Distribution of Predicted House Prices")

plt.tight_layout()
plt.savefig("prediction_distribution.png", dpi=300)

plt.show()


# ==========================================
# 16. FINAL SUMMARY
# ==========================================

print("\n===================================")
print("          PROJECT COMPLETED")
print("===================================")

print("Model: Random Forest Regressor")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

print("\nGenerated Files:")
print("1. model.pkl")
print("2. predictions.csv")
print("3. model_results.csv")
print("4. actual_vs_predicted.png")
print("5. prediction_distribution.png")

print("\nExecution completed successfully!")
