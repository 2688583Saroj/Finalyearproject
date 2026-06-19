import json
import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASETS = {
    "india_crop_yield": {
        "csv": "data/crop_yield.csv",
        "target": "Yield",
        "drop_columns": [],
        "description": "Indian crop yield dataset with Crop, Season, State, rainfall, fertilizer and pesticide data."
    }
}


def add_engineered_features(df):
    df = df.copy()

    if {"Production", "Area"}.issubset(df.columns):
        safe_area = df["Area"].replace(0, pd.NA)
        df["Production_per_Area"] = df["Production"] / safe_area

    if {"Fertilizer", "Area"}.issubset(df.columns):
        safe_area = df["Area"].replace(0, pd.NA)
        df["Fertilizer_per_Area"] = df["Fertilizer"] / safe_area

    if {"Pesticide", "Area"}.issubset(df.columns):
        safe_area = df["Area"].replace(0, pd.NA)
        df["Pesticide_per_Area"] = df["Pesticide"] / safe_area

    return df


def train_one_dataset(name, config):
    print(f"\nTraining: {name}")

    df = pd.read_csv(config["csv"])
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    target = config["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {config['csv']}")

    drop_columns = [c for c in config.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop_columns)

    df = df.dropna(subset=[target])
    df = add_engineered_features(df)

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=None,
        n_jobs=1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "dataset": name,
        "rows": int(len(df)),
        "features": list(X.columns),
        "target": target,
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mse ** 0.5), 4),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    os.makedirs("model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    with open(f"model/{name}_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open(f"model/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save prediction comparison for report
    comparison = X_test.copy()
    comparison["Actual_Yield"] = y_test.values
    comparison["Predicted_Yield"] = predictions
    comparison.head(100).to_csv(f"outputs/{name}_prediction_sample.csv", index=False)

    print(json.dumps(metrics, indent=4))


def main():
    for name, config in DATASETS.items():
        if os.path.exists(config["csv"]):
            train_one_dataset(name, config)
        else:
            print(f"Skipping {name}: file not found at {config['csv']}")

    print("\nTraining complete. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
