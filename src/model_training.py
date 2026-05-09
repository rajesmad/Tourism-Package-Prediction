import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import upload_file

from config import HF_DATASET_REPO, HF_MODEL_REPO

os.makedirs("models", exist_ok=True)

train_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/train.csv"
test_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob)
}

joblib.dump(best_model, "models/tourism_model.pkl")

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open("models/best_params.json", "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)

upload_file(
    path_or_fileobj="models/tourism_model.pkl",
    path_in_repo="tourism_model.pkl",
    repo_id=HF_MODEL_REPO,
    repo_type="model"
)

upload_file(
    path_or_fileobj="models/model_metrics.json",
    path_in_repo="model_metrics.json",
    repo_id=HF_MODEL_REPO,
    repo_type="model"
)

upload_file(
    path_or_fileobj="models/best_params.json",
    path_in_repo="best_params.json",
    repo_id=HF_MODEL_REPO,
    repo_type="model"
)

print("Model training completed.")
print("Best Parameters:", grid_search.best_params_)
print("Metrics:", metrics)