import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import upload_file
from config import HF_DATASET_REPO

os.makedirs("data", exist_ok=True)

raw_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/tourism.csv"

df = pd.read_csv(raw_url)

df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

upload_file(
    path_or_fileobj="data/train.csv",
    path_in_repo="train.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)

upload_file(
    path_or_fileobj="data/test.csv",
    path_in_repo="test.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)

print("Data preparation completed and train/test uploaded to Hugging Face.")