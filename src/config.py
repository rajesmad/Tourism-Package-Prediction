import os

HF_USERNAME = os.getenv("HF_USERNAME", "rajesmad")

HF_DATASET_REPO = f"{HF_USERNAME}/tourism-package-dataset"
HF_MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"
HF_SPACE_REPO = f"{HF_USERNAME}/tourism-package-space"