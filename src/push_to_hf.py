from huggingface_hub import upload_folder
from config import HF_SPACE_REPO

upload_folder(
    folder_path="deployment",
    repo_id=HF_SPACE_REPO,
    repo_type="space"
)

print("Deployment files pushed to Hugging Face Space.")