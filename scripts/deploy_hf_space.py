"""
Deploy the API to a Hugging Face Docker Space. Called by .github/workflows/ci.yml on push to main.

    HF_TOKEN=... HF_SPACE=<user>/<space-name> python scripts/deploy_hf_space.py

The Space gets only what the API image needs: the Dockerfile, the API code, four small
modules from src/models, the model file and its metadata. No data is uploaded.
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

FILES = ["api/__init__.py", "api/main.py", "api/requirements.txt",
         "src/models/__init__.py", "src/models/config.py", "src/models/data.py",
         "src/models/labels.py", "src/models/explanation_store.py",
         "models/fundifix_model.joblib", "models/model_meta.json"]


def main():
    token, space = os.environ["HF_TOKEN"], os.environ["HF_SPACE"]
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in FILES:
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / f, tmp / f)
        shutil.copy(root / "docker/Dockerfile.api", tmp / "Dockerfile")
        shutil.copy(root / "deploy/hf_space/README.md", tmp / "README.md")
        api = HfApi(token=token)
        api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        api.upload_folder(folder_path=str(tmp), repo_id=space, repo_type="space",
                          commit_message=f"Deploy API from GitHub commit {sha}")
    user, name = space.split("/")
    print(f"Deployed. API docs: https://{user}-{name}.hf.space/docs".lower())


if __name__ == "__main__":
    main()
