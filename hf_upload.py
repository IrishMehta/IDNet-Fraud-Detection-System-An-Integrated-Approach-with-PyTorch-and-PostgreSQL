import os
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from transformers import AutoImageProcessor, PretrainedConfig
from huggingface_hub import login, HfApi
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Load JSON configuration from file.
    """
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        return json.load(f)


def build_and_save_model(config: dict) -> Path:
    """
    Build the ResNet50 model, load weights, and save HuggingFace folder.
    Returns the folder path containing saved model files.
    """
    hf_folder = Path(config["HF_FOLDER_PATH"])
    hf_folder.mkdir(parents=True, exist_ok=True)

    # Instantiate model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load weights
    weights_path = Path(config.get("MODEL_WEIGHTS_PATH", "fraud_detection_model.pth"))
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state = torch.load(
        weights_path,
        map_location=torch.device("cpu"),
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # Save model weights for HF
    torch.save(model.state_dict(), hf_folder / "pytorch_model.bin")
    logger.info(f"Saved pytorch_model.bin in {hf_folder}")

    # Save transformers config
    hf_config = PretrainedConfig(
        model_type="resnet",
        num_labels=2,
        id2label={"0": "Non-Fraud", "1": "Fraud"},
        label2id={"Non-Fraud": 0, "Fraud": 1},
        hidden_size=2048,
        problem_type="single_label_classification",
        pipeline_tag="image-classification",
        library_name="transformers",
        auto_map={
            "AutoModelForImageClassification":
            "transformers.models.resnet.modeling_resnet.ResNetForImageClassification"
        }
    )
    hf_config.save_pretrained(hf_folder)
    logger.info(f"Saved HF config.json in {hf_folder}")

    # Save the image processor
    image_processor = AutoImageProcessor.from_pretrained(
        "microsoft/resnet-50",
        size=224,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    image_processor.save_pretrained(hf_folder)
    logger.info(f"Saved AutoImageProcessor in {hf_folder}")

    # Write README.md
    readme = hf_folder / "README.md"
    readme.write_text(
        """---
language: en
license: mit
library_name: transformers
pipeline_tag: image-classification
tags:
  - image-classification
  - computer-vision
  - fraud-detection
---

# Fraud Detection Image Classification Model

This model classifies images into two classes:
- `Non-Fraud` (label 0)
- `Fraud` (label 1)

## Usage

```python
from transformers import pipeline
classifier = pipeline(
    "image-classification",
    model="{repo_id}"
)
result = classifier("path/to/image.jpg")
print(result)
```""".replace("{repo_id}", config.get("HF_REPO_NAME"))
    )
    logger.info(f"Saved README.md in {hf_folder}")

    return hf_folder


def upload_to_hub(hf_folder: Path, config: dict):
    """
    Login to HF and upload the folder as a repo.
    """
    hf_token = os.getenv("HF_WRITE_TOKEN") or config.get("HF_WRITE_TOKEN")
    if not hf_token:
        logger.error("HF_WRITE_TOKEN not set in env or config.")
        raise EnvironmentError("HF_WRITE_TOKEN not set.")

    login(token=hf_token)
    api = HfApi()
    repo_id = os.getenv("HF_REPO_NAME") or config.get("HF_REPO_NAME")
    logger.info(f"Creating or updating repo: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(hf_folder),
        repo_id=repo_id,
        commit_message="Upload two-class fraud detection model"
    )
    logger.info("Upload to Hugging Face complete.")


def predict(image_path: str, config: dict) -> dict:
    """
    Call the HF Inference API for a single image.
    """
    hf_token = os.getenv("HF_READ_TOKEN") or config.get("HF_READ_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_READ_TOKEN not set in env or config.")

    repo_id = os.getenv("HF_REPO_NAME") or config.get("HF_REPO_NAME")
    api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {hf_token}",
               "Content-Type": "application/octet-stream",
               "x-wait-for-model": "true"
               }
    with open(image_path, "rb") as f:
        data = f.read()

    response = requests.post(api_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Inference error {response.status_code}: {response.text}")
        raise RuntimeError(f"Inference failed: {response.status_code}")


def main():
    base_path = Path(__file__).parent
    config = load_config(base_path / "config.json")

    hf_folder = build_and_save_model(config)
    upload_to_hub(hf_folder, config)

    # Optional: test inference
    sample_img = config.get("INFERENCE_IMAGE_PATH")
    if sample_img:
        result = predict(sample_img, config)
        logger.info(f"Sample prediction: {result}")


if __name__ == "__main__":
    main()
    logger.info("File uploaded to HF successfully.")