#!/usr/bin/env python3
"""
Data preparation script: extracts base64 images, splits into train/test/OOS directories.
"""
import json
import logging
import os
import shutil
import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

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
    with config_path.open('r') as f:
        cfg = json.load(f)
        cfg=cfg["data_prep"]
    return cfg


def extract_images(idimage_csv: Path, output_dir: Path) -> None:
    """
    Decode base64 images from CSV and save to output directory.
    """
    df = pd.read_csv(idimage_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        filename = row['name']
        img_data = row['imageData']
        img_bytes = base64.b64decode(img_data)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')

        dest = output_dir / filename
        image.save(dest, 'JPEG')

        if idx % 10 == 0:
            logger.info(f"Saved {idx+1} images to {output_dir}")

    logger.info("Image extraction complete.")


def find_image(base_name: str, src_dir: Path, extensions: list) -> Path:
    """
    Locate image file with any of the given extensions.
    """
    for ext in extensions:
        candidate = src_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def create_splits(idmeta_csv: Path,
                  idlabel_csv: Path,
                  extracted_dir: Path,
                  dataset_dir: Path,
                  oos_ratio: float,
                  test_ratio: float,
                  random_state: int) -> None:
    """
    Build binary labels, stratify-split, and copy images into class folders.
    """
    idmeta = pd.read_csv(idmeta_csv)
    idlabel = pd.read_csv(idlabel_csv)

    # Build 0/1 label map
    labels = {fname: 0 for fname in idmeta['id']}
    fraud_patterns = {'Fraud5_inpaint_and_rewrite', 'Fraud6_crop_and_replace'}
    for _, row in idlabel.iterrows():
        if row['fraudpattern'] in fraud_patterns:
            labels[row['id']] = 1

    df_labels = pd.DataFrame(
        [{'filename': k, 'label': v} for k, v in labels.items()]
    )

    # Stratified splits
    train_val, oos = train_test_split(
        df_labels, test_size=oos_ratio,
        stratify=df_labels['label'], random_state=random_state
    )
    train, test = train_test_split(
        train_val, test_size=test_ratio,
        stratify=train_val['label'], random_state=random_state
    )

    # Report distributions
    for name, subset in [('Total', df_labels), ('Train', train), ('Test', test), ('OOS', oos)]:
        counts = subset['label'].value_counts().sort_index()
        logger.info(f"{name}: 0={counts.get(0,0)} 1={counts.get(1,0)}")

    # Prepare directories
    for split in ['train', 'test', 'out_of_sample_test']:
        for cls in ['0', '1']:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Copy files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp']
    for subset, split in [(train, 'train'), (test, 'test'), (oos, 'out_of_sample_test')]:
        for _, row in subset.iterrows():
            src = find_image(row['filename'], extracted_dir, extensions)
            if not src:
                continue
            dest = dataset_dir / split / str(row['label']) / src.name
            shutil.copy(src, dest)
    logger.info("Data split and copy complete.")


def main():
    base = Path(__file__).parent
    cfg = load_config(base / 'config.json')

    # Paths & params from config
    idimage_csv = Path(cfg['IDIMAGE_CSV'])
    idmeta_csv  = Path(cfg['IDMETA_CSV'])
    idlabel_csv = Path(cfg['IDLABEL_CSV'])
    extracted_dir = Path(cfg['EXTRACTED_DIR'])
    dataset_dir   = Path(cfg['DATASET_DIR'])
    oos_ratio     = cfg.get('OOS_RATIO', 0.10)
    test_ratio    = cfg.get('TEST_RATIO', 0.067)
    random_state  = cfg.get('RANDOM_STATE', 42)

    extract_images(idimage_csv, extracted_dir)
    create_splits(
        idmeta_csv, idlabel_csv,
        extracted_dir, dataset_dir,
        oos_ratio, test_ratio,
        random_state
    )


if __name__ == '__main__':
    main()
    logger.info("Dataset creation completed successfully.")