#!/usr/bin/env python3
"""
Quick local inference script for the hand autoencoder.

Loads the saved model, runs a single image through it, prints the reconstruction
error, and optionally saves the reconstructed image for visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import tensorflow as tf


IMG_SIZE = (96, 96)  # model was trained on 96x96 RGB images
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_and_preprocess(image_path: Path) -> np.ndarray:
    """Load an image file and convert it to a normalized array."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr


def iter_images(directory: Path) -> Iterable[Path]:
    """Yield image files under a directory (recursively)."""
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            yield p


def run_inference(
    model: tf.keras.Model,
    image_path: Path,
    save_path: Path | None,
    threshold: float | None,
) -> float:
    """Run the autoencoder on one image and optionally apply a threshold check."""

    x = load_and_preprocess(image_path)
    x_batch = np.expand_dims(x, axis=0)

    reconstruction = model.predict(x_batch, verbose=0)[0]
    mae = float(np.mean(np.abs(reconstruction - x)))

    print(f"Reconstruction MAE on {image_path}: {mae:.6f}")
    if threshold is not None:
        verdict = "OK  (below threshold)" if mae <= threshold else "FLAG (above threshold)"
        print(f"Threshold {threshold:.6f} => {verdict}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        recon_img = np.clip(reconstruction * 255, 0, 255).astype("uint8")
        Image.fromarray(recon_img).save(save_path)
        print(f"Saved reconstructed image to {save_path}")

    return mae


def calibrate_threshold(
    model: tf.keras.Model, image_dir: Path, percentile: float
) -> float:
    """
    Compute a threshold from a directory of normal images.

    Uses the given percentile of reconstruction error; e.g., 95 -> flag the top 5% errors.
    """
    files = list(iter_images(image_dir))
    if not files:
        raise ValueError(f"No images found under {image_dir}")

    errors = []
    for img_path in files:
        x = load_and_preprocess(img_path)
        x_batch = np.expand_dims(x, axis=0)
        reconstruction = model.predict(x_batch, verbose=0)[0]
        mae = float(np.mean(np.abs(reconstruction - x)))
        errors.append(mae)

    threshold = float(np.percentile(errors, percentile))
    print(
        f"Calibrated threshold at p{percentile} over {len(errors)} images: {threshold:.6f}"
    )
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single image through the hand autoencoder model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/hand_hunter.keras"),
        help="Path to the saved Keras model",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("tests/test_imgs/closeup-of-black-female-hand-isolated-on-white-background.jpg"),
        help="Path to the input image to test",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("tests/test_imgs/reconstruction.png"),
        help="Optional path to save the reconstructed image",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a reconstructed image to disk",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Apply a manual MAE threshold to flag anomalies",
    )
    parser.add_argument(
        "--calibrate-dir",
        type=Path,
        help="Directory of normal images; will derive a threshold from reconstruction errors",
    )
    parser.add_argument(
        "--calibrate-percentile",
        type=float,
        default=95.0,
        help="Percentile used for threshold when calibrating (e.g., 95 -> top 5%% flagged)",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Automatically calibrate using --calibrate-dir if given, otherwise the image's parent directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = None if args.no_save else args.save

    model = tf.keras.models.load_model(args.model)

    threshold = args.threshold
    if args.calibrate_dir or args.auto_calibrate:
        calibrate_dir = args.calibrate_dir
        if calibrate_dir is None and args.auto_calibrate:
            calibrate_dir = args.image.parent
        threshold = calibrate_threshold(model, calibrate_dir, args.calibrate_percentile)

    run_inference(model, args.image, save_path, threshold)


if __name__ == "__main__":
    main()
