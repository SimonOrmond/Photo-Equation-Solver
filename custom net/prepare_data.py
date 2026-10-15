"""
Step 3a: turns the dataset into 28x28 crops and splits it 70/30 into train and test.

    python prepare_data.py

Every image goes through the same segment/normalise steps as the real photos, so the
network is trained on crops that look like the ones it will see in the app.
Saves Data/prepared.npz (used by train.py) and output/prepared_samples.png to check by eye.
"""
import glob
import os

import cv2
import numpy as np

from model import CLASSES
from preprocess import HERE, OUTPUT_DIR
from segment import normalise, segment

DATASET_DIR = os.path.join(HERE, "Data", "dataset")
PREPARED_FILE = os.path.join(HERE, "Data", "prepared.npz")
TEST_FRACTION = 0.3
SEED = 42  # fixed so the split is the same every time

# dataset folder -> class name. Folders not listed (dec, div, z) aren't used by the app, so they're skipped.
FOLDER_TO_CLASS = {str(d): str(d) for d in range(10)}
FOLDER_TO_CLASS.update({"x": "x", "mul": "x",  # the app's x is often written like a multiplication cross
                        "y": "y", "add": "+", "sub": "-", "eq": "="})


def remove_scan_lines(binary):
    """
    Some dataset images have a thin line across the whole image, at or just inside the top or bottom edge
    (left over from scanning).
    Paints those lines white so they aren't mistaken for part of the symbol.
    """
    height, width = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats((binary < 128).astype(np.uint8), connectivity=8)
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        if w >= 0.9 * width and h <= 0.05 * height and (y <= 0.05 * height or y + h >= 0.95 * height):
            binary[labels == i] = 255


def load_symbol(path):
    """Reads one dataset image and returns its 28x28 crop, or None if no symbol was found."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    remove_scan_lines(binary)
    symbols = segment(binary, ignore_edges=False)
    if not symbols:
        return None
    biggest = max(symbols, key=lambda s: int((s.mask > 0).sum()))  # drops stray marks like underlines
    return normalise(biggest)


def main():
    images, labels, skipped = [], [], 0
    for folder, name in FOLDER_TO_CLASS.items():
        for path in sorted(glob.glob(os.path.join(DATASET_DIR, folder, "*"))):
            crop = load_symbol(path)
            if crop is None:
                skipped += 1
                continue
            images.append(crop)
            labels.append(CLASSES.index(name))
    images, labels = np.array(images, np.uint8), np.array(labels, np.int64)

    # stratified split: 30% of every class goes to test, so both halves have the same class mix
    rng = np.random.default_rng(SEED)
    test_mask = np.zeros(len(labels), bool)
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.where(labels == c)[0])
        test_mask[idx[:round(len(idx) * TEST_FRACTION)]] = True

    np.savez_compressed(PREPARED_FILE,
                        train_x=images[~test_mask], train_y=labels[~test_mask],
                        test_x=images[test_mask], test_y=labels[test_mask])

    print(f"{len(labels)} crops ({skipped} images skipped, no symbol found)")
    print(f"train: {(~test_mask).sum()}   test: {test_mask.sum()}")
    for c, name in enumerate(CLASSES):
        print(f"  {name:>2}: {(labels == c).sum()}")

    # 12 random crops of each class, one row per class
    rows = []
    for c in range(len(CLASSES)):
        idx = rng.choice(np.where(labels == c)[0], 12, replace=False)
        rows.append(np.hstack(images[idx]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sheet = cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "prepared_samples.png"), sheet)


if __name__ == "__main__":
    main()
