"""
Shared helpers for the custom network experiments.

preprocess() does the same image preparation as Main.img_preprocess in the app,
so anything tested here sees exactly what the app would see.
"""
import csv
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DIR = os.path.join(HERE, "test set")
OUTPUT_DIR = os.path.join(HERE, "output")


def preprocess(image_path, source):
    """
    Returns a black-ink-on-white binary image, like morph_img in the app.

    source is how the image reached the app:
      "camera"    - a photo taken in the app (thresholded, then cropped to the viewfinder)
      "upload"    - a photo picked with the upload button (thresholded, not cropped)
      "processed" - an image the app already processed, e.g. from 'home images' (used as is)
    """
    if source == "processed":
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 38)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 5))
    morph_img = cv2.morphologyEx(adaptive_result, cv2.MORPH_OPEN, kernel)
    if source == "camera":  # same viewfinder crop as the app
        h, w = morph_img.shape
        cropped = morph_img[min(224, h):min(705, h), min(8, w):min(648, w)]
        if cropped.size:
            morph_img = cropped
    return morph_img


def load_test_set():
    """Reads 'test set/labels.csv' and returns a list of (image path, equation, source)."""
    with open(os.path.join(TEST_SET_DIR, "labels.csv"), newline="") as f:
        return [(os.path.join(TEST_SET_DIR, row["filename"]), row["equation"].upper().replace(" ", ""), row["source"])
                for row in csv.DictReader(f)]
