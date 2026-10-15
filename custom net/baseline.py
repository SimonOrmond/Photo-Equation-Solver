"""
Step 1: measures how well EasyOCR reads the test set, as the app currently uses it.

    python baseline.py

Runs two versions on every image in 'test set/labels.csv':
  plain     - EasyOCR exactly as Main.read_text_from_image uses it
  allowlist - the same, but EasyOCR may only answer with equation characters
Results are printed and saved to output/baseline_results.csv.
"""
import csv
import os
import re

import easyocr

from preprocess import OUTPUT_DIR, load_test_set, preprocess

ALLOWLIST = "0123456789xXyY+-="


def app_text(result):
    """Turns EasyOCR's output into the string the app would pass to find_coefficients."""
    text = "".join(re.sub(r"^\W+|\W+$", "", r[1]) for r in result)
    for old, new in [("_", "-"), (" ", ""), ("/", "1"), ("|", "1"), ("l", "1"), ("I", "1"), ("O", "0"), ("H", "4")]:
        text = text.replace(old, new)
    return text.upper()


def edit_distance(a, b):
    """Number of single-character insertions, deletions or swaps to turn a into b."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def main():
    tests = load_test_set()
    reader = easyocr.Reader(["en"], verbose=False)
    methods = {
        "plain": lambda img: reader.readtext(img),
        "allowlist": lambda img: reader.readtext(img, allowlist=ALLOWLIST),
    }
    rows = []
    for path, truth, source in tests:
        img = preprocess(path, source)
        row = {"image": os.path.basename(path), "truth": truth}
        for name, read in methods.items():
            row[name] = app_text(read(img))
        rows.append(row)

    print(f"{'image':<28} {'truth':<14} {'plain':<16} {'allowlist':<16}")
    for row in rows:
        marks = {m: ("ok " if row[m] == row["truth"] else "BAD") for m in methods}
        print(f"{row['image']:<28} {row['truth']:<14} {marks['plain']} {row['plain']:<12} {marks['allowlist']} {row['allowlist']:<12}")

    total_chars = sum(len(r["truth"]) for r in rows)
    print()
    for m in methods:
        exact = sum(r[m] == r["truth"] for r in rows)
        errors = sum(edit_distance(r[m], r["truth"]) for r in rows)
        print(f"{m:<10} whole equations right: {exact}/{len(rows)} ({100 * exact / len(rows):.0f}%)   "
              f"character error rate: {100 * errors / total_chars:.1f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "baseline_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "truth", *methods])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
