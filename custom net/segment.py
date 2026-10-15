"""
Step 2: splits a preprocessed equation image into one box per symbol.

    python segment.py

Run directly to check it on the test set. For every image it saves a picture to
output/segmentation/ with the boxes drawn on and the normalised crops underneath,
and prints whether the number of boxes matches the number of characters.
"""
import os

import cv2
import numpy as np

from preprocess import OUTPUT_DIR, load_test_set, preprocess

NOISE_RATIO = 0.25    # blobs smaller than this fraction of a typical symbol are specks, not writing
MERGE_OVERLAP = 0.5   # blobs sharing this much of their width are one symbol (e.g. the two bars of "=")
SPLIT_RATIO = 1.6     # tall boxes this many times wider than usual are probably two touching symbols
CROP_SIZE = 28        # size of the normalised crops given to the classifier (same as MNIST)
INK_SIZE = 20         # longest side of the symbol inside the crop, leaving a margin
STROKE_RATIO = 0.2    # thin strokes are thickened to this fraction of the symbol size, so all crops look alike


class Symbol:
    """One symbol found in the image: its box and the ink pixels inside it."""
    def __init__(self, x, y, w, h, mask):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.mask = mask  # 255 where there is ink, cropped to the box

    @property
    def right(self):
        return self.x + self.w


class _Group:
    """A box made of one or more connected blobs, used while segmenting."""
    def __init__(self, x, y, w, h, ids):
        self.x, self.y, self.w, self.h, self.ids = x, y, w, h, set(ids)

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    def merged_with(self, other):
        x, y = min(self.x, other.x), min(self.y, other.y)
        return _Group(x, y, max(self.right, other.right) - x, max(self.bottom, other.bottom) - y, self.ids | other.ids)


def segment(binary, ignore_edges=True):
    """
    Takes a black-ink-on-white image and returns its symbols as a list of Symbol, left to right.
    ignore_edges drops ink touching the image border (frames and shadows in photos); turn it off
    for images that are already cropped tightly around a symbol, like the training dataset.
    """
    ink = (binary < 128).astype(np.uint8)
    height, width = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    groups = []
    for i in range(1, count):  # 0 is the background
        x, y, w, h, _ = stats[i]
        if ignore_edges and (x == 0 or y == 0 or x + w >= width or y + h >= height):
            continue  # touches the edge of the image: a frame or shadow, not writing
        groups.append(_Group(x, y, w, h, [i]))
    if not groups:
        return []

    # remove specks, measured against the bigger half of the blobs so the specks don't skew it
    sizes = sorted(max(g.w, g.h) for g in groups)
    typical = np.median(sizes[len(sizes) // 2:])
    groups = [g for g in groups if max(g.w, g.h) >= NOISE_RATIO * typical]

    groups = _merge_overlapping(groups)
    groups = _split_wide(groups, labels)

    symbols = []
    for g in groups:
        region = labels[g.y:g.bottom, g.x:g.right]
        mask = np.isin(region, list(g.ids)).astype(np.uint8) * 255
        symbols.append(Symbol(g.x, g.y, g.w, g.h, mask))
    return symbols


def _merge_overlapping(groups):
    """Joins neighbouring blobs that sit above each other, like the bars of "=" or a broken "+"."""
    groups = sorted(groups, key=lambda g: g.x)
    merged = True
    while merged:
        merged = False
        for i in range(len(groups) - 1):
            a, b = groups[i], groups[i + 1]
            overlap = min(a.right, b.right) - max(a.x, b.x)
            if overlap >= MERGE_OVERLAP * min(a.w, b.w):
                groups[i:i + 2] = [a.merged_with(b)]
                groups.sort(key=lambda g: g.x)
                merged = True
                break
    return groups


def _split_wide(groups, labels):
    """Cuts boxes that are much wider than a normal symbol where the ink is thinnest."""
    heights = [g.h for g in groups]
    typical_h = np.median(heights)
    tall = [g for g in groups if g.h >= 0.6 * typical_h]  # ignore "-" and "=", which are meant to be wide
    if len(tall) < 2:
        return groups
    typical_w = np.median([g.w for g in tall])

    result = []
    for g in groups:
        if g.h >= 0.6 * typical_h and g.w > SPLIT_RATIO * typical_w:
            ink = np.isin(labels[g.y:g.bottom, g.x:g.right], list(g.ids))
            columns = ink.sum(axis=0)
            start, end = int(g.w * 0.3), int(g.w * 0.7)  # only cut in the middle part of the box
            cut = start + int(np.argmin(columns[start:end]))
            for left, right in ((0, cut), (cut, g.w)):
                part = ink[:, left:right]
                rows = np.where(part.any(axis=1))[0]
                if len(rows):
                    half = _Group(g.x + left, g.y + rows[0], right - left, rows[-1] - rows[0] + 1, g.ids)
                    result.append(half)
        else:
            result.append(g)
    return result


def normalise(symbol):
    """
    Turns a symbol into a CROP_SIZE x CROP_SIZE image, white ink on black like MNIST.
    The symbol is scaled to fit without stretching, so a "-" stays a thin bar.
    """
    mask = _thicken(symbol.mask)
    h, w = mask.shape
    scale = INK_SIZE / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop = np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
    top, left = (CROP_SIZE - new_h) // 2, (CROP_SIZE - new_w) // 2
    crop[top:top + new_h, left:left + new_w] = small
    return crop


def _thicken(mask):
    """Widens strokes thinner than STROKE_RATIO of the symbol's size (thick strokes are left alone)."""
    border = cv2.copyMakeBorder((mask > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # so a box full of ink still has an edge
    stroke = 2 * float(cv2.distanceTransform(border, cv2.DIST_L2, 3).max())
    grow = int(round(STROKE_RATIO * max(mask.shape) - stroke))
    if grow < 2:
        return mask
    padded = cv2.copyMakeBorder(mask, grow, grow, grow, grow, cv2.BORDER_CONSTANT, value=0)
    thick = cv2.dilate(padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    ys, xs = np.nonzero(thick)
    return thick[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def draw(binary, symbols):
    """Returns a picture of the image with numbered boxes, and the normalised crops along the bottom."""
    picture = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i, s in enumerate(symbols):
        cv2.rectangle(picture, (s.x, s.y), (s.right, s.y + s.h), (0, 0, 255), 2)
        cv2.putText(picture, str(i + 1), (s.x, max(12, s.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    tile = CROP_SIZE * 2
    per_row = max(1, (picture.shape[1] - 4) // (tile + 4))
    rows = max(1, -(-len(symbols) // per_row))  # rounds up
    strip = np.full((rows * (tile + 4) + 4, picture.shape[1], 3), 90, np.uint8)
    for i, s in enumerate(symbols):
        x, y = 4 + (i % per_row) * (tile + 4), 4 + (i // per_row) * (tile + 4)
        strip[y:y + tile, x:x + tile] = cv2.cvtColor(cv2.resize(normalise(s), (tile, tile),
                                                                interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    return np.vstack([picture, strip])


def main():
    out_dir = os.path.join(OUTPUT_DIR, "segmentation")
    os.makedirs(out_dir, exist_ok=True)
    right = 0
    tests = load_test_set()
    print(f"{'image':<28} {'truth':<14} {'expected':>8} {'found':>6}")
    for path, truth, source in tests:
        binary = preprocess(path, source)
        symbols = segment(binary)
        ok = len(symbols) == len(truth)  # every character in the label is one symbol, e.g. "X2" is x and a raised 2
        right += ok
        name = os.path.basename(path)
        print(f"{name:<28} {truth:<14} {len(truth):>8} {len(symbols):>6}  {'ok' if ok else 'WRONG'}")
        cv2.imwrite(os.path.join(out_dir, name), draw(binary, symbols))
    print(f"\nright number of symbols: {right}/{len(tests)}   (pictures saved to {out_dir})")


if __name__ == "__main__":
    main()
