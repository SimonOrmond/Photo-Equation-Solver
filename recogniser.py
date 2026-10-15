"""
Reads a handwritten equation from a preprocessed (black ink on white) image.

The image is split into one box per symbol, and each symbol is classified by SymbolNet,
the small neural network trained in 'custom net'. The network runs in plain numpy, using
the weights exported to symbolnet.npz (see 'custom net/export_model.py'), so the app
doesn't need PyTorch.

The segmentation and normalisation here are copied from 'custom net/segment.py'.
If you change them there and retrain, copy the changes here too.
"""
import os

import cv2
import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbolnet.npz")

NOISE_RATIO = 0.25    # blobs smaller than this fraction of a typical symbol are specks, not writing
MERGE_OVERLAP = 0.5   # blobs sharing this much of their width are one symbol (e.g. the two bars of "=")
SPLIT_RATIO = 1.6     # tall boxes this many times wider than usual are probably two touching symbols
CROP_SIZE = 28        # size of the normalised crops given to the classifier (same as MNIST)
INK_SIZE = 20         # longest side of the symbol inside the crop, leaving a margin
STROKE_RATIO = 0.2    # thin strokes are thickened to this fraction of the symbol size, so all crops look alike
MIN_CONFIDENCE = 0.5  # if the network is less sure than this about any symbol, the image is rejected


# ---------- reading an equation ----------

def recognise(binary):
    """
    Returns the equation in the image as the app's string format, e.g. "5X2+10X-30=0".
    Returns "" if no symbols were found or the network wasn't confident about one of them,
    which the app treats as an invalid image.
    """
    symbols = segment(binary)
    if not symbols:
        return ""
    crops = np.array([normalise(s) for s in symbols])
    probabilities = _network().predict(crops)
    best = probabilities.argmax(axis=1)
    text = "".join(_network().classes[i] for i in best).upper()
    print("Recognised: " + text + "   confidence per symbol: " +
          " ".join(f"{p:.2f}" for p in probabilities.max(axis=1)))  # for debugging
    if probabilities.max(axis=1).min() < MIN_CONFIDENCE:
        return ""
    return text


_loaded_network = None


def _network():
    """Loads the network the first time it's needed and reuses it afterwards."""
    global _loaded_network
    if _loaded_network is None:
        _loaded_network = SymbolNet(MODEL_FILE)
    return _loaded_network


# ---------- the neural network, in numpy ----------

class SymbolNet:
    """
    The same network as 'custom net/model.py':
    conv(1->16) ReLU maxpool, conv(16->32) ReLU maxpool, linear(1568->64) ReLU, linear(64->classes)
    """
    def __init__(self, path):
        data = np.load(path)
        self.w = {name: data[name] for name in data.files if name != "classes"}
        self.classes = [str(c) for c in data["classes"]]

    def predict(self, crops):
        """crops: uint8 array (N, 28, 28). Returns an (N, classes) array of probabilities."""
        x = crops.astype(np.float32)[:, None] / 255.0                    # (N, 1, 28, 28), scaled 0-1
        x = _max_pool(_relu(_conv3x3(x, self.w["conv1_w"], self.w["conv1_b"])))   # (N, 16, 14, 14)
        x = _max_pool(_relu(_conv3x3(x, self.w["conv2_w"], self.w["conv2_b"])))   # (N, 32, 7, 7)
        x = x.reshape(len(x), -1)                                        # flatten, same order as PyTorch
        x = _relu(x @ self.w["fc1_w"].T + self.w["fc1_b"])
        scores = x @ self.w["fc2_w"].T + self.w["fc2_b"]
        scores -= scores.max(axis=1, keepdims=True)                      # softmax: scores -> probabilities
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)


def _conv3x3(x, weights, bias):
    """3x3 convolution with 1 pixel of padding. x: (N, C, H, W), weights: (F, C, 3, 3)."""
    n, c, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # every 3x3 neighbourhood laid out as a column, so the convolution becomes one matrix multiply
    patches = np.stack([padded[:, :, i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=2)
    patches = patches.reshape(n, c * 9, h * w)
    out = np.einsum("fk,nkp->nfp", weights.reshape(len(weights), -1), patches) + bias[None, :, None]
    return out.reshape(n, len(weights), h, w)


def _relu(x):
    return np.maximum(x, 0)


def _max_pool(x):
    """2x2 max pooling: halves the width and height."""
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


# ---------- splitting the image into symbols ----------

class Symbol:
    """One symbol found in the image: its box and the ink pixels inside it."""
    def __init__(self, x, y, w, h, mask):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.mask = mask  # 255 where there is ink, cropped to the box


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


def segment(binary):
    """Takes a black-ink-on-white image and returns its symbols as a list of Symbol, left to right."""
    ink = (binary < 128).astype(np.uint8)
    height, width = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    groups = []
    for i in range(1, count):  # 0 is the background
        x, y, w, h, _ = stats[i]
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
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
                    result.append(_Group(g.x + left, g.y + rows[0], right - left, rows[-1] - rows[0] + 1, g.ids))
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
