"""
Copies the trained network into the app.

    python export_model.py      (run train.py first)

Converts output/symbolnet.pt into plain numpy arrays and saves them as symbolnet.npz in the
main app folder, where recogniser.py loads them. Then checks the numpy version gives the
same answers as PyTorch on the test split.
"""
import os
import sys

import numpy as np
import torch

from model import SymbolNet
from prepare_data import PREPARED_FILE
from preprocess import HERE
from train import MODEL_FILE, predict

APP_DIR = os.path.dirname(HERE)
sys.path.insert(0, APP_DIR)
import recogniser  # noqa: E402  (the app's numpy version of the network)

# PyTorch layer names -> names recogniser.py expects
LAYERS = {"features.0": "conv1", "features.3": "conv2", "classifier.1": "fc1", "classifier.4": "fc2"}


def main():
    saved = torch.load(MODEL_FILE)
    arrays = {}
    for torch_name, app_name in LAYERS.items():
        arrays[app_name + "_w"] = saved["state_dict"][torch_name + ".weight"].numpy()
        arrays[app_name + "_b"] = saved["state_dict"][torch_name + ".bias"].numpy()
    out_file = os.path.join(APP_DIR, "symbolnet.npz")
    np.savez(out_file, classes=np.array(saved["classes"]), **arrays)
    print(f"saved {out_file}")

    # check the numpy network matches PyTorch
    model = SymbolNet()
    model.load_state_dict(saved["state_dict"])
    test_x = np.load(PREPARED_FILE)["test_x"]
    torch_answers = predict(model, test_x)
    numpy_answers = recogniser.SymbolNet(out_file).predict(test_x).argmax(axis=1)
    same = (torch_answers == numpy_answers).mean()
    print(f"numpy and PyTorch agree on {100 * same:.2f}% of {len(test_x)} test crops")


if __name__ == "__main__":
    main()
