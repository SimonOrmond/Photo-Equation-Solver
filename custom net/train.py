"""
Step 3b: trains SymbolNet on the prepared crops and tests it.

    python train.py      (run prepare_data.py first)

Reports accuracy on the 30% test split, which symbols get confused, and how the network
does on the symbols in your real photos ('test set'). Saves the weights to output/symbolnet.pt.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import CLASSES, SymbolNet
from prepare_data import PREPARED_FILE
from preprocess import OUTPUT_DIR, load_test_set, preprocess
from segment import normalise, segment

EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MODEL_FILE = os.path.join(OUTPUT_DIR, "symbolnet.pt")


def to_tensor(images):
    """uint8 crops (N, 28, 28) -> float tensor (N, 1, 28, 28) scaled to 0-1."""
    return torch.from_numpy(images).float().unsqueeze(1) / 255.0


def predict(model, images):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(images)).argmax(dim=1).numpy()


def train(model, train_x, train_y, test_x, test_y):
    loader = DataLoader(TensorDataset(to_tensor(train_x), torch.from_numpy(train_y)), batch_size=BATCH_SIZE, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(labels)
        train_acc = (predict(model, train_x) == train_y).mean()
        test_acc = (predict(model, test_x) == test_y).mean()
        print(f"epoch {epoch:>2}   loss {total_loss / len(train_y):.3f}   train {100 * train_acc:.1f}%   test {100 * test_acc:.1f}%")


def report_test_split(model, test_x, test_y):
    predictions = predict(model, test_x)
    print(f"\ntest split accuracy: {100 * (predictions == test_y).mean():.2f}%  ({len(test_y)} crops)")
    print("per symbol:", "  ".join(f"{name} {100 * (predictions[test_y == c] == c).mean():.0f}%"
                                   for c, name in enumerate(CLASSES)))
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    for truth, guess in zip(test_y, predictions):
        confusion[truth, guess] += 1
    np.fill_diagonal(confusion, 0)
    worst = np.dstack(np.unravel_index(np.argsort(confusion, axis=None)[::-1], confusion.shape))[0][:5]
    print("most common mistakes:", ", ".join(f"{CLASSES[t]} read as {CLASSES[g]} ({confusion[t, g]})"
                                            for t, g in worst if confusion[t, g]))


def report_real_photos(model):
    """Classifies the segmented symbols from the photos in 'test set'."""
    print("\nreal photos:")
    right = total = equations_right = 0
    tests = load_test_set()
    for path, truth, source in tests:
        symbols = segment(preprocess(path, source))
        truth = truth.lower()
        if len(symbols) != len(truth):
            print(f"  {os.path.basename(path):<28} skipped, segmentation found {len(symbols)} symbols, expected {len(truth)}")
            continue
        guess = "".join(CLASSES[i] for i in predict(model, np.array([normalise(s) for s in symbols])))
        right += sum(a == b for a, b in zip(guess, truth))
        total += len(truth)
        equations_right += guess == truth
        print(f"  {os.path.basename(path):<28} truth {truth:<14} read {guess:<14} {'ok' if guess == truth else 'BAD'}")
    if total:
        print(f"symbols right: {right}/{total} ({100 * right / total:.0f}%)   whole equations right: {equations_right}/{len(tests)}")


def main():
    torch.manual_seed(0)
    data = np.load(PREPARED_FILE)
    train_x, train_y, test_x, test_y = data["train_x"], data["train_y"], data["test_x"], data["test_y"]
    print(f"train {len(train_y)} crops, test {len(test_y)} crops, {len(CLASSES)} classes\n")

    model = SymbolNet()
    train(model, train_x, train_y, test_x, test_y)
    report_test_split(model, test_x, test_y)
    report_real_photos(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, MODEL_FILE)
    print(f"\nsaved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
