"""Small onboard network. Trained here, run on the scout, never in the cloud."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veil.dataset import NAMES, generate

WEIGHTS = Path(__file__).resolve().parents[1] / "outputs" / "hazard_net.npz"
HIDDEN = 64
_cached: HazardNet | None = None


class HazardNet:
    def __init__(self, weights: dict):
        self.w1 = weights["w1"]
        self.b1 = weights["b1"]
        self.w2 = weights["w2"]
        self.b2 = weights["b2"]
        self.mean = weights["mean"]
        self.std = weights["std"]
        self.val_accuracy = float(weights["val_accuracy"])
        self.samples = int(weights["samples"])
        confusion = weights.get("confusion")
        self.confusion = None if confusion is None else np.asarray(confusion).astype(int).tolist()

    def predict(self, sample: np.ndarray) -> tuple[str, float]:
        probs = _forward_one(self, sample.astype(np.float32))
        index = int(np.argmax(probs))
        return NAMES[index], float(probs[index])


POLAR_WEIGHTS = Path(__file__).resolve().parents[1] / "outputs" / "hazard_net_polar.npz"


def load(path: Path | None = None) -> HazardNet:
    global _cached
    if path is None and _cached is not None:
        return _cached
    chosen = WEIGHTS if path is None else Path(path)
    if not chosen.exists():
        raise FileNotFoundError(f"missing onboard weights: {chosen}. Run python -m veil.train")
    blob = np.load(chosen)
    required = ("w1", "b1", "w2", "b2", "mean", "std", "val_accuracy", "samples")
    if any(key not in blob.files for key in required):
        raise ValueError("hazard_net.npz is missing required arrays")
    weights = {key: blob[key] for key in blob.files}
    if weights["w1"].ndim != 2 or weights["w2"].shape[0] != weights["w1"].shape[1]:
        raise ValueError("hazard_net.npz has inconsistent layer shapes")
    net = HazardNet(weights)
    if path is None:
        _cached = net
    return net


def arm_polar() -> HazardNet:
    """Serve the PolarDB weights. The painted mission still reads hazard_net.npz."""
    global _cached
    if not POLAR_WEIGHTS.exists():
        raise FileNotFoundError(f"missing PolarDB weights: {POLAR_WEIGHTS}")
    net = load(POLAR_WEIGHTS)
    _cached = net
    import veil.tracks as tracks

    tracks.BASELINE_M = 0.0
    return net


def active_note() -> str:
    net = load()
    if int(net.samples) >= 4000:
        return "PolarDB held-out accuracy. Terrains 9 and 12 were not used for training."
    return "Synthetic held-out accuracy. Not the PolarDB score."


def train(save: bool = True, seed: int = 7) -> HazardNet:
    train_x, train_y = generate(280, seed)
    val_x, val_y = generate(80, seed + 1000)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (train_x - mean) / std
    vx = (val_x - mean) / std
    rng = np.random.default_rng(seed)
    inputs = x.shape[1]
    outputs = len(NAMES)
    w1 = rng.normal(0, np.sqrt(2 / inputs), size=(inputs, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    w2 = rng.normal(0, np.sqrt(2 / HIDDEN), size=(HIDDEN, outputs)).astype(np.float32)
    b2 = np.zeros(outputs, np.float32)
    lr = 0.08
    batch = 64
    for _epoch in range(48):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            idx = order[start : start + batch]
            xb, yb = x[idx], train_y[idx]
            hidden = np.maximum(xb @ w1 + b1, 0)
            scores = hidden @ w2 + b2
            scores -= scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            probs = exp / exp.sum(axis=1, keepdims=True)
            target = np.zeros_like(probs)
            target[np.arange(len(yb)), yb] = 1.0
            grad_scores = (probs - target) / len(yb)
            grad_w2 = hidden.T @ grad_scores
            grad_b2 = grad_scores.sum(axis=0)
            grad_hidden = grad_scores @ w2.T
            grad_hidden[hidden <= 0] = 0
            w2 -= lr * grad_w2.astype(np.float32)
            b2 -= lr * grad_b2.astype(np.float32)
            w1 -= lr * (xb.T @ grad_hidden).astype(np.float32)
            b1 -= lr * grad_hidden.sum(axis=0).astype(np.float32)
        lr *= 0.96
    val_pred = _predict_many(w1, b1, w2, b2, vx)
    accuracy = float((val_pred == val_y).mean())
    matrix = np.zeros((outputs, outputs), np.int32)
    for truth, guess in zip(val_y, val_pred):
        matrix[int(truth), int(guess)] += 1
    weights = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "val_accuracy": np.array(accuracy, np.float32),
        "samples": np.array(len(train_x), np.int32),
        "confusion": matrix,
        "val_seed": np.array(seed + 1000, np.int32),
    }
    if save:
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        np.savez(WEIGHTS, **weights)
    global _cached
    _cached = HazardNet(weights)
    return _cached


def write_held_out_metrics(seed: int = 7) -> list:
    """Score the saved weights on the held-out generator. Does not change weights."""
    global _cached
    _cached = None
    net = load()
    samples, labels = generate(80, seed + 1000)
    normalized = ((samples - net.mean) / net.std).astype(np.float32)
    guessed = _predict_many(net.w1, net.b1, net.w2, net.b2, normalized)
    matrix = np.zeros((len(NAMES), len(NAMES)), np.int32)
    for truth, guess in zip(labels, guessed):
        matrix[int(truth), int(guess)] += 1
    blob = np.load(WEIGHTS)
    stored = {key: blob[key] for key in blob.files}
    stored["confusion"] = matrix
    stored["val_accuracy"] = np.array(float((guessed == labels).mean()), np.float32)
    stored["val_seed"] = np.array(seed + 1000, np.int32)
    np.savez(WEIGHTS, **stored)
    _cached = None
    return matrix.tolist()


def _predict_many(w1, b1, w2, b2, samples: np.ndarray) -> np.ndarray:
    hidden = np.maximum(samples @ w1 + b1, 0)
    scores = hidden @ w2 + b2
    return np.argmax(scores, axis=1)


def _forward_one(net: HazardNet, sample: np.ndarray) -> np.ndarray:
    z = (sample - net.mean) / net.std
    hidden = np.maximum(z @ net.w1 + net.b1, 0)
    scores = hidden @ net.w2 + net.b2
    scores = scores - scores.max()
    exp = np.exp(scores)
    return exp / exp.sum()
