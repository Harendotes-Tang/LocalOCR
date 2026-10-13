"""Train the crop classifier on grammar-round crops (CPU by default) and export ONNX.

Labels come from dataset kinds only: face -> face, plain/punctuation/emoji ->
bracketed-text, blank/graphic -> graphic. The threshold is chosen on the
development split for a bounded non-face false-positive rate; heldout is never
read. Runs inside a registered research round and binds the dataset freeze.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image

from tools.PretrainedOcr import common, round_profile as rp
from tools.RegionKaomoji import classifier as cl

LABELS = {'face': 0, 'plain': 1, 'punctuation': 1, 'emoji': 1, 'blank': 2, 'graphic': 2}
DEFAULT_FPR = 0.005


def load_rows(dataset, split):
    rows = common.read_split(dataset, split)
    root = Path(dataset).resolve()
    arrays, labels = [], []
    for row in rows:
        with Image.open(root / row['image']) as image:
            image.load()
            arrays.append(np.asarray(image.convert('RGB')))
        labels.append(LABELS[row['kind']])
    return arrays, np.asarray(labels, dtype=np.int64)


def jitter(array, rng):
    """Margin and scale jitter without altering label; mirrors the automatic candidate padding."""
    height, width = array.shape[:2]
    top, bottom = rng.randint(0, 4), rng.randint(0, 4)
    left, right = rng.randint(0, 4), rng.randint(0, 4)
    background = array[0, 0]
    padded = np.empty((height + top + bottom, width + left + right, 3), dtype=np.uint8)
    padded[:] = background
    padded[top:top + height, left:left + width] = array
    crop_top, crop_left = rng.randint(0, 3), rng.randint(0, 3)
    crop_bottom, crop_right = padded.shape[0] - rng.randint(0, 3), padded.shape[1] - rng.randint(0, 3)
    return np.ascontiguousarray(padded[crop_top:crop_bottom, crop_left:crop_right])


def build_model():
    import torch.nn as nn
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, len(cl.CLASSES)))


def tensors(arrays, rng=None):
    import torch
    batch = np.stack([cl.preprocess(jitter(a, rng) if rng is not None else a) for a in arrays]).astype(np.float32)
    return torch.from_numpy(batch)


def evaluate(model, arrays, labels, batch_size=256):
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            logits = model(tensors(arrays[start:start + batch_size]))
            outputs.append(torch.softmax(logits, dim=1).numpy())
    probs = np.concatenate(outputs) if outputs else np.zeros((0, len(cl.CLASSES)), dtype=np.float32)
    predicted = probs.argmax(axis=1)
    accuracy = float((predicted == labels).mean()) if len(labels) else 0.0
    return probs, accuracy


def select_threshold(probs, labels, maximum_fpr=DEFAULT_FPR):
    """Smallest face threshold whose non-face false-positive rate stays within the bound; recall is reported."""
    face = probs[:, 0]
    negatives = face[labels != 0]
    positives = face[labels == 0]
    candidates = sorted(set(np.round(np.linspace(0.05, 0.99, 95), 2)))
    table = []
    chosen = None
    for threshold in candidates:
        fpr = float((negatives >= threshold).mean()) if len(negatives) else 0.0
        recall = float((positives >= threshold).mean()) if len(positives) else 0.0
        table.append({'threshold': float(threshold), 'false_positive_rate': fpr, 'face_recall': recall})
        if chosen is None and fpr <= maximum_fpr:
            chosen = float(threshold)
    met = chosen is not None
    if chosen is None:
        chosen = 0.99
    return chosen, table, met


def export_onnx(model, output):
    import torch
    model.eval()
    example = torch.zeros(1, 1, cl.INPUT_HEIGHT, cl.INPUT_WIDTH)
    torch.onnx.export(model, example, str(output), input_names=['input'], output_names=['logits'],
                      dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17, dynamo=False)


def parity(model, onnx_path, arrays):
    import onnxruntime as ort
    import torch
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    batch = tensors(arrays)
    with torch.no_grad():
        reference = model(batch).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: batch.numpy()})[0]
    return float(np.max(np.abs(reference - actual)))


def train(dataset, output, identifier, seconds=600, batch_size=128, learning_rate=1e-3, seed=20261013, maximum_fpr=DEFAULT_FPR, epochs=40):
    import torch
    import torch.nn.functional as F
    output = Path(output).resolve()
    if rp.profile_for(output) is None:
        raise ValueError('Classifier training must write inside a registered research round')
    if output.exists():
        raise ValueError('Fresh classifier output directory required')
    if torch.cuda.is_available():
        raise RuntimeError('Classifier training runs on CPU only; hide CUDA devices before training')
    torch.manual_seed(seed)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    binding = common.dataset_binding(dataset)
    train_arrays, train_labels = load_rows(dataset, 'train')
    dev_arrays, dev_labels = load_rows(dataset, 'dev')
    model = build_model()
    counts = np.bincount(train_labels, minlength=len(cl.CLASSES)).astype(np.float64)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) / len(cl.CLASSES), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    started = time.perf_counter()
    history = []
    best = None
    order = np.arange(len(train_arrays))
    for epoch in range(epochs):
        if time.perf_counter() - started > seconds:
            break
        model.train()
        np_rng.shuffle(order)
        total, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = tensors([train_arrays[i] for i in indices], rng)
            target = torch.from_numpy(train_labels[indices])
            loss = F.cross_entropy(model(batch), target, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
            steps += 1
            if time.perf_counter() - started > seconds:
                break
        probs, accuracy = evaluate(model, dev_arrays, dev_labels)
        threshold, _, met = select_threshold(probs, dev_labels, maximum_fpr)
        recall = float((probs[dev_labels == 0, 0] >= threshold).mean()) if (dev_labels == 0).any() else 0.0
        record = {'epoch': epoch, 'train_loss': total / max(1, steps), 'dev_accuracy': accuracy, 'dev_threshold': threshold,
                  'false_positive_bound_met': met, 'dev_face_recall_at_threshold': recall, 'elapsed_seconds': time.perf_counter() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        # Only epochs whose threshold satisfies the false-positive bound can win; then recall, then accuracy.
        key = (met, recall, accuracy)
        if best is None or key > best['key']:
            best = {'epoch': epoch, 'key': key, 'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if best is None:
        raise RuntimeError('No training epoch completed within the time limit')
    model.load_state_dict(best['state'])
    probs, accuracy = evaluate(model, dev_arrays, dev_labels)
    threshold, table, met = select_threshold(probs, dev_labels, maximum_fpr)
    confusion = np.zeros((len(cl.CLASSES), len(cl.CLASSES)), dtype=int)
    for truth, predicted in zip(dev_labels, probs.argmax(axis=1)):
        confusion[truth, predicted] += 1
    output.mkdir(parents=True)
    model_path = output / 'model.onnx'
    export_onnx(model, model_path)
    difference = parity(model, model_path, dev_arrays[:64])
    if not difference < 1e-3:
        raise RuntimeError('ONNX export differs from the trained network')
    torch.save(model.state_dict(), output / 'model.pt')
    receipt = {'schema_version': cl.SCHEMA, 'id': identifier, 'classes': list(cl.CLASSES), 'preprocessing': cl.PREPROCESSING,
               'model': {'file': 'model.onnx', 'sha256': cl.digest(model_path), 'bytes': model_path.stat().st_size},
               'native_state': {'file': 'model.pt', 'sha256': cl.digest(output / 'model.pt')},
               'threshold': threshold, 'promote': False, 'enabled': False,
               'training': {'dataset': str(Path(dataset).resolve()), 'dataset_sha256': binding['dataset_sha256'],
                            'split_sha256': binding['split_sha256'], 'train_images': len(train_arrays), 'dev_images': len(dev_arrays),
                            'label_map': LABELS, 'class_counts': counts.astype(int).tolist(), 'seed': seed, 'batch_size': batch_size,
                            'learning_rate': learning_rate, 'time_limit_seconds': seconds, 'epochs_completed': len(history),
                            'selected_epoch': best['epoch'], 'history': history, 'device': 'cpu', 'cuda_available': False,
                            'gpu_used': False, 'threshold_selected_on': 'dev', 'maximum_false_positive_rate': maximum_fpr,
                            'false_positive_bound_met': met,
                            'threshold_table': table, 'dev_accuracy': accuracy, 'dev_confusion': confusion.tolist(),
                            'onnx_parity_max_abs_difference': difference, 'heldout_consumed': False,
                            'torch_version': torch.__version__, 'source_sha256': {
                                'train_classifier.py': cl.digest(Path(__file__)), 'classifier.py': cl.digest(Path(cl.__file__))},
                            'started_utc': common.utc()}}
    common.write(output / 'classifier.json', receipt)
    cl.classifier_binding(output / 'classifier.json')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--identifier', required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--maximum-fpr', type=float, default=DEFAULT_FPR)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args(argv)
    receipt = train(args.dataset, args.output, args.identifier, args.seconds, args.batch_size, args.learning_rate, args.seed, args.maximum_fpr, args.epochs)
    print(json.dumps({'id': receipt['id'], 'threshold': receipt['threshold'], 'dev_accuracy': receipt['training']['dev_accuracy'],
                      'selected_epoch': receipt['training']['selected_epoch'], 'gpu_used': False}))


if __name__ == '__main__':
    main()
