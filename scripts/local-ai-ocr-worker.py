"""Packaged CPU OCR: owned PNG in, original-coordinate TSV out, no network.

This helper runs once per image and exits. It accepts only local absolute paths;
the Qt owner enforces the image deadline and kills this process on cancellation.
RapidOCR/ONNX Runtime are fixed by tools/LocalAiOcr/requirements-build.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import re
import unicodedata
from types import SimpleNamespace

MODELS = {
    "PP-OCRv6_det_small.onnx": (9929594, "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "PP-OCRv6_rec_small.onnx": (21234383, "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (585532, "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
}
HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
VERSION = "MentorRecorder local AI OCR 2; RapidOCR 3.10.0; ONNX Runtime 1.31.0; CPU-only"
# Qt limits source files to 20 MiB before decoding. Its owned lossless PNG can
# be larger than a JPEG source; bound that copy by the already permitted pixels.
MAX_PIXELS = 50_000_000
MAX_OWNED_PNG_BYTES = MAX_PIXELS * 4 + 1024 * 1024
MAX_REREAD_GROUPS = 48
MAX_REREAD_PIXELS = 1_000_000


def text_anchors(text):
    """Keep letters and numbers unchanged when comparing symbol-only alternatives."""
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def symbol_count(text):
    return sum(unicodedata.category(c)[0] in "PS" for c in text)


def preserves_observed_text(original, candidate):
    """Keep original glyphs in order relative to text, with the reviewed bullet-eye alternative."""
    source = ''.join(original.split())
    target = ''.join(candidate.split())
    for expected in (source, ['●' if c=='•' else c for c in source]):
        remaining=iter(target)
        if all(any(c==symbol for c in remaining) for symbol in expected):
            return True
    return False


def visible_ring_prefix(image, rect, minimum_left=0):
    """Locate a run of tiny hollow stops before text; full-height zeros are excluded.

    This is crop selection evidence, not an independent text prediction. The
    recognizer must also report stops in two crop views. Pixels determine the
    count because weak first/last stops may be lost even in the repeated OCR.
    """
    import cv2
    import numpy as np
    x, y, r, b = rect
    h = b-y
    left, top, bottom = max(0, minimum_left, x-2*h), max(0, y-h//3), min(image.shape[0], b+h//3)
    if left >= x:
        return 0
    crop = image[top:bottom, left:x]
    if crop.size == 0 or h < 12:
        return 0
    mask = ((crop.min(axis=2)>160) & (np.ptp(crop, axis=2)<60)).astype(np.uint8)*255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    if hierarchy is None:
        return 0
    for contour, info in zip(contours, hierarchy[0]):
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if info[3] != -1 or info[2] == -1:
            continue
        if 3 <= cw <= h*.4 and 3 <= ch <= h*.4 and .7 <= cw/ch <= 1.4 and y+h*.5 <= top+cy+ch/2 <= b:
            rings.append((cx, cy, cw, ch))
    rings.sort()
    if not 2 <= len(rings) <= 6:
        return 0
    if max(cy for _,cy,_,_ in rings)-min(cy for _,cy,_,_ in rings)>2:
        return 0
    gaps=[rings[i+1][0]-rings[i][0] for i in range(len(rings)-1)]
    if min(gaps)<4 or max(gaps)>h or max(gaps)-min(gaps)>3:
        return 0
    return len(rings)


def choose_reread(original, variants):
    """Require independent crop views to agree; confidence alone cannot restore a glyph.

    Whitespace is ignored for voting only. The selected recognizer string is
    returned verbatim, without punctuation normalization or bracket completion.
    A reread may add symbols or replace a symbol with a longer visible spelling,
    but may not change the letters/numbers or remove original symbols.
    """
    votes = {}
    for text, score in variants:
        if not math.isfinite(float(score)) or score < 0.70 or not text.strip():
            continue
        if text_anchors(text) != text_anchors(original) or symbol_count(text) < symbol_count(original):
            continue
        if not preserves_observed_text(original, text):
            continue
        if symbol_count(text) == symbol_count(original):
            # A visible solid eye often becomes a small bullet in the first
            # pass. Do not use a tie to normalize full/half width punctuation.
            if ''.join(text.split()) != ''.join(original.replace('•', '●').split()):
                continue
        key = "".join(text.split())
        votes.setdefault(key, []).append((text.strip(), float(score)))
    eligible = [items for key, items in votes.items()
                if len(items) >= 2 and key != "".join(original.split())]
    if not eligible:
        return None
    items = max(eligible, key=lambda items: (symbol_count(items[0][0]), len(items)))
    # Keep the ordinary unpadded view's spelling if it participated in the vote.
    return items[0][0], min(score for _, score in items)


def reread_symbols(engine, image, result):
    """Retry a bounded number of horizontal symbol regions using the same recognizer.

    Detection runs once. Neighbour detections cap the crop so a retry cannot
    consume another field. Result boxes are actual crop envelopes in source
    coordinates, never estimates based on character counts. Blank areas cannot
    create candidates; unsupported/rotated layouts keep the first-pass result.
    """
    import cv2
    import numpy as np
    from rapidocr.ch_ppocr_rec import TextRecInput
    if result.boxes is None or len(result.boxes) > 4096:
        return result
    height, width = image.shape[:2]
    entries = []
    for index, (box, text, score) in enumerate(zip(result.boxes, result.txts, result.scores)):
        points = np.asarray(box)
        horizontal = abs(float(points[1, 1] - points[0, 1])) <= abs(float(points[1, 0] - points[0, 0])) * .2
        x, y = np.floor(points.min(axis=0)).astype(int)
        r, b = np.ceil(points.max(axis=0)).astype(int)
        x, y, r, b = max(0, x), max(0, y), min(width, r), min(height, b)
        if r > x and b > y:
            entries.append(dict(index=index, x=x, y=y, r=r, b=b, text=str(text), score=float(score), horizontal=horizontal))
    # Small face fragments have different box heights/centres. First establish
    # physical lines by overlap, then order within each line by x.
    lines = []
    for entry in sorted(entries, key=lambda e: (e['y'], e['x'])):
        for line in reversed(lines[-8:]):
            overlap = min(line['b'], entry['b']) - max(line['y'], entry['y'])
            if overlap >= min(line['b']-line['y'], entry['b']-entry['y'])*.5:
                line['entries'].append(entry)
                line['y'], line['b'] = min(line['y'],entry['y']), max(line['b'],entry['b'])
                break
        else:
            lines.append(dict(y=entry['y'],b=entry['b'],entries=[entry]))
    entries = [e for line in lines for e in sorted(line['entries'],key=lambda e:e['x'])]
    groups = []
    for entry in entries:
        if not entry['horizontal']:
            continue  # Still retained in entries to cap neighbouring retries.
        # Source time, level and delete marks must retain their original boxes.
        if re.search(r"\d{4}[-－]\d{2}[-－]\d{2}|\d{2}[:：]\d{2}[:：]\d{2}|^Lv\.", entry['text'], re.I) or entry['text'] in ('x', 'X', '×', '✕'):
            continue
        selected = None
        for group in reversed(groups[-8:]):
            overlap = min(group['b'], entry['b']) - max(group['y'], entry['y'])
            h = min(group['b'] - group['y'], entry['b'] - entry['y'])
            gap = entry['x'] - group['r']
            if overlap >= h * .5 and -h <= gap <= max(group['b']-group['y'],entry['b']-entry['y']) and ('(' in group['text'] or '（' in group['text']):
                selected = group
                break
        if selected is None:
            groups.append(dict(entry, indices=[entry['index']]))
        else:
            selected['indices'].append(entry['index'])
            selected['text'] += entry['text']
            selected['r'] = max(selected['r'], entry['r'])
            selected['y'] = min(selected['y'], entry['y'])
            selected['b'] = max(selected['b'], entry['b'])
    samples, requests = [], []
    pixels = 0
    for group in groups:
        h = group['b'] - group['y']
        if not 10 <= h <= 64 or group['r'] - group['x'] > 1600:
            continue
        left, right = max(0, group['x'] - 2*h), min(width, group['r'] + 2*h)
        for other in entries:
            if other['index'] in group['indices']:
                continue
            overlap = min(group['b'], other['b']) - max(group['y'], other['y'])
            if overlap < min(h, other['b'] - other['y']) * .5:
                continue
            if other['x'] < group['x']:
                left = max(left, other['r'] + 2)
            if other['r'] > group['r']:
                right = min(right, other['x'] - 2)
        if left > group['x'] or right < group['r']:
            continue  # Overlapping other fields are not safe to expand/re-read.
        rings = visible_ring_prefix(image, (group['x'],group['y'],group['r'],group['b']), left)
        if not rings and not any(c in group['text'] for c in '()（）Σ∑∀°;；'):
            continue
        top, bottom = max(0, group['y'] - h//3), min(height, group['b'] + h//3)
        safe = True
        for other in entries:
            if other['index'] in group['indices'] or other['r'] <= left or other['x'] >= right:
                continue
            if other['b'] <= group['y']:
                top = max(top, other['b'] + 1)
            elif other['y'] >= group['b']:
                bottom = min(bottom, other['y'] - 1)
            elif min(group['b'],other['b'])-max(group['y'],other['y']) < min(h,other['b']-other['y'])*.5:
                safe = False  # Neighbouring lines have overlapping envelopes.
                break
        if not safe or top > group['y'] or bottom < group['b']:
            continue
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        # Tighten to visible pale text on dark cards, retaining tiny strokes.
        # Colourful backgrounds/icons do not count as text in this optimization.
        pale = (crop.min(axis=2) > 160) & (np.ptp(crop, axis=2) < 60)
        if np.mean(crop) > 150 or not pale.any():
            continue
        ys, xs = np.where(pale)
        cx, cy = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 3)
        cr, cb = min(crop.shape[1], int(xs.max()) + 3), min(crop.shape[0], int(ys.max()) + 4)
        raw = crop[cy:cb, cx:cr]
        if len(requests) >= MAX_REREAD_GROUPS or pixels + raw.shape[0]*raw.shape[1]*3 > MAX_REREAD_PIXELS:
            break
        pixels += raw.shape[0]*raw.shape[1]*3
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        inverse = cv2.cvtColor(255-gray, cv2.COLOR_GRAY2BGR)
        padded = cv2.copyMakeBorder(raw, 4, 4, 6, 6, cv2.BORDER_REPLICATE)
        samples.extend((raw, inverse, padded))
        requests.append((group, (left+cx, top+cy, left+cr, top+cb), rings))
    if not samples:
        return result
    recognized = engine._load_rec_model()(TextRecInput(img=samples, return_word_box=False))
    if len(recognized.txts) != len(samples) or len(recognized.scores) != len(samples):
        raise ValueError('Invalid symbol reread output')
    replacements, removed = {}, set()
    for offset, (group, rect, rings) in enumerate(requests):
        variants = list(zip(recognized.txts[offset*3:offset*3+3], recognized.scores[offset*3:offset*3+3]))
        chosen = choose_reread(group['text'], variants)
        if rings:
            votes = [(text, float(score)) for text,score in variants
                     if float(score)>=.70 and text.startswith('。') and not text.startswith('。'*(rings+1))
                     and text_anchors(text)==text_anchors(group['text'])]
            chosen = None
            if len(votes)>=2:
                # The run is jointly supported by hollow pixels and OCR. Keep
                # the first pass's remaining text, including its punctuation.
                chosen = ('。'*rings + group['text'], min(score for _,score in votes))
        if chosen is None:
            continue
        x, y, r, b = rect
        replacements[group['indices'][0]] = ([[x,y],[r,y],[r,b],[x,b]], chosen[0], chosen[1])
        removed.update(group['indices'][1:])
    boxes, texts, scores = [], [], []
    for index, (box, text, score) in enumerate(zip(result.boxes, result.txts, result.scores)):
        if index in removed:
            continue
        box, text, score = replacements.get(index, (box, text, score))
        boxes.append(box)
        texts.append(text)
        scores.append(score)
    return SimpleNamespace(boxes=boxes, txts=texts, scores=scores)


def deny_network(event, _args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind"}:  # BOUNDARY-ALLOW(NET-008): Denied audit event identifier; every matching event raises before I/O.
        raise RuntimeError("Offline worker forbids network access")


def validate_models(root: Path):
    """Require the exact shipped bytes before RapidOCR can initialize sessions."""
    if not root.is_absolute():
        raise ValueError("Absolute model path is required")
    for name, (size, digest) in MODELS.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("OCR model is missing or incomplete")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("OCR model checksum mismatch")


def create_engine(root: Path):
    """Load all fixed CPU sessions eagerly so missing/corrupt models cannot hide behind an empty result."""
    validate_models(root)
    sys.addaudithook(deny_network)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import cv2
    cv2.setNumThreads(2)
    from rapidocr import RapidOCR
    from rapidocr.utils.download_file import DownloadFile
    def forbid_download(*_args, **_kwargs):
        raise RuntimeError("Offline worker forbids downloads")
    DownloadFile.run = forbid_download
    params = {
        "Global.log_level": "critical",
        "Global.return_word_box": False,
        "Global.return_single_char_box": False,
        "Det.model_path": str(root / "PP-OCRv6_det_small.onnx"),
        "Rec.model_path": str(root / "PP-OCRv6_rec_small.onnx"),
        "Cls.model_path": str(root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False,
        "EngineConfig.onnxruntime.use_coreml": False,
    }
    engine = RapidOCR(params=params)
    # These lazy-loading methods belong to the pinned RapidOCR 3.10 interface.
    # Eager verification also ensures the recognizer has its embedded dictionary.
    for load in (engine._load_det_model, engine._load_cls_model, engine._load_rec_model):
        if load().session.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Only the CPU execution provider is permitted")
    return engine


def serialize_result(result, width: int, height: int) -> bytes:
    """Represent each real detection box as one TSV word; never estimate per-character positions."""
    import numpy as np
    rows = [HEADER]
    if result.boxes is not None:
        if not (len(result.boxes) == len(result.txts) == len(result.scores)) or len(result.boxes) > 100_000:
            raise ValueError("Invalid OCR output")
        for index, (box, text, score) in enumerate(zip(result.boxes, result.txts, result.scores), 1):
            points = np.asarray(box)
            if points.shape != (4, 2) or not np.isfinite(points).all() or not math.isfinite(float(score)):
                raise ValueError("Invalid OCR geometry")
            x = max(0, math.floor(float(points[:, 0].min())))
            y = max(0, math.floor(float(points[:, 1].min())))
            right = min(width, math.ceil(float(points[:, 0].max())))
            bottom = min(height, math.ceil(float(points[:, 1].max())))
            text = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
            if not text or right <= x or bottom <= y:
                continue
            confidence = max(0.0, min(100.0, float(score) * 100.0))
            rows.append(f"5\t1\t{index}\t1\t1\t1\t{x}\t{y}\t{right-x}\t{bottom-y}\t{confidence:.5f}\t{text}\n")
    payload = "".join(rows).encode("utf-8")
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("OCR output exceeds limit")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Offline local CPU OCR")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--self-test", action="store_true", help="Verify models and all CPU sessions without an image")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--models", required=True)
    args = parser.parse_args()
    root = Path(args.models)
    if args.self_test:
        if args.input or args.output:
            raise ValueError("Self-test does not accept image or output paths")
        create_engine(root)
        print("Local CPU OCR models and dependencies verified.")
        return
    if not args.input or not args.output:
        raise ValueError("Input and output paths are required")
    source, output = Path(args.input), Path(args.output)
    if not source.is_absolute() or not output.is_absolute() or source.resolve() == output.resolve():
        raise ValueError("Separate absolute input and output paths are required")
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_OWNED_PNG_BYTES:
        raise ValueError("Invalid input size")
    if not output.parent.is_dir():
        raise ValueError("Output directory does not exist")
    from PIL import Image
    with Image.open(source) as image:
        width, height = image.size
        if image.format != "PNG" or width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError("Invalid PNG dimensions")
    engine = create_engine(root)
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError("Cannot decode input PNG")
    result = engine(image)
    serialize_result(result, width, height)  # Validate first-pass vectors before retrying regions.
    payload = serialize_result(reread_symbols(engine, image, result), width, height)
    temporary = output.with_name(output.name + ".partial")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Local AI OCR could not complete. Verify packaged models and input image.", file=sys.stderr)
        raise SystemExit(1)
