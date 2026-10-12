"""Owned CPU image decoder, terminated by its API owner on cancellation."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys
import warnings
from PIL import Image, UnidentifiedImageError

PIXEL_LIMIT = 50_000_000
IMAGE_LIMIT = 20 * 1024 * 1024

class ImageError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message

def save_png(raw, expected_format, target):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != expected_format:
                    raise ImageError(400, 'Image format does not match data URI')
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > PIXEL_LIMIT:
                    raise ImageError(413, 'Image exceeds 50 million pixels')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ImageError(400, 'Animated or multi-frame images are unsupported')
                image.load()
                image.convert('RGB').save(target, format='PNG')
                return width, height
    except ImageError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageError(413, 'Image exceeds decompression budget') from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageError(400, 'Malformed image') from None

def deny_network(event, args):
    if (event.startswith('socket.') and event != 'socket.gethostname') or event in {'os.system', 'subprocess.Popen'}:
        raise PermissionError('Image decoder networking and process execution are disabled')

def main():
    sys.addaudithook(deny_network)
    parser = argparse.ArgumentParser(description='Owned offline image preparation')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--format', required=True, choices=('PNG', 'JPEG'))
    args = parser.parse_args()
    try:
        if args.input.stat().st_size > IMAGE_LIMIT:
            raise ImageError(413, 'Decoded image exceeds 20 MiB')
        width, height = save_png(args.input.read_bytes(), args.format, args.output)
        result = {'width': width, 'height': height}
        code = 0
    except ImageError as error:
        result = {'error': {'status': error.status, 'message': error.message}}
        code = 1
    args.metadata.write_text(json.dumps(result), encoding='utf-8')
    return code

if __name__ == '__main__':
    raise SystemExit(main())
