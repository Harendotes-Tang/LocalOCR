"""Run the loopback service; updates are a separate local CLI."""
import argparse
from pathlib import Path
from .version import VERSION_TEXT

def main():
    parser = argparse.ArgumentParser(description='Local CPU OCR API')
    parser.add_argument('--version', action='version', version=VERSION_TEXT)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    import uvicorn
    from localocr.service import create_app
    uvicorn.run(create_app(args.store), host='127.0.0.1', port=args.port, workers=1,
                access_log=False, timeout_graceful_shutdown=1)

if __name__ == '__main__':
    main()
