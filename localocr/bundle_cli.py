"""Manual offline directory bundle management; no HTTP administration endpoint."""
import argparse
import json
from pathlib import Path
from .version import VERSION_TEXT
from .bundles import activate_bundle, create_baseline, install_bundle, load_active, validate_bundle, require_local_path, _directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION_TEXT)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-baseline")
    create.add_argument("--models", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--version", required=True)
    install = commands.add_parser("install")
    install.add_argument("--source", required=True)
    install.add_argument("--store", required=True)
    activate = commands.add_parser("activate", help="Activate or roll back to a validated installed version")
    activate.add_argument("--store", required=True)
    activate.add_argument("--version", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--store", required=True)
    args = parser.parse_args()
    try:
        if args.command == "create-baseline":
            bundle = create_baseline(args.models, args.output, args.version)
        elif args.command == "install":
            bundle = install_bundle(args.source, args.store)
        elif args.command == "activate":
            bundle = activate_bundle(args.store, args.version)
        else:
            store = Path(args.store).absolute()
            require_local_path(store)
            _directory(store / "versions")
            versions = [validate_bundle(path).version for path in sorted((store / "versions").iterdir())]
            try:
                active = load_active(store).version
            except FileNotFoundError:
                active = None
            print(json.dumps({"versions": versions, "active": active}))
            return
        print(json.dumps({"version": bundle.version, "root": str(bundle.root)}))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Bundle operation failed: {error}\n")


if __name__ == "__main__":
    main()
