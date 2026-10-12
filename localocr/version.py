"""Product and baseline identities, independent of protocol and dependency versions."""

SOFTWARE_VERSION = "0.1.0-rc.2"
BASELINE_MODEL_ID = "text-ppocrv6-1.0.0"
VERSION_TEXT = f"LocalOCR {SOFTWARE_VERSION}"


def artifact_names():
    """Return local delivery basenames without changing historical artifacts."""
    runtime = f"localocr-runtime-{SOFTWARE_VERSION}-windows-x64"
    return {
        "runtime_dir": runtime,
        "runtime_zip": runtime + ".zip",
        "source_zip": f"localocr-source-{SOFTWARE_VERSION}.zip",
        "model_zip": f"localocr-model-{BASELINE_MODEL_ID}.zip",
        "inventory": runtime + "-inventory.json",
        "delivery": f"localocr-delivery-{SOFTWARE_VERSION}.json",
        "build_dir": f"runtime-build-{SOFTWARE_VERSION}",
    }
