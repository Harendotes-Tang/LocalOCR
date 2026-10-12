"""Produce immutable local delivery ZIPs from the matching audited runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
import zipfile
from localocr.version import SOFTWARE_VERSION, BASELINE_MODEL_ID, VERSION_TEXT, artifact_names
from localocr.bundles import BASELINE_MODELS, _directory, _plain, require_local_path, validate_bundle
from tools.LocalAiOcr import build_runtime


def checked(root, relative):
    """Reject nonportable/escaping paths before accessing the declared input."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Invalid delivery relative path")
    pure = PurePosixPath(relative)
    reserved = {"CON", "PRN", "AUX", "NUL", *("COM"+str(i) for i in range(1,10)), *("LPT"+str(i) for i in range(1,10))}
    if pure.is_absolute() or pure.as_posix() != relative or any(p in {".",".."} or p.endswith(("."," ")) or p.split(".")[0].upper() in reserved or any(c in p for c in '<>"|?*') for p in pure.parts):
        raise ValueError("Invalid delivery relative path")
    path = root.joinpath(*pure.parts)
    require_local_path(path)
    _directory(path.parent)
    _plain(path)
    if not path.is_file():
        raise ValueError("Delivery input must be a regular file")
    return path


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_records(root, records):
    """Bind every archive entry to distinct current audited source bytes."""
    seen=set(); result=[]
    for row in records:
        if not isinstance(row,dict) or not isinstance(row.get("path"),str) or type(row.get("bytes")) is not int or row["bytes"]<0 or not isinstance(row.get("sha256"),str) or len(row["sha256"])!=64:
            raise ValueError("Invalid delivery inventory record")
        relative=row["path"]
        if relative.casefold() in seen:
            raise ValueError("Duplicate delivery path")
        seen.add(relative.casefold())
        path=checked(root,relative)
        if path.stat().st_size!=row["bytes"] or digest(path)!=row["sha256"]:
            raise ValueError("Delivery input differs from inventory: "+relative)
        result.append((relative,path))
    return result


def audit_archive(path, expected):
    """Verify the actual ZIP member bytes, not merely its internal CRCs."""
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(expected):
            raise ValueError("Archive member inventory mismatch")
        for name,(size,sha) in expected.items():
            h=hashlib.sha256();count=0
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):
                    count+=len(block);h.update(block)
            if count!=size or h.hexdigest()!=sha:
                raise ValueError("Archive member changed during packaging: "+name)


def verify_executable_versions(runtime):
    """Require actual fixed entrypoints to agree with declared product identity."""
    for name in ("local-ai-ocr.exe","localocr-models.exe"):
        executable=checked(runtime,name)
        result=subprocess.run([str(executable),"--version"],env=build_runtime.minimal_environment(),capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30)
        if result.returncode!=0 or result.stdout!=VERSION_TEXT+"\n":
            raise ValueError("Actual executable product version mismatch: "+name)


def produce(args):
    runtime=Path(args.runtime); inventory_path=Path(args.inventory); models=Path(args.model_package)
    output=Path(args.output_dir);notices=Path(args.notices)
    for directory in (runtime,models,output,notices):
        if not directory.is_absolute():raise ValueError("Use absolute local directories")
        require_local_path(directory);_directory(directory)
    if not inventory_path.is_absolute():raise ValueError("Use absolute inventory path")
    require_local_path(inventory_path);_directory(inventory_path.parent);_plain(inventory_path)
    if inventory_path.stat().st_size>8*1024*1024:raise ValueError("Inventory too large")
    inventory_bytes=inventory_path.read_bytes()
    inventory=json.loads(inventory_bytes.decode("utf-8"))
    names=artifact_names()
    if inventory.get("schema_version")!="localocr.runtime-inventory.v1" or inventory.get("software_version")!=SOFTWARE_VERSION or inventory.get("artifact_names")!=names:
        raise ValueError("Runtime inventory product version/names do not match producer")
    runtime_files=verify_records(runtime,inventory["files"])
    if {p.relative_to(runtime).as_posix() for p in build_runtime.audit_payload(runtime)}!={r for r,_ in runtime_files}:
        raise ValueError("Runtime contains undeclared files")
    verify_executable_versions(runtime)
    current=build_runtime.source_snapshot()
    if current["files"]!=inventory["source_snapshot"]["files"]:
        raise ValueError("Source changed since runtime freeze")
    sources=verify_records(build_runtime.REPO,current["files"])
    bundle=validate_bundle(models)
    bundle_manifest_bytes=json.dumps(bundle.manifest,ensure_ascii=False,indent=2).encode("utf-8")
    if bundle.version!=BASELINE_MODEL_ID:raise ValueError("Expected normalized baseline identity")
    for role,(_,size,sha) in BASELINE_MODELS.items():
        row=bundle.manifest["models"][role]
        if row["size"]!=size or row["sha256"]!=sha:raise ValueError("Baseline weights changed")
    prepared=[]
    for name,sha in [("flatbuffers-25.12.19-LICENSE.txt",build_runtime.FLATBUFFERS_LICENSE_SHA256),("Microsoft-Visual-C-V14-2026-LICENSE.txt",build_runtime.MSVC_LICENSE_SHA256),("Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx",build_runtime.MSVC_LICENSE_DOCX_SHA256)]:
        path=checked(notices,name)
        if digest(path)!=sha:raise ValueError("Prepared notice hash mismatch")
        prepared.append((name,path,sha,path.stat().st_size))
    for name in (names["runtime_zip"],names["source_zip"],names["model_zip"],names["delivery"]):
        destination=output/name
        try:destination.lstat()
        except FileNotFoundError:pass
        else:raise ValueError("Delivery already exists: "+name)
    owned=[]
    try:
        runtime_zip=output/names["runtime_zip"]
        with runtime_zip.open("xb") as stream:
            owned.append(runtime_zip)
            with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for relative,path in runtime_files:archive.write(path,names["runtime_dir"]+"/"+relative)
                archive.writestr(names["inventory"],inventory_bytes)
        source_zip=output/names["source_zip"]
        with source_zip.open("xb") as stream:
            owned.append(source_zip)
            with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for relative,path in sources:archive.write(path,relative)
                archive.writestr("provenance/source-snapshot.json",json.dumps({"schema_version":"localocr.source-snapshot.v1","source_snapshot":inventory["source_snapshot"]},ensure_ascii=False,indent=2).encode("utf-8"))
                for name,path,_,_ in prepared:archive.write(path,"build-notices/"+name)
                archive.writestr("BUILDING.txt",("LocalOCR "+SOFTWARE_VERSION+" source. Prepare CPython3.11.9 Windows x64 + tools/LocalAiOcr/requirements-build.txt independently.\nFrom extracted root: python -m tools.LocalAiOcr.build_runtime --output <new absolute output> --build-dir <new absolute work> --notices <absolute extracted build-notices>.\nNo private .git required: archive-snapshot provenance validates source hashes; no enclosing Git. Native runtime/library hashes must match pins. No downloads during build.\n").encode("utf-8"))
        model_zip=output/names["model_zip"]
        with model_zip.open("xb") as stream:
            owned.append(model_zip)
            with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                archive.writestr(BASELINE_MODEL_ID+"/manifest.json",bundle_manifest_bytes)
                for row in [*bundle.manifest["models"].values(),*bundle.manifest["licenses"]]:archive.write(models/row["file"],BASELINE_MODEL_ID+"/"+row["file"])
        runtime_expected={names["runtime_dir"]+"/"+r["path"]:(r["bytes"],r["sha256"]) for r in inventory["files"]}
        runtime_expected[names["inventory"]]=(len(inventory_bytes),hashlib.sha256(inventory_bytes).hexdigest())
        audit_archive(runtime_zip,runtime_expected)
        source_expected={r["path"]:(r["bytes"],r["sha256"]) for r in current["files"]}
        for name,path,sha,size in prepared:source_expected["build-notices/"+name]=(size,sha)
        # Small generated provenance/build instructions are bound to the object
        # already checked against the build, independently of live disk sources.
        with zipfile.ZipFile(source_zip) as archive:
            provenance=json.loads(archive.read("provenance/source-snapshot.json"))
            if provenance!={"schema_version":"localocr.source-snapshot.v1","source_snapshot":inventory["source_snapshot"]}:
                raise ValueError("Archive provenance mismatch")
            for name in ("provenance/source-snapshot.json","BUILDING.txt"):
                data=archive.read(name);source_expected[name]=(len(data),hashlib.sha256(data).hexdigest())
        audit_archive(source_zip,source_expected)
        model_expected={BASELINE_MODEL_ID+"/"+r["file"]:(r["size"],r["sha256"]) for r in [*bundle.manifest["models"].values(),*bundle.manifest["licenses"]]}
        model_expected[BASELINE_MODEL_ID+"/manifest.json"]=(len(bundle_manifest_bytes),hashlib.sha256(bundle_manifest_bytes).hexdigest())
        audit_archive(model_zip,model_expected)
        products=[]
        for role,path in [("runtime",runtime_zip),("source",source_zip),("models",model_zip)]:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:raise ValueError("ZIP integrity failed")
            products.append({"role":role,"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)})
        delivery={"schema_version":"localocr.delivery.v1","software_version":SOFTWARE_VERSION,"baseline_model_id":BASELINE_MODEL_ID,"artifact_names":names,"products":products,"inventory":str(inventory_path),"historical_delivery":"runtime-delivery-v2.json preserved; source-v1 incomplete and superseded","result_schema":"localocr.result.v1","activation_policy":"Packaging validates files but does not install/activate/migrate models"}
        destination=output/names["delivery"]
        with destination.open("x",encoding="utf-8") as stream:
            owned.append(destination);json.dump(delivery,stream,ensure_ascii=False,indent=2)
        return delivery
    except BaseException:
        for path in reversed(owned):path.unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("runtime","inventory","model-package","notices","output-dir"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    try:print(json.dumps(produce(args),ensure_ascii=False))
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:parser.exit(1,"Local delivery failed: "+str(error)+"\n")


if __name__=="__main__":main()
