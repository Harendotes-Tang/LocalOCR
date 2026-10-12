"""Delivery guards bind actual ZIP bytes to the matching product inventory."""
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import zipfile
import pytest
from tools.LocalAiOcr import package_runtime as package


@pytest.mark.parametrize("relative",["../outside", "a/../outside", "/outside", "a\\outside", "C:outside", "a//b", "a/./b", "CON.txt", "a/alias. "])
def test_path_rejection_precedes_input_access(tmp_path,relative):
    with patch.object(package,"_directory") as directory:
        with pytest.raises(ValueError):package.checked(tmp_path,relative)
        directory.assert_not_called()


def test_inventory_changed_and_case_alias_rejected(tmp_path):
    p=tmp_path/"input.txt";p.write_bytes(b"original")
    record={"path":"input.txt","bytes":8,"sha256":hashlib.sha256(b"original").hexdigest()}
    assert package.verify_records(tmp_path,[record])==[("input.txt",p)]
    with pytest.raises(ValueError,match="Duplicate"):package.verify_records(tmp_path,[record,{**record,"path":"INPUT.TXT"}])
    p.write_bytes(b"modified")
    with pytest.raises(ValueError,match="differs"):package.verify_records(tmp_path,[record])
    with pytest.raises(ValueError,match="Invalid"):package.verify_records(tmp_path,[{**record,"path":None}])


def test_zip_crc_is_not_inventory_identity(tmp_path):
    p=tmp_path/"valid.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr("data",b"wrong but CRC-valid")
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
    with pytest.raises(ValueError,match="changed during"):package.audit_archive(p,{"data":(5,hashlib.sha256(b"right").hexdigest())})
    with pytest.raises(ValueError,match="inventory mismatch"):package.audit_archive(p,{})


def test_old_runtime_identity_cannot_be_renamed_into_current_delivery(tmp_path):
    folders=[tmp_path/n for n in ("runtime","models","output","notices")]
    for p in folders:p.mkdir()
    inv=tmp_path/"inventory.json";inv.write_text(json.dumps({"schema_version":"localocr.runtime-inventory.v1","software_version":"old"}),encoding="utf-8")
    args=SimpleNamespace(runtime=folders[0],model_package=folders[1],output_dir=folders[2],notices=folders[3],inventory=inv)
    with patch.object(package,"validate_bundle") as validate:
        with pytest.raises(ValueError,match="product version"):package.produce(args)
        validate.assert_not_called()
    assert not list(folders[2].iterdir())


def test_actual_old_executable_cannot_be_hidden_by_current_inventory(tmp_path):
    for name in ("local-ai-ocr.exe","localocr-models.exe"):(tmp_path/name).write_bytes(b"fixture executable; not inference proof")
    result=SimpleNamespace(returncode=0,stdout="LocalOCR standalone CPU runtime 1; localocr.result.v1\n")
    with patch.object(package.subprocess,"run",return_value=result) as run:
        with pytest.raises(ValueError,match="Actual executable product version mismatch"):
            package.verify_executable_versions(tmp_path)
        assert run.call_args.args[0]==[str(tmp_path/"local-ai-ocr.exe"),"--version"]
        assert run.call_args.kwargs["timeout"]==30
