import hashlib
import json
import pytest
from tools.OcrDatasets.evaluate import frozen_candidate_binding

def test_self_consistent_other_candidate_is_rejected(tmp_path):
    model=b'other internally consistent model'
    alphabet=json.dumps({'alphabet':['x']}).encode()
    (tmp_path/'kaomoji.onnx').write_bytes(model)
    (tmp_path/'alphabet.json').write_bytes(alphabet)
    (tmp_path/'training.json').write_text(json.dumps({'model_sha256':hashlib.sha256(model).hexdigest(),
        'alphabet_sha256':hashlib.sha256(alphabet).hexdigest()}),encoding='utf-8')
    with pytest.raises(ValueError,match='not the approved frozen rejected v4'):
        frozen_candidate_binding(tmp_path)
