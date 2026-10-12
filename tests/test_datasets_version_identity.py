"""New report display identities retain frozen historical experiment authority."""
import json
from pathlib import Path
import pytest
from tools.OcrDatasets import evaluate

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_HASHES={
    'model_sha256':'d2bd044deaf4b83fdbb3a7c5eaa23c52cd6378e68d4901624601ae2cad03b54f',
    'alphabet_sha256':'1163a17b67609572b1e059d971be984edd051ec049994f03e681dc2c2af639ef',
    'training_sha256':'c24e1e503000be3a49e4628ff60b749541f783a8ca2934348dd2855d09c825e3',
}

def test_display_authority_preserves_historical_hashes_and_schema():
    authority=evaluate.frozen_candidate_authority()
    assert authority['schema_version']=='localocr.frozen-candidate.v1'
    assert authority['identity']=='rejected-v4'
    assert {key:authority[key] for key in EXPECTED_HASHES}==EXPECTED_HASHES
    assert authority['display_name']=='kaomoji-exp-20261011-004'
    assert authority['historical_marker']=='v4'


def test_new_report_projection_is_explicitly_unenabled_experiment():
    report=json.loads(json.dumps(evaluate.frozen_candidate_metadata()))
    assert report['candidate_display_name']=='kaomoji-exp-20261011-004'
    assert report['candidate_identity']=='rejected-v4'
    assert report['candidate_historical_marker']=='v4'
    assert report['candidate_identity_kind']=='experiment'
    assert report['candidate_enabled'] is False and report['promote'] is False
    assert report['candidate_status']=='rejected; not enabled; CPU comparison reference only'
    assert 'software_version' not in report and 'model_version' not in report


@pytest.mark.parametrize('changed',[
    {'enabled':True},{'status':'accepted'},{'identity_kind':'released-model'},
    {'display_name':'1.0.0'},{'historical_marker':'v5'},
])
def test_authority_cannot_turn_display_change_into_promotion(monkeypatch,changed):
    authority=evaluate.frozen_candidate_authority();authority.update(changed)
    monkeypatch.setattr(evaluate,'read',lambda path:authority)
    with pytest.raises(ValueError,match='rejected, unenabled identity'):
        evaluate.frozen_candidate_metadata()


def test_self_consistent_unapproved_weights_remain_rejected(tmp_path):
    import hashlib
    model=b'unapproved model with its own truthful report';alphabet=json.dumps({'alphabet':['x']}).encode()
    (tmp_path/'kaomoji.onnx').write_bytes(model);(tmp_path/'alphabet.json').write_bytes(alphabet)
    (tmp_path/'training.json').write_text(json.dumps({'model_sha256':hashlib.sha256(model).hexdigest(),
        'alphabet_sha256':hashlib.sha256(alphabet).hexdigest()}),encoding='utf-8')
    with pytest.raises(ValueError,match='not the approved frozen rejected v4'):
        evaluate.frozen_candidate_binding(tmp_path)
