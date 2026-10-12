import json
from tools.OcrDatasets.reproduce import reproduce
from tools.OcrDatasets.render import file_hash
import pytest

def test_replay_rejects_tampered_original_before_rendering(tmp_path):
    package=tmp_path/'package';package.mkdir()
    (package/'changed.png').write_bytes(b'changed original')
    (package/'regression.json').write_text(json.dumps([{'path':'changed.png','sha256':'0'*64}]),encoding='utf-8')
    (package/'dataset.json').write_text(json.dumps({'license':'CC0-1.0','splits':{'regression':{'sha256':file_hash(package/'regression.json'),'count':1}}}),encoding='utf-8')
    with pytest.raises(ValueError,match='Public regression image changed'):
        reproduce(package,tmp_path/'not-needed-fonts.json',tmp_path/'output')
