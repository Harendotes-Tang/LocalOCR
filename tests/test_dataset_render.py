"""Regression checks for original grouping and real licensed Unicode shaping."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.OcrDatasets.corpus import digest, originals
from tools.OcrDatasets.generate import allocate
from tools.OcrDatasets.render import Renderer, Unsupported

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / 'artifacts/ocr-data-fonts-v1/fonts.json'


def test_original_exact_unicode_and_exclusions():
    groups = originals(731942, 500)
    labels = [item['label'] for item in groups]
    assert len(set(labels)) == len(labels)
    assert any('\u0301' in label and '  ' in label for label in labels)
    assert any('───' in label for label in labels)
    assert any('\u200d' in label for label in labels)
    excluded = {groups[0]['label_sha256'], groups[1]['label_sha256']}
    new = originals(731942, 500, excluded)
    assert not excluded & {item['label_sha256'] for item in new}
    assert all(item['label_sha256'] == digest(item['label']) for item in new)
    assert groups == originals(731942, 500)


def test_all_original_variants_owned_by_one_split():
    splits = allocate(originals(731942, 500))
    owned = {}
    for split, values in splits.items():
        assert values
        for value in values:
            assert value['group'] not in owned
            owned[value['group']] = split
    assert len(owned) == 500
    assert all(any(item['kind'] == 'emoji' for item in values) for values in splits.values())


@pytest.fixture
def renderer():
    if not FONTS.exists():
        pytest.skip('Preparation-only official font profile unavailable')
    return Renderer(FONTS)


def test_real_combining_spaces_and_pixel_bounds(renderer):
    image, region, fonts, evidence = renderer.render('晴川 cafe\u0301  2037 ───')
    assert fonts and evidence['glyphs'] > 5
    x, y, width, height = region
    assert 0 <= x < x + width <= image.width
    assert 0 <= y < y + height <= image.height
    same = renderer.render('晴川 cafe\u0301  2037 ───')[0]
    assert image.tobytes() == same.tobytes()
    assert renderer.render('A  B')[0].width > renderer.render('A B')[0].width


@pytest.mark.parametrize('label', ['👩🏽\u200d🚀', '❤️', '👍🏽'])
def test_real_emoji_composes_and_has_color(renderer, label):
    image, region, fonts, evidence = renderer.render(label, emoji=True)
    assert evidence['glyphs'] == 1
    assert evidence['colored_glyphs'] == 1
    pixels = np.asarray(image)
    assert np.any((pixels[:, :, 0] != pixels[:, :, 1]) & (pixels[:, :, 1] != pixels[:, :, 2]))


def test_missing_glyph_never_becomes_labelled_box(renderer):
    with pytest.raises(Unsupported, match='missing-cluster'):
        renderer.render('A\U0010ffff')


def test_font_hash_tampering_rejected(tmp_path):
    fake = tmp_path / 'fake.otf'
    fake.write_bytes(b'changed font')
    license_file = tmp_path / 'LICENSE'
    license_file.write_text('SIL OPEN FONT LICENSE', encoding='utf-8')
    manifest = {'fonts': [{'id': 'fake', 'file': 'fake.otf', 'sha256': '0' * 64, 'license_file': 'LICENSE', 'license_sha256': hashlib.sha256(license_file.read_bytes()).hexdigest(), 'license': 'OFL-1.1'}]}
    path = tmp_path / 'fonts.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        Renderer(path)


def test_uncolored_family_excluded(renderer):
    with pytest.raises(Unsupported, match='missing-color'):
        renderer.render('👨\u200d👩\u200d👧', emoji=True)
