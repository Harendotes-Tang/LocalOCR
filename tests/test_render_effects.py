"""Declared rendering effects: validation, ink union and deterministic texture."""
from pathlib import Path

import numpy as np
import pytest

from tools.OcrDatasets.render import compose_effects, describe_effects, validate_effects
from PIL import Image

FONTS = Path(__file__).resolve().parents[1] / 'artifacts/ocr-data-fonts-v1/fonts.json'


def renderer():
    if not FONTS.exists():
        pytest.skip('Pinned local OFL fonts absent')
    from tools.PretrainedOcr.data import MixedRenderer
    return MixedRenderer(FONTS)


@pytest.mark.parametrize('bad', [
    {'outline': 4}, {'outline': True}, {'shadow': {'dx': 0, 'dy': 0}}, {'shadow': {'dx': 5, 'dy': 1}},
    {'shadow': {'dx': 1, 'dy': 1, 'alpha': 0}}, {'texture': {'amplitude': 0, 'seed': 1}}, {'texture': {'amplitude': 4}},
    {'nope': 1}, {'ink': (300, 0, 0)}, {'background': (1, 2)}, 'outline', {'outline_color': (1, 2, 3), 'outline': 1.5},
])
def test_unbounded_or_unknown_effects_fail_closed(bad):
    with pytest.raises(ValueError):
        validate_effects(bad)


def test_effects_normalize_to_declared_shapes():
    assert validate_effects(None) == {}
    value = validate_effects({'outline': 2, 'shadow': {'dx': -1, 'dy': 2}, 'texture': {'amplitude': 3, 'seed': 9}, 'ink': [1, 2, 3]})
    assert value['outline'] == 2 and value['outline_color'] == (0, 0, 0)
    assert value['shadow'] == {'dx': -1, 'dy': 2, 'color': (0, 0, 0), 'alpha': 160} and value['ink'] == (1, 2, 3)
    described = describe_effects(value)
    assert described['ink'] == [1, 2, 3] and described['shadow']['color'] == [0, 0, 0]


def test_compose_effects_unions_shadow_and_outline_ink():
    canvas = Image.new('RGBA', (40, 20), (0, 0, 0, 0))
    canvas.paste((25, 25, 25, 255), (10, 5, 20, 15))
    layer, union = compose_effects(canvas, validate_effects({'outline': 1, 'shadow': {'dx': 3, 'dy': 3, 'alpha': 255}}))
    text_box = canvas.getchannel('A').getbbox()
    assert union.getbbox() == (text_box[0] - 1, text_box[1] - 1, text_box[2] + 3, text_box[3] + 3)
    pixels = np.asarray(layer)
    assert tuple(pixels[10, 15][:3]) == (25, 25, 25)  # text keeps its ink on top
    assert tuple(pixels[5 + 3 + 2, 20 + 1][:3]) == (0, 0, 0) and pixels[5 + 3 + 2, 20 + 1][3] > 0  # shadow beyond the glyph


def test_default_render_is_unchanged_and_effects_extend_region():
    r = renderer()
    plain, region, fonts, evidence = r.render('(≧▽≦) river', size=24)
    assert evidence['effects'] == {} and evidence['bounds_basis'] == 'transformed-raster-alpha'
    decorated, region2, _, evidence2, alpha = r.render('(≧▽≦) river', size=24, with_alpha=True,
        effects={'outline': 1, 'shadow': {'dx': 2, 'dy': 2}, 'texture': {'amplitude': 8, 'seed': 3}, 'background': (40, 44, 52), 'ink': (230, 230, 230)})
    assert region2[2] >= region[2] and region2[3] >= region[3] and alpha.getbbox() is not None
    assert evidence2['effects']['outline'] == 1 and evidence2['bounds_basis'].endswith('with-effects')
    again, *_ = r.render('(≧▽≦) river', size=24, effects={'texture': {'amplitude': 8, 'seed': 3}})
    once_more, *_ = r.render('(≧▽≦) river', size=24, effects={'texture': {'amplitude': 8, 'seed': 3}})
    other, *_ = r.render('(≧▽≦) river', size=24, effects={'texture': {'amplitude': 8, 'seed': 4}})
    assert np.array_equal(np.asarray(again), np.asarray(once_more)) and not np.array_equal(np.asarray(again), np.asarray(other))
    background = np.asarray(decorated)[0, 0]
    assert abs(int(background[0]) - 40) <= 8
