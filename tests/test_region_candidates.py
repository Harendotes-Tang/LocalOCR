"""Deterministic geometry fixtures; these tests do not measure OCR accuracy."""
import copy
import numpy as np
from PIL import Image, ImageDraw
import pytest
from tools.RegionKaomoji import candidates as c


def face_fixture():
    image=Image.new('RGB',(160,80),'white');draw=ImageDraw.Draw(image)
    # Original geometric strokes used only to exercise evidence guards.
    draw.arc((20,20,34,50),90,270,fill='black',width=2)
    draw.arc((88,20,102,50),270,90,fill='black',width=2)
    draw.ellipse((43,31,50,38),outline='black',width=2)
    draw.ellipse((72,31,79,38),outline='black',width=2)
    draw.line((54,43,68,43),fill='black',width=2)
    return np.asarray(image)


def test_image_branch_finds_face_when_first_pass_empty():
    output=c.locate_candidates(face_fixture(),[])
    assert len(output['candidates'])==1
    candidate=output['candidates'][0]
    assert set(candidate['sources'])=={'image-components','image-bracket-pair'}
    assert candidate['baseline_indices']==[]
    assert candidate['evidence']['pixel_support']
    assert candidate['evidence']['extra_shape_support']
    assert not candidate['evidence']['edge_ink']
    assert not candidate['evidence']['neighbor_body_overlap']
    left,top,right,bottom=candidate['bbox']
    assert left<=20 and top<=20 and right>=102 and bottom>=51


def test_actual_baseline_geometry_merges_ocr_and_pixel_sources():
    baseline=[{'bbox':[20,20,103,51],'text':'(Д_Д)','score':.01}]
    output=c.locate_candidates(face_fixture(),baseline)
    assert len(output['candidates'])==1
    candidate=output['candidates'][0]
    assert set(candidate['sources'])=={'image-components','image-bracket-pair','first-pass-symbol-geometry'}
    assert candidate['baseline_indices']==[0]
    assert candidate['baseline_text']=='(Д_Д)'


def test_full_line_body_box_is_not_substring_coordinate():
    baseline=[{'bbox':[0,10,150,60],'text':'正文 (Д_Д) 123','score':.9}]
    image=Image.fromarray(face_fixture());ImageDraw.Draw(image).rectangle((130,25,140,42),fill='black')
    candidate=next(item for item in c.locate_candidates(image,baseline)['candidates'] if 'image-components' in item['sources'])
    assert candidate['baseline_indices']==[]
    assert candidate['evidence']['neighbor_body_overlap']
    assert candidate['evidence']['neighbor_regions'][0]['index']==0


def test_plain_brackets_are_only_routing_not_face_confirmation():
    assert not c.is_face_text('(ABC)')
    assert not c.is_face_text('(o)')
    assert not c.is_face_text('订单 (123)')
    assert c.is_face_text('(Д_Д)')
    assert c.is_face_text('(ω  _  ω)')


def test_blank_and_single_shape_have_no_candidates():
    blank=np.full((80,160,3),240,dtype=np.uint8)
    assert not c.locate_candidates(blank,[])['candidates']
    graphic=Image.fromarray(blank);ImageDraw.Draw(graphic).ellipse((20,20,110,65),outline='black',width=3)
    assert not c.locate_candidates(graphic,[])['candidates']


def test_candidate_budget_and_halfopen_limits():
    config=copy.deepcopy(c.DEFAULT_CONFIG);config['maximum_crop_pixels']=100
    assert not c.locate_candidates(face_fixture(),[],config)['candidates']
    for box in ([0,0,161,20],[-1,0,10,20],[10,5,10,15],[0.,0,10,20]):
        with pytest.raises(ValueError):
            c.locate_candidates(face_fixture(),[{'bbox':box,'text':'()','score':.5}])


def test_nonfinite_scores_and_unsafe_configs_rejected():
    with pytest.raises(ValueError):c.locate_candidates(face_fixture(),[{'bbox':[20,20,103,51],'text':'()','score':float('nan')}])
    config=dict(c.DEFAULT_CONFIG);config['maximum_candidates']=49
    with pytest.raises(ValueError):c.locate_candidates(face_fixture(),[],config)
    config=dict(c.DEFAULT_CONFIG);config['unknown']=1
    with pytest.raises(ValueError):c.configuration(config)
    config=dict(c.DEFAULT_CONFIG);config['padding']=False
    with pytest.raises(ValueError):c.configuration(config)


def test_rgb_contract_and_canvas_digest():
    with pytest.raises(ValueError):c.locate_candidates(np.zeros((5,5),np.uint8),[])
    with pytest.raises(ValueError):c.locate_candidates(np.zeros((5,5,3),np.float32),[])
    image=face_fixture()
    assert c.image_digest(image)==c.image_digest(Image.fromarray(image))
    resized=Image.fromarray(image).resize((80,40))
    assert c.image_digest(image)!=c.image_digest(resized)


def test_discovery_and_route_eligibility_are_distinct():
    image=face_fixture()
    ordinary=[{'bbox':[20,20,103,51],'text':'订单 (ABC) 123','score':.1}]
    candidates=c.locate_candidates(image,ordinary)['candidates']
    assert candidates
    assert not any(item['route_eligible'] for item in candidates)
    assert any(item['route_reason']=='observed-ordinary-body-anchors' for item in candidates)
    assert c.locate_candidates(image,[])['route_eligible_count']==1
    ordinary=[{'bbox':[20,20,103,51],'text':'(o)','score':.1}]
    assert c.locate_candidates(image,ordinary)['route_eligible_count']==0


def test_bracket_pair_source_bridges_internal_pixel_gaps():
    image=Image.new('RGB',(240,90),'white');draw=ImageDraw.Draw(image)
    draw.arc((20,20,34,50),90,270,fill='black',width=2)
    draw.arc((180,20,194,50),270,90,fill='black',width=2)
    draw.ellipse((47,31,54,38),outline='black',width=2)
    draw.ellipse((155,31,162,38),outline='black',width=2)
    draw.line((90,43,112,43),fill='black',width=2)
    candidates=c.locate_candidates(image,[])['candidates']
    complete=[item for item in candidates if item['bbox'][0]<=20 and item['bbox'][2]>=194]
    assert complete and any('image-bracket-pair' in item['sources'] for item in complete)
    assert any(item['route_eligible'] for item in complete)


def test_observed_ink_coverage_can_ignore_ocr_background_padding():
    baseline=[{'bbox':[0,10,150,60],'text':'(Д_Д)','score':.5}]
    candidate=next(item for item in c.locate_candidates(face_fixture(),baseline)['candidates'] if 'image-components' in item['sources'])
    assert candidate['baseline_indices']==[0]
    assert candidate['evidence']['baseline_pixel_coverage'][0]['box_coverage']<.8
    assert candidate['evidence']['baseline_pixel_coverage'][0]['ink_coverage']==1.
    assert candidate['route_eligible']


def test_face_grammar_does_not_absorb_independent_tokens_or_expressions():
    for text in ('o  (Д_Д)', '(Д_Д) o', '(o Д_Д)', '(Д_Д)(ω_ω)',
                 '(Д_Д) (ω_ω)', '(Д_Д', 'Д_Д)', 'Д_Д', '(oo_oo)', '(o)'):
        assert not c.is_face_text(text), text
    for text in ('(Д_Д)', ' (ω  _  ω) ', '(^_^)', 'つ(Д_Д)つ'):
        assert c.is_face_text(text), text
    assert not c.is_face_text('つ(Д_Д)つ', allow_arms=False)



def test_parent_digest_preserves_raw_unicode_geometry_and_index():
    parent=c.baseline_regions([{'bbox':[10,10,100,50],
                               'text':'BODY e\u0301  (Д_Д)  123','score':.4}],(160,80))[0]
    frozen=c.baseline_parent_digest(parent)
    assert frozen==c.baseline_parent_digest(copy.deepcopy(parent))
    for field,value in [('text','BODY é  (Д_Д)  123'),('text','BODY e\u0301 (Д_Д)  123'),
                        ('text','BODY e\u0301  (ω_ω)  123'),('bbox',[11,10,100,50]),('index',1)]:
        changed=copy.deepcopy(parent);changed[field]=value
        assert c.baseline_parent_digest(changed)!=frozen


def test_mixed_candidate_binds_actual_parent_raw_text():
    image=Image.new('RGB',(360,80),'white');image.paste(Image.fromarray(face_fixture()),(100,0))
    draw=ImageDraw.Draw(image);draw.rectangle((15,23,55,46),fill='black');draw.rectangle((245,23,285,46),fill='black')
    baseline=[{'bbox':[10,10,291,61],'text':'LEFT  (Д_Д)  RIGHT','score':.4}]
    mixed=next(item for item in c.locate_candidates(image,baseline)['candidates'] if item['splice_baseline_index'] is not None)
    assert mixed['baseline_parent_sha256']==c.baseline_parent_digest(c.baseline_regions(baseline,image.size)[0])
    baseline[0]['text']='LEFT  (^_^)  RIGHT'
    changed=next(item for item in c.locate_candidates(image,baseline)['candidates'] if item['candidate_id']==mixed['candidate_id'])
    assert changed['bbox']==mixed['bbox'] and changed['context_crops']==mixed['context_crops']
    assert changed['baseline_parent_sha256']!=mixed['baseline_parent_sha256']
