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



def angular_face_fixture(height=24, stroke=1, style='square'):
    """Original pixel controls; no font/model or expected-coordinate input."""
    top=10; left=10; right=left+height*2; bottom=top+height-1
    image=Image.new('RGB',(right+20,bottom+20),'white'); draw=ImageDraw.Draw(image)
    cap=max(3,height//4)
    if style=='square':
        draw.line([(left+cap-1,top),(left,top),(left,bottom),(left+cap-1,bottom)],fill='black',width=stroke)
        draw.line([(right-cap+1,top),(right,top),(right,bottom),(right-cap+1,bottom)],fill='black',width=stroke)
    elif style=='small-curve':
        points=[(left+(1 if y in (top,bottom) else 0),y) for y in range(top,bottom+1)]
        draw.line(points,fill='black',width=1)
        draw.line([(right-(x-left),y) for x,y in points],fill='black',width=1)
    elif style=='brace':
        mid=(top+bottom)//2
        draw.line([(left+cap-1,top),(left+2,top+2),(left+2,mid-3),(left,mid),(left+2,mid+3),(left+2,bottom-2),(left+cap-1,bottom)],fill='black',width=stroke)
        draw.line([(right-(x-left),y) for x,y in [(left+cap-1,top),(left+2,top+2),(left+2,mid-3),(left,mid),(left+2,mid+3),(left+2,bottom-2),(left+cap-1,bottom)]],fill='black',width=stroke)
    else:
        raise AssertionError('unknown control style')
    eye_y=top+max(2,height//3)
    for x in (left+height*2//3,left+height*4//3):
        draw.rectangle((x,eye_y,x+1,eye_y+1),fill='black')
    draw.line((left+height-1,top+height-3,left+height+1,top+height-3),fill='black',width=1)
    return np.asarray(image)


@pytest.mark.parametrize('height,stroke', [(8,1),(12,1),(16,1),(24,1),(32,2)])
def test_quality_thin_right_angle_pair_has_auditable_pixel_support(height,stroke):
    image=angular_face_fixture(height,stroke)
    output=c.locate_candidates(image,[])
    qualified=[item for item in output['candidates'] if item['route_eligible']]
    assert qualified
    assert any(item['evidence']['bracket_shape_counts']['right-angle']>=1 for item in qualified)
    assert all(item['evidence']['extra_shape_support'] for item in qualified)
    pair=next(p for item in qualified for p in item['evidence']['paired_bracket_shapes'] if 'right-angle' in p['shape_kinds'])
    assert pair['left_shape']['measurements']['stem_coverage']>=.85
    assert pair['right_shape']['measurements']['top_cap_extension']>=2
    assert pair['right_shape']['measurements']['bottom_cap_extension']>=2
    assert c.DEFAULT_CONFIG['minimum_height']==7


def test_quality_small_curved_extrema_are_separate_from_global_curvature():
    output=c.locate_candidates(angular_face_fixture(13,style='small-curve'),[])
    selected=[item for item in output['candidates'] if item['route_eligible']]
    assert selected
    assert any(item['evidence']['bracket_shape_counts']['small-curve'] for item in selected)
    assert all(item['evidence']['bracket_shape_counts']['right-angle']==0 for item in selected)


def test_quality_old_curve_evidence_and_brace_strokes_remain_observed():
    for image in (face_fixture(),angular_face_fixture(24,style='brace')):
        selected=[item for item in c.locate_candidates(image,[])['candidates'] if item['route_eligible']]
        assert selected
        assert any(item['evidence']['bracket_shape_counts']['curve'] for item in selected)


@pytest.mark.parametrize('pair', c.BRACKETS)
def test_quality_all_declared_brackets_and_numeric_mouth_are_role_limited(pair):
    left,right=pair
    assert c.is_face_text(left+'ω3ω'+right)
    assert c.is_face_text(left+'Д益Д'+right)
    for part in ('123','333','3_3','o','ABC','o  ω3ω','ω3ω  o'):
        assert not c.is_face_text(left+part+right)
    assert not c.is_face_text('3'+left+'ω3ω'+right)
    assert not c.is_face_text(left+'ω3ω'+right+'3')
    assert '3' not in c.FACE_LETTERS


@pytest.mark.parametrize('ordinary', ['[123]','[333]','[ABC]','[o]','[T]','o [ω3ω]','[ω3ω] o'])
def test_quality_right_angles_never_bypass_ordinary_body_anchor_guard(ordinary):
    image=angular_face_fixture()
    baseline=[{'bbox':[10,10,59,34],'text':ordinary,'score':.1}]
    output=c.locate_candidates(image,baseline)
    assert output['candidates']
    assert not any(item['route_eligible'] for item in output['candidates'])


def test_quality_straight_stems_without_two_caps_and_closed_frames_are_not_faces():
    image=Image.fromarray(angular_face_fixture()); draw=ImageDraw.Draw(image)
    # Remove bottom caps: two L-like stems do not provide paired closed brackets.
    draw.rectangle((11,33,15,33),fill='white');draw.rectangle((53,33,57,33),fill='white')
    evidence=c.pixel_evidence(np.asarray(image),[5,5,64,40])
    assert not evidence['extra_shape_support']
    image=Image.new('RGB',(90,55),'white'); draw=ImageDraw.Draw(image)
    draw.rectangle((10,10,58,33),outline='black',width=1)
    draw.rectangle((25,18,26,19),fill='black');draw.rectangle((42,18,43,19),fill='black')
    draw.line((32,29,35,29),fill='black')
    evidence=c.pixel_evidence(np.asarray(image),[5,5,64,40])
    assert not evidence['extra_shape_support']


def test_quality_right_angle_crop_clipping_and_tight_neighbor_keep_refusal():
    image=angular_face_fixture()
    clipped=c.locate_candidates(image[:,10:],[])
    assert not any(item['route_eligible'] for item in clipped['candidates'])
    extended=Image.fromarray(image);ImageDraw.Draw(extended).rectangle((62,16,70,30),fill='black')
    baseline=[{'bbox':[9,8,75,37],'text':'BODY [ω3ω] 123','score':.6}]
    output=c.locate_candidates(np.asarray(extended),baseline)
    assert not any(item['route_eligible'] for item in output['candidates'])


def thick_cap_mask(height, width, stem_width, cap_rows):
    """Pixel geometry reproducing the observed thick CJK square stem ties."""
    mask=np.zeros((height,width),dtype=np.uint8)
    mask[:,:stem_width]=1
    mask[:cap_rows,:]=1
    mask[-cap_rows:,:]=1
    return mask


def thick_cap_face_fixture(height, width, stem_width, cap_rows):
    mask=thick_cap_mask(height,width,stem_width,cap_rows)
    right=10+height*2
    canvas=np.full((height+20,right+width+10,3),255,dtype=np.uint8)
    for x,part in ((10,mask),(right,np.fliplr(mask))):
        canvas[10:10+height,x:x+width][part.astype(bool)]=0
    canvas[10+height//3:12+height//3,10+height*2//3:12+height*2//3]=0
    canvas[10+height//3:12+height//3,10+height*4//3:12+height*4//3]=0
    canvas[10+height-3,10+height:13+height]=0
    return canvas


@pytest.mark.parametrize('height,width,stem_width,cap_rows',
                         [(23,6,3,2),(12,3,2,1),(8,3,2,1),(32,8,3,2)])
def test_quality_thick_square_stem_ties_are_mirror_symmetric(height,width,stem_width,cap_rows):
    left=thick_cap_mask(height,width,stem_width,cap_rows)
    left_profile=c._right_angle_profile(left)
    right_profile=c._right_angle_profile(np.fliplr(left))
    assert left_profile is not None and right_profile is not None
    assert left_profile['direction']=='left'
    assert right_profile['direction']=='right'
    first=left_profile['measurements'];second=right_profile['measurements']
    assert first['stem_column']<stem_width
    assert second['stem_column']>=width-stem_width
    assert first['stem_coverage']==second['stem_coverage']==1.
    required=max(2,int(np.ceil(width*.50)))
    assert min(first['top_cap_extension'],second['top_cap_extension'])>=required
    assert min(first['bottom_cap_extension'],second['bottom_cap_extension'])>=required
    assert first['maximum_middle_ink_width']==second['maximum_middle_ink_width']==stem_width


@pytest.mark.parametrize('height,width,stem_width,cap_rows',
                         [(23,6,3,2),(12,3,2,1),(8,3,2,1),(32,8,3,2)])
def test_quality_thick_square_pair_routes_from_complete_original_pixels(height,width,stem_width,cap_rows):
    image=thick_cap_face_fixture(height,width,stem_width,cap_rows)
    candidates=c.locate_candidates(image,[])['candidates']
    eligible=[item for item in candidates if item['route_eligible']]
    assert eligible
    full=[item for item in eligible if item['bbox'][0]<=10 and item['bbox'][2]>=10+height*2+width]
    assert full
    assert any(item['evidence']['bracket_shape_counts']['right-angle'] for item in full)
    assert all(not item['evidence']['edge_ink'] and not item['evidence']['neighbor_body_overlap'] for item in full)


@pytest.mark.parametrize('ordinary',['[ABC]','[123]','[333]','[o]','o [ω3ω]','[ω3ω] o'])
def test_quality_thick_square_pixel_confirmation_preserves_body_refusal(ordinary):
    image=thick_cap_face_fixture(23,6,3,2)
    baseline=[{'bbox':[10,10,62,33],'text':ordinary,'score':.01}]
    result=c.locate_candidates(image,baseline)
    assert any(item['evidence']['bracket_shape_counts']['right-angle'] for item in result['candidates'])
    assert result['route_eligible_count']==0


def test_quality_thick_square_tie_cannot_manufacture_caps_or_closed_frame():
    left=thick_cap_mask(23,6,3,2)
    left[-2:]=0;left[-2:,:3]=1
    assert c._right_angle_profile(left) is None
    assert c._right_angle_profile(np.fliplr(left)) is None
    closed=thick_cap_mask(23,6,3,2)
    closed[:,-2:]=1
    assert c._right_angle_profile(closed) is None
    assert c._right_angle_profile(np.fliplr(closed)) is None


def test_quality_thick_square_tie_keeps_adjacent_expressions_separate():
    image=thick_cap_face_fixture(23,6,3,2)
    joined=np.full((image.shape[0],image.shape[1]*2+18,3),255,dtype=np.uint8)
    joined[:,:image.shape[1]]=image
    offset=image.shape[1]+18
    joined[:,offset:offset+image.shape[1]]=image
    baseline=[{'bbox':[10,10,62,33],'text':'[ω3ω]','score':.1},
              {'bbox':[10+offset,10,62+offset,33],'text':'[Д益Д]','score':.1}]
    result=c.locate_candidates(joined,baseline)
    qualified=[item for item in result['candidates'] if item['route_eligible']]
    assert qualified
    assert all(len(item['baseline_indices'])<=1 for item in qualified)
    assert all(item['baseline_text'] in ('[ω3ω]','[Д益Д]') for item in qualified)
