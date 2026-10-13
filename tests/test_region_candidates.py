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
    # Measured bracket strokes and interior ink route the crop whatever the baseline read.
    ordinary=[{'bbox':[20,20,103,51],'text':'订单 (ABC) 123','score':.1}]
    candidates=c.locate_candidates(image,ordinary)['candidates']
    assert candidates and any(item['route_eligible'] for item in candidates)
    assert c.locate_candidates(image,[])['route_eligible_count']==1
    # Without strokes nothing routes, and a lone interior token is body.
    blank=np.full((80,160,3),255,dtype=np.uint8)
    assert c.locate_candidates(blank,[{'bbox':[20,20,103,51],'text':'(o)','score':.1}])['route_eligible_count']==0
    lone=Image.fromarray(image);draw=ImageDraw.Draw(lone)
    draw.rectangle((40,28,82,46),fill='white');draw.ellipse((55,31,68,41),outline='black',width=2)
    result=c.locate_candidates(np.asarray(lone),[{'bbox':[20,20,103,51],'text':'(o)','score':.1}])
    assert result['route_eligible_count']==0 and any(item['route_reason']=='insufficient-interior-ink' for item in result['candidates'])


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


@pytest.mark.parametrize('ordinary', ['[123]','[333]','[o]','[T]','o [ω3ω]','[ω3ω] o'])
def test_quality_right_angles_route_on_pixels_and_lone_token_stays_body(ordinary):
    image=angular_face_fixture()
    baseline=[{'bbox':[10,10,59,34],'text':ordinary,'score':.1}]
    output=c.locate_candidates(image,baseline)
    assert output['candidates'] and any(item['route_eligible'] for item in output['candidates'])
    # The same strokes around a single interior blob are an ordinary bracketed token.
    lone=Image.fromarray(image);draw=ImageDraw.Draw(lone)
    draw.rectangle((16,11,52,32),fill='white');draw.ellipse((28,17,40,27),outline='black',width=2)
    result=c.locate_candidates(np.asarray(lone),baseline)
    assert result['candidates'] and not any(item['route_eligible'] for item in result['candidates'])
    assert any(item['route_reason']=='insufficient-interior-ink' for item in result['candidates'])


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
    # The tight neighbour is exterior ink: routing may proceed on measured strokes, but the
    # recorded geometry forces fusion to demand an exterior atom on that side.
    eligible=[item for item in output['candidates'] if item['route_eligible']]
    assert all(item['evidence']['pair_geometry']['exterior_ink']['right'] for item in eligible)


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


@pytest.mark.parametrize('ordinary',['[123]','[333]','[o]','o [ω3ω]','[ω3ω] o'])
def test_quality_thick_square_pixel_confirmation_routes_on_strokes(ordinary):
    image=thick_cap_face_fixture(23,6,3,2)
    baseline=[{'bbox':[10,10,62,33],'text':ordinary,'score':.01}]
    result=c.locate_candidates(image,baseline)
    assert any(item['evidence']['bracket_shape_counts']['right-angle'] for item in result['candidates'])
    assert result['route_eligible_count']>=1
    assert all(item['evidence']['pair_geometry']['interior_components']>=2 for item in result['candidates'] if item['route_eligible'])


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


def dark_card_fixture(face=True, gap_cards=False):
    """Light page, dark rounded cards, light strokes: the record-list polarity of real screenshots."""
    image=Image.new('RGB',(400,160),(232,230,226));draw=ImageDraw.Draw(image)
    if gap_cards:
        draw.rounded_rectangle((10,8,390,68),radius=10,fill=(92,46,44))
        draw.rounded_rectangle((10,80,390,150),radius=10,fill=(58,92,42))
    else:
        draw.rounded_rectangle((10,20,390,140),radius=10,fill=(92,46,44))
    if face:
        ink=(242,240,236);ox,oy=120,40
        draw.arc((ox+20,oy+20,ox+34,oy+50),90,270,fill=ink,width=2)
        draw.arc((ox+88,oy+20,ox+102,oy+50),270,90,fill=ink,width=2)
        draw.ellipse((ox+43,oy+31,ox+50,oy+38),outline=ink,width=2)
        draw.ellipse((ox+72,oy+31,ox+79,oy+38),outline=ink,width=2)
        draw.line((ox+54,oy+43,ox+68,oy+43),fill=ink,width=2)
    return np.asarray(image)


def test_foreground_is_estimated_against_the_local_background_not_the_page_border():
    array=dark_card_fixture()
    mask=c.foreground(array,c.DEFAULT_CONFIG['contrast_threshold'])
    card=np.zeros(mask.shape,dtype=bool);card[30:130,20:380]=True
    # The dark card body must not become ink; only the light strokes inside it are foreground.
    assert mask[card].mean()<.05 and mask[60:95,140:225].sum()>60
    light=face_fixture()
    border=np.concatenate((light[0],light[-1],light[:,0],light[:,-1]),axis=0)
    global_mask=(np.max(np.abs(light.astype(np.int16)-np.median(border.astype(np.int16),axis=0)),axis=2)>=c.DEFAULT_CONFIG['contrast_threshold'])
    assert np.array_equal(c.foreground(light,c.DEFAULT_CONFIG['contrast_threshold']).astype(bool),global_mask)


def test_light_face_on_dark_card_is_discovered_from_pixels():
    output=c.locate_candidates(dark_card_fixture(),[])
    assert len(output['candidates'])==1
    candidate=output['candidates'][0]
    assert {'image-components','image-bracket-pair'}<=set(candidate['sources'])
    assert candidate['evidence']['pixel_support'] and candidate['evidence']['extra_shape_support']
    left,top,right,bottom=candidate['bbox']
    assert left<=140 and top<=60 and right>=222 and bottom>=91


def test_card_edges_and_gaps_never_become_candidates():
    output=c.locate_candidates(dark_card_fixture(face=False,gap_cards=True),[])
    assert output['candidates']==[]
    output=c.locate_candidates(dark_card_fixture(face=False),[])
    assert output['candidates']==[]


def test_background_window_configuration_is_bounded():
    config=dict(c.DEFAULT_CONFIG,background_window=50)
    with pytest.raises(ValueError):
        c.configuration(config)
    config=dict(c.DEFAULT_CONFIG,background_window=301)
    with pytest.raises(ValueError):
        c.configuration(config)
    legacy={key:value for key,value in c.DEFAULT_CONFIG.items() if key!='background_window'}
    assert c.configuration(legacy)['background_window']==c.DEFAULT_CONFIG['background_window']


def test_line_height_ignores_tall_icons_when_joining_chunks():
    text = lambda x: {'bbox': [x, 40, x + 12, 60], 'pixels': 100}
    icon = {'bbox': [0, 20, 50, 80], 'pixels': 2000}
    # Two text chunks separated by 16px (0.8 of the 20px text height, 0.27 of the 60px icon height).
    components = [icon] + [text(x) for x in (70, 84, 98)] + [text(x) for x in (126, 140, 154)]
    groups = c._line_groups(components, c.configuration())
    boxes = sorted(g['bbox'] for g in groups)
    assert [b[0] for b in boxes] == [0, 70, 126]
    # Without the icon the grouping is unchanged.
    assert sorted(g['bbox'][0] for g in c._line_groups(components[1:], c.configuration())) == [70, 126]
    # Bracket strokes taller than an expression's eyes and mouth still set the scale: the 16px gap
    # between two tall brackets and a following stroke joins when the brackets are 30px tall.
    face = [{'bbox': [70, 30, 76, 60], 'pixels': 60}, {'bbox': [82, 42, 88, 48], 'pixels': 20}, {'bbox': [94, 44, 104, 47], 'pixels': 15},
            {'bbox': [110, 42, 116, 48], 'pixels': 20}, {'bbox': [122, 30, 128, 60], 'pixels': 60}, {'bbox': [144, 32, 156, 58], 'pixels': 80}]
    assert len(c._line_groups(face, c.configuration())) == 1


# --- improve-region-candidate-localization: structural components, per-chunk pairs, pair-first candidates ---

def page_face(draw, ox, oy, ink=(0, 0, 0)):
    """The face_fixture strokes translated to (ox, oy): brackets 30px tall spanning x ox..ox+82."""
    draw.arc((ox, oy, ox+14, oy+30), 90, 270, fill=ink, width=2)
    draw.arc((ox+68, oy, ox+82, oy+30), 270, 90, fill=ink, width=2)
    draw.ellipse((ox+23, oy+11, ox+30, oy+18), outline=ink, width=2)
    draw.ellipse((ox+52, oy+11, ox+59, oy+18), outline=ink, width=2)
    draw.line((ox+34, oy+23, ox+48, oy+23), fill=ink, width=2)


def bordered_card_fixture():
    """Dark card with two faces on separate rows and a page-high separator line beside the card."""
    image=Image.new('RGB',(420,200),(232,230,226));draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((30,10,410,190),radius=10,fill=(92,46,44))
    draw.line((8,0,8,199),fill=(40,40,40),width=2)  # 200px tall structural component
    ink=(242,240,236)
    page_face(draw,120,30,ink)
    page_face(draw,120,120,ink)
    return np.asarray(image)


def test_structural_components_are_excluded_before_line_construction():
    output=c.locate_candidates(bordered_card_fixture(),[])
    assert output['structural_components']>=1
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert len(faces)==2
    for item in faces:
        left,top,right,bottom=item['bbox']
        assert left>=100 and left<=120 and right>=202 and bottom-top<=45
    # Without the separator the same two faces are found: exclusion only removes the structural component.
    plain=bordered_card_fixture().copy();plain[:,4:12]=(232,230,226)
    without=c.locate_candidates(plain,[])
    assert without['structural_components']==0 and output['structural_components']>=1
    assert sorted(i['bbox'] for i in without['candidates'] if 'image-bracket-pair' in i['sources'])==sorted(i['bbox'] for i in faces)


def test_bracket_pairs_are_detected_per_chunk_beside_taller_line_elements():
    image=Image.new('RGB',(360,100),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,15,76,71),fill='black')  # 56px icon on the same physical line
    page_face(draw,150,28)                      # 30px brackets: below 60% of the line's ink height
    output=c.locate_candidates(np.asarray(image),[])
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert len(faces)==1 and faces[0]['evidence']['extra_shape_support']
    left,top,right,bottom=faces[0]['bbox']
    assert left<=150 and right>=232 and left>80


def arm_body_fixture(arm_gap=10, body=True, ratio_key=None):
    image=Image.new('RGB',(320,80),'white');draw=ImageDraw.Draw(image)
    page_face(draw,40,20)                                   # brackets x 40..122, y 20..50 (H=30)
    ax=122+arm_gap
    draw.line((ax,28,ax+10,44),fill='black',width=2)        # exterior arm: 16px tall, 10px wide
    if body:
        draw.rectangle((ax+10+6,22,ax+10+6+26,48),fill='black')  # full-square body glyph 6px beyond the arm
    return np.asarray(image)


def test_pair_candidate_keeps_exterior_arm_but_stops_at_full_square_body():
    output=c.locate_candidates(arm_body_fixture(),[])
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert len(faces)==1
    left,top,right,bottom=faces[0]['bbox']
    assert right>=143 and right<149  # the arm (ends at x 142) is inside, the body glyph (starts at 148) is outside


def test_exterior_bridge_ratio_is_optional_and_bounded():
    # Arm 16px beyond the pair (0.53H) followed by a body glyph: the body forces the pair box, so the arm
    # is kept only when the bridge ratio reaches it. Without body the container keeps the arm regardless.
    wide=arm_body_fixture(arm_gap=16,body=True)
    default=[i for i in c.locate_candidates(wide,[])['candidates'] if 'image-bracket-pair' in i['sources']]
    assert default and default[0]['bbox'][2]<138
    config=dict(c.DEFAULT_CONFIG,exterior_bridge_ratio=.6)
    bridged=[i for i in c.locate_candidates(wide,[],config)['candidates'] if 'image-bracket-pair' in i['sources']]
    assert bridged and 149<=bridged[0]['bbox'][2]<154
    free=[i for i in c.locate_candidates(arm_body_fixture(arm_gap=16,body=False),[])['candidates'] if 'image-bracket-pair' in i['sources']]
    assert free and free[0]['bbox'][2]>=149
    legacy={key:value for key,value in c.DEFAULT_CONFIG.items() if key not in c.OPTIONAL_CONFIG}
    assert c.configuration(legacy)['exterior_bridge_ratio']==c.OPTIONAL_CONFIG['exterior_bridge_ratio']==.45
    for invalid in (.2,.9,'0.5',True):
        with pytest.raises(ValueError):
            c.configuration(dict(c.DEFAULT_CONFIG,exterior_bridge_ratio=invalid))


def test_bridging_does_not_cross_physical_lines():
    image=Image.new('RGB',(240,120),'white');draw=ImageDraw.Draw(image)
    page_face(draw,40,10)                                   # line 1: y 10..40
    draw.line((128,50,138,66),fill='black',width=2)        # line 2 stroke 10px below, within 0.45H horizontally
    output=c.locate_candidates(np.asarray(image),[])
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert len(faces)==1 and faces[0]['bbox'][3]<=46


def test_pair_box_replaces_a_containing_mixed_first_pass_box():
    image=Image.new('RGB',(320,80),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,22,46,48),fill='black');draw.rectangle((52,22,78,48),fill='black')  # body glyphs
    page_face(draw,92,20)                                                                    # face 6px after the body
    baseline=[{'bbox':[18,18,178,52],'text':'正文(ω_ω)','score':.2}]
    output=c.locate_candidates(np.asarray(image),baseline)
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert len(faces)==1
    left,top,right,bottom=faces[0]['bbox']
    assert left>=84 and right<=180 and 'first-pass-symbol-geometry' in faces[0]['sources']
    assert faces[0]['evidence']['neighbor_body_overlap'] and faces[0]['evidence']['mixed_region_separable']
    assert faces[0]['splice_baseline_index']==0 and faces[0]['baseline_indices']==[]


def test_container_without_body_keeps_its_extent_but_body_containers_yield_to_the_pair():
    """An OCR box that only adds a far arm to the pair is kept; one that adds a body glyph is replaced by the pair box."""
    image=Image.new('RGB',(320,80),'white');draw=ImageDraw.Draw(image)
    page_face(draw,40,20)                                   # brackets x 40..122 (H=30)
    draw.line((138,28,148,44),fill='black',width=2)         # arm 16px beyond: 0.53H, outside the default bridge but inside the chunk gap
    arm_only=[{'bbox':[36,16,152,54],'text':'(ω_ω)ノ','score':.3}]
    kept=[i for i in c.locate_candidates(np.asarray(image),arm_only)['candidates'] if 'image-bracket-pair' in i['sources']]
    assert kept and kept[0]['bbox'][2]>=149 and kept[0]['baseline_indices']==[0]
    body=Image.new('RGB',(320,80),'white');draw=ImageDraw.Draw(body)
    page_face(draw,40,20)
    draw.rectangle((138,22,164,48),fill='black')            # full-square body glyph 16px beyond
    with_body=[{'bbox':[36,16,168,54],'text':'(ω_ω)字','score':.3}]
    pair=[i for i in c.locate_candidates(np.asarray(body),with_body)['candidates'] if 'image-bracket-pair' in i['sources']]
    assert pair and pair[0]['bbox'][2]<138 and pair[0]['evidence']['neighbor_body_overlap']


def test_thin_full_square_outline_arm_is_absorbed_but_dense_glyph_is_body():
    """⊂-like outline arms are full-square yet sparse: they bridge; a dense full-square glyph is body and stops the bridge."""
    image=Image.new('RGB',(320,80),'white');draw=ImageDraw.Draw(image)
    page_face(draw,40,20)                                   # brackets x 40..122 (H=30)
    draw.arc((132,22,158,48),90,270,fill='black',width=2)   # 26x26 outline arm 10px beyond the pair
    output=c.locate_candidates(np.asarray(image),[])
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert faces and faces[0]['bbox'][2]>=146  # the left-half arc spans x 132..~145 and is bridged (plus 3px padding)
    dense=Image.new('RGB',(320,80),'white');draw=ImageDraw.Draw(dense)
    page_face(draw,40,20)
    draw.rectangle((132,22,158,48),fill='black')            # same cell, filled: density 1.0
    output=c.locate_candidates(np.asarray(dense),[])
    faces=[item for item in output['candidates'] if 'image-bracket-pair' in item['sources']]
    assert faces and faces[0]['bbox'][2]<132
