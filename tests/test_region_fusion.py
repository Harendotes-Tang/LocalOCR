"""Exact multi-view, shape and ordinary-body guard tests; no model claim."""
import copy
from PIL import Image,ImageDraw
import numpy as np
import pytest
from tools.RegionKaomoji import candidates as c
from tools.RegionKaomoji import fusion as f


def fixture():
    image=Image.new('RGB',(160,80),'white');draw=ImageDraw.Draw(image)
    draw.arc((20,20,34,50),90,270,fill='black',width=2)
    draw.arc((88,20,102,50),270,90,fill='black',width=2)
    draw.ellipse((43,31,50,38),outline='black',width=2)
    draw.ellipse((72,31,79,38),outline='black',width=2)
    draw.line((54,43,68,43),fill='black',width=2)
    return np.asarray(image)


def request(original='(Д_Д)',text='(ω_ω)',baseline=None):
    image=fixture()
    if baseline is None:baseline=[{'bbox':[20,20,103,51],'text':original,'score':.0001}]
    if baseline and baseline[0]['bbox']==[0,10,150,60]:
        extended=Image.fromarray(image);ImageDraw.Draw(extended).rectangle((130,25,140,42),fill='black');image=np.asarray(extended)
    candidates=c.locate_candidates(image,baseline)['candidates']
    views={}
    for candidate in candidates:
        records=[]
        for transform in ('raw','edge-pad-6'):
            pixels=f.view_image(image,candidate['bbox'],transform)
            records.append({'model_id':'research-specialist','view_id':transform,'transform':transform,
                            'image_sha256':c.image_digest(pixels),'text':text,'score':.9999})
        views[candidate['candidate_id']]=records
    return image,baseline,candidates,views


def decision(args):
    return f.fuse_candidates(*args)


def test_confirmed_face_allows_D_omega_correction():
    output=decision(request())
    assert output['accepted_count']==1
    assert output['regions'][0]['text']=='(ω_ω)'
    assert output['decisions'][0]['reason']=='exact-views-and-independent-pixels'


def test_body_anchors_preserved_even_with_high_specialist_score():
    # Pixels are the evidence: the fixture is a face, so any baseline reading may be replaced,
    # while a real lone token or unaccounted exterior ink keeps the raw baseline text.
    for misread in ('(ABC)','(A_A)','(o)','订单123','o (A_A)','(A1A)','(0_0)'):
        output=decision(request(original=misread))
        assert output['accepted_count']==1 and output['regions'][0]['text']=='(ω_ω)', misread
    lone=Image.fromarray(fixture());draw=ImageDraw.Draw(lone)
    draw.rectangle((40,28,82,46),fill='white');draw.ellipse((55,31,68,41),outline='black',width=2)
    image=np.asarray(lone);baseline=[{'bbox':[20,20,103,51],'text':'(o)','score':.0001}]
    candidates=c.locate_candidates(image,baseline)['candidates']
    views={item['candidate_id']:[{'model_id':'m','view_id':t,'transform':t,'image_sha256':c.image_digest(f.view_image(image,item['bbox'],t)),'text':'(ω_ω)','score':.9} for t in ('raw','edge-pad-6')] for item in candidates}
    output=f.fuse_candidates(image,baseline,candidates,views)
    assert output['accepted_count']==0 and output['regions'][0]['text']=='(o)' and output['regions'][0]['score']==.0001
    assert any(item['reason']=='insufficient-interior-ink' for item in output['decisions'])


def test_whitespace_unicode_and_bracket_disagreement_not_normalized():
    for variant in ('(ω_ω) ','(ω _ ω)','（ω_ω）','(ω_ω'):
        args=request();args[3][args[2][0]['candidate_id']][1]['text']=variant
        output=decision(args)
        assert output['accepted_count']==0
        assert output['regions'][0]['text']=='(Д_Д)'
        assert output['decisions'][0]['reason']=='exact-unicode-view-disagreement'


def test_agreed_output_keeps_unicode_and_internal_whitespace_exact():
    output=decision(request(original='(Д  _  Д)',text='(ω  _  ω)'))
    assert output['accepted_count']==1
    assert output['regions'][0]['text']=='(ω  _  ω)'


def test_duplicate_or_forged_view_and_cross_model_votes_rejected():
    for change,reason in [('duplicate','duplicate-view-pixels'),('forged','view-pixels-differ'),('model','cross-model-vote-not-supported')]:
        args=request();records=args[3][args[2][0]['candidate_id']]
        if change=='duplicate':records[1].update(transform='raw',image_sha256=records[0]['image_sha256'])
        elif change=='forged':records[1]['image_sha256']='0'*64
        else:records[1]['model_id']='other-model'
        output=decision(args)
        assert output['accepted_count']==0
        assert output['decisions'][0]['reason']==reason


def test_empty_output_and_missing_actual_views_keep_original():
    for text in ('','   ','\n'):
        output=decision(request(text=text))
        assert output['accepted_count']==0
    args=request();args[3][args[2][0]['candidate_id']].pop()
    assert decision(args)['decisions'][0]['reason']=='insufficient-independent-views'


def test_body_overlap_never_replaces_or_inserts():
    baseline=[{'bbox':[0,10,150,60],'text':'正文 (Д_Д) 123','score':.9}]
    output=decision(request(baseline=baseline))
    assert output['accepted_count']==0
    assert output['regions'][0]['text']==baseline[0]['text']
    assert any(item['reason'] in ('neighbor-body-overlap','mixed-baseline-context-required') for item in output['decisions'])


def test_pixel_only_detected_region_inserts_only_with_extra_shapes():
    output=decision(request(baseline=[]))
    assert output['accepted_count']==1
    assert output['regions'][0]['text']=='(ω_ω)'
    config=dict(f.DEFAULT_CONFIG);config['allow_pixel_only_insertions']=False
    output=f.fuse_candidates(*request(baseline=[]),config=config)
    assert output['accepted_count']==0


def test_forged_candidate_evidence_and_duplicate_candidate_rejected():
    args=request();args[2][0]['evidence']['neighbor_body_overlap']=True
    with pytest.raises(ValueError,match='differs'):decision(args)
    args=request();args[2].append(copy.deepcopy(args[2][0]))
    with pytest.raises(ValueError,match='Duplicate'):decision(args)


def test_incomplete_canvas_candidate_and_raw_view_required():
    args=request();records=args[3][args[2][0]['candidate_id']]
    records[0].update(transform='edge-pad-4',image_sha256=c.image_digest(f.view_image(args[0],args[2][0]['bbox'],'edge-pad-4')))
    assert decision(args)['decisions'][0]['reason']=='raw-view-required'
    image=fixture()[:,20:]
    baseline=[];cs=c.locate_candidates(image,baseline)['candidates']
    assert cs and cs[0]['evidence']['clipped_canvas']
    views={cs[0]['candidate_id']:[{'model_id':'m','view_id':t,'transform':t,'image_sha256':c.image_digest(f.view_image(image,cs[0]['bbox'],t)),'text':'(ω_ω)','score':.8} for t in ('raw','edge-pad-6')]}
    assert f.fuse_candidates(image,baseline,cs,views)['accepted_count']==0


def test_configuration_cannot_disable_body_or_pixel_protection():
    for key in ('require_raw_view','require_extra_shape_support','protect_body_letters_numbers'):
        config=dict(f.DEFAULT_CONFIG);config[key]=False
        with pytest.raises(ValueError):f.configuration(config)


def test_nonfinite_view_score_is_retained_as_json_safe_failure():
    import json
    args=request();args[3][args[2][0]['candidate_id']][0]['score']=float('nan')
    output=decision(args)
    assert output['accepted_count']==0
    assert output['decisions'][0]['reason']=='nonfinite-specialist-score'
    assert output['decisions'][0]['view_scores'][0] is None
    json.dumps(output,allow_nan=False)


def test_ineligible_discovery_keeps_route_reason_without_model_views():
    # Without measured strokes nothing routes; the route reason is kept and no specialist view exists.
    plain=np.full((80,160,3),255,dtype=np.uint8)
    canvas=Image.fromarray(plain);draw=ImageDraw.Draw(canvas)
    draw.rectangle((45,31,50,38),fill='black');draw.rectangle((72,31,79,38),fill='black');draw.line((54,43,68,43),fill='black',width=2)
    image=np.asarray(canvas);baseline=[{'bbox':[20,20,103,51],'text':'o (ABC)','score':.0001}]
    candidates=c.locate_candidates(image,baseline)['candidates']
    assert candidates and not any(item['route_eligible'] for item in candidates)
    output=f.fuse_candidates(image,baseline,candidates,{})
    assert output['accepted_count']==0
    assert output['decisions'][0]['reason']=='unconfirmed-independent-pixel-face-shape'


def test_unknown_candidate_view_mapping_rejected():
    args=request();args[3]['not-an-automatic-candidate']=[]
    with pytest.raises(ValueError,match='unknown automatic'):decision(args)


def mixed_request(original='LEFT  (Д_Д)  RIGHT',replacement='(ω_ω)',context_omission=False):
    image=Image.new('RGB',(360,80),'white')
    image.paste(Image.fromarray(fixture()),(100,0))
    draw=ImageDraw.Draw(image)
    draw.rectangle((15,23,55,46),fill='black')
    draw.rectangle((245,23,285,46),fill='black')
    if context_omission:
        # Independent round glyph outside the face ROI and inside left context.
        draw.ellipse((80,30,86,36),outline='black',width=2)
    image=np.asarray(image)
    baseline=[{'bbox':[10,10,291,61],'text':original,'score':.4}]
    candidates=c.locate_candidates(image,baseline)['candidates']
    candidate=next(item for item in candidates if item['splice_baseline_index'] is not None and item['route_eligible'])
    views={candidate['candidate_id']:[{'model_id':'m','view_id':t,'transform':t,'image_sha256':c.image_digest(f.view_image(image,candidate['bbox'],t)),'text':replacement,'score':.9} for t in ('raw','edge-pad-6')]}
    context=[]
    for declaration in candidate['context_crops']:
        context.append({'role':declaration['role'],'bbox':list(declaration['bbox']),'image_sha256':c.image_digest(f.view_image(image,declaration['bbox'],'raw')),
                        'text':'LEFT' if declaration['role']=='left' else 'RIGHT','score':.2,'model_id':'production-baseline',
                        'baseline_parent_sha256':candidate['baseline_parent_sha256']})
    return image,baseline,candidates,views,{candidate['candidate_id']:context}


def mixed_output(args):
    return f.fuse_candidates(*args[:4],contexts=args[4])


def test_actual_blank_bands_and_contexts_allow_unique_face_splice():
    args=mixed_request();candidate=next(item for item in args[2] if item['route_eligible'] and item['splice_baseline_index'] is not None)
    assert candidate['baseline_indices']==[]
    assert candidate['evidence']['neighbor_body_overlap']
    assert candidate['evidence']['mixed_region_separable']
    assert len(candidate['context_crops'])==2
    assert all(item['blank_band_pixels']>=2 for item in candidate['context_crops'])
    output=mixed_output(args)
    assert output['accepted_count']==1
    assert output['regions'][0]['text']=='LEFT  (ω_ω)  RIGHT'
    assert output['regions'][0]['bbox']==args[1][0]['bbox']
    assert output['regions'][0]['score']==.4
    assert output['regions'][0]['splice']['preserved_prefix']=='LEFT  '
    assert output['regions'][0]['splice']['preserved_suffix']=='  RIGHT'
    assert output['regions'][0]['splice']['original_text_span']==[6,11]


def test_context_missing_case_difference_or_repetition_preserves_whole_line():
    args=mixed_request();args[4].clear();output=mixed_output(args)
    assert output['accepted_count']==0
    assert output['regions'][0]['text']==args[1][0]['text']
    args=mixed_request();next(iter(args[4].values()))[0]['text']='left';output=mixed_output(args)
    assert output['accepted_count']==0
    assert any(item['reason']=='context-not-unique-raw-prefix-suffix' for item in output['decisions'])
    args=mixed_request(original='LEFT LEFT  (Д_Д)  RIGHT');output=mixed_output(args)
    assert output['accepted_count']==0
    assert output['regions'][0]['text']==args[1][0]['text']


def test_context_geometry_and_pixel_sha_cannot_be_forged():
    args=mixed_request();next(iter(args[4].values()))[0]['image_sha256']='0'*64
    assert mixed_output(args)['accepted_count']==0
    args=mixed_request();next(iter(args[4].values()))[0]['bbox'][0]+=1
    assert mixed_output(args)['accepted_count']==0


def test_ordinary_letters_in_mixed_middle_never_change():
    # Digits, single interior tokens and extra interior tokens never change; letters the
    # baseline placed inside a pixel-separated bracket span may be misread face glyphs.
    for middle in ('(o)','123','(A1A)','(o Д_Д)','(ABCD)'):
        args=mixed_request(original='LEFT  '+middle+'  RIGHT')
        output=mixed_output(args)
        assert output['accepted_count']==0, middle
        assert output['regions'][0]['text']==args[1][0]['text']
    for middle in ('(ABC)','(A_A)'):
        args=mixed_request(original='LEFT  '+middle+'  RIGHT')
        output=mixed_output(args)
        assert output['accepted_count']==1 and output['regions'][0]['text']=='LEFT  (ω_ω)  RIGHT', middle


def test_specialist_spaces_remain_raw_during_context_splice():
    args=mixed_request(replacement=' (ω _ ω) ')
    output=mixed_output(args)
    assert output['accepted_count']==1
    assert output['regions'][0]['text']=='LEFT   (ω _ ω)   RIGHT'
    assert output['regions'][0]['regional_text']==' (ω _ ω) '


def test_overlapping_raw_or_anchor_occurrences_are_ambiguous():
    assert not f._unique_occurrence('ABABA','ABA')
    assert not f._unique_occurrence('TTTT','TTT')
    assert f._unique_occurrence('LEFT (Д_Д) RIGHT','LEFT')
    args=mixed_request(original='ABABA (Д_Д) RIGHT')
    next(iter(args[4].values()))[0]['text']='ABA'
    output=mixed_output(args)
    assert output['accepted_count']==0
    assert any(item['reason']=='context-not-unique-raw-prefix-suffix' for item in output['decisions'])


def test_real_producer_residual_body_token_cannot_be_deleted():
    # Candidate geometry and pixel evidence are produced from actual fixture
    # pixels. These declared OCR outputs simulate a context recognizer omission;
    # they are not a claim that a model recognized this synthetic geometry.
    args=mixed_request(original='LEFT o  (Д_Д)  RIGHT',context_omission=True)
    candidate=next(item for item in args[2] if item['splice_baseline_index'] is not None)
    assert candidate['route_eligible']
    assert candidate['evidence']['extra_shape_support']
    left_crop=next(item for item in candidate['context_crops'] if item['role']=='left')
    assert left_crop['bbox'][2]>86 and candidate['bbox'][0]>86
    assert next(iter(args[4].values()))[0]['text']=='LEFT'
    output=mixed_output(args)
    assert output['accepted_count']==0
    assert output['regions'][0]['text']=='LEFT o  (Д_Д)  RIGHT'
    assert any(item['reason'] in ('mixed-region-not-confirmed-face-fragment','mixed-region-ordinary-letter-number-protected') for item in output['decisions'])


@pytest.mark.parametrize('residual', ['o (Д_Д)', '(Д_Д) o', '(o Д_Д)',
                                      'つ(Д_Д)', '(Д_Д)つ', '(Д_Д) (ω_ω)',
                                      'Д_Д', '(Д_Д', 'Д_Д)', 'o Д_Д', '(^_^)(^_^)'])
def test_ambiguous_mixed_residual_preserves_raw_parent(residual):
    args=mixed_request(original='LEFT  '+residual+'  RIGHT')
    output=mixed_output(args)
    assert output['accepted_count']==0
    assert output['regions'][0]['text']==args[1][0]['text']


def test_specialist_cannot_add_an_independent_body_token():
    for replacement in ('o (ω_ω)', '(ω_ω) o', '(o ω_ω)', '(ω_ω) (ω_ω)'):
        args=mixed_request(replacement=replacement)
        output=mixed_output(args)
        assert output['accepted_count']==0
        assert output['regions'][0]['text']==args[1][0]['text']



def test_stale_mixed_candidate_subset_rejects_changed_parent_text():
    args=mixed_request();args[2][:]=[item for item in args[2] if item['splice_baseline_index'] is not None]
    args[1][0]['text']='LEFT  (^_^)  RIGHT'
    with pytest.raises(ValueError,match='Candidate differs'):
        mixed_output(args)


@pytest.mark.parametrize('parent_sha256', [None, '0'*64])
def test_missing_or_forged_context_parent_digest_preserves_raw_line(parent_sha256):
    args=mixed_request();record=next(iter(args[4].values()))[0]
    if parent_sha256 is None:record.pop('baseline_parent_sha256')
    else:record['baseline_parent_sha256']=parent_sha256
    output=mixed_output(args)
    assert output['accepted_count']==0 and output['regions'][0]['text']==args[1][0]['text']
    assert any(item['reason']=='baseline-context-parent-provenance-differs' for item in output['decisions'])


def test_context_receipt_cannot_be_reused_for_a_different_parent():
    original=mixed_request();changed=mixed_request(original='LEFT  (^_^)  RIGHT');fresh_contexts=changed[4]
    changed=(*changed[:4],original[4]);output=mixed_output(changed)
    assert output['accepted_count']==0 and output['regions'][0]['text']==changed[1][0]['text']
    assert any(item['reason']=='baseline-context-parent-provenance-differs' for item in output['decisions'])
    valid=mixed_output((*changed[:4],fresh_contexts));assert valid['accepted_count']==1
    splice=next(item['splice'] for item in valid['decisions'] if item['accepted'])
    parent=c.baseline_regions(changed[1],(changed[0].shape[1],changed[0].shape[0]))[0]
    assert splice['baseline_parent_sha256']==c.baseline_parent_digest(parent)


def test_two_physical_symbol_faces_cannot_collapse_two_baseline_regions():
    small=fixture()[15:56,16:107];image=Image.new('RGB',(200,65),'white')
    image.paste(Image.fromarray(small),(8,8));image.paste(Image.fromarray(small),(102,8));image=np.asarray(image)
    baseline=[{'bbox':[12,13,95,44],'text':'(^_^)','score':.5},{'bbox':[106,13,189,44],'text':'(^_^)','score':.5}]
    candidates=c.locate_candidates(image,baseline)['candidates'];assert any(len(item['baseline_indices'])==2 for item in candidates)
    views={item['candidate_id']:[{'view_id':view,'transform':view,'image_sha256':c.image_digest(f.view_image(image,item['bbox'],view)),
                                'text':'(^_^)','score':.2,'model_id':'declared-public-output'}
                               for view in ('raw','edge-pad-6')] for item in candidates}
    output=f.fuse_candidates(image,baseline,candidates,views)
    assert output['accepted_count']==0 and len(output['regions'])==2
    assert [(item['bbox'],item['text'],item['score']) for item in output['regions']]==[(item['bbox'],item['text'],item['score']) for item in baseline]
    assert any(item['reason'] in ('multiple-baseline-regions-not-one-face-expression','multiple-bracket-pairs') for item in output['decisions'])


def test_one_complete_expression_in_two_ocr_fragments_remains_supported():
    baseline=[{'bbox':[20,20,65,51],'text':'(^_','score':.5},{'bbox':[65,20,103,51],'text':'^)','score':.5}]
    args=request(baseline=baseline)
    assert any(len(item['baseline_indices'])==2 and c.is_face_text(item['baseline_text']) for item in args[2])
    output=decision(args)
    assert output['accepted_count']==1 and len(output['regions'])==1 and output['regions'][0]['text']=='(ω_ω)'



@pytest.mark.parametrize('original,replacement', [('[ω3ω]','[Д3Д]'),('(ω3ω)','(Д3Д)'),('{Д益Д}','{ω皿ω}'),('（ω  3  ω）','（Д  3  Д）')])
def test_quality_numeric_mouth_and_supported_letter_mouth_preserve_raw_outputs(original,replacement):
    output=decision(request(original=original,text=replacement))
    assert output['accepted_count']==1
    assert output['regions'][0]['text']==replacement
    assert '3' not in c.FACE_LETTERS


@pytest.mark.parametrize('residual', ['3 (ω3ω)','(ω3ω) 3','(3)','[123]','[333]','[3_3]'])
def test_quality_independent_digits_and_numeric_body_never_receive_face_exception(residual):
    args=mixed_request(original='LEFT  '+residual+'  RIGHT',replacement='(Д3Д)')
    output=mixed_output(args)
    assert output['accepted_count']==0
    assert output['regions'][0]['text']==args[1][0]['text']


def test_quality_mixed_numeric_mouth_maps_only_actual_bare_expression():
    args=mixed_request(original='LEFT  (ω3ω)  RIGHT',replacement='(Д3Д)')
    output=mixed_output(args)
    assert output['accepted_count']==1
    assert output['regions'][0]['text']=='LEFT  (Д3Д)  RIGHT'
    assert output['regions'][0]['splice']['preserved_prefix']=='LEFT  '
    assert output['regions'][0]['splice']['preserved_suffix']=='  RIGHT'


def test_quality_numeric_mouth_does_not_relax_multiview_or_parent_provenance():
    args=request(original='(ω3ω)',text='(Д3Д)')
    for records in args[3].values():records[1]['text']='(Д 3 Д)'
    output=decision(args)
    assert output['accepted_count']==0 and output['regions'][0]['text']=='(ω3ω)'
    args=mixed_request(original='LEFT  (ω3ω)  RIGHT',replacement='(Д3Д)')
    next(iter(args[4].values()))[0]['baseline_parent_sha256']='0'*64
    output=mixed_output(args)
    assert output['accepted_count']==0 and output['regions'][0]['text']==args[1][0]['text']


def three_view_request(texts):
    """Raw, edge-pad-4 and edge-pad-6 views carrying the given texts in that order."""
    image,baseline,candidates,views=request()
    candidate=candidates[0]
    records=[]
    for transform,text in zip(('raw','edge-pad-4','edge-pad-6'),texts):
        pixels=f.view_image(image,candidate['bbox'],transform)
        records.append({'model_id':'research-specialist','view_id':transform,'transform':transform,
                        'image_sha256':c.image_digest(pixels),'text':text,'score':.99})
    return image,baseline,candidates,{candidate['candidate_id']:records}


def test_majority_of_three_views_must_include_the_raw_view():
    output=decision(three_view_request(['(ω_ω)','(ω_ω)','(ω _ω)']))
    assert output['accepted_count']==1 and output['regions'][0]['text']=='(ω_ω)'
    output=decision(three_view_request(['(ω_ω)','(ω _ω)','(ω_ω)']))
    assert output['accepted_count']==1 and output['regions'][0]['text']=='(ω_ω)'
    output=decision(three_view_request(['(ω _ω)','(ω_ω)','(ω_ω)']))
    assert output['accepted_count']==0 and output['decisions'][0]['reason']=='raw-view-outside-agreement'
    output=decision(three_view_request(['(ω_ω)','(ω _ω)','(ω__ω)']))
    assert output['accepted_count']==0 and output['decisions'][0]['reason']=='exact-unicode-view-disagreement'


def test_agreement_count_is_configurable_but_never_below_two():
    config=f.configuration();config['minimum_agreeing_views']=3
    args=three_view_request(['(ω_ω)','(ω_ω)','(ω _ω)'])
    output=f.fuse_candidates(*args,config=config)
    assert output['accepted_count']==0 and output['decisions'][0]['reason']=='exact-unicode-view-disagreement'
    config['minimum_agreeing_views']=1
    with pytest.raises(ValueError):f.configuration(config)


def whole_reading_request(original, texts, baseline_bbox=(20, 20, 103, 51)):
    image = fixture()
    baseline = [{'bbox': list(baseline_bbox), 'text': original, 'score': .0001}]
    candidates = c.locate_candidates(image, baseline)['candidates']
    candidate = candidates[0]
    records = []
    for transform, text in zip(('raw', 'edge-pad-4', 'edge-pad-6'), texts):
        pixels = f.view_image(image, candidate['bbox'], transform)
        records.append({'model_id': 'research-specialist', 'view_id': transform, 'transform': transform,
                        'image_sha256': c.image_digest(pixels), 'text': text, 'score': .99})
    return image, baseline, candidates, {candidate['candidate_id']: records}


def test_whole_reading_face_splices_into_the_baseline_bytes():
    # A trailing tail glyph would need its own exterior ink; the fixture draws none, so a CJK suffix stands for body.
    output = decision(whole_reading_request('AB(Д_Д)了', ['AB(ω_ω)了', 'AB(ω_ω)了', 'AB(ω_ω)了']))
    assert output['accepted_count'] == 1
    region = output['regions'][0]
    assert region['source'] == 'baseline-with-regional-splice' and region['text'] == 'AB(ω_ω)了' and region['regional_text'] == '(ω_ω)'
    assert region['bbox'] == [20, 20, 103, 51] and region['splice']['preserved_prefix'] == 'AB' and region['splice']['preserved_suffix'] == '了'
    # A tail read by the specialist without exterior ink on that side stays rejected.
    output = decision(whole_reading_request('AB(Д_Д)!', ['AB(ω_ω)!', 'AB(ω_ω)!', 'AB(ω_ω)!']))
    assert output['accepted_count'] == 0 and output['decisions'][0]['reason'] == 'exterior-ink-and-arms-disagree'
    assert region['splice']['original_text_span'] == [2, 7] and region['splice']['method'] == 'whole-reading-agreement'
    assert region['regional_ink_bbox'][0] >= 20 and region['regional_ink_bbox'][2] <= 103


def test_whole_reading_votes_on_the_face_while_body_misreads_differ():
    output = decision(whole_reading_request('AB(Д_Д)', ['AB(ω_ω)', 'A8(ω_ω)', 'AB(ω_ω)']))
    assert output['accepted_count'] == 1 and output['regions'][0]['text'] == 'AB(ω_ω)'
    output = decision(whole_reading_request('AB(Д_Д)', ['A8(ω_ω)', 'AB(ω_ω)', 'AB(ω_ω)']))
    assert output['accepted_count'] == 0 and output['decisions'][0]['reason'] == 'raw-view-outside-agreement'


def test_whole_reading_never_rewrites_body_letters_or_numbers():
    for original, texts, reason in [
        ('AB(Д_Д)', ['AX(ω_ω)', 'AX(ω_ω)', 'AX(ω_ω)'], 'specialist-output-not-confirmed-face-grammar'),
        ('AB(1_1)', ['AB(ω_ω)', 'AB(ω_ω)', 'AB(ω_ω)'], 'mixed-region-ordinary-letter-number-protected'),
        ('ABCD', ['AB(ω_ω)', 'AB(ω_ω)', 'AB(ω_ω)'], 'whole-reading-baseline-middle-not-bracketed'),
        ('AB(Д_Д)', ['AB(ω_ω)', 'AB(ω _ω)', 'AB(ω__ω)'], 'exact-unicode-view-disagreement'),
    ]:
        output = decision(whole_reading_request(original, texts))
        assert output['accepted_count'] == 0, (original, texts)
        assert output['decisions'][0]['reason'] == reason, (original, texts, output['decisions'][0]['reason'])
        assert output['regions'][0]['text'] == original


def test_whole_reading_keeps_surrounding_spaces_in_the_preserved_bytes():
    output = decision(whole_reading_request('AB (Д_Д)', ['AB (ω_ω)', 'AB (ω_ω)', 'AB (ω_ω)']))
    assert output['accepted_count'] == 1
    region = output['regions'][0]
    assert region['text'] == 'AB (ω_ω)' and region['regional_text'] == '(ω_ω)' and region['splice']['preserved_prefix'] == 'AB '
