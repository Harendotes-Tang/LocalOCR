"""Conservative exact-view fusion with original-pixel and body-anchor guards.

This research module records raw model scores without treating them as comparable
probabilities. Uncertain crops keep all original baseline regions unchanged.
"""
from __future__ import annotations
import copy
import math
import unicodedata
import numpy as np
from tools.RegionKaomoji import candidates as c

SCHEMA = 'localocr.region-fusion.v1'
DEFAULT_CONFIG = {'minimum_agreeing_views': 2, 'maximum_output_characters': 256,
                  'require_raw_view': True, 'allow_pixel_only_insertions': True,
                  'require_extra_shape_support': True, 'protect_body_letters_numbers': True}


def configuration(config=None):
    value = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_CONFIG):
        raise ValueError('Unknown or incomplete fusion configuration')
    for key, default in DEFAULT_CONFIG.items():
        item = value[key]
        if type(item) is not type(default):
            raise ValueError('Fusion configuration type differs')
    if not 2 <= value['minimum_agreeing_views'] <= 8 or not 1 <= value['maximum_output_characters'] <= 256:
        raise ValueError('Fusion output/view ceiling outside range')
    if not value['require_raw_view'] or not value['require_extra_shape_support'] or not value['protect_body_letters_numbers']:
        raise ValueError('Research fusion protection cannot be disabled')
    return value


def anchors(text):
    return ''.join(char for char in text if unicodedata.category(char)[0] in 'LN')


def view_image(image, bbox, transform):
    """Produce declared crop views from actual pixels, preserving RGB identity."""
    array = c.rgb_image(image)
    left, top, right, bottom = bbox
    if not 0 <= left < right <= array.shape[1] or not 0 <= top < bottom <= array.shape[0]:
        raise ValueError('View box outside original image')
    raw = np.ascontiguousarray(array[top:bottom,left:right])
    if transform == 'raw':
        return raw
    if transform in ('edge-pad-4', 'edge-pad-6'):
        padding = int(transform.rsplit('-',1)[1])
        return np.ascontiguousarray(np.pad(raw, ((padding,padding),(padding,padding),(0,0)), mode='edge'))
    raise ValueError('Undeclared independent view transform')


def _face_span(text):
    """Longest substring that is one complete expression (arms and tails included), or None."""
    if not isinstance(text, str) or len(text) > 256:
        return None
    length = len(text)
    for size in range(length, 1, -1):
        for start in range(0, length - size + 1):
            if c.is_face_text(text[start:start + size]):
                # Surrounding whitespace belongs to the reading around the expression, not to it.
                a, b = start, start + size
                while a < b and text[a].isspace():
                    a += 1
                while b > a and text[b - 1].isspace():
                    b -= 1
                return a, b
    return None


def _reading_middle(original, text):
    """Split one specialist reading of a mixed box into (prefix, face, suffix, baseline middle).

    The prefix and suffix must be the baseline's own leading and trailing bytes;
    only the baseline bytes between them may change. None when no complete
    expression exists or the surrounding reading differs from the baseline.
    """
    span = _face_span(text)
    if span is None:
        return None
    prefix, face, suffix = text[:span[0]], text[span[0]:span[1]], text[span[1]:]
    if len(prefix) + len(suffix) > len(original) or not original.startswith(prefix) or not original.endswith(suffix):
        return None
    return prefix, face, suffix, original[len(prefix):len(original) - len(suffix)]


def _agreement(image, bbox, records, config, key=None):
    if not isinstance(records, list) or len(records) > 8:
        return None, 'invalid-view-count'
    if len(records) < config['minimum_agreeing_views']:
        return None, 'insufficient-independent-views'
    names = set(); transforms = set(); hashes = set(); texts = []; model_ids = set()
    for record in records:
        if not isinstance(record, dict) or not {'view_id','transform','image_sha256','text','score','model_id'} <= set(record):
            return None, 'invalid-view-record'
        if not isinstance(record['view_id'], str) or not record['view_id'] or record['view_id'] in names:
            return None, 'duplicate-view-identity'
        names.add(record['view_id'])
        if not isinstance(record['model_id'], str) or not record['model_id']:
            return None, 'missing-specialist-identity'
        model_ids.add(record['model_id'])
        try:
            expected = c.image_digest(view_image(image,bbox,record['transform']))
        except (ValueError, TypeError):
            return None, 'undeclared-view-transform'
        if expected != record['image_sha256']:
            return None, 'view-pixels-differ'
        if record['transform'] in transforms or expected in hashes:
            return None, 'duplicate-view-pixels'
        transforms.add(record['transform']);hashes.add(expected)
        text = record['text']
        if not isinstance(text,str) or len(text)>config['maximum_output_characters'] or not text or not any(not char.isspace() for char in text) or any(unicodedata.category(char)=='Cc' for char in text):
            return None, 'empty-or-invalid-specialist-text'
        try:
            if not math.isfinite(float(record['score'])):
                return None, 'nonfinite-specialist-score'
        except (TypeError,ValueError,OverflowError):
            return None, 'invalid-specialist-score'
        texts.append(text)
    if len(model_ids) != 1:
        return None, 'cross-model-vote-not-supported'
    if 'raw' not in transforms:
        return None, 'raw-view-required'
    raw_text = next(record['text'] for record in records if record['transform'] == 'raw')
    # ``key`` projects each view's text onto the compared value; a None projection is not a vote.
    values = [text if key is None else key(text) for text in texts]
    raw_value = raw_text if key is None else key(raw_text)
    counts = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    agreeing = max(counts.values()) if counts else 0
    if agreeing < config['minimum_agreeing_views']:
        return None, 'exact-unicode-view-disagreement'
    # Exact Unicode agreement of at least the configured number of views; the raw view must carry the agreed value.
    if raw_value is None or counts[raw_value] < config['minimum_agreeing_views'] or counts[raw_value] < agreeing:
        return None, 'raw-view-outside-agreement'
    return raw_value, None


def _recorded_score(record):
    if not isinstance(record,dict):
        return None
    try:
        value=float(record.get('score'))
        return value if math.isfinite(value) else None
    except (ValueError,TypeError,OverflowError):
        return None


def _unique_occurrence(text, part):
    first=text.find(part)
    return first>=0 and text.find(part,first+1)<0


def _context_splice(image, candidate, baseline, records, specialist):
    """Map actual remainder OCR to one raw baseline string, preserving its bytes."""
    index=candidate['splice_baseline_index']
    declarations=candidate['context_crops']
    parent_sha256=c.baseline_parent_digest(baseline[index])
    if candidate['baseline_parent_sha256']!=parent_sha256:
        return None,'baseline-parent-provenance-differs',None
    if not isinstance(records,list) or len(records)!=len(declarations):
        return None,'mixed-baseline-context-required',None
    by_role={}
    for record in records:
        if not isinstance(record,dict) or not {'role','bbox','image_sha256','text','score','model_id'}<=set(record):
            return None,'invalid-baseline-context-record',None
        role=record['role']
        if role in by_role or role not in ('left','right'):
            return None,'duplicate-or-unknown-baseline-context',None
        by_role[role]=record
    for declaration in declarations:
        record=by_role.get(declaration['role'])
        if record is None or record['bbox']!=declaration['bbox'] or record['model_id']!='production-baseline':
            return None,'baseline-context-geometry-or-model-differs',None
        if record.get('baseline_parent_sha256')!=parent_sha256:
            return None,'baseline-context-parent-provenance-differs',None
        if c.image_digest(view_image(image,declaration['bbox'],'raw'))!=record['image_sha256']:
            return None,'baseline-context-pixels-differ',None
        text=record['text']
        if not isinstance(text,str) or not text or len(text)>4096 or any(unicodedata.category(char)=='Cc' for char in text) or not anchors(text):
            return None,'empty-or-unreliable-baseline-context',None
        if _recorded_score(record) is None:
            return None,'invalid-baseline-context-score',None
    original=baseline[index]['text']
    left=by_role['left']['text'] if 'left' in by_role else ''
    right=by_role['right']['text'] if 'right' in by_role else ''
    if (left and (not original.startswith(left) or not _unique_occurrence(original,left))) or (right and (not original.endswith(right) or not _unique_occurrence(original,right))):
        return None,'context-not-unique-raw-prefix-suffix',None
    old_anchors=anchors(original)
    if (left and not _unique_occurrence(old_anchors,anchors(left))) or (right and not _unique_occurrence(old_anchors,anchors(right))):
        return None,'context-letter-number-mapping-ambiguous',None
    start=len(left);end=len(original)-len(right) if right else len(original)
    if start>end:
        return None,'context-prefix-suffix-overlap',None
    # Existing separator whitespace is retained byte-for-byte. Specialist output
    # remains verbatim; no stripping, bracket construction or normalization.
    while start<end and original[start].isspace():
        start+=1
    while end>start and original[end-1].isspace():
        end-=1
    observed=original[start:end]
    outside,digits=c.ordinary_anchor_evidence(observed)
    confirmed=c.is_face_text(observed, allow_arms=False)
    # Digits and letters outside the observed bracket span are ordinary-body
    # evidence; letters inside the span may be baseline misreads of face glyphs.
    # An omitted independent body token such as "o" next to the brackets must
    # never be absorbed into the regional edit.
    if not confirmed and (digits or outside):
        return None,'mixed-region-ordinary-letter-number-protected',None
    # The pixel-separated ROI must map to exactly one bracketed fragment whose
    # interior holds at least two and at most the replacement's glyphs: exterior
    # arms, extra interior tokens or multiple expressions cannot map uniquely.
    if not confirmed and (not c.single_bracket_pair(observed) or not 2 <= c.interior_glyphs(observed) <= c.interior_glyphs(specialist)):
        return None,'mixed-region-not-confirmed-face-fragment',None
    result=original[:start]+specialist+original[end:]
    if result[:start]!=original[:start] or result[start+len(specialist):]!=original[end:]:
        raise RuntimeError('Baseline outside regional splice changed')
    return result,None,{'baseline_index':index,'original_text_span':[start,end],
                         'original_face_text':observed,'replacement_text':specialist,
                         'preserved_prefix':original[:start],'preserved_suffix':original[end:],
                         'context_crops':declarations,'context_records':records,
                         'baseline_parent_sha256':parent_sha256,
                          'ordinary_prefix_suffix_preserved':True}


def _whole_reading_proof(parent, index, whole):
    """Splice proof for a face agreed inside the specialist's whole reading of one baseline region."""
    prefix,face,suffix,middle=whole
    original=parent['text'];start=len(prefix);end=len(original)-len(suffix)
    result=original[:start]+face+original[end:]
    if result[:start]!=original[:start] or result[start+len(face):]!=original[end:]:
        raise RuntimeError('Baseline outside whole-reading splice changed')
    return result,{'baseline_index':index,'original_text_span':[start,end],'original_face_text':middle,'replacement_text':face,
                   'preserved_prefix':prefix,'preserved_suffix':suffix,'context_crops':[],'context_records':[],
                   'baseline_parent_sha256':c.baseline_parent_digest(parent),'ordinary_prefix_suffix_preserved':True,
                   'method':'whole-reading-agreement'}


def fuse_candidates(image, baseline, candidates, views, config=None, candidate_config=None, contexts=None, classifier=None):
    """Return complete region output and per-crop acceptance/retention evidence.

    Baseline indices refer only to actual OCR regions. A mixed parent can be
    spliced only after real pixel-separated context crops bind unique raw body
    strings; character offsets identify text edits, never pixel coordinates.
    """
    config = configuration(config)
    candidate_config = c.configuration(candidate_config)
    array = c.rgb_image(image,candidate_config['maximum_pixels'])
    height,width = array.shape[:2]
    entries = c.baseline_regions(baseline,(width,height),candidate_config['maximum_baseline_regions'])
    contexts={} if contexts is None else contexts
    if not isinstance(contexts,dict) or len(contexts)>candidate_config['maximum_candidates']:
        raise ValueError('Unbounded baseline context collection')
    if not isinstance(candidates,list) or len(candidates)>candidate_config['maximum_candidates'] or not isinstance(views,dict):
        raise ValueError('Unbounded candidate/view collection')
    # Recompute localization from the original inputs. Evidence fields supplied
    # by a caller cannot manufacture a confirmed region or omit a body neighbor.
    regenerated = c.locate_candidates(array,baseline,candidate_config,classifier=classifier)
    expected = {item['candidate_id']:item for item in regenerated['candidates']}
    if len(views)>candidate_config['maximum_candidates'] or not set(views).issubset(expected) or not set(contexts).issubset(expected):
        raise ValueError('View collection includes unknown automatic candidates')
    replacements={};removed=set();insertions=[];decisions=[];seen=set();consumed=set()
    for candidate in candidates:
        if not isinstance(candidate,dict) or 'candidate_id' not in candidate:
            raise ValueError('Missing candidate identity')
        identity=candidate['candidate_id']
        if identity in seen:
            raise ValueError('Duplicate candidate identity')
        seen.add(identity)
        audited=expected.get(identity)
        if audited is None or candidate != audited:
            raise ValueError('Candidate differs from automatic original-image localization')
        bbox=audited['bbox'];indices=audited['baseline_indices'];original=audited['baseline_text'];evidence=audited['evidence']
        whole=None
        if audited['route_eligible']:
            text,reason=_agreement(array,bbox,views.get(identity,[]),config)
            # A candidate that owns exactly one mixed baseline region with measured bracket strokes may
            # agree on the face inside its whole reading while the surrounding bytes equal the baseline's.
            mixed_reading=reason=='exact-unicode-view-disagreement' or (reason is None and not c.is_face_text(text))
            if mixed_reading and audited['splice_baseline_index'] is None and len(indices)==1 and evidence.get('pair_geometry') is not None:
                parts,whole_reason=_agreement(array,bbox,views.get(identity,[]),config,key=lambda value:_reading_middle(original,value))
                if whole_reason is None:
                    whole=parts;text=parts[1];reason=None
                elif reason is None:
                    reason='specialist-output-not-confirmed-face-grammar'
        else:
            text,reason=None,audited['route_reason']
        if reason is None and (evidence['edge_ink'] or evidence['clipped_canvas']):
            reason='incomplete-or-edge-clipped-crop'
        if reason is None and evidence['neighbor_body_overlap'] and not evidence['mixed_region_separable']:
            reason='neighbor-body-overlap'
        # Owned regions accept bracket-pair shapes or frozen-threshold classifier
        # evidence; a pixel-only insertion (no baseline region) still needs shapes.
        independent = evidence.get('independent_face_evidence', evidence['extra_shape_support'])
        if reason is None and (not evidence['pixel_support'] or not independent or (not indices and not evidence['extra_shape_support'])):
            reason='insufficient-independent-pixel-shape-support'
        if reason is None and not c.is_face_text(text):
            reason='specialist-output-not-confirmed-face-grammar'
        splice=None;spliced_text=None;mixed_index=audited['splice_baseline_index']
        if reason is None and mixed_index is not None:
            spliced_text,reason,splice=_context_splice(array,audited,entries,contexts.get(identity,[]),text)
        if reason is None and whole is not None:
            # The baseline bytes being replaced must themselves be one bracketed non-body fragment.
            middle=whole[3]
            outside,digits=c.ordinary_anchor_evidence(middle)
            if not middle or not any(char in c.ATOMS.bracket_chars for char in middle):
                reason='whole-reading-baseline-middle-not-bracketed'
            elif not c.is_face_text(middle) and (digits or outside or not c.single_bracket_pair(middle)):
                reason='mixed-region-ordinary-letter-number-protected'
            else:
                spliced_text,splice=_whole_reading_proof(entries[indices[0]],indices[0],whole)
        claim_indices=indices if mixed_index is None else [mixed_index]
        if reason is None and any(index in consumed for index in claim_indices):
            reason='baseline-region-already-consumed'
        if reason is None and mixed_index is None and not indices and not config['allow_pixel_only_insertions']:
            reason='pixel-only-insertion-disabled'
        if reason is None and len(indices)>1 and not c.is_face_text(original):
            # Multiple OCR fragments may describe one expression. Two complete
            # expressions or independent symbol fields cannot collapse into one.
            reason='multiple-baseline-regions-not-one-face-expression'
        geometry=evidence.get('pair_geometry')
        if reason is None and indices and geometry is not None:
            # Measured bracket strokes exist: the replacement must account for the
            # pixels. Exterior ink on a side requires exterior atoms (arms/tails)
            # on that side and vice versa, so an independent body token beside
            # the face can never be absorbed and arms are never invented; the
            # interior must hold at least two components (a lone (o) is body).
            sides=c.exterior_atoms(text)
            # Baseline body bytes kept beside a whole-reading face explain that side's exterior ink.
            explained={'left':bool(whole and whole[0]),'right':bool(whole and whole[2])}
            if sides is None or any(not explained[side] and sides[side]!=geometry['exterior_ink'][side] for side in ('left','right')):
                reason='exterior-ink-and-arms-disagree'
            elif geometry['interior_components']<2:
                reason='insufficient-interior-ink'
        elif reason is None and indices and not c.is_face_text(original):
            # Without measured strokes (classifier-only evidence) the baseline's
            # raw reading keeps its conservative text guards: digits and exterior
            # letters are ordinary-body anchors, the span must be one bracket
            # pair, and it cannot hold more glyphs than the face it becomes.
            outside,digits=c.ordinary_anchor_evidence(original)
            core=c.strip_exterior_atoms(original)
            inside=c.interior_glyphs(core)
            if digits:
                reason='ordinary-letter-number-anchor-change'
            elif outside or not c.single_bracket_pair(core) or inside < 2 or inside > c.interior_glyphs(text):
                reason='ordinary-body-region-protected'
        if reason is None and text==original and indices:
            reason='unchanged-baseline'
        accepted=reason is None
        decision={'candidate_id':identity,'bbox':bbox,'accepted':accepted,'reason':reason or 'exact-views-and-independent-pixels',
                  'original_text':entries[mixed_index]['text'] if mixed_index is not None else original,'specialist_text':text,'baseline_indices':claim_indices,
                  'splice':splice,'pixel_evidence':evidence,'view_scores':[_recorded_score(record) for record in views.get(identity,[])],
                  'score_semantics':'Uncalibrated scores are recorded; no cross-model score comparison.'}
        decisions.append(decision)
        if accepted:
            score=float(views[identity][0]['score'])
            region={'bbox':bbox,'text':text,'score':score,'source':'specialist-exact-views','ink_bbox':list(evidence['ink_bbox']),
                    'candidate_id':identity,'model_id':views[identity][0]['model_id'],'baseline_indices':list(claim_indices)}
            if mixed_index is not None:
                parent=entries[mixed_index]
                region.update(bbox=parent['bbox'],text=spliced_text,score=parent['score'],source='baseline-with-regional-splice',baseline_index=mixed_index,regional_bbox=bbox,regional_ink_bbox=list(evidence['ink_bbox']),regional_text=text,splice=splice)
                replacements[mixed_index]=region;consumed.add(mixed_index)
            elif whole is not None:
                parent=entries[indices[0]];geometry=evidence['pair_geometry']
                regional=[geometry['left_bbox'][0],min(geometry['left_bbox'][1],geometry['right_bbox'][1]),geometry['right_bbox'][2],max(geometry['left_bbox'][3],geometry['right_bbox'][3])]
                region.update(bbox=parent['bbox'],text=spliced_text,score=parent['score'],source='baseline-with-regional-splice',baseline_index=indices[0],regional_bbox=regional,regional_ink_bbox=regional,regional_text=text,splice=splice)
                replacements[indices[0]]=region;consumed.add(indices[0])
            elif indices:
                replacements[indices[0]]=region;removed.update(indices[1:]);consumed.update(indices)
            else:
                insertions.append(region)
    output=[]
    for entry in entries:
        index=entry['index']
        if index in removed:
            continue
        output.append(replacements.get(index,{'bbox':entry['bbox'],'text':entry['text'],'score':entry['score'],'source':'baseline','baseline_index':index}))
    output.extend(insertions)
    output.sort(key=lambda item:(item['bbox'][1],item['bbox'][0]))
    return {'schema_version':SCHEMA,'image_size':[width,height],'image_sha256':c.image_digest(array),'config':config,
            'candidate_config':candidate_config,'regions':output,'decisions':decisions,'accepted_count':sum(item['accepted'] for item in decisions)}
