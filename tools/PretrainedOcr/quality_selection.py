"""Raw Unicode development evidence and conservative face checkpoint selection.

This module ranks research checkpoints only. It never authorizes production or
replaces the actual CPU region/fusion protection gate. Character-family filtered
metrics are diagnostic projections; reference labels and predictions stay raw.
"""
from __future__ import annotations
import math
import re
import unicodedata
from collections import Counter

SCHEMA = 'localocr.quality.development.v1'
REQUIRED_KINDS = frozenset(('face', 'plain', 'punctuation', 'blank', 'graphic'))
ALLOWED_KINDS = REQUIRED_KINDS | {'emoji'}
FAMILIES = ('PS', 'decimal', 'ASCIIspace', 'letters', 'combining_joiner', 'other')
COUNT_FIELDS = ('count', 'exact', 'characters', 'character_errors', 'symbol_characters',
                'symbol_errors', 'negative_rows', 'negative_nonempty',
                'reference_correct_to_wrong', 'erroneous_rewrites', 'new_negative_nonempty')


def _text(value):
    if not isinstance(value, str) or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError('Evidence must contain raw Unicode scalar strings')
    return value


def family(c):
    category = unicodedata.category(c)
    if c == ' ': return 'ASCIIspace'
    if category[0] in 'PS': return 'PS'
    if c.isdecimal(): return 'decimal'
    if category[0] == 'L': return 'letters'
    if category[0] == 'M' or c in ('\u200c', '\u200d'): return 'combining_joiner'
    return 'other'


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def _pairs(rows, predictions, require_all=True):
    if not isinstance(rows, (list, tuple)) or not rows or len(rows) != len(predictions):
        raise ValueError('Nonempty development rows and complete predictions required')
    ids, result, kinds = set(), [], set()
    for row, prediction in zip(rows, predictions):
        if not isinstance(row, dict) or not isinstance(prediction, dict):
            raise ValueError('Rows and predictions must be identity-bound records')
        identity = row.get('id')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Missing or duplicate row identity')
        ids.add(identity)
        kind = row.get('kind'); kinds.add(kind)
        if kind not in ALLOWED_KINDS:
            raise ValueError('Unsupported development kind')
        expected, actual = _text(row.get('text')), _text(prediction.get('text'))
        if (kind in ('blank', 'graphic')) != (expected == ''):
            raise ValueError('Negative kinds require empty labels; positive kinds require text')
        if 'id' not in prediction and 'image' not in prediction:
            raise ValueError('Prediction identity is missing')
        if 'id' in prediction and prediction['id'] != identity:
            raise ValueError('Prediction row identity differs')
        if 'image' in prediction and prediction['image'] != row.get('image'):
            raise ValueError('Prediction image identity differs')
        result.append((row, expected, actual))
    if require_all and not REQUIRED_KINDS <= kinds:
        raise ValueError('Required development kinds are missing')
    return result


def _metric(pairs, reference=None, project=None):
    result = dict.fromkeys(COUNT_FIELDS, 0)
    for row, expected, actual in pairs:
        old = reference[row['id']] if reference is not None else None
        if project is not None:
            expected = ''.join(c for c in expected if family(c) == project)
            actual = ''.join(c for c in actual if family(c) == project)
            if old is not None: old = ''.join(c for c in old if family(c) == project)
        symbols = lambda s: ''.join(c for c in s if unicodedata.category(c)[0] in 'PS')
        result['count'] += 1; result['exact'] += expected == actual
        result['characters'] += len(expected)
        result['character_errors'] += distance(expected, actual)
        result['symbol_characters'] += len(symbols(expected))
        result['symbol_errors'] += distance(symbols(expected), symbols(actual))
        negative = row['kind'] in ('blank', 'graphic')
        result['negative_rows'] += negative
        result['negative_nonempty'] += negative and bool(actual)
        if old is not None:
            result['reference_correct_to_wrong'] += old == expected and actual != expected
            result['erroneous_rewrites'] += old != expected and actual != expected and old != actual
            result['new_negative_nonempty'] += negative and not old and bool(actual)
    result['exact_accuracy'] = result['exact'] / result['count']
    result['cer'] = result['character_errors'] / result['characters'] if result['characters'] else None
    result['symbol_cer'] = result['symbol_errors'] / result['symbol_characters'] if result['symbol_characters'] else None
    return result


def summarize(rows, predictions, reference=None):
    pairs = _pairs(rows, predictions)
    baseline = None if reference is None else {r['id']: actual for r, _, actual in _pairs(rows, reference)}
    result = {'schema_version': SCHEMA, 'normalization': 'none; raw Unicode and ASCII spaces',
              'reference_available': reference is not None, 'overall': _metric(pairs, baseline),
              'by_kind': {kind: _metric([p for p in pairs if p[0]['kind'] == kind], baseline)
                          for kind in sorted({r['kind'] for r in rows})},
              'by_character_family': {f: _metric(pairs, baseline, f) for f in FAMILIES}}
    validate_summary(result)
    return result


def _validate_metric(metric):
    if not isinstance(metric, dict): raise ValueError('Missing development metric')
    for key in COUNT_FIELDS:
        value = metric.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Development counts must be nonnegative integers')
    n = metric['count']
    if (n < 1 or metric['exact'] > n or metric['negative_rows'] > n
            or metric['negative_nonempty'] > metric['negative_rows']
            or any(metric[k] > n for k in ('reference_correct_to_wrong', 'erroneous_rewrites'))
            or metric['new_negative_nonempty'] > metric['negative_nonempty']
            or metric['symbol_characters'] > metric['characters']):
        raise ValueError('Development counts disagree')
    for key, numerator, denominator in [('exact_accuracy', metric['exact'], n),
            ('cer', metric['character_errors'], metric['characters']),
            ('symbol_cer', metric['symbol_errors'], metric['symbol_characters'])]:
        value = metric.get(key)
        if not denominator:
            if value is not None: raise ValueError('Zero denominator requires null rate')
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or value < 0
              or not math.isclose(value, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Development rate is nonfinite or differs from counts')


def validate_summary(summary):
    if (not isinstance(summary, dict) or summary.get('schema_version') != SCHEMA
            or summary.get('normalization') != 'none; raw Unicode and ASCII spaces'
            or not isinstance(summary.get('reference_available'), bool)):
        raise ValueError('Unsupported raw development evidence')
    kinds, families = summary.get('by_kind'), summary.get('by_character_family')
    if (not isinstance(kinds, dict) or not REQUIRED_KINDS <= kinds.keys()
            or not kinds.keys() <= ALLOWED_KINDS or not isinstance(families, dict)
            or set(families) != set(FAMILIES)):
        raise ValueError('Required development kinds/families missing')
    _validate_metric(summary.get('overall'))
    for kind, value in kinds.items():
        _validate_metric(value)
        if kind in ('blank', 'graphic'):
            if value['characters'] or value['negative_rows'] != value['count']:
                raise ValueError('Negative development evidence differs')
        elif not value['characters'] or value['negative_rows']:
            raise ValueError('Positive development evidence differs')
    if not kinds['face']['symbol_characters']:
        raise ValueError('Face development lacks punctuation/symbol reference evidence')
    for key in COUNT_FIELDS:
        if sum(v[key] for v in kinds.values()) != summary['overall'][key]:
            raise ValueError('Kind aggregates differ from actual overall counts')
    for value in families.values():
        _validate_metric(value)
        if value['count'] != summary['overall']['count']:
            raise ValueError('Character-family sample denominator differs')
    if sum(v['characters'] for v in families.values()) != summary['overall']['characters']:
        raise ValueError('Character-family reference counts differ')
    return summary


def face_better(candidate, incumbent):
    """Require Pareto face improvement and no increased empty-label false output.

    Overall accuracy never participates. This research comparison is not the
    final ordinary-text, two-view or automatic-region acceptance decision.
    """
    validate_summary(candidate); validate_summary(incumbent)
    if candidate['by_kind'].keys() != incumbent['by_kind'].keys():
        raise ValueError('Development class inventory differs')
    for kind in candidate['by_kind']:
        for key in ('count', 'characters', 'symbol_characters'):
            if candidate['by_kind'][kind][key] != incumbent['by_kind'][kind][key]:
                raise ValueError('Development reference denominator differs')
    a, b = candidate['by_kind']['face'], incumbent['by_kind']['face']
    no_regression = a['exact'] >= b['exact'] and a['cer'] <= b['cer'] and a['symbol_cer'] <= b['symbol_cer']
    improvement = a['exact'] > b['exact'] or a['cer'] < b['cer'] or a['symbol_cer'] < b['symbol_cer']
    negatives_ok = all(candidate['by_kind'][k]['negative_nonempty'] <= incumbent['by_kind'][k]['negative_nonempty']
                       for k in ('blank', 'graphic'))
    return no_regression and improvement and negatives_ok


def original_class_coverage(training_rows, original_tokens):
    if (not isinstance(original_tokens, (list, tuple)) or not original_tokens
            or any(not isinstance(t, str) or not t for t in original_tokens)
            or len(set(original_tokens)) != len(original_tokens)):
        raise ValueError('Original token identities must be nonempty and unique')
    observed = Counter(c for row in training_rows for c in _text(row.get('text')))
    if not observed: raise ValueError('Positive training evidence required')
    original = {_text(t) for t in original_tokens if len(t) == 1}
    groups = {}
    for f in FAMILIES:
        tokens = {c for c in original if family(c) == f}
        groups[f] = {'original_codepoints': len(tokens), 'supported_original_codepoints': len(tokens & observed.keys()),
                     'train_occurrences': sum(observed[c] for c in tokens)}
    return {'source': 'provided training rows only; never development or heldout',
            'original_codepoints': len(original), 'supported_original_codepoints': len(original & observed.keys()),
            'train_codepoints': len(observed), 'families': groups,
            'basic_symbols': {c: {'in_original_vocabulary': c in original, 'train_occurrences': observed[c]}
                              for c in '()[]{}（）［］｛｝：~'}}


def two_view_statistics(rows, first, second, first_sha256, second_sha256):
    a, b = _pairs(rows, first, False), _pairs(rows, second, False)
    if len(first_sha256) != len(rows) or len(second_sha256) != len(rows):
        raise ValueError('Two-view pixel identity count differs')
    result = {'pairs': len(rows), 'raw_agreement': 0, 'both_exact': 0, 'agreement_and_exact': 0}
    for (_, expected, x), (_, _, y), sha1, sha2 in zip(a, b, first_sha256, second_sha256):
        if any(not isinstance(s, str) or not re.fullmatch('[0-9a-f]{64}', s) for s in (sha1, sha2)) or sha1 == sha2:
            raise ValueError('Two views require distinct bound pixel identities')
        result['raw_agreement'] += x == y
        result['both_exact'] += x == y == expected
        result['agreement_and_exact'] += x == y == expected
    return result
