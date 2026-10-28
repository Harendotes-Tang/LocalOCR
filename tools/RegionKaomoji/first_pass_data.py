"""Metadata-only fresh-data lineage guard for the optional first-pass round.

The exposure loader replays only its explicitly authorized historical sources;
this helper never opens new or unused heldout labels/images. Its receipt proves
bound generation/exclusion lineage, not OCR quality or first-use authorization.
"""
from tools.PretrainedOcr import common, quality_evaluate as quality, round_profile as rp
from . import evaluate as scene, exposure


SCHEMA = 'localocr.first-pass-fresh-data-proof.v1'
PREVIOUS_EXPOSURE_SHA256 = '430203b31983d8729d05dcaa2a13b11ca14ea1e1f75e5f5bfc949047ab4e7de9'
SEED = 20261027
GENERATORS = {'fresh_crop': 'localocr.browser-crop-data.20261016.v1',
              'fresh_scene': 'localocr.browser-card-scenes.20261015.v1'}


def _identity(source):
    if (not isinstance(source, dict) or source.get('split') not in ('train', 'dev', 'heldout')
            or not isinstance(source.get('dataset'), str) or not source['dataset']):
        raise ValueError('Invalid exposure source identity')
    for field in ('manifest', 'dataset_metadata', 'freeze'):
        ref = source.get(field)
        if (not isinstance(ref, dict) or set(ref) != {'file', 'sha256'} or not isinstance(ref['file'], str)
                or not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in ref['sha256'])):
            raise ValueError('Original exposure source references are required')
    return str(scene.safe_path(source.get('dataset', ''))), source['split']


def verify_fresh_data(crop_dataset, scene_dataset):
    """Verify metadata/exposure lineage; unreadable or malformed evidence rejects."""
    try:
        return _verify_fresh_data(crop_dataset, scene_dataset)
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('Unreadable or malformed fresh-data evidence') from error


def _verify_fresh_data(crop_dataset, scene_dataset):
    """Return a reproducible bound lineage receipt, or reject before exposure replay.

    Only exact v6 roots and the current before-dev freeze are supported. Source
    rows outside the immutable predecessor inventory and the three approved dev/
    contract additions are rejected before ``exposure.load_bound`` can read any
    labels. Existing consumed heldouts retain their original first-use binding.
    No file is written and no model/browser session is created.
    """
    root = scene.safe_path(rp.FIRST_PASS_POSITION.root)
    paths = {'fresh_crop': scene.safe_path(crop_dataset), 'fresh_scene': scene.safe_path(scene_dataset)}
    if paths != {'fresh_crop': root / 'data/browser-crops-v6', 'fresh_scene': root / 'scenes/browser-cards-v6'}:
        raise ValueError('Fresh first-pass datasets require the exact new v6 roots')
    captured = {}

    def capture(path):
        path = scene.safe_path(path)
        value = quality.reference(path)
        if str(path) in captured and captured[str(path)] != value:
            raise ValueError('Fresh-data evidence changed during verification')
        captured[str(path)] = value
        return value

    freeze_path = root / 'validation/source-inventory-before-dev.json'
    freeze_ref = capture(freeze_path)
    frozen = common.read(freeze_path)
    sources = quality.source_inventory()
    if (not isinstance(frozen, dict) or frozen.get('schema_version') != 'localocr.first-pass-before-dev-freeze.v1'
            or frozen.get('source_inventory') != sources):
        raise ValueError('Fresh-data guard requires the current before-dev source freeze')
    references = frozen.get('references', [])
    datasets = frozen.get('datasets', {})
    if not isinstance(references, list) or not isinstance(datasets, dict):
        raise ValueError('Before-dev dataset/reference bindings are required')
    bindings, metadata = {}, {}
    for role, path in paths.items():
        meta_ref, data_ref = capture(path / 'dataset.json'), capture(path / 'freeze.json')
        meta = common.read(path / 'dataset.json')
        if (not isinstance(meta, dict) or meta.get('browser_rendered') is not True or meta.get('synthetic_only') is not True
                or type(meta.get('seed')) is not int or meta['seed'] != SEED
                or meta.get('generator_id') != GENERATORS[role]):
            raise ValueError('Fresh-data generator, rendering or seed identity differs')
        binding = common.dataset_binding(path) if role == 'fresh_crop' else scene.scene_binding(path)
        if (scene.hash_object(datasets.get(role)) != scene.hash_object({'freeze': data_ref, 'binding': binding})
                or meta_ref not in references or data_ref not in references):
            raise ValueError('Fresh dataset differs from the before-dev frozen references')
        bindings[role], metadata[role] = binding, meta
    excluded = metadata['fresh_crop'].get('exposure_exclusions')
    if not isinstance(excluded, dict) or excluded != metadata['fresh_scene'].get('exposure_exclusions'):
        raise ValueError('Both fresh datasets must bind the same exposure exclusions')
    excluded_path = root / 'exposure-exclusions.json'
    excluded_ref = capture(excluded_path)
    if (excluded.get('file') != str(excluded_path) or excluded.get('sha256') != excluded_ref['sha256']
            or excluded_ref not in references):
        raise ValueError('Fresh exposure inventory path/bytes are not bound before development')
    prior_path = scene.safe_path(rp.OWNED_POSITION.root / 'exposure-exclusions.json')
    prior_ref = capture(prior_path)
    if prior_ref['sha256'] != PREVIOUS_EXPOSURE_SHA256 or prior_ref not in references:
        raise ValueError('Immutable predecessor exposure inventory is required')
    prior, current = common.read(prior_path), common.read(excluded_path)
    if (not isinstance(prior, dict) or not isinstance(current, dict)
            or prior.get('schema_version') != exposure.SCHEMA or current.get('schema_version') != exposure.SCHEMA):
        raise ValueError('Unknown exposure inventory schema')
    old_sources, new_sources = prior.get('sources'), current.get('sources')
    if not isinstance(old_sources, list) or not old_sources or not isinstance(new_sources, list):
        raise ValueError('Original exposure source bindings are required')
    old = {_identity(source): source for source in old_sources}
    new = {_identity(source): source for source in new_sources}
    required = {(str(scene.safe_path(rp.OWNED_POSITION.root / 'scenes/browser-cards-v5')), 'dev'),
                (str(scene.safe_path(rp.OWNED_POSITION.root / 'data/browser-crops-v5')), 'train'),
                (str(scene.safe_path(rp.OWNED_POSITION.root / 'data/browser-crops-v5')), 'dev')}
    if (len(old) != len(old_sources) or len(new) != len(new_sources) or not required <= set(new)
            or set(new) != set(old) | required
            or any(scene.hash_object(new.get(key)) != scene.hash_object(value) for key, value in old.items())
            or any('first_use' in new[key] for key in required - set(old))):
        raise ValueError('Fresh exposure loses/changes original sources or adds an unauthorized source')
    seen_metadata = {source['dataset_metadata']['sha256'] for source in new_sources}
    seen_splits = {source['manifest']['sha256'] for source in new_sources}
    if any(binding['dataset_sha256'] in seen_metadata
           or set(binding['split_sha256'].values()) & seen_splits for binding in bindings.values()):
        raise ValueError('Fresh dataset copies a previously exposed metadata/split digest')
    loaded = exposure.load_bound(excluded)
    if loaded.binding != excluded:
        raise ValueError('Replayed exposure inventory binding differs')
    if quality.source_inventory() != sources or any(quality.reference(ref['file']) != ref for ref in captured.values()):
        raise ValueError('Fresh-data sources or bound references changed during verification')
    return {'schema_version': SCHEMA, 'source_inventory': sources, 'before_dev': freeze_ref,
            'datasets': bindings, 'exposure_exclusions': excluded, 'references': list(captured.values()),
            'promote': False, 'enabled': False, 'app_acceptance': False}
