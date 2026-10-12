"""Real CPU probe failure, atomic activation and explicit rollback evidence."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localocr.bundles import validate_bundle, install_bundle, activate_bundle, load_active

def main():
    p=argparse.ArgumentParser(); p.add_argument('--store',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    store,source,output=args.store.resolve(),args.source.resolve(),args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    pointer=(store/'active.json').read_bytes(); original=load_active(store)
    report={'scope':'real baseline versions and genuine invalid ONNX probe, no candidate promotion','checks':{}}
    with tempfile.TemporaryDirectory(prefix='bad-bundle-',dir=output) as tmp:
        bad=Path(tmp)/'package'; shutil.copytree(source,bad)
        manifest=json.loads((bad/'manifest.json').read_text()); manifest['version']='invalid-probe-test'
        entry=manifest['models']['rec']; model=bad/entry['file']; model.write_bytes(b'invalid ONNX, correct manifest hash')
        entry['size']=model.stat().st_size; entry['sha256']=hashlib.sha256(model.read_bytes()).hexdigest()
        (bad/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        validate_bundle(bad)  # integrity succeeds; actual model load must fail
        try: install_bundle(bad,store)
        except ValueError as error:
            assert 'probe' in str(error).lower(),str(error)
        else: raise AssertionError('Invalid ONNX was installed')
        assert (store/'active.json').read_bytes()==pointer
        assert not (store/'versions'/'invalid-probe-test').exists()
        assert not list(store.glob('.install-*')) and not (store/'.bundle.lock').exists()
        report['checks']['invalid_model_real_probe_preserves_active']=True
    captured=load_active(store)
    switched=activate_bundle(store,'baseline-2')
    assert captured.version=='baseline-1' and switched.version=='baseline-2' and load_active(store).version=='baseline-2'
    assert captured.root.is_dir() and validate_bundle(captured.root).version=='baseline-1'
    report['checks']['immutable_old_snapshot_survives_switch']=True
    rolled=activate_bundle(store,'baseline-1')
    assert rolled.version==original.version and load_active(store).version==original.version
    report['checks']['real_probe_and_atomic_rollback']=True
    assert not list(store.glob('.active-*')) and not (store/'.bundle.lock').exists()
    report['checks']['owned_staging_cleanup']=True
    (output/'bundle-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
