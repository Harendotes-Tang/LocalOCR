"""Real frozen Windows EXE acceptance; harness Python never executes OCR code.

The two packaged EXEs run with a system-only PATH and no Python environment.
Outputs and models remain in a fresh ignored artifacts directory. This verifies
this machine's standalone contract, not another machine or APP acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import psutil
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def owned_output(path):
    target = Path(path).resolve()
    artifact = (ROOT / 'artifacts').resolve()
    if artifact.parent != ROOT.resolve() or artifact not in target.parents:
        raise ValueError('Verification output must belong to repository artifacts')
    return target


def process_alive(identity):
    pid, created = identity
    try:
        process = psutil.Process(pid)
        return process.create_time() == created and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class Evidence:
    def __init__(self, runtime, output):
        self.runtime, self.output = runtime, output
        self.environment = os.environ.copy()
        for name in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'LOCALOCR_CORE_NONCE', 'LOCALOCR_CORE_PARENT'):
            self.environment.pop(name, None)
        system = Path(os.environ.get('SystemRoot', r'C:\Windows'))
        self.environment['PATH'] = os.pathsep.join([str(system / 'System32'), str(system)])
        self.environment['PYTHONNOUSERSITE'] = '1'
        self.commands = []
        self.checks = {}
        self.monitor = []

    def spawn(self, executable, arguments, label):
        binary = self.runtime / executable
        if not binary.is_file():
            raise ValueError('Standalone executable missing: ' + executable)
        log = (self.output / (label + '.log')).open('wb')
        process = subprocess.Popen([str(binary), *map(str, arguments)], cwd=self.output, env=self.environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        self.commands.append({'label': label, 'executable': executable, 'arguments': list(map(str, arguments)), 'pid': process.pid})
        return process, log

    def run(self, executable, arguments, label, expected=0, timeout=110, watch=False):
        process, log = self.spawn(executable, arguments, label)
        observed = {'samples': 0, 'process_ids': [], 'executables': [], 'listening_connections': []}
        stop = threading.Event()
        monitor_errors = []
        def observe():
            identities = set();executables=set()
            while not stop.is_set():
                try:
                    parent = psutil.Process(process.pid)
                    processes = [parent, *parent.children(recursive=True)]
                    for child in processes:
                        try:
                            identities.add((child.pid, child.create_time()))
                            image=Path(child.exe()).resolve()
                            console_host=(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/conhost.exe').resolve()
                            if image.parent!=self.runtime.resolve() and image!=console_host:
                                raise AssertionError('Unexpected host executable in runtime subtree: '+str(image))
                            executables.add(str(image))
                            for connection in child.net_connections(kind='inet'):
                                if connection.status == psutil.CONN_LISTEN:
                                    observed['listening_connections'].append({'pid': child.pid, 'local': list(connection.laddr)})
                        except psutil.NoSuchProcess:
                            continue
                    observed['samples'] += 1
                except psutil.NoSuchProcess:
                    break
                except Exception as error:
                    monitor_errors.append(str(error))
                    break
                stop.wait(.03)
            observed['process_ids'] = sorted(pid for pid, created in identities)
            observed['executables'] = sorted(executables)
        thread = threading.Thread(target=observe, daemon=True)
        if watch:
            thread.start()
        started = time.monotonic()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=10)
            raise AssertionError('Harness deadline exceeded for ' + label)
        finally:
            stop.set()
            if watch:
                thread.join(timeout=5)
            log.close()
        self.commands[-1].update(exit_code=code, elapsed_seconds=time.monotonic()-started)
        if watch:
            observed['label'] = label
            self.monitor.append(observed)
            assert not monitor_errors, monitor_errors
            assert observed['samples'] > 0, 'No process monitoring samples'
            assert not observed['listening_connections'], observed
        assert code == expected, f'{label}: expected {expected}, got {code}; inspect {label}.log'
        return (self.output / (label + '.log')).read_text(encoding='utf-8', errors='replace')

    def mark(self, key):
        self.checks[key] = True


def assert_json_result(path, expected_version, dimensions):
    raw = Path(path).read_bytes();value = json.loads(raw.decode('utf-8'))
    assert value['schema_version'] == 'localocr.result.v1'
    assert value['model_version'] == expected_version
    assert value['image'] == dict(zip(('width','height'), dimensions))
    assert 'Not a probability' in value['score_semantics']
    assert value['regions'], 'Expected actual recognized text regions'
    joined = ''.join(row['text'] for row in value['regions'])
    assert 'HELLO' in joined and '123' in joined, joined
    assert any('\u4e00' <= char <= '\u9fff' for char in joined), 'Real Unicode recognition missing'
    assert any(ord(char)>127 for char in raw.decode('utf-8')), 'JSON does not retain UTF-8 Unicode'
    assert any(row['score'] != round(row['score'], 4) for row in value['regions']), 'All JSON scores unexpectedly rounded'
    for row in value['regions']:
        assert isinstance(row['text'], str)
        assert len(row['box']) == 4 and all(len(point)==2 for point in row['box'])
        assert all(type(v) in (int,float) and math.isfinite(v) for point in row['box'] for v in point)
        assert type(row['score']) in (int,float) and math.isfinite(row['score'])
        rect=row['rect'];x=max(0,math.floor(min(p[0] for p in row['box'])));y=max(0,math.floor(min(p[1] for p in row['box'])))
        right=min(dimensions[0],math.ceil(max(p[0] for p in row['box'])));bottom=min(dimensions[1],math.ceil(max(p[1] for p in row['box'])))
        assert rect=={'x':x,'y':y,'width':right-x,'height':bottom-y}
        assert rect['width']>0 and rect['height']>0
    return {'regions':len(value['regions']),'recognized_text':joined,'utf8_bytes':len(raw)}


def sample_image(fonts, destination, large=False):
    fonts = Path(fonts).resolve();profile=read(fonts/'fonts.json')
    cjk=next(record for record in profile['fonts'] if record['id']=='cjk')
    font=fonts/cjk['file']
    assert cjk['license']=='OFL-1.1'
    assert digest(font)==cjk['sha256'], 'Official font bytes changed'
    assert digest(fonts/cjk['license_file'])==cjk['license_sha256'], 'Official font license bytes changed'
    image=Image.new('RGB',(5000,3000) if large else (680,210),'white')
    draw=ImageDraw.Draw(image);size=56 if not large else 72
    face=ImageFont.truetype(str(font),size)
    draw.text((28,15),'HELLO 123',font=face,fill='black')
    draw.text((28,95),'离线文字测试',font=face,fill='black')
    if large:
        for y in range(280,2800,180):
            draw.text((30,y),'HELLO 123  本机文字',font=face,fill='black')
    image.save(destination)
    return image.size


def cancel_parent(evidence, arguments, output_path):
    process,log=evidence.spawn('local-ai-ocr.exe',arguments,'forced-parent-cancel')
    tracked={};deadline=time.monotonic()+20
    try:
        while time.monotonic()<deadline:
            assert process.poll() is None, 'OCR completed before Job cancellation probe'
            children=psutil.Process(process.pid).children(recursive=True)
            for child in children:
                tracked[(child.pid,child.create_time())]=child.cmdline()
            # Broker creates its core only after assigning itself to kill-on-close Job.
            if any('--_runtime-core' in command for command in tracked.values()):
                break
            time.sleep(.02)
        assert any('--_runtime-core' in command for command in tracked.values()), 'No real authenticated core descendant observed'
        foreign_output=b'Foreign concurrently created output must survive cancellation'
        foreign_partial=b'Foreign concurrently created partial must survive cancellation'
        with output_path.open('xb') as stream:stream.write(foreign_output)
        with output_path.with_name(output_path.name+'.partial').open('xb') as stream:stream.write(foreign_partial)
        process.kill();process.wait(timeout=10)
        end=time.monotonic()+10
        while time.monotonic()<end and any(process_alive(key) for key in tracked):
            time.sleep(.03)
        assert not any(process_alive(key) for key in tracked), 'Job left live descendants'
        result={'parent_pid':process.pid,'descendants_observed':[{'pid':pid,'command':cmd} for (pid,created),cmd in tracked.items()],
                'all_observed_descendants_reaped':True,'exit_code':process.returncode,
                'foreign_output_preserved':output_path.read_bytes()==foreign_output,
                'foreign_partial_preserved':output_path.with_name(output_path.name+'.partial').read_bytes()==foreign_partial}
        assert result['foreign_output_preserved'] and result['foreign_partial_preserved']
        claim=output_path.with_name(output_path.name+'.runtime-claim')
        assert claim.is_dir(), 'Forced cancellation should leave the caller-owned claim for caller cleanup'
        result['owned_claim_present_after_forced_termination']=True
        result['claim_files_before_caller_cleanup']=[p.name for p in claim.iterdir()]
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=10)
        log.close()
        # Forced termination cannot execute Python finally. Calling owner removes its own request files.
        claim=output_path.with_name(output_path.name+'.runtime-claim')
        if claim.exists():
            assert claim.resolve().parent==evidence.output.resolve() and not claim.is_symlink(), 'Caller cleanup must stay in its owned request directory'
            shutil.rmtree(claim)
    assert not output_path.with_name(output_path.name+'.runtime-claim').exists()
    assert output_path.read_bytes()==foreign_output and output_path.with_name(output_path.name+'.partial').read_bytes()==foreign_partial
    result['caller_owned_claim_cleanup']=True
    evidence.commands[-1].update(exit_code=process.returncode)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=ROOT/'artifacts/localocr-runtime-win-x64-v1')
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/baseline-package')
    parser.add_argument('--fonts',type=Path,default=ROOT/'artifacts/ocr-data-fonts-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/runtime-verification-v1')
    args=parser.parse_args()
    if os.name!='nt':raise ValueError('Real Windows Job acceptance requires Windows')
    runtime=args.runtime.resolve();source=args.source.resolve();output=owned_output(args.output)
    if output.exists():raise ValueError('Verification output already exists; supply a fresh owned directory')
    output.mkdir(parents=True);evidence=Evidence(runtime,output)
    before={p.relative_to(runtime).as_posix():digest(p) for p in runtime.rglob('*') if p.is_file()}
    store=output/'model-store';package1=output/'package-v1';package2=output/'package-v2'
    report={'schema_version':'localocr.runtime-verification.v1','scope':'real packaged Windows EXEs on this machine; no host Python OCR execution; no APP or cross-machine acceptance',
            'environment':{'PATH':evidence.environment['PATH'],'PYTHONHOME_removed':True,'PYTHONPATH_removed':True},'checks':evidence.checks}
    try:
        evidence.run('local-ai-ocr.exe',['--version'],'runtime-version')
        evidence.run('localocr-models.exe',['create-baseline','--models',source,'--output',package1,'--version','runtime-baseline-1'],'create-baseline')
        assert read(package1/'manifest.json')['version']=='runtime-baseline-1'
        assert (package1/'MODEL_LICENSES.md').is_file() and (package1/'APACHE_LICENSE.txt').is_file()
        evidence.mark('frozen_license_resources_create_baseline')
        evidence.run('localocr-models.exe',['install','--source',package1,'--store',store],'install-v1')
        evidence.run('localocr-models.exe',['activate','--store',store,'--version','runtime-baseline-1'],'activate-v1')
        listing=evidence.run('localocr-models.exe',['list','--store',store],'list-v1')
        assert json.loads(listing)['active']=='runtime-baseline-1'
        evidence.mark('real_cpu_probe_install_activate_list')
        active_before=(store/'active.json').read_bytes()
        for category in ('incompatible','bad-onnx'):
            package=output/category;shutil.copytree(package1,package)
            manifest=read(package/'manifest.json');manifest['version']='runtime-'+category
            if category=='incompatible':manifest['compatibility']['architecture']='unsupported-architecture'
            else:
                entry=manifest['models']['rec'];model=package/entry['file'];model.write_bytes(b'Not an ONNX protobuf; manifest integrity is valid')
                entry.update(size=model.stat().st_size,sha256=digest(model))
            write(package/'manifest.json',manifest)
            message=evidence.run('localocr-models.exe',['install','--source',package,'--store',store],'reject-'+category,expected=1)
            if category=='bad-onnx':assert 'probe' in message.lower(),message
            assert (store/'active.json').read_bytes()==active_before
            assert not (store/'versions'/manifest['version']).exists()
            assert not list(store.glob('.install-*')) and not (store/'.bundle.lock').exists()
            evidence.mark(category+'_rejected_preserves_active_and_cleans_stage')
        shutil.copytree(package1,package2);manifest=read(package2/'manifest.json');manifest['version']='runtime-baseline-2';write(package2/'manifest.json',manifest)
        evidence.run('localocr-models.exe',['install','--source',package2,'--store',store],'install-v2')
        evidence.run('localocr-models.exe',['activate','--store',store,'--version','runtime-baseline-2'],'activate-v2')
        assert read(store/'active.json')['version']=='runtime-baseline-2'
        evidence.run('localocr-models.exe',['activate','--store',store,'--version','runtime-baseline-1'],'rollback-v1')
        assert read(store/'active.json')['version']=='runtime-baseline-1'
        evidence.mark('compatible_model_update_and_explicit_rollback')
        installed=store/'versions/runtime-baseline-1';base=['--models',installed,'--manifest',installed/'manifest.json','--model-version','runtime-baseline-1']
        image=output/'原创离线-ofl.png';dimensions=sample_image(args.fonts,image);result=output/'真实结果.json'
        evidence.run('local-ai-ocr.exe',[*base,'--input',image,'--output',result,'--format','json'],'actual-json-inference',watch=True)
        report['actual_json']=assert_json_result(result,'runtime-baseline-1',dimensions)
        rejected=output/'rejected-result.json'
        illegal=output/'illegal.png';illegal.write_bytes(b'Not PNG or JPEG')
        oversized=output/'over-20-mib.png'
        with oversized.open('wb') as stream:
            stream.seek(20*1024*1024);stream.write(b'X')
        too_many_pixels=output/'over-50m-pixels.png'
        Image.new('1',(10000,5001),1).save(too_many_pixels)
        for name,source_path,options in (
                ('illegal-image',illegal,[]),('encoded-input-limit',oversized,[]),
                ('decoded-pixel-limit',too_many_pixels,[]),('unsupported-deadline',image,['--timeout','91']),
                ('unauthenticated-core',image,['--_runtime-core'])):
            evidence.run('local-ai-ocr.exe',[*base,'--input',source_path,'--output',rejected,'--format','json',*options],name,expected=1)
            assert not rejected.exists() and not rejected.with_name(rejected.name+'.partial').exists()
            assert not rejected.with_name(rejected.name+'.runtime-claim').exists()
            evidence.mark(name+'_rejected_without_output')
        foreign_output=output/'foreign-output.json';foreign_partial=foreign_output.with_name(foreign_output.name+'.partial')
        foreign_output.write_bytes(b'Existing foreign output');foreign_partial.write_bytes(b'Existing foreign partial')
        evidence.run('local-ai-ocr.exe',[*base,'--input',image,'--output',foreign_output,'--format','json'],'existing-output-rejected',expected=1)
        assert foreign_output.read_bytes()==b'Existing foreign output' and foreign_partial.read_bytes()==b'Existing foreign partial'
        foreign_partial_only=output/'foreign-partial-only.json.partial';foreign_partial_only.write_bytes(b'Existing partial only')
        missing_output=output/'foreign-partial-only.json'
        evidence.run('local-ai-ocr.exe',[*base,'--input',image,'--output',missing_output,'--format','json'],'existing-partial-rejected',expected=1)
        assert foreign_partial_only.read_bytes()==b'Existing partial only' and not missing_output.exists()
        evidence.mark('existing_output_and_partial_preserved')
        collided=output/'claim-collision.json';foreign_claim=collided.with_name(collided.name+'.runtime-claim');foreign_claim.mkdir()
        (foreign_claim/'foreign.txt').write_bytes(b'Foreign claim owner')
        evidence.run('local-ai-ocr.exe',[*base,'--input',image,'--output',collided,'--format','json'],'existing-claim-rejected',expected=1)
        assert (foreign_claim/'foreign.txt').read_bytes()==b'Foreign claim owner' and not collided.exists()
        evidence.mark('existing_foreign_claim_preserved')
        report['output_limit_evidence']='8 MiB output ceiling is covered by source contract unit checks; no giant synthetic OCR result was fabricated here'

        assert not result.with_name(result.name+'.partial').exists()
        assert not result.with_name(result.name+'.runtime-claim').exists()
        evidence.mark('actual_unicode_json_raw_coordinates_scores')
        tsv=output/'result.tsv'
        evidence.run('local-ai-ocr.exe',[*base,'--input',image,'--output',tsv,'--format','tsv'],'legacy-tsv-inference')
        assert '\t' in tsv.read_text(encoding='utf-8-sig') and 'HELLO' in tsv.read_text(encoding='utf-8-sig')
        evidence.mark('legacy_tsv_entry')
        audit=evidence.run('local-ai-ocr.exe',[*base,'--network-audit-self-test'],'network-audit-self-test')
        audit_json=json.loads(audit.strip().splitlines()[-1])
        assert set(audit_json['denied'])=={'creation','dns','connect','sendto','bind'} and audit_json['cpu_probe'] is True
        report['network_audit']=audit_json;evidence.mark('real_cpu_probe_network_audit_denies_all_five_actions')
        timed=output/'timed-result.json'
        evidence.run('local-ai-ocr.exe',[*base,'--input',image,'--output',timed,'--format','json','--timeout','0.1'],'short-deadline',expected=124,watch=True)
        assert not timed.exists() and not timed.with_name(timed.name+'.partial').exists()
        assert not timed.with_name(timed.name+'.runtime-claim').exists()
        evidence.mark('short_deadline_124_and_owned_output_cleanup')
        large=output/'large-original.png';sample_image(args.fonts,large,large=True);cancelled=output/'cancelled.json'
        report['forced_cancel']=cancel_parent(evidence,[*base,'--input',large,'--output',cancelled,'--format','json'],cancelled)
        evidence.mark('parent_kill_job_reaps_descendants_caller_cleans_claim_foreign_outputs_preserved')
        assert all(not item['listening_connections'] for item in evidence.monitor)
        evidence.mark('psutil_no_inference_listening_sockets')
        after={p.relative_to(runtime).as_posix():digest(p) for p in runtime.rglob('*') if p.is_file()}
        assert before==after,'Model update or execution modified runtime files'
        report['runtime_file_hashes']=after;evidence.mark('model_update_preserves_all_runtime_hashes')
        assert not list(store.glob('.active-*')) and not list(store.glob('.install-*')) and not (store/'.bundle.lock').exists()
        report['passed']=True
    except Exception as error:
        report.update(passed=False,failure=type(error).__name__+': '+str(error))
        raise
    finally:
        report['commands']=evidence.commands;report['process_monitor']=evidence.monitor
        write(output/'runtime-verification.json',report)
    print(json.dumps({'passed':True,'checks':report['checks'],'summary':str(output/'runtime-verification.json')},indent=2))


if __name__=='__main__':main()
