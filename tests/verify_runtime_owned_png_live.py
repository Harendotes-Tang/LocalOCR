"""Real frozen EXE acceptance of caller-owned lossless PNG expansion.

Reads APP synthetic boundary evidence without modifying that repository. OCR
runs only through the frozen Windows EXE. This proves this machine's runtime
boundary, not caller provenance enforcement or a complete APP lifecycle.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import msvcrt
import json
import math
import os
from pathlib import Path
import shutil
import struct
from PIL import Image
from verify_runtime_live import Evidence, owned_output, digest, write, read, sample_image, assert_json_result, ROOT

MAX_OWNED_BYTES=201048576
SOURCE=ROOT/'artifacts/owned-png-boundary'


def clean_request(output):
    assert not output.exists(), 'Unexpected rejected request output'
    assert not output.with_name(output.name+'.partial').exists(), 'Unexpected rejected request partial'
    assert not output.with_name(output.name+'.runtime-claim').exists(), 'Owned claim not cleaned'


def result_contract(path, dimensions, version):
    raw=path.read_bytes();value=json.loads(raw.decode('utf-8'))
    assert value['schema_version']=='localocr.result.v1' and value['model_version']==version
    assert value['image']==dict(zip(('width','height'),dimensions))
    assert isinstance(value['regions'],list) and 'Not a probability' in value['score_semantics']
    for row in value['regions']:
        assert isinstance(row['text'],str) and isinstance(row['score'],(int,float)) and math.isfinite(row['score'])
        assert len(row['box'])==4 and all(len(point)==2 for point in row['box'])
        assert all(isinstance(v,(int,float)) and math.isfinite(v) for point in row['box'] for v in point)
        x=max(0,math.floor(min(p[0] for p in row['box'])));y=max(0,math.floor(min(p[1] for p in row['box'])))
        right=min(dimensions[0],math.ceil(max(p[0] for p in row['box'])));bottom=min(dimensions[1],math.ceil(max(p[1] for p in row['box'])))
        assert row['rect']=={'x':x,'y':y,'width':right-x,'height':bottom-y}
        assert right>x and bottom>y
    assert not path.with_name(path.name+'.partial').exists()
    assert not path.with_name(path.name+'.runtime-claim').exists()
    return {'regions':len(value['regions']),'utf8_bytes':len(raw),'image':value['image'],'schema_version':value['schema_version']}


def corrupt_crc(source,destination):
    payload=bytearray(source.read_bytes());offset=8
    while offset+12<=len(payload):
        size=struct.unpack('>I',payload[offset:offset+4])[0];kind=payload[offset+4:offset+8]
        if kind==b'IDAT':
            payload[offset+8+size]^=1
            destination.write_bytes(payload);return
        offset+=size+12
    raise AssertionError('Generated PNG lacks IDAT chunk')


def sparse_ceiling_counterexample(path):
    """Set actual NTFS/ReFS sparse attribute before writing the last byte."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.DeviceIoControl.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,
                                    ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
    kernel.DeviceIoControl.restype=wintypes.BOOL
    kernel.GetFileAttributesW.argtypes=[wintypes.LPCWSTR];kernel.GetFileAttributesW.restype=wintypes.DWORD
    with path.open('wb') as stream:
        returned=wintypes.DWORD()
        if not kernel.DeviceIoControl(msvcrt.get_osfhandle(stream.fileno()),0x900C4,None,0,None,0,ctypes.byref(returned),None):
            raise ctypes.WinError(ctypes.get_last_error())
        stream.seek(MAX_OWNED_BYTES);stream.write(b'X')
    attrs=kernel.GetFileAttributesW(str(path))
    assert attrs!=0xffffffff and attrs&0x200,'Sparse file attribute not established'
    assert path.stat().st_size==MAX_OWNED_BYTES+1
    return {'bytes':path.stat().st_size,'sparse_attribute_verified':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=ROOT/'artifacts/localocr-runtime-win-x64-v2')
    parser.add_argument('--models',type=Path,default=ROOT/'artifacts/runtime-verification-v2/model-store/versions/runtime-baseline-1')
    parser.add_argument('--source',type=Path,default=SOURCE,
                        help='Caller-created synthetic JPEG/PNG/facts directory; read only')
    parser.add_argument('--fonts',type=Path,default=ROOT/'artifacts/ocr-data-fonts-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/runtime-owned-png-verification-v2')
    args=parser.parse_args()
    if os.name!='nt':raise ValueError('This acceptance requires the actual frozen Windows runtime')
    output=owned_output(args.output);runtime=args.runtime.resolve();models=args.models.resolve();source=args.source.resolve()
    if output.exists():raise ValueError('Acceptance output must be fresh')
    output.mkdir(parents=True);evidence=Evidence(runtime,output)
    before={p.relative_to(runtime).as_posix():digest(p) for p in runtime.rglob('*') if p.is_file()}
    manifest=read(models/'manifest.json');version=manifest['version']
    models_before={p.name:digest(p) for p in models.iterdir() if p.is_file()}
    base=['--models',models,'--manifest',models/'manifest.json','--model-version',version,'--format','json']
    report={'schema_version':'localocr.owned-png-verification.v1','scope':'actual frozen CPU EXE; read-only APP synthetic boundary evidence; no complete APP/cross-machine acceptance',
            'environment':{'PATH':evidence.environment['PATH'],'PYTHONHOME_removed':True,'PYTHONPATH_removed':True},
            'checks':evidence.checks,'owned_png_limit_bytes':MAX_OWNED_BYTES,'ordinary_limit_bytes':20*1024*1024,
            'provenance_boundary':'The flag cannot prove the original-source limits; caller must enforce them. Source/copy hashes below bind this synthetic acceptance.'}
    try:
        copied={};original_hashes={}
        for name in ('source-under-20mib.jpg','owned-decoded.png','image-facts.json'):
            origin=source/name;target=output/name;sha=digest(origin);shutil.copyfile(origin,target)
            assert digest(target)==sha,'Read-only boundary copy changed'
            original_hashes[name]=sha;copied[name]={'source_sha256':sha,'copy_sha256':digest(target),'bytes':target.stat().st_size}
        jpeg=output/'source-under-20mib.jpg';owned=output/'owned-decoded.png'
        assert jpeg.stat().st_size==19352892 and jpeg.stat().st_size<=20*1024*1024
        assert owned.stat().st_size==24033316 and 20*1024*1024<owned.stat().st_size<=MAX_OWNED_BYTES
        for file,kind in ((jpeg,'JPEG'),(owned,'PNG')):
            with Image.open(file) as image:
                assert image.format==kind and image.size==(4000,2000) and getattr(image,'n_frames',1)==1
                image.verify()
        report['source_copy_binding']=copied;evidence.mark('read_only_app_synthetic_source_and_copy_hashes_match')
        rejected=output/'default-rejected.json'
        message=evidence.run('local-ai-ocr.exe',[*base,'--input',owned,'--output',rejected],'default-over-20mib-rejected',expected=1)
        assert '20 MiB' in message;clean_request(rejected)
        evidence.mark('ordinary_png_20mib_limit_unchanged')
        actual=output/'owned-expanded-result.json'
        evidence.run('local-ai-ocr.exe',[*base,'--owned-png','--input',owned,'--output',actual],'actual-owned-expanded-png',watch=True)
        report['actual_expanded_png']=result_contract(actual,(4000,2000),version)
        evidence.mark('actual_cpu_owned_24mb_png_success_original_4000x2000_dimensions')
        small=output/'原创文字-owned.png';dimensions=sample_image(args.fonts,small)
        small_result=output/'small-text-result.json'
        evidence.run('local-ai-ocr.exe',[*base,'--owned-png','--input',small,'--output',small_result],'actual-owned-small-ofl-text',watch=True)
        report['actual_ofl_text']=assert_json_result(small_result,version,dimensions)
        assert report['actual_ofl_text']['regions']==2
        result_contract(small_result,dimensions,version)
        evidence.mark('actual_owned_ofl_unicode_text_two_regions')
        disguised=output/'jpeg-bytes-with-png-extension.png';shutil.copyfile(jpeg,disguised)
        apng=output/'animated.png';frames=[Image.new('RGB',(64,32),(255,255,255)),Image.new('RGB',(64,32),(0,0,0))]
        frames[0].save(apng,save_all=True,append_images=frames[1:],duration=100,loop=0)
        with Image.open(apng) as image:assert image.n_frames==2
        bad_crc=output/'bad-crc.png';corrupt_crc(small,bad_crc)
        pristine=small.read_bytes();assert pristine[-8:-4]==b'IEND'
        bad_iend_crc=output/'bad-iend-crc.png';iend=bytearray(pristine);iend[-4]^=1;bad_iend_crc.write_bytes(iend)
        missing_iend_crc=output/'missing-iend-crc.png';missing_iend_crc.write_bytes(pristine[:-4])
        trailing=output/'after-iend-data.png';trailing.write_bytes(pristine+b'Unexpected data after IEND')
        empty=output/'empty.png';empty.touch()
        oversized=output/'owned-ceiling-plus-one.png'
        report['sparse_byte_ceiling_counterexample']=sparse_ceiling_counterexample(oversized)
        too_many=output/'over-50m-pixels.png';Image.new('1',(10000,5001),1).save(too_many)
        for label,file in (('jpeg-content',jpeg),('png-extension-jpeg-content',disguised),('apng',apng),('bad-crc',bad_crc),
                           ('bad-iend-crc',bad_iend_crc),('missing-iend-crc',missing_iend_crc),('after-iend-data',trailing),
                           ('empty',empty),('owned-byte-ceiling',oversized),('owned-pixel-ceiling',too_many)):
            target=output/(label+'-rejected.json')
            evidence.run('local-ai-ocr.exe',[*base,'--owned-png','--input',file,'--output',target],label+'-owned-rejected',expected=1)
            clean_request(target);evidence.mark(label+'_owned_rejected_without_output_partial_claim')
        timed=output/'owned-timeout.json'
        evidence.run('local-ai-ocr.exe',[*base,'--owned-png','--input',owned,'--output',timed,'--timeout','0.1'],'owned-short-timeout',expected=124,watch=True)
        clean_request(timed);evidence.mark('owned_png_short_deadline_124_cleans_request_claim')
        assert all(digest(source/name)==sha for name,sha in original_hashes.items()),'APP source was changed'
        assert models_before=={p.name:digest(p) for p in models.iterdir() if p.is_file()},'Read-only baseline model snapshot changed'
        after={p.relative_to(runtime).as_posix():digest(p) for p in runtime.rglob('*') if p.is_file()}
        assert before==after,'Runtime files changed while accepting owned PNG'
        report['runtime_file_hashes']=after;report['baseline_model_hashes']=models_before
        evidence.mark('app_sources_baseline_snapshot_and_runtime_hashes_unchanged')
        evidence.mark('real_inference_no_host_python_or_listening_sockets')
        report['passed']=True
    except Exception as error:
        report.update(passed=False,failure=type(error).__name__+': '+str(error));raise
    finally:
        report['commands']=evidence.commands;report['process_monitor']=evidence.monitor
        write(output/'owned-png-verification.json',report)
    print(json.dumps({'passed':True,'checks':report['checks'],'summary':str(output/'owned-png-verification.json')},indent=2))


if __name__=='__main__':main()
