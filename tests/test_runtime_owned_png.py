"""Owned transport bounds/format validation; envelope fixtures are not inference."""
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from localocr import runtime_entry as entry


def parsed(tmp_path, **updates):
    models=tmp_path/'models';models.mkdir(exist_ok=True)
    source=tmp_path/'input.png';Image.new('RGB',(64,32),'white').save(source)
    value=entry.parser().parse_args(['--models',str(models),'--input',str(source),'--output',str(tmp_path/'out.json'),'--format','json','--owned-png'])
    for key,v in updates.items():setattr(value,key,v)
    return value


def test_owned_envelope_uses_worker_ceiling_and_default_stays_20mib(tmp_path):
    args=parsed(tmp_path);worker=entry.load_worker()
    assert worker.MAX_OWNED_PNG_BYTES == worker.MAX_PIXELS*4+1024*1024
    with Path(args.input).open('wb') as f:f.truncate(worker.MAX_OWNED_PNG_BYTES)
    assert entry.validate(args)[1]
    args.owned_png=False
    with pytest.raises(ValueError,match='20 MiB'):entry.validate(args)
    args.owned_png=True
    with Path(args.input).open('wb') as f:f.truncate(worker.MAX_OWNED_PNG_BYTES+1)
    with pytest.raises(ValueError,match='byte limit'):entry.validate(args)


def test_owned_empty_input_rejected(tmp_path):
    args=parsed(tmp_path);Path(args.input).write_bytes(b'')
    with pytest.raises(ValueError,match='empty'):entry.validate(args)


@pytest.mark.parametrize('flag',['self_test','network_audit_self_test'])
def test_owned_mode_cannot_be_self_test(tmp_path,flag):
    args=parsed(tmp_path, input=None, output=None);setattr(args,flag,True)
    with pytest.raises(ValueError,match='not self-test'):entry.validate(args)


def test_owned_valid_png_verified_without_pixel_change(tmp_path):
    args=parsed(tmp_path);source=Path(args.input);before=source.read_bytes()
    entry.verify_owned_png(source,entry.load_worker());assert source.read_bytes()==before


def test_owned_rejects_actual_jpeg_despite_png_extension(tmp_path):
    args=parsed(tmp_path);Image.new('RGB',(64,32),'white').save(args.input,format='JPEG')
    with pytest.raises(ValueError,match='format or integrity'):entry.verify_owned_png(Path(args.input),entry.load_worker())


def test_owned_rejects_apng(tmp_path):
    args=parsed(tmp_path);Image.new('RGB',(64,32),'white').save(args.input,format='PNG',save_all=True,append_images=[Image.new('RGB',(64,32),'black')],duration=100,loop=0)
    with pytest.raises(ValueError,match='single-frame PNG'):entry.verify_owned_png(Path(args.input),entry.load_worker())


def test_owned_rejects_bad_png_crc(tmp_path):
    args=parsed(tmp_path);source=Path(args.input);data=bytearray(source.read_bytes());index=data.index(b'IDAT');data[index+4]^=1;source.write_bytes(data)
    with pytest.raises(ValueError,match='integrity'):entry.verify_owned_png(source,entry.load_worker())


def test_owned_rejects_more_than_50m_pixels_before_engine(tmp_path):
    args=parsed(tmp_path);Image.new('1',(10000,5001)).save(args.input)
    with pytest.raises(ValueError,match='pixel limit'):entry.verify_owned_png(Path(args.input),entry.load_worker())


def test_authenticated_core_revalidates_owned_before_engine(tmp_path):
    args=parsed(tmp_path);worker=entry.load_worker()
    with patch.object(entry,'load_worker',return_value=worker),patch.object(entry,'verify_owned_png',side_effect=ValueError('Owned validation called')) as verify,patch.object(worker,'main') as main:
        with pytest.raises(ValueError,match='Owned validation called'):entry.core(args,[])
        verify.assert_called_once_with(Path(args.input),worker);main.assert_not_called()
    assert '--owned-png' in entry.child_command(['--owned-png'])


@pytest.mark.parametrize("damage", ["iend_crc", "missing_iend_crc", "missing_iend", "trailing", "fake_second_iend"])
def test_owned_rejects_invalid_png_termination(tmp_path, damage):
    args = parsed(tmp_path)
    source = Path(args.input)
    data = bytearray(source.read_bytes())
    if damage == "iend_crc":
        data[-1] ^= 1
    elif damage == "missing_iend_crc":
        del data[-4:]
    elif damage == "missing_iend":
        del data[-12:]
    elif damage == "trailing":
        data.extend(b"unexpected bytes")
    else:
        # A good final IEND must not hide a corrupt first terminal chunk.
        terminal = bytes(data[-12:])
        data[-1] ^= 1
        data.extend(terminal)
    source.write_bytes(data)
    with pytest.raises(ValueError, match="integrity"):
        entry.verify_owned_png(source, entry.load_worker())
