"""Network targets must be rejected before any exists/mkdir target access."""
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from localocr import bundles

@pytest.mark.skipif(os.name!='nt',reason='Windows UNC/mapped drive semantics')
@pytest.mark.parametrize('remote',[r'\\unit-no-network\share\output',r'Z:\offline-models\output'])
@pytest.mark.parametrize('operation',['baseline','install','activate'])
def test_remote_target_rejected_before_target_filesystem_access(tmp_path,remote,operation):
    target=Path(remote)
    with patch('ctypes.windll.kernel32.GetDriveTypeW',side_effect=lambda root:4 if root.upper().startswith('Z:') else 3), \
         patch.object(Path,'exists',side_effect=AssertionError('Unexpected exists before rejection')) as exists, \
         patch.object(Path,'mkdir',side_effect=AssertionError('Unexpected mkdir before rejection')) as mkdir, \
         patch.object(Path,'lstat',side_effect=AssertionError('Unexpected lstat before rejection')) as lstat:
        with pytest.raises(ValueError,match='Network filesystem|Mapped network'):
            if operation=='baseline':bundles.create_baseline(tmp_path,target,'baseline-1')
            elif operation=='install':bundles.install_bundle(tmp_path,target)
            else:bundles.activate_bundle(target,'baseline-1')
        exists.assert_not_called();mkdir.assert_not_called();lstat.assert_not_called()

@pytest.mark.skipif(os.name!='nt',reason='Windows UNC/mapped drive semantics')
@pytest.mark.parametrize('remote',[r'\\unit-no-network\share\output',r'Z:\offline-models\output'])
def test_cli_list_rejects_remote_before_directory_listing(remote):
    from localocr.bundle_cli import main
    with patch('sys.argv',['bundle_cli','list','--store',remote]), \
         patch('ctypes.windll.kernel32.GetDriveTypeW',side_effect=lambda root:4 if root.upper().startswith('Z:') else 3), \
         patch.object(Path,'iterdir',side_effect=AssertionError('Unexpected directory listing')) as listing:
        with pytest.raises(SystemExit) as raised:
            main()
        assert raised.value.code==1
        listing.assert_not_called()

@pytest.mark.parametrize('operation',['lock','baseline','list'])
def test_reparse_parent_rejected_before_descendant_lookup_or_creation(tmp_path,operation):
    link=tmp_path/'junction'
    target=link/'missing-parent'/'store'
    original_plain=bundles._plain
    def inspect(path):
        if path==link:
            raise ValueError('Symlinks and reparse points are forbidden')
        if link in path.parents:
            raise AssertionError('Descendant lookup crossed reparse parent')
        return original_plain(path)
    with patch.object(bundles,'_plain',side_effect=inspect), \
         patch.object(Path,'mkdir',side_effect=AssertionError('Creation before parent validation')) as mkdir:
        with pytest.raises((ValueError,SystemExit)):
            if operation=='lock':
                with bundles._lock(target):pass
            elif operation=='baseline':bundles.create_baseline(tmp_path,target,'baseline-1')
            else:
                from localocr.bundle_cli import main
                with patch('sys.argv',['bundle_cli','list','--store',str(target)]):main()
        mkdir.assert_not_called()

@pytest.mark.parametrize('operation',['create-app','token','directory-alias','install-versions','install-destination'])
def test_all_owned_path_entries_check_existing_parents_first(tmp_path,operation):
    from types import SimpleNamespace
    from localocr.service import create_app,_token
    junction=tmp_path/'junction'
    original_plain=bundles._plain
    def inspect(path):
        if path==junction:
            raise ValueError('Symlinks and reparse points are forbidden')
        if junction in path.parents:
            raise AssertionError('Lookup crossed reparse parent')
        return original_plain(path)
    bundle=SimpleNamespace(version='junction',root=tmp_path)
    if operation=='install-versions':junction=tmp_path/'versions'
    with patch.object(bundles,'_plain',side_effect=inspect), patch.object(bundles,'validate_bundle',return_value=bundle), \
         patch.object(Path,'mkdir',side_effect=AssertionError('Creation before validation')) as mkdir:
        with pytest.raises(ValueError,match='reparse'):
            if operation=='create-app':create_app(junction/'store')
            elif operation=='token':_token(junction/'token')
            elif operation=='directory-alias':bundles._directory(tmp_path/'missing'/'..'/'junction'/'store',allow_missing=True)
            else:
                # Skip only the already-verified store lock to inspect inner
                # versions/destination validation before mkdir/exists.
                from contextlib import nullcontext
                store=tmp_path if operation=='install-versions' else tmp_path.parent
                with patch.object(bundles,'_lock',return_value=nullcontext()):
                    if operation=='install-destination':
                        junction=store/'versions'/'junction'
                        original_inspect=inspect
                        def missing_versions(path):
                            if path==store/'versions':
                                import stat
                                return SimpleNamespace(st_mode=stat.S_IFDIR)
                            return original_inspect(path)
                        with patch.object(bundles,'_plain',side_effect=missing_versions),patch.object(Path,'mkdir'):
                            bundles.install_bundle(tmp_path,store)
                    else:bundles.install_bundle(tmp_path,store)
        if operation!='install-destination':mkdir.assert_not_called()
