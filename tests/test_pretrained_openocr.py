"""Exact CTC identities and head-transfer failures, independent of private data."""
import unittest
import contextlib
import tempfile
import zipfile
import hashlib
import importlib
import importlib.util
import marshal
import random
import struct
import copy
import sys
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tools.PretrainedOcr import openocr_adapter as adapter


class ExactCTCTests(unittest.TestCase):
    def test_fixed_archive_rejects_inventory_rewrite_and_extra_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            original = b'official source'
            (source / 'model.py').write_bytes(original)
            archive = root / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('OpenOCR-' + adapter.COMMIT + '/model.py', original)
            weight = root / 'openocr_repsvtr_ch.pth'; weight.write_bytes(b'w')
            data = {'source_commit': adapter.COMMIT, 'extracted_root': str(source),
                'downloads': [{'path': str(archive)}, {'path': str(weight)}],
                'source_file_inventory': [{'path': 'model.py', 'bytes': len(original),
                    'sha256': hashlib.sha256(original).hexdigest()}]}
            receipt = root / 'receipt.json'
            with mock.patch.multiple(adapter, SOURCE_SHA=adapter.common.digest(archive), SOURCE_BYTES=archive.stat().st_size,
                                      WEIGHT_SHA=adapter.common.digest(weight), WEIGHT_BYTES=1):
                adapter.common.write(receipt, data)
                self.assertEqual(adapter.validate_inputs(receipt)[0], source.resolve())
                changed = copy.deepcopy(data); changed['source_file_inventory'] = []
                adapter.common.write(receipt, changed)
                with self.assertRaises(ValueError): adapter.validate_inputs(receipt)
                altered = b'attacker source'
                (source / 'model.py').write_bytes(altered)
                changed = copy.deepcopy(data)
                changed['source_file_inventory'][0].update(bytes=len(altered), sha256=hashlib.sha256(altered).hexdigest())
                adapter.common.write(receipt, changed)
                with self.assertRaises(ValueError): adapter.validate_inputs(receipt)
                (source / 'model.py').write_bytes(original)
                adapter.common.write(receipt, data)
                (source / 'unlisted.py').write_text('raise RuntimeError()', encoding='utf-8')
                with self.assertRaises(ValueError): adapter.validate_inputs(receipt)
                (source / 'unlisted.py').unlink()
                archive.write_bytes(b'changed archive')
                with self.assertRaises(ValueError): adapter.validate_inputs(receipt)

    def test_fixed_source_loader_ignores_valid_header_forged_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'openrec'; package.mkdir()
            source = package / '__init__.py'; raw = b"VALUE = 'official'\n"; source.write_bytes(raw)
            cache = Path(importlib.util.cache_from_source(str(source))); cache.parent.mkdir()
            forged = compile("VALUE = 'forged'", str(source), 'exec')
            cache.write_bytes(importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, int(source.stat().st_mtime), len(raw)) + marshal.dumps(forged))
            expected = {'openrec/__init__.py': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
            finder = adapter.VerifiedSourceFinder(root, expected)
            old = sys.modules.pop('openrec', None); sys.path.insert(0, str(root)); sys.meta_path.insert(0, finder)
            try:
                module = importlib.import_module('openrec')
                self.assertEqual(module.VALUE, 'official')
                self.assertIsInstance(module.__loader__, adapter.VerifiedSourceLoader)
                self.assertEqual(marshal.loads(cache.read_bytes()[16:]).co_consts, forged.co_consts)
                source.write_bytes(b"VALUE = 'modified'\n")
                with self.assertRaises(ValueError): module.__loader__.get_code('openrec')
            finally:
                sys.modules.pop('openrec', None)
                if old is not None: sys.modules['openrec'] = old
                sys.meta_path.remove(finder); sys.path.remove(str(root))

    def test_source_importer_rejects_preimported_unproved_regular_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            loaded = SimpleNamespace(__spec__=SimpleNamespace(origin=str(root / 'openrec/__init__.py'), loader=object()))
            with mock.patch.dict(adapter._VERIFIED_SOURCES, {str(root): {'openrec/__init__.py': {'bytes': 1, 'sha256': 'x'}}}):
                with mock.patch.dict(sys.modules, {'openrec': loaded}):
                    with self.assertRaises(ValueError): adapter.install_source_importer(root)

    def test_region_balanced_sampling_is_deterministic_and_records_empty_kind(self):
        rows = [{'kind': kind, 'text': '' if kind == 'graphic' else 'x'}
                for kind in ('face', 'punctuation', 'plain', 'emoji', 'graphic') for _ in range(2000)]
        first = adapter.sampling_order(rows, 'region-balanced', random.Random(20261012))
        second = adapter.sampling_order(rows, 'region-balanced', random.Random(20261012))
        self.assertEqual(first, second); self.assertEqual(len(first), len(rows))
        for kind, expected in adapter.REGION_SAMPLING.items():
            observed = sum(adapter.sampling_kind(rows[index]) == kind for index in first) / len(first)
            self.assertAlmostEqual(observed, expected, delta=0.02)
        with self.assertRaises(ValueError): adapter.sampling_order(rows[:1], 'region-balanced', random.Random(1))
        uniform = adapter.sampling_order(rows, 'uniform', random.Random(1))
        self.assertEqual(set(uniform), set(range(len(rows))))

    def test_namespace_source_binding_accepts_only_verified_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            namespace = SimpleNamespace(__file__=None, __path__=[str(root / 'openrec')])
            regular = SimpleNamespace(__file__=str(root / 'openrec/modeling/__init__.py'))
            with mock.patch.dict(sys.modules, {'openrec': namespace, 'openrec.modeling': regular}):
                adapter.verify_imported_source(root)
                namespace.__path__.append(str(root.parent / 'foreign-openrec'))
                with self.assertRaises(ValueError): adapter.verify_imported_source(root)
                namespace.__path__.pop()
                regular.__spec__ = SimpleNamespace(origin=str(root.parent / 'foreign-openrec/spec.py'))
                with self.assertRaises(ValueError): adapter.verify_imported_source(root)
                regular.__spec__ = None
                regular.__file__ = str(root.parent / 'foreign-openrec/child.py')
                with self.assertRaises(ValueError): adapter.verify_imported_source(root)

    def test_unknown_is_rejected_and_spaces_combining_preserved(self):
        tokens = [adapter.BLANK, 'a', ' ', '\u0301', '\U0001f600']
        self.assertEqual(adapter.encode(' a\u0301\U0001f600 ', tokens), [2, 1, 3, 4, 2])
        with self.assertRaises(ValueError): adapter.encode('b', tokens)

    def test_blank_separates_identical_tokens(self):
        probabilities = np.eye(3, dtype=np.float32)[[1, 1, 0, 1, 2]][None]
        self.assertEqual(adapter.decode(probabilities, [adapter.BLANK, 'A', ' ']), [('AA ', 1.0)])

    def test_empty_prediction_has_zero_score(self):
        self.assertEqual(adapter.decode(np.array([[[1, 0], [1, 0]]]), [adapter.BLANK, 'A']), [('', 0.0)])

    def test_rgb_pixels_and_zero_normalized_padding(self):
        image = Image.new('RGB', (11, 48), (255, 0, 0))
        array = adapter.preprocess(image)
        self.assertEqual(array.shape, (3, 48, 16))
        np.testing.assert_array_equal(array[:, 0, 0], [1, -1, -1])
        np.testing.assert_array_equal(array[:, :, 11:], 0)

    def test_shared_evaluator_uses_identical_pixels(self):
        from tools.PretrainedOcr.evaluate import prepare_image
        for size in [(11, 48), (71, 21), (2000, 48)]:
            image = Image.new('RGB', size, (13, 127, 243))
            np.testing.assert_array_equal(adapter.preprocess(image), prepare_image(image, adapter.PREPROCESSING))

    def test_width_limits_and_batch_padding(self):
        batch = adapter.batch_images([Image.new('RGB', (20, 48)), Image.new('RGB', (100, 48))])
        self.assertEqual(batch.shape, (2, 3, 48, 100))
        np.testing.assert_array_equal(batch[0, :, :, 20:], 0)
        self.assertEqual(adapter.preprocess(Image.new('RGB', (2000, 48))).shape[-1], 1024)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Research Torch environment required for tensor transfer')
class HeadTransferTests(unittest.TestCase):
    def test_explicit_negative_weight_changes_only_empty_target_contribution(self):
        logits = torch.zeros(1, 8, 3, requires_grad=True)
        weighted = adapter.exact_ctc_loss(logits, [[]], 3.0)
        normal = adapter.exact_ctc_loss(logits, [[]])
        self.assertAlmostEqual(float(weighted.detach()), float(normal.detach()) * 3, places=6)
        weighted.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        positive = torch.zeros(1, 8, 3)
        self.assertEqual(float(adapter.exact_ctc_loss(positive, [[1]], 3)), float(adapter.exact_ctc_loss(positive, [[1]], 1)))
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): adapter.exact_ctc_loss(logits, [[]], invalid)

    def test_warm_start_preserves_parent_and_creates_independent_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root / 'parent'; parent.mkdir()
            torch.save({'state_dict': {'w': torch.zeros(1)}, 'steps': 0}, parent / 'initial.pth')
            torch.save({'state_dict': {'w': torch.ones(1)}, 'steps': 2}, parent / 'best.pth')
            checkpoint_sha = adapter.common.digest(parent / 'best.pth')
            adapter.common.write(parent / 'training.json', {'steps': 2, 'nonzero_gradient': True,
                'changed_tensors': ['w'], 'checkpoint_sha256': checkpoint_sha,
                'initial_checkpoint_sha256': adapter.common.digest(parent / 'initial.pth')})
            (parent / 'prepare.json').write_text('{}', encoding='utf-8')
            (parent / 'vocabulary.json').write_text('{"tokens":["<blank>"," "]}', encoding='utf-8')
            info = {'dataset_root': 'frozen-data', 'dataset_binding': {'fixed': True},
                'vocabulary_sha256': adapter.common.digest(parent / 'vocabulary.json'), 'preprocessing': {'fixed': True}}
            adapter.common.write(parent / 'prepare.json', info)
            training = adapter.common.read(parent / 'training.json')
            training.update(preparation_sha256=adapter.common.digest(parent / 'prepare.json'), vocabulary_sha256=info['vocabulary_sha256'])
            adapter.common.write(parent / 'training.json', training)
            output = root / 'new-run'
            args = SimpleNamespace(prepared=parent, checkpoint='best.pth', checkpoint_sha256=checkpoint_sha, output=output)
            with mock.patch.object(adapter, 'load_prepared', return_value=(None, None, info)), mock.patch.object(adapter.common, 'dataset_binding', return_value={'fixed': True}):
                adapter.warm_start(args)
                self.assertEqual(adapter.common.digest(output / 'initial.pth'), checkpoint_sha)
                self.assertEqual(adapter.common.digest(parent / 'best.pth'), checkpoint_sha)
                receipt = adapter.common.read(output / 'warm-start.json')
                self.assertFalse(receipt['training_started']); self.assertFalse(receipt['heldout_read'])
                self.assertEqual(receipt['parent_finetune']['steps'], 2)
                self.assertEqual((output / 'vocabulary.json').read_bytes(), (parent / 'vocabulary.json').read_bytes())
                with self.assertRaises(FileExistsError): adapter.warm_start(args)
                adapter.validate_warm_start(output, info)
                changed = copy.deepcopy(info); changed['preprocessing'] = {'width': {'min': 512}}
                warm = adapter.common.read(output / 'warm-start.json'); warm['preprocessing'] = changed['preprocessing']
                adapter.common.write(output / 'warm-start.json', warm)
                with self.assertRaises(ValueError): adapter.validate_warm_start(output, changed)
                warm['preprocessing'] = info['preprocessing']; adapter.common.write(output / 'warm-start.json', warm)
                (output / 'initial.pth').write_bytes(b'local modified copy')
                with self.assertRaises(ValueError): adapter.validate_warm_start(output, info)
                self.assertEqual(adapter.common.digest(parent / 'best.pth'), checkpoint_sha)

    def test_warm_start_hash_mismatch_never_creates_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'best.pth').write_bytes(b'checkpoint')
            output = root / 'new-run'
            with self.assertRaises(ValueError):
                adapter.warm_start(SimpleNamespace(prepared=root, checkpoint='best.pth', checkpoint_sha256='wrong', output=output))
            self.assertFalse(output.exists())

    def test_finetuned_label_rejects_untrained_or_unchanged_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = {'state_dict': {'weight': torch.ones(2)}, 'steps': 0}
            torch.save(initial, root / 'initial.pth')
            torch.save(initial, root / 'best.pth')
            with self.assertRaises(ValueError): adapter.finetune_evidence(root, 'best.pth')
            initial['steps'] = 10; torch.save(initial, root / 'best.pth')
            with self.assertRaises(ValueError): adapter.finetune_evidence(root, 'best.pth')

    def model(self, count):
        model = torch.nn.Module(); model.encoder = torch.nn.Linear(2, 2)
        model.decoder = torch.nn.Module(); model.decoder.fc = torch.nn.Linear(2, count)
        return model

    def test_forward_failure_cleans_resources_before_budget_unlock(self):
        events = []
        class FailingModel(torch.nn.Module):
            def cuda(self):
                events.append('model-cuda'); return self
            def cpu(self):
                events.append('model-cpu'); return self
            def forward(self, batch):
                raise RuntimeError('controlled forward failure')
        @contextlib.contextmanager
        def lease(*args, **kwargs):
            try:
                yield SimpleNamespace(expired=lambda: False, remaining=lambda: 120)
            finally:
                events.append('budget-unlock')
        state = {'live-optimizer-state': object()}
        optimizer = SimpleNamespace(state=state, zero_grad=lambda **kw: events.append('gradient-clear'))
        budget = SimpleNamespace(run=lease)
        args = SimpleNamespace(prepared='unused', budget='unused', batch=1, lr=1e-4)
        info = {'dataset_root': 'unused', 'dataset_binding': {}, 'seed': 48, 'preprocessing': {}}
        rows = [{'text': 'a'}, {'text': ''}]
        with contextlib.ExitStack() as stack:
            for target, value in [
                ('load_prepared', (FailingModel(), [adapter.BLANK, 'a'], info)),
                ('read_images', []), ('batch_images', np.zeros((1, 3, 48, 16), dtype=np.float32)),
            ]:
                stack.enter_context(mock.patch.object(adapter, target, return_value=value))
            stack.enter_context(mock.patch.object(adapter.common, 'read_split', return_value=rows))
            stack.enter_context(mock.patch.object(adapter.common, 'dataset_binding', return_value={}))
            stack.enter_context(mock.patch.object(adapter.common, 'disk_guard'))
            stack.enter_context(mock.patch.object(adapter.common, 'digest', return_value='initial'))
            stack.enter_context(mock.patch.object(adapter.common, 'GPUBudget', return_value=budget))
            stack.enter_context(mock.patch.object(torch.optim, 'Adam', return_value=optimizer))
            stack.enter_context(mock.patch.object(torch.Tensor, 'cuda', lambda tensor: tensor))
            for name in ('synchronize', 'empty_cache', 'reset_peak_memory_stats'):
                stack.enter_context(mock.patch.object(torch.cuda, name, side_effect=lambda: events.append('cuda-cleanup')))
            stack.enter_context(mock.patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(mock.patch.object(torch.cuda, 'memory_allocated', return_value=0))
            stack.enter_context(mock.patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(mock.patch.object(torch._C, '_cuda_clearCublasWorkspaces', create=True, side_effect=lambda: events.append('blas-clear')))
            stack.enter_context(mock.patch.object(adapter.common, 'write'))
            with self.assertRaisesRegex(RuntimeError, 'controlled forward failure'):
                adapter.train(args, probe=True)
        self.assertFalse(state)
        self.assertLess(events.index('model-cpu'), events.index('budget-unlock'))
        self.assertLess(events.index('blas-clear'), events.index('budget-unlock'))
        self.assertEqual(events[-1], 'budget-unlock')

    def test_space_row_identity_and_backbone_retained(self):
        old = self.model(3); new = self.model(4)
        state = old.state_dict()
        state['decoder.gtc_head.unused'] = torch.ones(2)
        receipt = adapter.transplant(new, state, [adapter.BLANK, 'a', ' '], [adapter.BLANK, 'a', 'x', ' '])
        self.assertTrue(torch.equal(new.decoder.fc.weight[3], old.decoder.fc.weight[2]))
        self.assertTrue(torch.equal(new.encoder.weight, old.encoder.weight))
        self.assertIn('decoder.gtc_head.unused', receipt['discarded_training_only_sgm'])

    def test_empty_loss_normalized_by_timesteps_retains_gradient(self):
        for t in (4, 8):
            logits = torch.zeros(2, t, 3, requires_grad=True)
            loss = adapter.exact_ctc_loss(logits, [[], []])
            self.assertAlmostEqual(loss.item(), np.log(3), places=5)
            loss.backward()
            self.assertGreater(logits.grad.abs().sum().item(), 0)
        mixed = torch.randn(2, 8, 3, requires_grad=True)
        value = adapter.exact_ctc_loss(mixed, [[1, 1, 2], []])
        self.assertTrue(torch.isfinite(value)); value.backward()
        self.assertTrue(torch.isfinite(mixed.grad).all())

    def test_empty_negative_has_finite_ctc_backward(self):
        logits = torch.randn(4, 2, 3, requires_grad=True)
        loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
            logits.log_softmax(-1), torch.empty(0, dtype=torch.long),
            torch.tensor([4, 4]), torch.tensor([0, 0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_unexpected_backbone_and_missing_tensor_refused(self):
        old = self.model(3); new = self.model(3)
        state = old.state_dict(); state['encoder.unknown'] = torch.ones(1)
        with self.assertRaises(ValueError): adapter.transplant(new, state, [adapter.BLANK, 'a', ' '], [adapter.BLANK, 'a', ' '])
        del state['encoder.unknown']; del state['encoder.weight']
        with self.assertRaises(ValueError): adapter.transplant(new, state, [adapter.BLANK, 'a', ' '], [adapter.BLANK, 'a', ' '])


if __name__ == '__main__': unittest.main()
