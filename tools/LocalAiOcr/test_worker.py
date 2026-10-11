"""Contract regressions for the fixed offline helper; no inference/training needed."""
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

WORKER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "local-ai-ocr-worker.py"
spec = importlib.util.spec_from_file_location("mr_local_ai_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerContractTests(unittest.TestCase):
    def result(self, box=None, text="中文 OCR 2024-09-22 23:04:51", score=0.91):
        box = box if box is not None else [[10, 20], [210, 20], [210, 47], [10, 47]]
        return SimpleNamespace(boxes=[box], txts=[text], scores=[score])

    def test_detection_box_and_combined_timestamp_stay_whole(self):
        result = self.result()
        payload = worker.serialize_result(result, 400, 100)
        rows = payload.decode("utf-8").splitlines()
        self.assertEqual(len(rows), 2)
        columns = rows[1].split("\t")
        self.assertEqual(len(columns), 12)
        self.assertEqual(columns[6:10], ["10", "20", "200", "27"])
        self.assertEqual(columns[11], result.txts[0])

    def test_real_fractional_box_envelope_is_clipped_to_image(self):
        result = self.result([[-0.5, 19.7], [210.2, 20.4], [210.1, 47.3], [-0.2, 46.7]])
        columns = worker.serialize_result(result, 200, 40).decode().splitlines()[1].split("\t")
        self.assertEqual(columns[6:10], ["0", "19", "200", "21"])

    def test_tabs_and_linebreaks_cannot_inject_tsv_rows(self):
        columns = worker.serialize_result(self.result(text="心得\tAAA\r\nBBB"), 400, 100).decode().splitlines()[1].split("\t")
        self.assertEqual(len(columns), 12)
        self.assertEqual(columns[11], "心得 AAA  BBB")

    def test_invalid_numeric_geometry_and_nonfinite_score_are_rejected(self):
        for result in [self.result([[0, np.nan], [10, 0], [10, 10], [0, 10]]),
                       self.result([[0, 0], [10, 10]]), self.result(score=float("inf"))]:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    worker.serialize_result(result, 400, 100)

    def test_mismatched_output_vectors_are_rejected(self):
        result = self.result()
        result.scores = []
        with self.assertRaises(ValueError):
            worker.serialize_result(result, 400, 100)

    def test_no_detections_does_not_invent_text(self):
        result = SimpleNamespace(boxes=None, txts=None, scores=None)
        self.assertEqual(worker.serialize_result(result, 400, 100), worker.HEADER.encode())

    def test_output_limit_is_enforced_before_file_write(self):
        with self.assertRaises(ValueError):
            worker.serialize_result(self.result(text="字" * (3 * 1024 * 1024)), 400, 100)

    def test_audit_hook_blocks_connect_and_dns_before_network_access(self):
        # Audit hooks are process-global and cannot be removed, so isolate this
        # proof from the test runner instead of changing its network policy.
        code = """
import importlib.util, socket, sys  # BOUNDARY-ALLOW(NET-010): Test-only module import to assert that the installed audit hook denies I/O.
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
sys.addaudithook(worker.deny_network)
for operation in (lambda: socket.getaddrinfo('example.com', 443),  # BOUNDARY-ALLOW(NET-008): Audit hook is installed above; this probe must raise before DNS resolution.
                  lambda: socket.socket().connect(('127.0.0.1', 9))):
    try:
        operation()
    except RuntimeError as error:
        assert str(error) == 'Offline worker forbids network access'
    else:
        raise AssertionError('Network operation was not blocked')
"""
        subprocess.run([sys.executable, "-c", code, str(WORKER_PATH)], check=True, timeout=5)


@dataclass
class StubTextRecInput:
    """Input carrier for fake recognizers; actual inference is checked in the frozen worker."""
    img: object = None
    return_word_box: bool = False


class SymbolRereadTests(unittest.TestCase):
    def setUp(self):
        # These tests inject fake recognizers and verify crop/vote geometry. Avoid loading
        # RapidOCR and inference libraries just to construct its two-field input carrier.
        # Restore any real installed modules after each case; production imports are intact.
        parent = ModuleType('rapidocr')
        parent.__path__ = []
        recognition = ModuleType('rapidocr.ch_ppocr_rec')
        recognition.TextRecInput = StubTextRecInput
        input_modules = patch.dict(sys.modules, {'rapidocr': parent,
                                                'rapidocr.ch_ppocr_rec': recognition})
        input_modules.start()
        self.addCleanup(input_modules.stop)

    def test_two_views_restore_observed_face_tail_without_changing_letters(self):
        self.assertEqual(worker.choose_reread('(;', [('(; ~)', .82), ('(;~)', .79), ('6;)', .99)]), ('(; ~)', .79))

    def test_single_view_and_high_confidence_letter_change_are_rejected(self):
        self.assertIsNone(worker.choose_reread('Σ(っ°;)っ', [('Σ(っ°Δ°;)っ', .99), ('Σ(っ°Δ°;)っ', .99), ('Σ(っ°°;)っ', .90)]))
        self.assertIsNone(worker.choose_reread('正文', [('0000正文', .99), ('0000正文', .99)]))

    def test_full_half_width_punctuation_is_not_normalized_on_a_tie(self):
        self.assertIsNone(worker.choose_reread('030（怎么', [('030(怎么', .99), ('030(怎么', .99)]))

    def test_more_symbols_cannot_delete_or_normalize_existing_symbols(self):
        for original,candidate in [('(;)','[!!]'),('030（怎么','030(怎么!)'),('正文？？？','正文...!')]:
            with self.subTest(original=original):
                self.assertIsNone(worker.choose_reread(original,[(candidate,.99),(candidate,.99)]))

    def test_existing_punctuation_order_is_preserved_when_symbols_are_added(self):
        self.assertEqual(worker.choose_reread('(;)',[('(; ~)',.82),('(;~)',.79)]),('(; ~)',.79))
        self.assertIsNone(worker.choose_reread('(;)',[(');(!',.99),(');(!',.99)]))

    def test_symbols_cannot_move_relative_to_original_letters(self):
        for original,candidate in [('foo(bar)','(foo)bar!'),('(ab)','a(b)!')]:
            with self.subTest(original=original):
                self.assertIsNone(worker.choose_reread(original,[(candidate,.99),(candidate,.99)]))

    def test_solid_face_eye_has_consistent_two_view_support(self):
        self.assertEqual(worker.choose_reread('(●∀•)', [('(●∀●)', .96), ('(●∀●)', .95)]), ('(●∀●)', .95))

    def test_empty_or_nonfinite_views_do_not_complete_brackets(self):
        self.assertIsNone(worker.choose_reread('(', [('', .99), ('()', float('nan')), ('()', .4)]))

    def test_hollow_prefix_count_comes_from_pixels(self):
        import cv2
        image = np.full((60,160,3), (35,70,30), dtype=np.uint8)
        for x in (62,72,82,92):
            cv2.circle(image,(x,34),2,(245,245,245),1)
        self.assertEqual(worker.visible_ring_prefix(image,(100,18,150,40)),4)

    def test_digits_filled_dots_and_blank_pixels_are_not_hollow_stops(self):
        import cv2
        for variant in ('zeros','filled','blank'):
            image = np.full((60,160,3), (35,70,30), dtype=np.uint8)
            for x in (62,72,82,92):
                if variant=='zeros':
                    cv2.ellipse(image,(x,29),(3,9),0,0,360,(245,245,245),1)
                elif variant=='filled':
                    cv2.circle(image,(x,34),2,(245,245,245),-1)
            with self.subTest(variant=variant):
                self.assertEqual(worker.visible_ring_prefix(image,(100,18,150,40)),0)

    def test_prefix_already_covered_by_another_detection_is_not_duplicated(self):
        import cv2
        image=np.full((60,160,3), (35,70,30), dtype=np.uint8)
        for x in (62,72,82,92):
            cv2.circle(image,(x,34),2,(245,245,245),1)
        result=SimpleNamespace(boxes=[[[58,30],[96,30],[96,38],[58,38]],[[100,18],[150,18],[150,40],[100,40]]],txts=['oooo','正文'],scores=[.9,.9])
        def forbidden():
            self.fail('Covered marks must not trigger a second prefix recovery')
        self.assertIs(worker.reread_symbols(SimpleNamespace(_load_rec_model=forbidden),image,result),result)

    def test_vertical_retry_crop_does_not_consume_the_next_detected_line(self):
        image=np.full((90,180,3), (35,70,30), dtype=np.uint8)
        image[22:39,105:109]=(245,245,245)
        image[43:62,120:124]=(200,200,200)
        result=SimpleNamespace(boxes=[[[100,20],[130,20],[130,40],[100,40]],[[100,43],[130,43],[130,63],[100,63]]],txts=['(;',')'],scores=[.9,.9])
        calls=[]
        def recognize(input):
            calls.extend(input.img)
            return SimpleNamespace(txts=['']*len(input.img),scores=[.99]*len(input.img))
        worker.reread_symbols(SimpleNamespace(_load_rec_model=lambda:recognize),image,result)
        self.assertEqual(len(calls),6)
        self.assertFalse(np.any(np.all(calls[0]==[200,200,200],axis=2)))
        self.assertFalse(np.any(np.all(calls[3]==[245,245,245],axis=2)))

    def test_rotated_neighbour_remains_a_crop_boundary(self):
        image=np.full((90,220,3),(35,70,30),dtype=np.uint8)
        image[22:39,105:109]=(245,245,245)
        image[26:53,144:151]=(200,200,200)
        result=SimpleNamespace(boxes=[[[100,20],[130,20],[130,40],[100,40]],[[145,25],[165,40],[157,56],[138,41]]],txts=['(;',')'],scores=[.9,.9])
        calls=[]
        def recognize(input):
            calls.extend(input.img)
            return SimpleNamespace(txts=['(;)' for _ in input.img],scores=[.99]*len(input.img))
        output=worker.reread_symbols(SimpleNamespace(_load_rec_model=lambda:recognize),image,result)
        self.assertEqual(len(calls),3)
        self.assertFalse(np.any(np.all(calls[0]==[200,200,200],axis=2)))
        np.testing.assert_equal(output.boxes[-1],result.boxes[-1])

    def test_rereads_are_bounded_and_blank_outputs_keep_original_boxes(self):
        import cv2
        image=np.full((2200,220,3), (35,70,30), dtype=np.uint8)
        boxes=[]
        for i in range(100):
            y=10+i*21
            boxes.append([[70,y],[120,y],[120,y+14],[70,y+14]])
            cv2.putText(image,'(;',(75,y+12),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
        result=SimpleNamespace(boxes=boxes,txts=['(;']*100,scores=[.9]*100)
        seen=[]
        def recognize(input):
            seen.extend(input.img)
            return SimpleNamespace(txts=['']*len(input.img),scores=[.99]*len(input.img))
        engine=SimpleNamespace(_load_rec_model=lambda:recognize)
        output=worker.reread_symbols(engine,image,result)
        self.assertEqual(output.txts,result.txts)
        np.testing.assert_equal(output.boxes,result.boxes)
        self.assertLessEqual(len(seen),worker.MAX_REREAD_GROUPS*3)
        self.assertGreater(len(seen),0)
        self.assertLess(sum(c.shape[0]*c.shape[1] for c in seen),worker.MAX_REREAD_PIXELS+100_000)


class ImageOnlyRuntimeBuildTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('mr_ocr_builder', Path(__file__).with_name('build_worker.py'))
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.cv2 = self.output / '_internal' / 'cv2'
        self.cv2.mkdir(parents=True)

    def test_optional_video_dll_is_removed_without_changing_image_dependencies(self):
        video = self.cv2 / 'opencv_videoio_ffmpeg500_64.dll'
        image = self.cv2 / 'cv2.pyd'
        video.write_bytes(b'optional video DLL')
        image.write_bytes(b'required still-image extension')
        records = self.builder.remove_unused_video_runtime(self.output)
        self.assertFalse(video.exists())
        self.assertEqual(image.read_bytes(), b'required still-image extension')
        self.assertEqual(records[0]['path'], '_internal/cv2/opencv_videoio_ffmpeg500_64.dll')
        self.assertEqual(records[0]['bytes'], len(b'optional video DLL'))

    def test_unfamiliar_ffmpeg_payload_refuses_before_any_file_is_removed(self):
        known = self.cv2 / 'opencv_videoio_ffmpeg500_64.dll'
        unknown = self.output / 'another_ffmpeg_backend.dll'
        known.write_bytes(b'known optional video DLL')
        unknown.write_bytes(b'unknown video DLL')
        with self.assertRaisesRegex(RuntimeError, 'Unexpected FFmpeg payload'):
            self.builder.remove_unused_video_runtime(self.output)
        self.assertTrue(known.exists())
        self.assertTrue(unknown.exists())


if __name__ == "__main__":
    unittest.main()
