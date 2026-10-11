"""Meaningful development gates: Unicode fidelity, leakage and rejection paths."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from common import decode,decide,edit_distance,local_output,metrics,prepare,sha256,write_json,dataset_fingerprint
from synthetic import make_texts,load_corpus
from evaluate import load_candidate,compare,summarize
from types import SimpleNamespace
from corpus import extract
from compatibility import audit_texts

class Gates(unittest.TestCase):
    def test_public_corpus_parser_preserves_entities_spaces_and_symbols(self):
        page='<ul class="symbol_list"><li><span> (Д  ω)&gt; <br>caption</span></li><li><span>(Д  ω)&gt;<br>duplicate</span></li></ul>'
        texts,audit=extract(page)
        self.assertEqual(texts,["(Д  ω)>"])
        self.assertEqual(audit["raw_entries"],2)
    def test_public_corpus_rejects_tags_and_controls(self):
        texts,audit=extract('<ul class="symbol_list"><span><script>x</script><br>x</span><span>a\u200bb<br>x</span></ul>')
        self.assertEqual(texts,[])
        self.assertEqual(audit["rejected"],2)
    def test_empty_split_has_no_invented_zero_error_rate(self):
        result=metrics([],"candidate")
        self.assertIsNone(result["cer"])
        self.assertIsNone(result["symbol_cer"])
    def test_emoji_joiner_is_required_for_exact_representability(self):
        self.assertFalse(audit_texts(["🙂\u200d🙂"],"🙂")[0]["representable"])
    def test_corpus_reserved_text_and_missing_glyph_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"corpus.json"
            write_json(path,dict(schema=1,texts=["(Дω)","☃"],source="https://example.org",html_sha256="0"*64))
            renderer=SimpleNamespace(maps={"test":{ord(c) for c in "(Дω)"}})
            rows,meta=load_corpus(path,renderer,{"(Дω)"})
            self.assertEqual(rows,[])
            self.assertEqual(meta["exclusions"]["private_reserved"],1)
            self.assertEqual(meta["exclusions"]["unsupported_font"],1)
    def test_ctc_repeated_symbols_keep_blank_separation(self):
        logits=np.full((5,4),-20.)
        for row,c in enumerate([1,1,0,1,2]):
            logits[row,c]=20.
        self.assertEqual(decode(logits,"Дω!")[0],"ДДω")
    def test_raw_unicode_and_spaces_not_normalized(self):
        self.assertEqual(edit_distance("Дω","Δw"),2)
        rows=[dict(label="(Д ω)",candidate="（Дω）")]
        self.assertEqual(metrics(rows,"candidate")["exact"],0)
        self.assertGreater(metrics(rows,"candidate")["cer"],0)
    def test_missing_evidence_cannot_promote(self):
        self.assertFalse(decide({}, {}, {})["promote"])
        self.assertIn("real_accuracy",decide({}, {}, {})["failed"])
    def test_synthetic_gain_does_not_replace_real_labels(self):
        good=dict(baseline=dict(count=10,exact=2,cer=.5,symbol_cer=.5),
                  candidate=dict(count=10,exact=8,cer=.1,symbol_cer=.1))
        self.assertFalse(decide({"face":good},{},{})["promote"])
    def test_raw_quality_cannot_prove_production_region_safety(self):
        good=dict(baseline=dict(count=10,exact=2,cer=.5,symbol_cer=.5),
                  candidate=dict(count=10,exact=8,cer=.1,symbol_cer=.1))
        dataset=dict(face=good,plain=good,face_regressions=0,face_erroneous_changes=0,plain_regressions=0,plain_erroneous_changes=0,
                     negative_count=10,negative_false_outputs=0)
        resources=dict(model_bytes=1,providers=["CPUExecutionProvider"],threads=2,
                       forty_eight_crops_seconds=1,groups=48,probe_total_pixels=990000,peak_extra_bytes=1,peak_method="windows_process_peak_wset")
        self.assertFalse(decide(dataset,dataset,resources)["promote"])
        self.assertEqual(decide(dataset,dataset,resources)["failed"],
                         ["production_region_and_lifecycle_evidence"])
    def test_new_plain_error_and_negative_output_reject(self):
        good=dict(baseline=dict(count=10,exact=2,cer=.5,symbol_cer=.5),
                  candidate=dict(count=10,exact=8,cer=.1,symbol_cer=.1))
        dataset=dict(face=good,plain=good,face_regressions=0,plain_regressions=1,
                     negative_count=10,negative_false_outputs=1)
        self.assertIn("no_added_plain_errors",decide(dataset,dataset,{})["failed"])
        self.assertIn("negative_safety",decide(dataset,dataset,{})["failed"])
    def test_already_wrong_prose_changed_to_another_error_rejects(self):
        rows=[dict(kind="plain",label="abcd",baseline="abce",candidate="XXXXXX")]
        report=summarize(rows)
        self.assertEqual(report["plain_regressions"],0)
        self.assertEqual(report["plain_erroneous_changes"],1)
        self.assertFalse(decide(report,report,{})["gates"]["no_added_plain_errors"])
    def test_already_wrong_face_changed_to_another_error_rejects(self):
        report=summarize([dict(kind="face",label="(Д)",baseline="(A)",candidate="(X)")])
        self.assertEqual(report["face_regressions"],0)
        self.assertEqual(report["face_erroneous_changes"],1)
        self.assertFalse(decide(report,report,{})["gates"]["no_real_regressions"])
    def test_text_pool_is_unique_and_seeded(self):
        rows=make_texts(42,300)
        self.assertEqual(rows,make_texts(42,300))
        self.assertEqual(len({text for text,_ in rows}),300)
        self.assertTrue(all(t==t.strip() for t,_ in rows))
        self.assertFalse(set(t for t,_ in rows[:100]) & set(t for t,_ in rows[100:]))
    def test_blank_stays_empty_image(self):
        self.assertEqual(prepare(Image.new("RGB",(30,20),"white")).sum(),0)
    def test_invalid_ctc_rejects_nan(self):
        with self.assertRaises(ValueError):
            decode(np.array([[np.nan,0.]]),"a")
    def test_frozen_inventory_rejects_changed_image_or_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            image=root/"image.png"
            Image.new("L",(10,10)).save(image)
            for name in ("train","dev","test"):
                write_json(root/f"{name}.json",[dict(path="image.png",label=name,kind="plain",sha256=sha256(image))])
            write_json(root/"dataset.json",dict(schema=1,synthetic_only=True,
                image_rows={n:1 for n in ("train","dev","test")},
                split_sha256={n:sha256(root/f"{n}.json") for n in ("train","dev","test")}))
            frozen=dataset_fingerprint(root)
            self.assertIn("test",frozen["split_sha256"])
            original=image.read_bytes()
            image.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                dataset_fingerprint(root)
            image.write_bytes(original)
            write_json(root/"test.json",[dict(path="image.png",label="post hoc",sha256=sha256(image))])
            with self.assertRaises(ValueError):
                dataset_fingerprint(root)
    def test_missing_peak_measurement_rejects_resource_gate(self):
        resource=dict(model_bytes=1,threads=2,providers=["CPUExecutionProvider"],
                      forty_eight_crops_seconds=1,session_rss_delta_bytes=1)
        self.assertFalse(decide({}, {}, resource)["gates"]["cpu_budget"])
    def test_frozen_alphabet_tamper_rejects_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_json(root/"alphabet.json",dict(alphabet="Дω"))
            (root/"kaomoji.onnx").write_bytes(b"not loaded")
            write_json(root/"training.json",dict(
                alphabet_sha256=sha256(root/"alphabet.json"),model_sha256=sha256(root/"kaomoji.onnx")))
            write_json(root/"alphabet.json",dict(alphabet="ωД"))
            with self.assertRaisesRegex(ValueError,"alphabet changed"):
                load_candidate(root)
    def test_baseline_score_vector_missing_does_not_silently_drop_rows(self):
        fake=SimpleNamespace(_load_rec_model=lambda:
            lambda _:SimpleNamespace(txts=["x"],scores=[]))
        with self.assertRaisesRegex(ValueError,"count mismatch"):
            compare([Image.new("RGB",(30,20))],[dict(label="x")],None,"x",fake)
    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            local_output(Path(__file__).resolve().parent/"candidate.onnx")

if __name__=="__main__":
    unittest.main()
