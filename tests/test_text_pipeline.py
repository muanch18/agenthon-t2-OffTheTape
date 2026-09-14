"""Cutoff, strict JSON, evidence, cache, prior, and endpoint tests for text state."""

import io
import json
import shutil
import unittest
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

from off_the_tape.macro_extractor import extract_macro_state, extract_with_prior
from off_the_tape.macro_heuristic import OfflineHeuristicModel
from off_the_tape.macro_llm import EndpointModel
from off_the_tape.macro_state import MacroState
from off_the_tape.text_corpus import TextCorpus, load_corpus
from off_the_tape.text_selector import DocumentSelector

UNIT = Path(__file__).parent / "fixtures" / "text_unit"


class FakeModel:
    mode = "llm"

    def __init__(self, response=None):
        self.calls = 0
        self.response = response
        self.prompts = []

    def complete(self, *, system, user, documents, model_name, seed):
        self.calls += 1
        self.prompts.append((system, user, seed))
        if self.response is not None:
            return self.response
        state = MacroState.neutral().to_dict()
        selected = {doc.doc_id for doc in documents}
        state["policy_stance"] = 2.0 if "new-policy" in selected else -0.4
        state["confidence"] = 1.5
        doc_id = "new-policy" if "new-policy" in selected else "old-policy"
        return json.dumps({
            "schema_version": "1",
            "state": state,
            "evidence": [{
                "signal": "policy_stance", "value": state["policy_stance"],
                "supporting_docs": [doc_id], "summary": "Policy language changed.",
            }],
        })


class TextPipelineTest(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path("reports") / f"text_test_cache_{uuid.uuid4().hex}"
        self.addCleanup(lambda: shutil.rmtree(self.cache_dir, ignore_errors=True))

    def test_loader_uses_release_cutoff_without_reading_future_file(self):
        corpus = load_corpus(UNIT, date(2024, 2, 1))
        self.assertEqual({doc.doc_id for doc in corpus.documents}, {"old-policy", "new-policy"})
        self.assertTrue(all(doc.timestamp <= corpus.asof for doc in corpus.documents))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_corpus(UNIT, date(2024, 2, 3))

    def test_selector_is_deterministic_relevant_and_budgeted(self):
        corpus = load_corpus(UNIT, date(2024, 2, 1))
        selector = DocumentSelector(max_documents=1, max_doc_chars=90, max_total_chars=90)
        chosen = selector.select(corpus, family="T2-F1", panel_id="rates_daily")
        self.assertEqual(chosen[0].doc_id, "new-policy")
        self.assertEqual(chosen, selector.select(corpus, family="T2-F1", panel_id="rates_daily"))
        self.assertLessEqual(len(chosen[0].excerpt), 90)

    def test_clamps_valid_json_requires_evidence_and_reuses_exact_cache(self):
        model = FakeModel()
        kwargs = dict(
            asof=date(2024, 2, 1), family="T2-F1", panel_id="rates_daily",
            client=model, model_name="fake-v1", cache_dir=self.cache_dir,
            selector=DocumentSelector(max_documents=1),
        )
        first = extract_macro_state(UNIT, **kwargs)
        self.assertEqual(first.source, "llm")
        self.assertEqual(first.state.policy_stance, 1.0)
        self.assertEqual(first.state.confidence, 1.0)
        self.assertEqual(first.evidence[0].supporting_docs, ("new-policy",))
        self.assertIn("Do not forecast any asset price", model.prompts[0][0])
        self.assertEqual(model.prompts[0][2], 2026)
        second = extract_macro_state(UNIT, **kwargs)
        self.assertEqual(second.source, "cache")
        self.assertEqual(model.calls, 1)
        self.assertEqual(first.state, second.state)

        # An unselected document changed: the full information-set hash still invalidates cache.
        corpus = load_corpus(UNIT, date(2024, 2, 1))
        changed = replace(corpus.documents[0], sha256="changed")
        changed_corpus = TextCorpus(corpus.card_id, corpus.asof, (changed, corpus.documents[1]))
        with patch("off_the_tape.macro_extractor.load_corpus", return_value=changed_corpus):
            third = extract_macro_state(UNIT, **kwargs)
        self.assertEqual(third.source, "llm")
        self.assertEqual(model.calls, 2)
        self.assertNotEqual(first.cache_key, third.cache_key)

    def test_invalid_output_and_missing_evidence_fall_back_neutrally(self):
        invalid = FakeModel("not JSON")
        kwargs = dict(asof=date(2024, 2, 1), client=invalid, model_name="bad", cache_dir=self.cache_dir)
        first = extract_macro_state(UNIT, **kwargs)
        self.assertEqual(first.source, "fallback")
        self.assertEqual(first.state, MacroState.neutral())
        extract_macro_state(UNIT, **kwargs)
        self.assertEqual(invalid.calls, 2)  # Failure was not cached.

        state = MacroState.neutral().to_dict()
        state["market_stress"] = 0.8
        missing_evidence = FakeModel(json.dumps({"schema_version": "1", "state": state, "evidence": []}))
        result = extract_macro_state(
            UNIT, asof=date(2024, 2, 1), client=missing_evidence,
            model_name="missing-evidence", cache_dir=None,
        )
        self.assertEqual(result.state, MacroState.neutral())

        # Even a small non-neutral signal requires a citation to a selected doc.
        state = MacroState.neutral().to_dict()
        state["inflation_change"] = 0.01
        tiny_signal = FakeModel(json.dumps({"schema_version": "1", "state": state, "evidence": []}))
        self.assertEqual(extract_macro_state(
            UNIT, asof=date(2024, 2, 1), client=tiny_signal,
            model_name="tiny-signal", cache_dir=None,
        ).source, "fallback")

        state["inflation_change"] = 0.3
        future_citation = FakeModel(json.dumps({
            "schema_version": "1", "state": state,
            "evidence": [{"signal": "inflation_change", "value": 0.3,
                          "supporting_docs": ["future-policy"], "summary": "Claim."}],
        }))
        self.assertEqual(extract_macro_state(
            UNIT, asof=date(2024, 2, 1), client=future_citation,
            model_name="future-citation", cache_dir=None,
        ).source, "fallback")

    def test_ablation_and_configurable_prior(self):
        model = FakeModel()
        ablated = extract_macro_state(
            Path("nonexistent"), asof=date(2024, 2, 1), text_enabled=False,
            client=model, cache_dir=None,
        )
        self.assertEqual(ablated.source, "ablated")
        self.assertEqual(ablated.state, MacroState.neutral())
        self.assertEqual(model.calls, 0)

        result = extract_with_prior(
            UNIT, asof=date(2024, 2, 1), window_days=31,
            client=model, model_name="fake-prior", cache_dir=None,
        )
        self.assertEqual(result.previous.asof, date(2024, 1, 1))
        self.assertAlmostEqual(result.delta.policy_stance_delta, 1.4)
        with self.assertRaisesRegex(ValueError, "positive"):
            extract_with_prior(UNIT, asof=date(2024, 2, 1), window_days=0, client=model)

    def test_offline_proxy_is_stable_and_explicit(self):
        model = OfflineHeuristicModel()
        kwargs = dict(asof=date(2024, 2, 1), client=model, model_name="offline-v1", cache_dir=None)
        first = extract_macro_state(UNIT, **kwargs)
        second = extract_macro_state(UNIT, **kwargs)
        self.assertEqual(first.source, "heuristic")
        self.assertEqual(first.state, second.state)
        self.assertLess(first.state.confidence, 1.0)

    def test_endpoint_uses_organizer_style_deterministic_chat_request(self):
        result = {"choices": [{"message": {"content": "{}"}}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(result).encode())) as send:
            raw = EndpointModel("http://model:8000/v1").complete(
                system="system", user="user", documents=(), model_name="house", seed=17,
            )
        self.assertEqual(raw, "{}")
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://model:8000/v1/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual((payload["temperature"], payload["seed"]), (0, 17))
        self.assertEqual(payload["model"], "house")
        self.assertNotIn("tools", payload)


if __name__ == "__main__":
    unittest.main()
