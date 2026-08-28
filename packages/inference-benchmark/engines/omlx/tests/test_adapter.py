from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from magnitude_omlx_benchmark.config import AdapterConfiguration, detected_backend, write_configuration
from magnitude_omlx_benchmark import instrumentation
from magnitude_omlx_benchmark.identity import _installed_git_revision
from magnitude_omlx_benchmark.qualify import _verify_locked_snapshot


class AdapterConfigurationTests(unittest.TestCase):
    def model(self, root: Path, model_type: str = "qwen3_5") -> Path:
        model = root / "snapshot"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": model_type}))
        (model / "model.safetensors").write_bytes(b"locked")
        return model

    def test_baseline_explicitly_disables_every_speculative_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = AdapterConfiguration(
                model=self.model(root), served_model="target", base_path=root / "base",
                context_capacity=4096, max_concurrent_requests=2, speculative_method="none",
            )
            write_configuration(config)
            settings = json.loads((config.base_path / "model_settings.json").read_text())["models"]["target"]
            self.assertFalse(settings["mtp_enabled"])
            self.assertFalse(settings["vlm_mtp_enabled"])
            self.assertFalse(settings["dflash_enabled"])
            self.assertFalse(settings["specprefill_enabled"])
            self.assertFalse(settings["turboquant_kv_enabled"])
            global_settings = json.loads((config.base_path / "settings.json").read_text())
            self.assertEqual(global_settings["version"], "1.0")
            self.assertEqual(global_settings["server"]["burst_decode_mode"], "off")
            self.assertFalse(global_settings["huggingface"]["hf_cache_enabled"])
            self.assertEqual((config.base_path / "models" / "target").resolve(), config.model.resolve())

    def test_dflash_and_dspark_are_explicit_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = self.model(root, "deepseek_v4_dspark")
            document = json.loads((target / "config.json").read_text())
            document["dspark_block_size"] = 4
            document["dspark_target_layer_ids"] = [1, 2]
            (target / "config.json").write_text(json.dumps(document))
            dspark = AdapterConfiguration(
                model=target, served_model="target", base_path=root / "dspark",
                context_capacity=4096, max_concurrent_requests=1,
                speculative_method="dspark", max_draft_tokens=3,
            )
            self.assertEqual(detected_backend(dspark), "dspark")
            draft = root / "draft"
            draft.mkdir()
            (draft / "config.json").write_text("{}")
            dflash = AdapterConfiguration(
                model=target, served_model="target", base_path=root / "dflash",
                context_capacity=4096, max_concurrent_requests=1,
                speculative_method="dflash", dflash_draft=draft, dflash_block_size=4,
            )
            write_configuration(dflash)
            settings = json.loads((dflash.base_path / "model_settings.json").read_text())["models"]["target"]
            self.assertTrue(settings["dflash_enabled"])
            self.assertEqual(settings["dflash_block_size"], 4)
            self.assertFalse(settings["dflash_in_memory_cache"])
            self.assertFalse(settings["dflash_draft_quant_enabled"])

    def test_rejects_settings_for_the_wrong_speculative_method(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = self.model(root)
            with self.assertRaisesRegex(ValueError, "baseline mode"):
                write_configuration(AdapterConfiguration(
                    model=target, served_model="target", base_path=root / "base",
                    context_capacity=4096, max_concurrent_requests=1,
                    speculative_method="none", max_draft_tokens=1,
                ))
            with self.assertRaisesRegex(ValueError, "positive max draft tokens"):
                write_configuration(AdapterConfiguration(
                    model=target, served_model="target", base_path=root / "mtp",
                    context_capacity=4096, max_concurrent_requests=1,
                    speculative_method="mtp", max_draft_tokens=-1,
                ))


class RuntimeIdentityTests(unittest.TestCase):
    def test_reads_the_installed_commit_instead_of_reporting_a_constant(self) -> None:
        revision = "a" * 40

        class Distribution:
            def read_text(self, name: str) -> str | None:
                if name != "direct_url.json":
                    return None
                return json.dumps({"vcs_info": {"commit_id": revision}})

        with patch("importlib.metadata.distribution", return_value=Distribution()):
            self.assertEqual(_installed_git_revision("omlx"), revision)

    def test_rejects_a_non_hex_commit_identity(self) -> None:
        class Distribution:
            def read_text(self, _name: str) -> str:
                return json.dumps({"vcs_info": {"commit_id": "z" * 40}})

        with patch("importlib.metadata.distribution", return_value=Distribution()):
            with self.assertRaisesRegex(RuntimeError, "invalid Git revision"):
                _installed_git_revision("omlx")


class LockedSnapshotTests(unittest.TestCase):
    def test_verifies_every_file_against_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            files = []
            for name in ("config.json", "model.safetensors"):
                content = (root / name).read_bytes()
                files.append({
                    "path": name,
                    "sizeBytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": files}))
            _verify_locked_snapshot(root, manifest)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "wrong digest"):
                _verify_locked_snapshot(root, manifest)

    def test_rejects_manifest_paths_outside_the_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside = root.parent / "outside.safetensors"
            outside.write_bytes(b"weights")
            try:
                content = outside.read_bytes()
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"files": [{
                    "path": "../outside.safetensors",
                    "sizeBytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }]}))
                with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                    _verify_locked_snapshot(root, manifest)
            finally:
                outside.unlink(missing_ok=True)


class TerminalNormalizationTests(unittest.TestCase):
    def test_seconds_are_converted_and_cached_tokens_are_excluded_from_prompt_n(self) -> None:
        key = "test"
        instrumentation._backend = "none"
        self.assertAlmostEqual(instrumentation._elapsed_ms(10.0, 10.012), 12.0)
        self.assertEqual(instrumentation._dflash_generation_ms(25_000, 7_000), 18.0)
        instrumentation._metrics[key] = instrumentation.RequestMetrics(
            prompt_ms=12.0, generation_ms=25.0
        )
        payload = instrumentation._terminal_payload(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
            key,
        )
        self.assertEqual(payload["timings"]["prompt_n"], 6)
        self.assertEqual(payload["timings"]["prompt_ms"], 12.0)
        self.assertEqual(payload["timings"]["predicted_ms"], 25.0)

    def test_first_batched_step_finishes_prompt_evaluation(self) -> None:
        metric = instrumentation.RequestMetrics(
            prompt_ms=8.0, decode_started_at=10.0
        )
        instrumentation._record_batched_response_time(
            metric, 10.003, has_uncached_prompt_token=True
        )
        self.assertAlmostEqual(metric.prompt_ms or 0.0, 11.0)
        self.assertEqual(metric.generation_ms, 0.0)
        instrumentation._record_batched_response_time(
            metric, 10.008, has_uncached_prompt_token=True
        )
        self.assertAlmostEqual(metric.generation_ms or 0.0, 5.0)

    def test_missing_request_local_duration_is_rejected(self) -> None:
        key = "missing"
        instrumentation._backend = "none"
        instrumentation._metrics[key] = instrumentation.RequestMetrics()
        with self.assertRaisesRegex(RuntimeError, "prompt evaluation duration"):
            instrumentation._terminal_payload(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "prompt_eval_duration": 99.0,
                        "generation_duration": 99.0,
                    },
                }, key,
            )

    def test_inconsistent_cached_token_count_is_rejected(self) -> None:
        key = "cached"
        instrumentation._backend = "none"
        instrumentation._metrics[key] = instrumentation.RequestMetrics(
            prompt_ms=1.0, generation_ms=2.0
        )
        with self.assertRaisesRegex(RuntimeError, "inconsistent token counts"):
            instrumentation._terminal_payload(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                }, key,
            )

    def test_speculative_counters_are_request_local_and_required(self) -> None:
        key = "mtp"
        instrumentation._backend = "mtp"
        instrumentation._metrics[key] = instrumentation.RequestMetrics(
            prompt_ms=1.0, generation_ms=2.0, draft_tokens=5, accepted_draft_tokens=3
        )
        payload = instrumentation._terminal_payload(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }, key,
        )
        self.assertEqual(payload["timings"]["draft_n"], 5)
        self.assertEqual(payload["timings"]["draft_n_accepted"], 3)
        self.assertEqual(payload["timings"]["speculative_backend"], "mtp")

        instrumentation._metrics[key] = instrumentation.RequestMetrics(
            prompt_ms=1.0, generation_ms=2.0, draft_tokens=2, accepted_draft_tokens=3
        )
        with self.assertRaisesRegex(RuntimeError, "accepted more draft tokens"):
            instrumentation._terminal_payload(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }, key,
            )


if __name__ == "__main__":
    unittest.main()
