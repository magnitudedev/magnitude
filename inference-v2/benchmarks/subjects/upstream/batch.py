"""Stock MLX-VLM continuous batching at fixed prompt and output work.

This runs the upstream generator, including its prompt scheduling and cache
batching. It uses no Magnitude model, state, generation, or scheduler adapter.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import mlx.core as mx
from mlx_vlm.generate.ar import BatchGenerator
from mlx_vlm.utils import StoppingCriteria, load_model, load_processor

from benchmarks.contracts import Observation
from magnitude_engine.artifacts.source import LocalArtifact


class BatchTrace:
    def __init__(
        self, *, artifact: LocalArtifact, prompt_text: str, prompt_tokens: int,
        output_tokens: int, rows: int, prefill_tokens: int, cache_limit_bytes: int,
    ):
        if min(prompt_tokens, output_tokens, rows, prefill_tokens) < 1 or rows > 64:
            raise ValueError("invalid upstream batch geometry")
        self.rows, self.output_tokens, self.prefill_tokens = rows, output_tokens, prefill_tokens
        self.model = self.processor = self.generator = None
        self.expected = None
        self.outputs, self.waves = [], []
        self.closed = False
        self._cache_limit = mx.set_cache_limit(cache_limit_bytes)
        try:
            self.model = load_model(Path(artifact.path), lazy=False, strict=True)
            self.processor = load_processor(Path(artifact.path), trust_remote_code=False)
            tokenizer: Any = getattr(self.processor, "tokenizer", self.processor)
            # Fixed-output mechanism control, just like EngineWaves: EOS does not
            # truncate the declared work and no hidden stop condition is inherited.
            tokenizer.stopping_criteria = StoppingCriteria([], tokenizer)
            unit = tokenizer.encode(prompt_text)
            if not unit:
                raise ValueError("upstream benchmark prompt is empty")
            self.prompt = (unit * ((prompt_tokens + len(unit) - 1) // len(unit)))[:prompt_tokens]
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        self.complete()
        self.outputs, self.waves = [], []
        mx.reset_peak_memory()

    def invoke(self) -> None:
        assert self.model is not None
        embed_inputs = self.model.get_input_embeddings
        if embed_inputs is None:
            raise ValueError("upstream batching requires input embedding preparation")
        for _ in range(2):
            start = perf_counter_ns()
            self.generator = BatchGenerator(
                self.model.language_model, self.processor,
                max_tokens=self.output_tokens, completion_batch_size=self.rows,
                prefill_batch_size=self.rows, prefill_step_size=self.prefill_tokens,
                compute_logprobs=False, greedy_sampling=True,
            )
            try:
                prompt_kwargs = []
                for _ in range(self.rows):
                    features = embed_inputs(mx.array([self.prompt]), None)
                    prompt_kwargs.append({
                        key: value for key, value in features.to_dict().items() if value is not None
                    })
                uids = self.generator.insert(
                    [list(self.prompt) for _ in range(self.rows)], prompt_kwargs=prompt_kwargs,
                )
                outputs = {uid: [] for uid in uids}
                first, finished = {}, {}
                for _ in range(len(self.prompt) + self.output_tokens + self.rows + 10):
                    _, responses = self.generator.next()
                    now = perf_counter_ns() - start
                    for response in responses:
                        outputs[response.uid].append(response.token)
                        first.setdefault(response.uid, now)
                        if response.finish_reason is not None:
                            if response.finish_reason != "length":
                                raise ValueError("upstream ended before its fixed output allowance")
                            finished[response.uid] = now
                    if len(finished) == self.rows:
                        break
                else:
                    raise RuntimeError("upstream batch exceeded its declared work bound")
                self.complete()
                self.outputs.extend(outputs[uid] for uid in uids)
                self.waves.append({
                    "elapsed_ns": perf_counter_ns() - start,
                    "first_token_ns": [first[uid] for uid in uids],
                    "finished_ns": [finished[uid] for uid in uids],
                })
            finally:
                self.generator.close()
                self.generator = None

    def complete(self) -> None:
        mx.synchronize()

    def observe(self) -> Observation:
        if len(self.outputs) != 2 * self.rows or any(
            len(row) != self.output_tokens for row in self.outputs
        ):
            raise ValueError("upstream batch did not complete the declared output work")
        if self.expected is None:
            self.expected = self.outputs
        if self.outputs != self.expected:
            raise ValueError("upstream continuous batching is not repeatable")
        return Observation(
            hashlib.sha256(json.dumps(self.outputs).encode()).hexdigest(),
            {"requests": len(self.outputs), "actual_prompt_tokens_per_request": len(self.prompt),
             "output_tokens": sum(map(len, self.outputs)), "cached_tokens": 0,
             "mlx_peak_bytes": mx.get_peak_memory()},
            {"outputs": self.outputs, "waves": self.waves,
             "prompt_digest": hashlib.sha256(json.dumps(self.prompt).encode()).hexdigest()},
        )

    def close(self) -> None:
        if self.closed:
            return
        self.complete()
        if self.generator is not None:
            self.generator.close()
            self.generator = None
        self.model = self.processor = None
        self.outputs = []
        self.expected = None
        mx.set_cache_limit(self._cache_limit)
        self.closed = True
