import io
import json
import os
import select
import signal
import struct
import subprocess
import sys

import psutil
import pytest

from magnitude_engine import blueprints as bp
from magnitude_engine.engine.delivery import Finished, Tokens
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.worker.framing import MAX_FRAME_BYTES, Frame, read_frame, write_frame
from magnitude_engine.worker.host import Worker, WorkerUnavailable
from tests.models.architectures.qwen35.test_construction import artifact_pair


class Fragmented(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 3))

    def write(self, value):
        return super().write(value[:2])


def test_private_framing_handles_short_io_and_rejects_invalid_envelopes_before_allocation():
    stream = Fragmented()
    frame = Frame("generation", {"type": "hello", "unicode": "世界"})
    write_frame(stream, frame)
    stream.seek(0)
    assert read_frame(stream, "generation") == frame
    stream.seek(0)
    with pytest.raises(ValueError, match="generation"):
        read_frame(stream, "other")
    with pytest.raises(ValueError, match="size"):
        read_frame(io.BytesIO(struct.pack(">I", MAX_FRAME_BYTES + 1)))
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(b"\0\0\0\x20{}"))
    invalid = b'{"version":1,"version":1,"generation":"g","message":{}}'
    with pytest.raises(ValueError, match="duplicate"):
        read_frame(io.BytesIO(struct.pack(">I", len(invalid)) + invalid))
    invalid = b'{"version":1,"generation":"g","message":{"value":NaN}}'
    with pytest.raises(ValueError, match="non-finite"):
        read_frame(io.BytesIO(struct.pack(">I", len(invalid)) + invalid))
    for value in (True, 2):
        invalid = json.dumps({"version": value, "generation": "g", "message": {}}).encode()
        with pytest.raises(ValueError, match="version"):
            read_frame(io.BytesIO(struct.pack(">I", len(invalid)) + invalid))


def collect(request):
    output = []
    for _ in range(100):
        event = request.next(timeout=5)
        if isinstance(event, Finished):
            return tuple(output), event
        output.extend(event.values)
    raise AssertionError("worker request exceeded its output bound")


def compose_engine(
    target,
    head=None,
    *,
    memory_bytes=28 << 30,
    context_tokens=32768,
    output_capacity=64,
    max_active=8,
    prefill_tokens=512,
    retained_prefixes=32,
):
    artifact = bp.model.artifacts.Local(path=target)
    program = bp.model.programs.qwen35.Program(artifact=artifact)
    executor = bp.model.Executor(program=program, state=bp.model.state.PagedHybrid())
    method = bp.generation.methods.Plain()
    if head is not None:
        head_program = bp.model.programs.mtp.Head(
            artifact=bp.model.artifacts.Local(path=head),
            target=program,
        )
        method = bp.generation.methods.MTP(
            drafter=bp.model.Executor(
                program=head_program,
                state=bp.model.state.Native(source=head_program),
            )
        )
    return bp.engine.Engine(
        generation=bp.generation.Generation(target=executor, method=method),
        memory=bp.engine.memory.Budgeted(limit_bytes=memory_bytes),
        scheduler=bp.engine.scheduling.TimeShared(
            max_active=max_active, prefill_tokens=prefill_tokens
        ),
        prefixes=bp.engine.prefixes.Radix(
            retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=retained_prefixes),
        ),
        context_tokens=context_tokens,
        output_capacity=output_capacity,
    )


def configuration(tmp_path):
    target, head, _, _ = artifact_pair(tmp_path)
    return compose_engine(
        str(target),
        str(head),
        memory_bytes=64 << 20,
        context_tokens=128,
        output_capacity=2,
        max_active=2,
    )


def test_real_worker_runs_independent_requests_and_cancels_a_nonreading_consumer(tmp_path):
    config = configuration(tmp_path)
    with Worker(config, startup_timeout=20) as host:
        assert host.properties["pid"] == host.process.pid != os.getpid()
        blocked = host.submit((1, 2, 3), SamplingPolicy(temperature=0), 40)
        active = host.submit((1, 2, 3), SamplingPolicy(temperature=0), 12)
        output, done = collect(active)
        assert len(output) == 12 and done.reason == "length"
        assert blocked._event is None  # no unsolicited output or growing host token buffer
        blocked.cancel(timeout=5)
        repeated = host.submit((1, 2, 3), SamplingPolicy(temperature=0), 12)
        warm, done = collect(repeated)
        assert warm == output and done.cached_tokens == 2
        with pytest.raises(StopIteration):
            repeated.next(0)
    assert host.process.poll() == 0, host.stderr


def test_invalid_request_is_local_and_worker_loss_wakes_pending_output(tmp_path):
    with Worker(configuration(tmp_path), startup_timeout=20) as host:
        invalid = host.submit((99999,), SamplingPolicy(temperature=0), 2)
        with pytest.raises(WorkerUnavailable, match="vocabulary"):
            invalid.next(5)
        invalid.cancel()  # rejected requests have no worker resources left to cancel
        valid = host.submit((1,), SamplingPolicy(temperature=0), 50)
        assert isinstance(valid.next(5), Tokens)
        os.kill(host.process.pid, signal.SIGKILL)
        with pytest.raises(WorkerUnavailable):
            valid.next(5)
    assert host.process.poll() is not None


def test_host_imports_do_not_initialize_mlx():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from magnitude_engine.worker.host import Worker; "
                "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_worker_enforces_constraints_and_keeps_bad_grammar_request_local(tmp_path):
    from pathlib import Path

    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    from magnitude_engine.generation.constraint_spec import ConstraintSpec

    config = configuration(tmp_path)
    vocab = {chr(32 + i): i for i in range(127)}
    vocab["[EOS]"] = 127
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[EOS]"])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="[EOS]").save_pretrained(
        config.generation.target.program.artifact.path
    )
    (Path(config.generation.target.program.artifact.path) / "generation_config.json").write_text(
        '{"eos_token_id":127}'
    )
    with Worker(config, startup_timeout=20) as host:
        invalid = host.submit(
            (1,),
            SamplingPolicy(temperature=0),
            10,
            constraint=ConstraintSpec("start: unknown_rule"),
        )
        tokens, done = collect(invalid)
        assert not tokens and done.reason == "error" and "unknown_rule" in done.message
        for _ in range(2):
            request = host.submit(
                (1,),
                SamplingPolicy(temperature=0),
                10,
                (127,),
                constraint=ConstraintSpec('start: "hello"'),
            )
            tokens, done = collect(request)
            assert tokens == (*[vocab[c] for c in "hello"], 127)
            assert done.reason == "stop" and done.forced_tokens == 5
            assert done.accepted_tokens == done.proposed_tokens == 0
        assert host.process.poll() is None


def test_read_timeout_keeps_one_outstanding_request_instead_of_resubmitting(tmp_path, monkeypatch):
    with Worker(configuration(tmp_path), startup_timeout=20) as host:
        request = host.submit((1,), SamplingPolicy(temperature=0), 3)
        with request._condition:
            assert request._condition.wait_for(lambda: request._accepted, 5)
        sent = []
        original = host._send

        def record(message):
            sent.append(message["type"])
            original(message)

        monkeypatch.setattr(host, "_send", record)
        os.kill(host.process.pid, signal.SIGSTOP)
        try:
            for _ in range(2):
                with pytest.raises(TimeoutError):
                    request.next(0.05)
            assert sent == ["read"] and request._pending
        finally:
            os.kill(host.process.pid, signal.SIGCONT)
        assert len(collect(request)[0]) == 3


def test_failed_load_disposes_the_exact_child_before_raising(tmp_path):
    host = object.__new__(Worker)
    with pytest.raises(WorkerUnavailable):
        host.__init__(compose_engine(str(tmp_path / "missing")), startup_timeout=10)
    assert host.process.poll() is not None


def test_abrupt_parent_loss_terminates_its_model_worker(tmp_path):
    config = configuration(tmp_path)
    script = (
        "import json,sys,time; "
        "from magnitude_engine.worker.host import Worker; "
        "from magnitude_engine import blueprints as bp; "
        "h=Worker(bp.loads(sys.argv[1]), startup_timeout=10); "
        "print(h.process.pid, flush=True); time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script, bp.dumps(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child = None
    try:
        assert select.select([parent.stdout], [], [], 15)[0], (
            "parent did not publish worker readiness"
        )
        line = parent.stdout.readline()
        assert line, parent.stderr.read().decode()
        child = psutil.Process(int(line))
        assert child.ppid() == parent.pid and child.is_running()
        parent.kill()
        parent.wait(timeout=3)
        gone, alive = psutil.wait_procs([child], timeout=5)
        assert gone and not alive, "worker survived abrupt parent loss"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if child is not None and child.is_running():
            child.kill()
        parent.stdout.close()
        parent.stderr.close()


def test_private_worker_progress_is_opt_in_and_preserves_warm_request_tokens(tmp_path):
    from dataclasses import replace

    from magnitude_engine.engine.delivery import PrefillProgress

    config = configuration(tmp_path)
    config = replace(config, scheduler=replace(config.scheduler, prefill_tokens=1))
    with Worker(config, startup_timeout=20) as host:
        outputs = []
        for cached in (0, 7):
            request = host.submit(tuple(range(1, 9)), SamplingPolicy(temperature=0), 4,
                                  progress=True)
            progress, values = [], []
            while True:
                event = request.next(5)
                if isinstance(event, PrefillProgress):
                    assert not values
                    progress.append(event)
                elif isinstance(event, Finished):
                    assert event.cached_tokens == cached
                    break
                else:
                    values.extend(event.values)
            assert progress and progress[-1].completed_tokens == 7
            assert all(p.total_tokens == 7 and p.cached_tokens == cached for p in progress)
            assert [p.completed_tokens for p in progress] == sorted(
                p.completed_tokens for p in progress
            )
            assert len(values) == 4
            outputs.append(values)
        assert outputs[0] == outputs[1]
    assert host.process.poll() == 0, host.stderr
