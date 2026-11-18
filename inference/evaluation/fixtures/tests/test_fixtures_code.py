import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from magnitude_fixtures import Template, bigcodebench
from magnitude_fixtures.bigcodebench import CorpusLock, Task
from magnitude_fixtures.sources import DATA


def test_tasks_are_answered_in_fenced_blocks_with_the_probe_last(tasks):
    sessions = bigcodebench.build(tasks, lanes=3)
    for lane, session in enumerate(sessions):
        assert session.key == "code"
        probe_idx = len(tasks) - 1 - lane
        asked = [session.messages[start]["content"] for start, _ in session.units()]
        order = tasks[:probe_idx] + [tasks[probe_idx]]
        assert asked == [task.instruct_prompt.strip() for task in order]
        assert [m for m in session.messages if m.get("scored")] == [session.messages[-1]]
        assert session.generation.prefix == len(session.messages) - 1
    expected_probe = tasks[-1]
    assert (
        sessions[0].messages[-1]["content"]
        == f"```python\n{expected_probe.code_prompt}{expected_probe.canonical_solution.rstrip()}\n```"
    )
    with pytest.raises(ValueError, match="lanes"):
        bigcodebench.build(tasks, lanes=len(tasks))


def test_lock_is_valid_and_cached_parquet_loads_without_network(tmp_path, monkeypatch, tasks):
    lock = CorpusLock.model_validate_json((DATA / "bigcodebench.lock.json").read_text())
    assert lock.license == "Apache-2.0"

    # Write a small synthetic parquet and point a temporary lock at it to verify cache + hash check.
    table = pa.table(
        {
            "task_id": [t.task_id for t in tasks],
            "instruct_prompt": [t.instruct_prompt for t in tasks],
            "code_prompt": [t.code_prompt for t in tasks],
            "canonical_solution": [t.canonical_solution for t in tasks],
        }
    )
    cached = tmp_path / "cache" / "bigcodebench" / lock.commit / lock.file
    cached.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cached)
    sha = hashlib.sha256(cached.read_bytes()).hexdigest()

    fake_data = tmp_path / "data"
    fake_data.mkdir()
    (fake_data / "bigcodebench.lock.json").write_text(
        json.dumps({**lock.model_dump(), "sha256": sha})
    )
    monkeypatch.setattr(bigcodebench, "DATA", fake_data)

    loaded, source = bigcodebench.load(cache=tmp_path / "cache")
    assert loaded == tasks
    assert source.name == "bigcodebench"


def test_a_code_template_names_its_context():
    template = Template.parse("code:8k")
    assert (template.key, template.context, template.name) == ("code", 8192, "code:8k")
