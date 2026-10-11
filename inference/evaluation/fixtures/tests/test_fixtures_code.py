from pathlib import Path

from magnitude_fixtures import Source, code

SOURCE = Source(name="test", digest="0" * 64)


def test_files_are_answered_in_fenced_blocks_with_the_probe_last(files):
    fixtures = code.build(files, SOURCE, count=3)
    assert [f.id for f in fixtures] == [f"code/magnitude/{f.path}" for f in files]
    for lane, fixture in enumerate(fixtures):
        assert fixture.generation is None
        asked = [fixture.messages[start]["content"] for start, _ in fixture.units()]
        order = files[lane + 1 :] + files[:lane] + [files[lane]]
        assert asked == [f"Show me the contents of `{file.path}`." for file in order]
        assert [m for m in fixture.messages if m.get("scored")] == [fixture.messages[-1]]
    assert fixtures[0].messages[-1]["content"] == "```rust\nfn a() {}\n```"


def test_pinned_files_load_from_the_checkout_and_match_the_lock():
    files, source = code.load(checkout=Path(__file__).parent)
    assert len(files) == len({file.path for file in files}) >= 2
    assert source.name == "magnitude-code"
    assert all(file.content for file in files)
