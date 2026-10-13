import pytest
from magnitude_fixtures import Source, prose
from magnitude_fixtures.prose import (
    CONTINUATION_WORDS,
    CONTINUE_PASSAGE_WORDS,
    REPEAT_COPY_WORDS,
    REPEAT_PASSAGE_WORDS,
    SHIFT_COPY_WORDS,
    Book,
)

SOURCE = Source(name="test", digest="0" * 64)


def test_book_starts_at_opening_with_straight_quotes_and_paragraphs(book):
    body = Book(book).body
    assert body.startswith('Call me Ishmael. "paragraph0word0\'s" ')
    assert not any(quote in body for quote in "‘’“”")
    assert "\n\n" in body and "front0" not in body


def test_continue_answers_with_the_next_book_words(book):
    fixture = prose.build(book, SOURCE, "prose-continue", count=2)[0]
    text = Book(book)
    window = CONTINUE_PASSAGE_WORDS + CONTINUATION_WORDS
    for index, (start, end) in enumerate(fixture.units()):
        user, answer = fixture.messages[start:end]
        split = index * window + CONTINUE_PASSAGE_WORDS
        assert user["content"].endswith(text.passage(index * window, split))
        assert answer["content"] == text.passage(split, split + CONTINUATION_WORDS)
    assert fixture.generation.expected == {}
    assert fixture.generation.max_tokens == 256


def test_lanes_probe_consecutive_late_windows(book):
    first, second = prose.build(book, SOURCE, "prose-repeat", count=2)
    assert len(second.units()) == len(first.units()) + 1
    assert second.messages[: len(first.messages)] != first.messages  # distinct system message
    assert second.messages[1 : len(first.messages)] == [
        {key: value for key, value in message.items() if key != "scored"}
        for message in first.messages[1:]
    ]


def test_repeat_answer_is_the_latest_sufficient_paragraph_suffix(book):
    for fixture in prose.build(book, SOURCE, "prose-repeat", count=3):
        for start, end in fixture.units():
            user, answer = fixture.messages[start:end]
            supplied = user["content"].removeprefix("Supplied text:\n\n").split("\n\n")
            instruction = supplied.pop()
            paragraphs = ["\n\n".join(supplied[i:]) for i in range(1, len(supplied))]
            qualifying = [p for p in paragraphs if len(p.split()) >= REPEAT_COPY_WORDS]
            assert answer["content"] == (qualifying[-1] if qualifying else "\n\n".join(supplied))
            assert len(answer["content"].split()) >= REPEAT_COPY_WORDS
            assert " ".join(answer["content"].split()[:6]) in instruction
            assert instruction.endswith("Output only the passage text.")
        assert fixture.generation.expected == {"text": fixture.messages[-1]["content"]}


def test_shift_probe_asks_for_the_passage_end_twice_around_new_writing(book):
    shift, repeat = (prose.build(book, SOURCE, family, count=2)[0] for family in ("prose-shift", "prose-repeat"))
    # The history is a repeat session's; only the probe differs.
    assert shift.messages[1:-2] == repeat.messages[1:-2]
    user, answer = shift.messages[-2:]
    supplied = user["content"].removeprefix("Supplied text:\n\n")
    passage, _, instruction = supplied.rpartition("\n\n")
    assert passage.endswith(answer["content"])
    assert len(answer["content"].split()) == SHIFT_COPY_WORDS
    assert instruction.count(" ".join(answer["content"].split()[:6])) == 2
    assert shift.generation.expected == {"text": answer["content"]}
    assert shift.generation.max_tokens == 768


def test_repeat_single_paragraph_copies_whole_passage():
    text = "Call me Ishmael. " + " ".join(f"word{i}" for i in range(4 * REPEAT_PASSAGE_WORDS))
    fixture = prose.build(text, SOURCE, "prose-repeat", count=1)[0]
    user, answer = fixture.messages[-2:]
    assert user["content"].startswith("Supplied text:\n\n" + answer["content"] + "\n\n")


def test_exhaustion_fails_clearly():
    text = "Call me Ishmael. " + " ".join(f"word{i}" for i in range(REPEAT_PASSAGE_WORDS))
    with pytest.raises(ValueError, match="probes"):
        prose.build(text, SOURCE, "prose-repeat", count=1)


def test_vendored_book_matches_its_lock_without_network():
    text, source = prose.load()
    assert "PROJECT GUTENBERG EBOOK" not in text
    assert Book(text).body.startswith("Call me Ishmael.")
    assert source.name == "moby-dick"
