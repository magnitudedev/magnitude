import pytest
from magnitude_fixtures import moby
from magnitude_fixtures.moby import (
    REPEAT_COPY_WORDS,
    REPEAT_PASSAGE_WORDS,
    TWIST_COPY_WORDS,
    Book,
)


def test_book_starts_at_opening_with_straight_quotes_and_paragraphs(book):
    body = Book(book).body
    assert body.startswith('Call me Ishmael. "paragraph0word0\'s" ')
    assert not any(quote in body for quote in "‘’“”")
    assert "\n\n" in body and "front0" not in body


def test_lanes_probe_windows_back_from_the_end(book):
    first, second = moby.build(book, "repeat", lanes=2)
    assert len(first.units()) == len(second.units()) + 1
    assert first.messages[: len(second.messages)] != second.messages  # distinct system message
    assert first.messages[1 : len(second.messages)] == [
        {key: value for key, value in message.items() if key != "scored"}
        for message in second.messages[1:]
    ]


def test_a_lane_does_not_depend_on_how_many_lanes_are_built(book):
    assert moby.build(book, "repeat", lanes=1)[0] == moby.build(book, "repeat", lanes=3)[0]


def test_repeat_answer_is_the_latest_sufficient_paragraph_suffix(book):
    for session in moby.build(book, "repeat", lanes=3):
        for start, end in session.units():
            user, answer = session.messages[start:end]
            supplied = user["content"].removeprefix("Supplied text:\n\n").split("\n\n")
            instruction = supplied.pop()
            paragraphs = ["\n\n".join(supplied[i:]) for i in range(1, len(supplied))]
            qualifying = [p for p in paragraphs if len(p.split()) >= REPEAT_COPY_WORDS]
            assert answer["content"] == (qualifying[-1] if qualifying else "\n\n".join(supplied))
            assert len(answer["content"].split()) >= REPEAT_COPY_WORDS
            assert " ".join(answer["content"].split()[:6]) in instruction
            assert instruction.endswith("Output only the passage text.")
        assert session.generation.expected == {"text": session.messages[-1]["content"]}


def test_twist_probe_asks_for_the_passage_end_twice_around_new_writing(book):
    shift, repeat = (moby.build(book, kind, lanes=2)[0] for kind in ("twist", "repeat"))
    # The history is a repeat session's; only the probe differs.
    assert shift.messages[1:-2] == repeat.messages[1:-2]
    user, answer = shift.messages[-2:]
    supplied = user["content"].removeprefix("Supplied text:\n\n")
    passage, _, instruction = supplied.rpartition("\n\n")
    assert passage.endswith(answer["content"])
    assert len(answer["content"].split()) == TWIST_COPY_WORDS
    assert instruction.count(" ".join(answer["content"].split()[:6])) == 2
    assert shift.generation.expected == {"text": answer["content"]}
    assert shift.generation.max_tokens == 768


def test_repeat_single_paragraph_copies_whole_passage():
    text = "Call me Ishmael. " + " ".join(f"word{i}" for i in range(4 * REPEAT_PASSAGE_WORDS))
    session = moby.build(text, "repeat", lanes=1)[0]
    user, answer = session.messages[-2:]
    assert user["content"].startswith("Supplied text:\n\n" + answer["content"] + "\n\n")


def test_exhaustion_fails_clearly():
    text = "Call me Ishmael. " + " ".join(f"word{i}" for i in range(REPEAT_PASSAGE_WORDS))
    with pytest.raises(ValueError, match="lanes"):
        moby.build(text, "repeat", lanes=1)


def test_vendored_book_matches_its_lock_without_network():
    text, source = moby.load()
    assert "PROJECT GUTENBERG EBOOK" not in text
    assert Book(text).body.startswith("Call me Ishmael.")
    assert source.name == "moby-dick"
