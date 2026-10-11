import pytest
from magnitude_fixtures.code import File
from magnitude_fixtures.tools import materialize


def bfcl(index: int, name: str, value: int) -> tuple[dict, dict]:
    question = {
        "id": f"simple_python_{index}",
        "question": [[{"role": "user", "content": f"Call {name} with value {value}."}]],
        "function": [
            {
                "name": name,
                "description": f"Echo an integer ({index})",
                "parameters": {
                    "type": "dict",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            }
        ],
    }
    answer = {"id": question["id"], "ground_truth": [{name: {"value": [value]}}]}
    return question, answer


@pytest.fixture
def interactions():
    return [materialize(*bfcl(i, f"echo{i % 3}", i), "simple-python") for i in range(6)]


@pytest.fixture
def book():
    front = " ".join(f"front{i}" for i in range(300))
    paragraphs = [
        " ".join(f"“paragraph{paragraph}word{word}’s”" for word in range(80))
        for paragraph in range(200)
    ]
    return f"{front}\n\nCHAPTER 1. Loomings.\n\nCall me Ishmael. " + "\n\n".join(paragraphs)


@pytest.fixture
def files():
    return [
        File(path="src/a.rs", content="fn a() {}\n"),
        File(path="src/b.ts", content="export const b = 1;\n"),
        File(path="src/c.py", content="def c():\n    return 3\n"),
    ]
