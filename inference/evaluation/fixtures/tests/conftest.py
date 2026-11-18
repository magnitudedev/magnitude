import pytest
from magnitude_fixtures.bfcl import materialize
from magnitude_fixtures.bigcodebench import Task


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
def tasks():
    return [
        Task(
            task_id=f"BigCodeBench/{i}",
            instruct_prompt=f"Return {i}.\nYou should write self-contained code starting with:\n```\ndef task_func():\n```",
            code_prompt="def task_func():\n",
            canonical_solution=f"    return {i}\n",
        )
        for i in range(4)
    ]
