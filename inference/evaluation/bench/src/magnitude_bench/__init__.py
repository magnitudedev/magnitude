"""magnitude-bench: the serving benchmark (`cli`).

Its tests run in the evaluation workspace's one environment, whose dev group owns pytest and
pytest-asyncio for every member (and whose `asyncio_mode` runs the async tests):

    uv run --project inference/evaluation pytest inference/evaluation/bench/tests
"""
