from src.utils import chunk_list, flatten


def test_chunk_list_basic():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_list_exact():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_list_single():
    assert chunk_list([1, 2, 3], 1) == [[1], [2], [3]]


def test_chunk_list_larger_than_list():
    assert chunk_list([1, 2], 5) == [[1, 2]]


def test_flatten():
    assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]


def test_flatten_empty():
    assert flatten([]) == []
