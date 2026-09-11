"""Physical layout of logical projection segments.

Each segment is a row-major matrix.  A grouped projection concatenates those
matrices rather than interleaving their columns, so consumers can view every
logical result without copying it.
"""

import tilelang.language as T

from magnitude_engine.weights.representation import WeightLayout


def widths_and_outputs(widths: int | tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    logical = (widths,) if isinstance(widths, int) else widths
    if not logical or any(width <= 0 for width in logical):
        raise ValueError("projection widths must be positive")
    return logical, sum(logical)


def output_index(rows: int, widths: tuple[int, ...]):
    """Map a logical ``(row, combined-column)`` to grouped flat storage."""

    def index(row, col):
        result = row * widths[0] + col
        start = widths[0]
        for width in widths[1:]:
            result = T.if_then_else(col >= start, rows * start + row * width + col - start, result)
            start += width
        return result

    return index


def weight_index(layout: WeightLayout):
    """Map a logical matrix coordinate to its row in the shared backing."""

    def index(row, column):
        return (layout.first_row + row) * layout.columns + column

    return index
