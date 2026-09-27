"""Resource demands shared by formula accounting and device characterization."""

from enum import StrEnum


class Resource(StrEnum):
    EXECUTION_COPY = "execution-copy"
    SOURCE_IMPORT = "source-import"
    MATRIX_ARITHMETIC = "matrix-arithmetic"
    VECTOR_ARITHMETIC = "vector-arithmetic"
    INTEGER_ARITHMETIC = "integer-arithmetic"
    SPECIAL_FUNCTIONS = "special-functions"
    COMPARISONS = "comparisons"
