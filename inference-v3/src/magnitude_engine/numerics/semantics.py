"""Numerical meanings shared by operation contracts and their compiled schedules."""

from enum import StrEnum


class Pointwise(StrEnum):
    ADD = "add"
    MULTIPLY = "multiply"
    SILU_PRODUCT = "silu_product"
    SIGMOID_PRODUCT = "sigmoid_product"


class HeadMapping(StrEnum):
    GROUPED = "grouped"
    TILED = "tiled"
