"""Portable row movement around bounded streamed expert computation."""

from dataclasses import dataclass

import tilelang.language as T


@T.macro
def _gather(hidden, routes, gathered, step, width, selected, threads):
    with T.Kernel(T.ceildiv(step * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            index = block * threads + lane
            if index < step * width:
                row, column = index // width, index % width
                route = routes[row]
                gathered[row, column] = T.if_then_else(route >= 0, hidden[T.max(route, 0) // selected, column], 0)


@T.macro
def _scatter(projected, routes, contributions, step, width, threads):
    with T.Kernel(T.ceildiv(step * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            index = block * threads + lane
            if index < step * width:
                row, column = index // width, index % width
                route = routes[row]
                if route >= 0:
                    contributions[route, column] = projected[row, column]


@T.macro
def _combine(contributions, scores, output, rows, selected, width, threads):
    with T.Kernel(T.ceildiv(rows * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            index = block * threads + lane
            if index < rows * width:
                row, column = index // width, index % width
                total = T.alloc_local((1,), "float32")
                total[0] = 0
                for rank in T.serial(selected):
                    total[0] += T.cast(contributions[row * selected + rank, column], "float32") * scores[row, rank]
                output[row, column] = T.cast(total[0], output.dtype)


@dataclass(frozen=True, slots=True)
class GatherExpertRows:
    step: int
    width: int
    selected: int
    threads: int

    def __call__(self, operands):
        _gather(*operands, self.step, self.width, self.selected, self.threads)


@dataclass(frozen=True, slots=True)
class ScatterExpertRows:
    step: int
    width: int
    threads: int

    def __call__(self, operands):
        _scatter(*operands, self.step, self.width, self.threads)


@dataclass(frozen=True, slots=True)
class CombineExpertRows:
    rows: int
    selected: int
    width: int
    threads: int

    def __call__(self, operands):
        _combine(*operands, self.rows, self.selected, self.width, self.threads)
