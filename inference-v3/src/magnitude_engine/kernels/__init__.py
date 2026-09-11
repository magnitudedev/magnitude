"""Portable TileLang schedules, named by strategy and gated by capability.

Nothing here names a backend or a container. A factory takes its shape extents,
then a ``Precision`` if it rounds, a ``Capability`` if it launches a grid,
reduces across lanes or sizes shared memory, and a ``Representation`` if it
reads weights. Which factory runs is decided by a candidate table, never here.
"""
