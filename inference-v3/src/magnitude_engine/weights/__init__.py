"""The weight-format axis: stored layouts, resident representations, residency.

``formats/`` reads containers. ``representation.py`` says what a kernel reads.
``binding.py`` is the one place that decides which representation a stored
weight becomes. ``residency.py`` owns the resident weights of one device context.
"""
