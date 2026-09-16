"""Host elapsed-time observations, separate from worker numerical service time."""

from engine.data import Record


class PreparationMetrics(Record):
    template_create_ns: int = 0
    effort_probe_ns: int = 0
    profile_cache_hit: bool = False
    render_ns: int = 0
    tokenize_ns: int = 0


class ParsingMetrics(Record):
    elapsed_ns: int = 0
    input_bytes: int = 0
    calls: int = 0
