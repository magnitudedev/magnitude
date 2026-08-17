from magnitude_mlx_benchmark_server.protocol import terminal_chunk


def test_terminal_evidence_is_internally_consistent():
    result = terminal_chunk(
        "request",
        "model",
        0,
        prompt_tokens=20,
        cached_tokens=8,
        completion_tokens=6,
        prompt_ms=10,
        generation_ms=20,
        peak_memory_bytes=123,
    )
    assert result["usage"]["total_tokens"] == 26
    assert result["timings"]["prompt_n"] == 12
    assert result["timings"]["predicted_n"] == 6
    assert result["mlx"]["peak_memory_bytes"] == 123
