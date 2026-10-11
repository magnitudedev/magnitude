"""Performance options: how a reference is evaluated, never what it computes. They are outside the
implementation digest (`digest.NUMERICAL`), so tuning them never invalidates a cached reference.

`BATCH_TOKENS`: padded tokens per streamed forward. Every batch dequantizes the whole model once, so
larger batches amortize it, but activations grow with them (65536 drove a 48 GB Apple host into
swap at 16k context); one 16k-context fixture per batch by default, `--batch-tokens` per run.
`HEAD_ROWS`: readout rows per head matmul ([HEAD_ROWS, vocabulary] float32 at a time).
"""
BATCH_TOKENS = 16384
HEAD_ROWS = 256
