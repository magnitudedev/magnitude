"""Run an ordinary model request through the public import surface."""

import argparse

from magnitude import ModelRequest, SpecialTokens, load_model

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Local supported Qwen3.5 GGUF file or MLX directory")
parser.add_argument("--memory-bytes", type=int, required=True)
args = parser.parse_args()

with load_model(args.model, memory_bytes=args.memory_bytes) as loaded:
    tokens = loaded.tokenizer.encode("The capital of France is", special=SpecialTokens.LITERAL)
    source = loaded.input(tokens)
    sequence = source.open()
    try:
        batch = loaded.executor.prepare((ModelRequest(sequence, tokens),))
        try:
            logits = batch.advances[0].read_logits()[0]
            token = max(range(len(logits)), key=logits.__getitem__)
            print(loaded.tokenizer.decode((token,)))
            batch.advances[0].commit()
        finally:
            batch.close()
    finally:
        sequence.close()
        source.close()
