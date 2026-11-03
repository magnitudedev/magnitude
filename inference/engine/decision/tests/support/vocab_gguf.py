"""Write a vocabulary-only GGUF (tokenizer keys only) from a Hugging Face tokenizer directory, with
llama.cpp's convert_hf_to_gguf token types: added tokens that are special or look special CONTROL, other
added tokens USER_DEFINED, padding ids UNUSED. The parity tests read these (`tests/parity.rs`): `d3u`,
`mam` and `pplx` with pre `qwen35` and 248320 ids, `d1` with pre `lfm2` and 128000.

    PYTHONPATH=<llama.cpp>/gguf-py python vocab_gguf.py <tokenizer dir> <pre> <vocab size> <out.gguf>
"""
import json, sys
import gguf
from transformers import AutoTokenizer

src, pre, size, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False)
data = json.load(open(f"{src}/tokenizer.json"))
reverse = {i: t for t, i in tok.vocab.items()}
added = tok.get_added_vocab()
decoder = tok.added_tokens_decoder
def looks_special(t):
    return t in ("<pad>", "<mask>", "<2mass>", "[@BOS@]") or (t.startswith("<|") and t.endswith("|>")) or (t.startswith("<unused") and t.endswith(">"))
tokens, types = [], []
for i in range(size):
    if i not in reverse:
        tokens.append(f"[PAD{i}]"); types.append(gguf.TokenType.UNUSED); continue
    t = reverse[i]
    if t in added:
        types.append(gguf.TokenType.CONTROL if decoder[i].special or looks_special(t) else gguf.TokenType.USER_DEFINED)
    else:
        types.append(gguf.TokenType.NORMAL)
    tokens.append(t)
merges = data["model"]["merges"]
merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
w = gguf.GGUFWriter(out, "vocab")
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre(pre)
w.add_token_list(tokens)
w.add_token_types(types)
w.add_token_merges(merges)
cfg = json.load(open(f"{src}/tokenizer_config.json"))
if tok.bos_token_id is not None: w.add_bos_token_id(tok.bos_token_id)
if tok.eos_token_id is not None: w.add_eos_token_id(tok.eos_token_id)
if tok.pad_token_id is not None: w.add_pad_token_id(tok.pad_token_id)
w.add_add_bos_token(bool(cfg.get("add_bos_token", False)))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(out, len(tokens), sum(t == gguf.TokenType.CONTROL for t in types), "control")
