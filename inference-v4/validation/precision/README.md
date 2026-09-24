# D4 precision harness: reference side

Spec: `specs/26-09-23/seismic-native-kernel-program.md` D4 (§1) and Q1 (§10 M0).
The fixed prompt set is 512 chunks × 64 evaluated positions: 171 prose, 171 code,
170 tool JSON. The reference is llama.cpp's CPU forward, captured as
`llama-perplexity --kl-divergence-base` files. **Use the F32-dequantized GGUF**
(`reference.py f32-gguf`, then `base --model $G_F32 --source-model $G`): the
CPU forward on the Q4_K_M file itself quantizes activations to Q8_K and is
~13× further (in mean KL) from the F32 forward than llama.cpp Metal is (see "Local
results"). The llama.cpp spread (Metal, CUDA) and every V4 candidate are
compared against these files with llama.cpp's own KL and "same top"
definitions.

| File | Purpose |
|---|---|
| `build_corpus.py` | Writes `results/precision/corpus/{prose,code,tool_json}.txt` and `manifest.json` (sha256s, sources). Deterministic, offline. |
| `reference.py base` | CPU reference base file per category (`results/precision/<label>/`). `--categories code,tool_json` (re)produces only those; the label's existing `reference.json` entries for the other categories are kept when model, corpus and llama.cpp build match. |
| `reference.py f32-gguf` | Dequantizes the pinned GGUF to an all-F32 GGUF, the true F32 reference model. |
| `reference.py spread` | Runs `--kl-divergence` on a GPU backend against a reference, parses its summary to `spread.json`; `--save-base` also writes that backend's own base files and cross-checks with `kl_base.py`. |
| `spread_host.sh` | `spread --save-base` one category at a time; with `GPU_LOCK` set, each category runs under that shared lock directory. |
| `kl_base.py` | NumPy reader/writer for the base format; computes mean KL, same-top, PPL between two base files or a base file and a raw-logits `.npy`; `self-test`. |

All outputs are under the git-ignored `inference-v4/validation/results/precision/`.

## Chunking and positions

`llama-perplexity` tokenizes the whole corpus file once
(`common_tokenize(ctx, text, add_special=true)`), then slices the token stream
into consecutive, non-overlapping chunks of `n_ctx` tokens. Chunk `c` is
`tokens[c*n_ctx : (c+1)*n_ctx]`; `n_chunk = min(--chunks, len(tokens) / n_ctx)`
(it silently truncates, so `reference.py` checks the header's `n_chunk`). Each
chunk is evaluated from an empty KV/recurrent state at positions `0..n_ctx-1`.

With `first = n_ctx / 2` (integer division), the stored rows are the logits at
positions `first .. n_ctx - 2`: `n_eval = n_ctx - 1 - first` rows. Row `r` is the
distribution at position `first + r`, scored against the next token
`tokens[c*n_ctx + first + r + 1]`. The last position (`n_ctx - 1`) is computed but
not stored (it has no next token inside the chunk).

**`n_ctx = 130`**: `first = 65`, `n_eval = 130 - 1 - 65 = 64`. Each chunk is 65
context tokens (positions 0..64, prefill) followed by 64 scored positions
(65..128) whose inputs are the corpus tokens (teacher forcing), plus one unscored
trailing token. A V4 runner may produce the 64 rows by prefill or by decoding
tokens 65..128 one at a time; the inputs are identical.

**BOS**: llama.cpp overwrites (does not insert) token 0 of every chunk with BOS
only when the vocabulary's `add_bos` is true. For Qwen3.5 (`tokenizer.ggml.model =
gpt2`, `pre = qwen35`, no `add_bos_token` key) `add_bos` is false: no BOS is used,
the file's token 0 is the first corpus token (verified: the stored token array
equals `llama-tokenize --ids` of the corpus). A runner must feed exactly the
stored tokens. (For a model with `add_bos`, the file still stores the original
token at position 0 and the runner must substitute BOS there.)

**Parallel sequences**: with the default `-b 2048`, `llama-perplexity` evaluates
`n_seq = n_batch / n_ctx = 15` chunks per `llama_decode` as separate sequences
(`params.n_ctx` becomes `15 * n_ctx`). Chunks are still independent; this only
affects batching.

## Base file format (`--kl-divergence-base`)

Written by `perplexity()` in `tools/perplexity/perplexity.cpp` when
`--kl-divergence-base` is given without `--kl-divergence`. Identical in b8680
(Homebrew, commit 15f786e65), b9994, b10809 (Metal build) and b10998
(CUDA build); the only changes between b8680 and later are `size_t` casts in
offset arithmetic (a >2 GiB overflow fix for huge `n_ctx * nv`, not a format
change). All integers and floats are little-endian; there is no padding
or alignment between sections.

| Offset (bytes) | Type | Field |
|---|---|---|
| 0 | `char[8]` | magic `_logits_` (no NUL) |
| 8 | `i32` (read back as `u32`) | `n_ctx` |
| 12 | `i32` | `n_vocab` (248320 for Qwen3.5) |
| 16 | `i32` | `n_chunk` |
| 20 | `i32[n_chunk * n_ctx]` | all chunk tokens, chunk-major (token ids as fed, BOS not substituted) |
| `R = 20 + 4*n_chunk*n_ctx` | `u16[n_chunk][n_eval][nv]` | rows |

`nv = 2*((n_vocab + 1)/2) + 4` (n_vocab rounded up to even, plus 4). One row is
`2*nv` bytes, row `(c, r)` starts at `R + 2*nv*(c*n_eval + r)`:

| Row offset | Type | Field |
|---|---|---|
| 0 | `f32` | `scale` |
| 4 | `f32` | `min_log_prob` |
| 8 | `u16[n_vocab]` | `q[i]` |
| 8 + 2*n_vocab | `u16` | padding (only when `n_vocab` is odd; value 0) |

File size = `20 + 4*n_chunk*n_ctx + 2*nv*n_eval*n_chunk`. The two f32 fields sit
at byte offsets that are multiples of 4 relative to the file start only because
`20 + 4k` and `2*nv` (nv even, `2*nv ≡ 0 mod 4`) are; read them with unaligned
loads anyway.

Encoding of one row from raw float32 logits `l[0..n_vocab)` (all arithmetic
float32 except the double-precision exp sum):

```
max_logit    = max_i l[i]
min_logit    = max(min_i l[i], max_logit - 16)
sum_exp      = Σ_i (double) expf(l[i] - max_logit)
log_sum_exp  = (float) log(sum_exp)
min_log_prob = min_logit - max_logit - log_sum_exp      // stored
scale        = (max_logit - min_logit) / 65535.f        // stored
q[i]         = l[i] > min_logit ? nearest_int((1/scale) * (l[i] - min_logit)) : 0
               // nearest_int: round-to-nearest-even via the 1.5*2^23 trick
               // if scale == 0, all q[i] = 0
```

Decoding: `log_p[i] = scale * q[i] + min_log_prob` (float32). This is the
log-softmax, quantized with step `scale ≤ 16/65535 ≈ 2.44e-4` nats (error
≤ `scale/2`), floored at `max_log_p - 16` (every token more than 16 nats below
the top token decodes to the floor value). The top token decodes to
`-log_sum_exp`.

For Qwen3.5-4B: `nv = 248324`, row = 496,648 B, one chunk (64 rows) =
31,785,472 B. With `n_ctx = 130`: 171 chunks = 5,435,404,652 B, 170 chunks =
5,403,618,660 B, the full 512-chunk set = 16.27 GB (sum of the three files:
16,274,427,964 B).

## Metrics (`--kl-divergence`), as `kl_base.py` computes them

`--kl-divergence --kl-divergence-base F` reads `n_ctx`, `n_vocab`, `n_chunk` and
the tokens from `F` (no corpus file is needed, `-c` must be ≥ the file's
`n_ctx`), re-evaluates every chunk with the candidate backend, and for each
stored row, with candidate float32 logits `l` and decoded base `b`:

- candidate `log_softmax_i = l[i] - max(l) - log Σ expf(l - max(l))`;
- **KL(base ‖ candidate)** `= Σ_{i : b[i] > -16} exp(b[i]) * (b[i] - log_softmax_i)`
  (terms with base log-prob ≤ -16 are skipped, per-position sum in double);
- **same top**: `argmax(l) == argmax(b)`, each the *first* index attaining the
  maximum (strict `>` scan). Ties in the quantized base resolve to the lower id;
- `NLL = -log_softmax[target]`, `NLL_base = -b[target]` (floored as above);
  `Δp = exp(-NLL) - exp(-NLL_base)`.

Summary (printed only when at least 100 positions were evaluated):
`Mean KLD` = mean over all positions (± `sqrt(var/(n-1))`), KLD percentiles and
maximum, `Mean PPL(Q)`/`PPL(base)` = exp(mean NLL), `RMS Δp`, and
`Same top p` = `n_same_top / count` in percent with 3 decimals
(± `sqrt(p(1-p)/(n-1))`). The per-chunk table prints running values with the
same formulas. D4's "top-1 agreement" is `same top p` and "mean KL" is
`Mean KLD`, both over all 32,768 positions (512 × 64).

A V4 candidate supplies float32 logits `[n_chunk, 64, n_vocab]` per category
(`.npy`), or writes its own base file with the encoding above (then the
candidate side is quantized too: argmax ties and ≤ 1.2e-4 nats of error).
`kl_base.py compare BASE CAND` reports `mean_kld`, `same_top`, `ppl`,
`ppl_base`, percentiles and Δp. It (and `forward_bench qualify`) also reports a
diagnostic that D4 does not gate: `same_top_above_margin`, same-top over only the
positions whose reference top-two log-probability gap exceeds 0.01/0.05/0.1/0.2/0.5
nats, and `flip_gap_percentiles`, the reference gaps at the positions whose top
token flipped. It separates near-tie flips from real ranking errors.

## Reference compute path

`reference.py base` runs, per category:

```
llama-perplexity -m $G_F32 -f corpus/<cat>.txt -c 130 --chunks <N> \
  --kl-divergence-base <label>/<cat>.bin -ngl 0 -dev none -fa off -ctk f32 -ctv f32
```

- `-dev none` keeps the model off every GPU device. `-ngl 0` alone is not
  enough on a GPU build: the Metal and CUDA devices implement `offload_op`, so
  the scheduler still offloads MUL_MAT/MUL_MAT_ID with a large batch to the
  GPU (op offload). The log must show only `CPU_Mapped`/`CPU_REPACK` model
  buffers and a `CPU compute buffer` (no `MTL0`/`CUDA0`). The Homebrew build
  also creates an Accelerate `BLAS` backend (ACCEL devices are always added);
  it takes MUL_MAT when the weight is in a plain host buffer and `ne0`, `ne1`,
  `ne10` are all ≥ 32 (dequantizing non-F32 weights to F32 for sgemm). With
  the Q4_K_M file the quantized weights are repacked into `CPU_REPACK`, so BLAS
  takes no node (`graph splits = 1`). With the F32 file there is no repack and
  BLAS (Accelerate sgemm) runs the batched matmuls (`graph splits = 434 (with
  bs=512)`); both BLAS and the CPU F32 path are F32 × F32. Linux CPU builds
  without BLAS do the same products in ggml's F32 kernels.
- Attention is F32 end to end: `-fa off -ctk f32 -ctv f32` (both defaults in
  `reference.py`). Flash attention must be off: with FA on, `llama-graph.cpp`
  casts F32 K and V to F16 before `ggml_flash_attn_ext` (so `-ctk f32` is a
  no-op; measured: FA-on f16-KV and f32-KV base files are bit-identical), the
  mask is cast to F16, Q is converted to F16 for the K dot product, and the CPU
  kernel accumulates softmax·V in an **F16** accumulator (`VKQ16`) when V is
  F16. With FA off, KQ and KQV are F32 `mul_mat`s over the F32 cache and the
  softmax is F32. Measured on 4 code chunks (Q4_K_M file): FA-off vs FA-on
  CPU mean KL 0.0017, same top 93.8 % (the positions that flip are near-ties,
  top-2 probabilities within a few percent). Recurrent
  (Gated DeltaNet) state and its update are F32 in llama.cpp regardless.
- **What "F32" means here with the Q4_K_M file**: the CPU backend does not
  dequantize weights to F32. `ggml_mul_mat` quantizes the activation operand to
  the weight type's `vec_dot_type` and runs an integer dot product with F32
  scale accumulation: Q4_K/Q5_K/Q6_K weights → activations quantized to
  **Q8_K** (per 256-block f32 scale, int8 values); Q8_0 weights → activations
  **Q8_0**; F32 weights (norms, small tensors) → F32. The Homebrew build repacks
  Q4_K (and other supported types) into interleaved `CPU_REPACK` layouts using
  the i8mm/dotprod kernels; activations are still Q8_K. So the Q4_K_M
  reference is "F32 accumulate, int8 activations", like the CPU backend of
  every llama.cpp build.
- **True F32 reference**: `reference.py f32-gguf --model $G --output $G_F32`
  runs `llama-quantize --allow-requantize $G $G_F32 F32`, dequantizing every
  tensor to F32 once (exactly the values the Q4_K_M codes represent). On that
  file every CPU matmul is F32 × F32 (`vec_dot_type` F32), so the forward has
  no activation quantization. Size ≈ 4.21 B params × 4 B = **16.8 GB** (plus
  metadata). Run `reference.py base --model $G_F32 --source-model $G ...`; the
  record keeps both hashes, and `spread`/candidates still use `$G` (the spread
  check accepts the source model's hash).

The CPU reference is backend-independent in the sense that matters (no GPU
kernels), but not bitwise identical across CPU ISAs/builds (repack kernels,
threading of reductions). Record the llama.cpp build with every reference; the
harness writes it into `reference.json`.

## Spread

```
reference.py spread --model $G --reference results/precision/<ref-label> --label <backend-label> [--save-base]
# per category:
llama-perplexity -m $G -c 130 --kl-divergence --kl-divergence-base <ref>/<cat>.bin -ngl 99
```

`spread.json` holds, per category, the parsed `Mean KLD`, percentiles, `Same top
p`, `PPL(Q)`, `PPL(base)`, `RMS Δp` (with uncertainties), plus a
position-weighted overall mean KL and same-top. GPU defaults (F16 KV cache,
flash attention auto) are what the spread measures; pass `--extra` to change
flags. Recorded backends: Metal on an Apple M4 Pro (Homebrew llama.cpp b10964,
`--binary-dir /opt/homebrew/bin`), CUDA on an NVIDIA GB10 (llama.cpp b10998 release
build, `--binary-dir <llama.cpp dir>` with `LD_LIBRARY_PATH` set to that directory).
To measure on another machine, copy `precision/`, the corpus and the reference
directory into an `inference-v4/validation/` tree there first. The base files compress
~8× with `zstd -3` (5.4 GB → 0.4–0.9 GB), which matters over slow links; check the
decompressed sha256 against `reference.json`. A spread saturates the GPU: on a
machine shared with other timing work, hold a shared lock while it runs, one category
per lock hold (`GPU_LOCK=<dir> spread_host.sh`; `spread --categories` merges into the
label's `spread.json`).

## Local results (2026-09-23, M4 Max, Homebrew llama.cpp b8680 `15f786e65`)

Corpus sha256 (from `build_corpus.py`, also in `results/precision/corpus/manifest.json`):
prose `0bc4b2e45cb79c65438f63bfaa218249d28e2f2340353f3d314aca1dd8a5e515`
(243,428 B, 61,674 tokens), code
`dfc10486f27891b53d06223f8f289da4a9fd5f74d74499170d302d6d68e65ebf`
(170,535 B, 42,451 tokens), tool JSON
`8a1b265ccc5223f163d0e2d855d79acd557d67af9c246de23a93a28c73d1afb9`
(242,763 B, 69,746 tokens). Each needs 171 × 130 = 22,230 tokens. The Mac was heavily loaded by
other work (load average ~50), so the times are upper bounds.

Comparator check: on the full 512-chunk set, `kl_base.py compare` of the
CPU-Q4_K_M reference against the Metal run's own base file reproduces
llama-perplexity's printed numbers for the same pair (Metal raw logits):

| Category | llama.cpp KL / same top | kl_base.py KL / same top | PPL(base) both |
|---|---|---|---|
| prose | 0.003391 / 96.619 % | 0.003390 / 96.619 % | 20.718618 |
| code | 0.003094 / 97.816 % | 0.003092 / 97.816 % | 4.608315 |
| tool JSON | 0.004887 / 96.259 % | 0.004880 / 96.268 % | 5.251220 |

The small differences come from quantizing the candidate side as well (a file
against a file). PPL(base) is decoded from the file and agrees to every printed
digit, which confirms the row/target alignment. Metal is run-to-run
deterministic (KL 0 between two runs) and batch-size invariant to 2e-7.

Distances between llama.cpp paths (mean KL nats / same top):

| Pair | Positions | KL | Same top |
|---|---|---|---|
| CPU-Q4_K_M ref vs Metal (Q4_K_M, FA auto, F16 KV) | 32,768 | 0.00379 | 96.90 % |
| CPU-F32 ref vs Metal | 10,944 (prose) | 0.00025 | 98.93 % |
| CPU-F32 ref vs CPU-Q4_K_M | 10,944 (prose) | 0.00314 | 96.71 % |
| CPU-F32 ref vs Metal, 15 chunks/category | 960 each | 0.00017–0.00043 | 98.75–99.69 % |

So the CPU forward on the Q4_K_M file (Q8_K activations) is the outlier; Metal
(which dequantizes weights and computes F16/F32) sits close to the F32
forward. This is why the reference is the F32-dequantized GGUF. Note that with
the F32 reference, Metal's top-1 agreement is ~99 %, right at the provisional
D4 threshold; the tie-sensitivity of "same top" (flips are near-ties) matters
when M0 sets the thresholds.

Times on the loaded machine: CPU Q4_K_M reference 41 min for 512 chunks
(26 min prose, 10 min code, 5 min tool JSON, varying with load); CPU F32
reference 19.5 min for 171 prose chunks (loaded), 5.0 min code and 3.5 min tool
JSON (idle); Metal spread with `--save-base` 6.5 min for all three categories.
Disk: 16.27 GB per 512-chunk base set, 16.8 GB for the F32 GGUF.

## F32 reference and llama.cpp spreads (2026-09-24)

The F32 reference `results/precision/ref-cpu-f32-b8680/` is complete (all 512
chunks; sha256s in its `reference.json`). Against it, all 32,768 positions:

| Candidate | Mean KL overall | Category max | Same top overall | Category min | KL p99 max |
|---|---|---|---|---|---|
| llama.cpp Metal, M4 Max, b8680 | 0.000331 | 0.000556 (tool JSON) | 99.203 % | 98.931 % (prose) | 0.0032 |
| llama.cpp Metal, M4 Pro, b10964 | 0.000331 | 0.000555 | 99.225 % | 98.958 % | 0.0032 |
| llama.cpp CUDA, GB10, b10998 | 0.004264 | 0.005821 (tool JSON) | 97.040 % | 96.446 % (prose) | 0.0496 |
| llama.cpp CPU on the Q4_K_M file (Q8_K activations), b8680 | 0.003554 | 0.004673 | 96.960 % | 96.296 % | 0.0460 |

CUDA sits with the CPU-Q4_K_M forward, not with Metal: its quantized matmuls
quantize activations to q8_1 (MMQ/MMVQ), where Metal dequantizes weights and
computes in float. Metal's top-1 flips are near-ties (99.97 % same top where the
reference's top-two gap exceeds 0.1 nats); CUDA keeps 275 flips above that gap
(99.10 %). The D4 thresholds (spec §1) are set from the CUDA spread.

## V4 qualification

```
# from validation/results/precision (CUDA: export SEISMIC_NVRTC_DIRECTORY=<CUDA toolkit>/lib64)
forward_bench qualify --model $G --reference ref-cpu-f32-b8680 --output v4-qualify/<label>.json [--chunks N]
```

`engine/examples/forward_bench.rs qualify` loads V4 once (tuning at load takes
minutes) and runs every category of the reference directory: per chunk a
prefill of the first 65 tokens, then 64 teacher-forced one-row decodes with
logits exported, compared with the stored rows by llama.cpp's KL and same-top
definitions. The report is rewritten after each category; with all three it
carries `overall` and a `d4` verdict (each threshold, value, pass). `--chunks N`
caps the chunks per category for a quick look. `--storage-gib` must hold the
model (35B: 25 on an M4 Max, 30 on a GB10). The 35B F32 reference is
`ref-cpu-f32-35b-b8680` (139 GB F32 GGUF, larger than RAM; built with
`--extra -b 2048 -ub 2048` so each pass streams the file once). On a shared host, hold the GPU
lock for the run. First result (2026-09-24, local M4 Max, Metal): mean KL
0.000244, same top 99.39 %, D4 pass — closer to F32 than llama.cpp Metal.
