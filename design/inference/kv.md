---
applies_to:
  - inference/crates/icn-engine/src/radix_cache.rs
  - inference/crates/icn-engine/src/scheduler.rs
  - inference/crates/icn-engine/src/lib.rs
  - inference/crates/icn-contracts/src/lib.rs
  - inference/crates/icn-server/src/main.rs
  - inference/native/llama-cpp-rs/llama-cpp-2/src/context/kv_cache.rs
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/include/llama.h
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/src/llama-context.cpp
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/src/llama-kv-cache.cpp
---

# KV cache design

Status: implemented, with the limitations called out below.

The inference KV system turns computed prompt prefixes into reusable, page-granular cache entries. Its main purpose is to avoid repeating prefill work and duplicating physical K/V tensors when requests share a prefix. The cache is owned by one model executor, so entries cannot cross model, tokenizer, context, adapter, or process boundaries accidentally.

This document describes KV ownership and movement. Admission and batching are described in [scheduler.md](./scheduler.md), and the containing runtime is described in [engine.md](./engine.md).

## Influences

The design combines ideas from three serving systems rather than reproducing any one of them:

- **llama.cpp** supplies the underlying KV memory model. Magnitude extends its unified KV implementation with page pin, attach, release, export, import, and accounting operations. The legacy per-sequence checkpoint cache and the dense unified-attention behavior also come from llama.cpp's execution model.
- **SGLang** inspired token-prefix radix lookup, locking the matched ancestor path while requests use it, publishing prefixes dynamically, and evicting only unreferenced suffixes. Magnitude uses fixed-size page edges instead of SGLang's variable-length compressed edges.
- **vLLM** inspired fixed-size physical cache units, explicit page ownership/reference safety, exact free-capacity accounting, and treating cached-but-unreferenced pages as immediately reclaimable. Magnitude does not currently implement vLLM's block-table paged-attention kernel.

## Architecture

```text
token prefix radix
       │
       ├── page ── device: pinned native llama.cpp KV cells
       ├── page ── host: opaque serialized page in RAM
       └── page ── disk: checksummed page file
                         │
                         └── eviction: discard; prompt remains canonical
```

The logical index is a radix tree over token pages. Each edge represents exactly `page_tokens` consecutive tokens and each non-root node owns one KV page. The default page size is 16 tokens. Fixed pages align the logical tree with independently pinnable, spillable, promotable, and evictable physical units. Lookup is proportional to the number of complete pages in the matching prefix.

The tree stores topology independently of a page's physical residency. Common prefixes are represented once, branches begin at the first differing page, and every descendant implicitly depends on its ancestors. A lookup updates per-node hit and recency information.

Only complete pages are cached. The final prompt token is deliberately excluded from matching so the engine always evaluates a token that produces logits for the first sample. As a result, an exact prompt hit normally still performs a small replay, currently up to one page boundary rather than exactly one token.

## Native page interface

Magnitude's llama.cpp fork exposes immutable, context-scoped KV page handles:

- `pin` retains a contiguous position range already computed for a sequence;
- `attach` adds a retained page to another sequence through metadata, without copying K/V tensor data;
- `release` removes one cache pin while preserving any live sequence attachment;
- `export` serializes a page into an opaque, self-describing blob;
- `import` reconstructs a retained page in free native cells;
- `stats` reports exact page, used-cell, free-cell, and pinned-cell counts.

Handles carry the originating context identity in the safe Rust binding. A page therefore cannot be attached to a different model or draft context by mistake. Native cells remain valid while owned by either a live sequence, a page pin, or both.

This interface is supported only for unified, ordinary full-attention KV memory with one normal token cell per position. Per-sequence KV streams, recurrent state, sliding-window/hybrid layouts, multidimensional positions, and MTP/draft execution do not use radix pages. Unsupported configurations fall back to the legacy sequence cache rather than approximating unsafe reuse.

## Residency tiers

Every radix node has exactly one current residency:

| Tier | Representation | Reuse operation |
| --- | --- | --- |
| Device | Pinned native page | Metadata-only attach |
| Host RAM | Immutable serialized blob | Import, then attach |
| Local disk | Length- and SHA-256-verified file | Read, verify, import, then attach |
| Evicted | No radix entry | Ordinary cold prefill |

Device capacity is the native context's KV cell allocation. Host and disk are optional, strict byte budgets allocated lazily; a zero budget disables that tier. An object larger than a tier bypasses it.

When native cells are needed, the cache chooses a cold, unlocked suffix page. It prefers one-hit pages before reused pages and then uses recency, with deterministic depth tie-breaking. Demotion proceeds in commit order:

1. Export the device page.
2. Make room in RAM, cascading a RAM suffix to disk when necessary.
3. Atomically persist a disk page when disk is selected.
4. Release the native page only after the lower-tier copy is committed.
5. If no lower tier can accept it, remove its unlocked suffix subtree.

Disk writes use a temporary file, `sync_all`, and rename. Each process uses a unique cache directory beneath the configured root, and that directory is removed when the cache owner is dropped. Disk storage is therefore a bounded spill tier, not a restart-persistent cache.

Promotion reverses the path: reserve native cells, read and validate lower-tier bytes, import the page, switch the node to device residency, and remove the old lower-tier accounting and file. Promotion is synchronous on the model executor today.

An unreadable, corrupt, or unimportable suffix is acceleration failure, not inference failure. The cache removes that suffix and the request resumes with ordinary prefill from the last successfully attached page.

## Lookup, publication, and lifetime

For a text request, the engine matches all complete pages before the final prompt token. It locks the matched path, reserves native capacity for non-device pages, promotes them as needed, clears the destination sequence, and attaches each device page. The request begins prefill at the attached prefix length.

As a request computes new complete pages, it can publish them while still active. Publication pins native ranges, inserts only missing radix edges, and extends the request's path lock. This allows queued siblings to share a prefix produced in the same serving burst. When a cacheable request finishes, its remaining complete history is retained and its path is unlocked. A non-cacheable request publishes no retained history.

Path locks are the eviction safety protocol. A request locks every ancestor it depends on; an eviction or subtree removal may select only nodes whose paths are unlocked. Radix mutations and native page operations are serialized on the single model executor, so page ownership cannot race with sequence mutation.

## Capacity and accounting invariants

- A token match is reusable only inside the same model executor and native context.
- Device allocation uses exact native free-cell accounting, not an estimate of bytes.
- RAM and disk counters must never exceed their configured maxima.
- Lower-tier hits save model evaluation but still consume device cells after promotion.
- Destination residency is committed before source residency is released.
- A locked prefix cannot be demoted, deleted, or truncated.
- Cache state is disposable; request tokens and content remain authoritative.
- Metrics distinguish device-, RAM-, and disk-sourced tokens and record promotion time.

## Configuration

The resolved execution plan contains:

- `enabled`, default `true`;
- `page_tokens`, default `16`;
- `host_bytes`, default `0`;
- `disk_bytes`, default `0`;
- `disk_path`, optional.

The server exposes `--no-radix-cache`, `--radix-cache-host-bytes`, `--radix-cache-disk-bytes`, and `--radix-cache-dir`. Lower tiers require `--kv-unified`. Radix configuration can be enabled while the runtime still declines to activate it when the native memory layout is unsupported.

## Current limitations

- Tier reads, writes, hashing, export, and import are synchronous.
- Restore selection always favors the longest logical prefix; it does not yet compare transfer cost with recomputation cost.
- Disk entries are process-scoped and have no restart index or recovery protocol.
- There are no dedicated in-flight transfer states or transfer-worker limits.
- The native attention kernel still scans llama.cpp's dense unified KV arena. It does not use a vLLM/SGLang-style per-request block table, so independent concurrent requests can be slower than separate KV streams.
- Radix reuse is disabled for multimodal prompts, recurrent/hybrid memory, and MTP/draft execution.
