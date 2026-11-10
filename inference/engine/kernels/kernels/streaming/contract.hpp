// Expert streaming's device protocol (routed.seismic's streaming section),
// written once, in the language MSL, CUDA C++ and GLSL have in common, over
// each backend's shim (`metal/lib/streaming/control.h`,
// `cuda/lib/streaming/control.cuh`, `vulkan/lib/streaming/control.glsl`),
// which includes this file last.
//
// The shim provides:
// - `STREAMING_FN`, a protocol function's qualifier; `u32` and `u64`.
// - Word handles: `STREAMING_SYSTEM_WORDS` and `STREAMING_SYSTEM_READ` for
//   host-shared words (read-write, read-only), `STREAMING_DEVICE_WORDS` and
//   `STREAMING_DEVICE_READ` for device words; `STREAMING_AT(handle, index)`, the handle of word `index`
//   past `handle`; `streaming_load(handle)` and `streaming_store(handle, v)`
//   of a host-shared word, `streaming_device_load` and
//   `streaming_device_store` of a device word, and
//   `streaming_device_add(word, v)`, a relaxed device-scope atomic add of
//   `v` to a device word, returning the word before it (a landing launch's
//   counters, streaming/land.hpp), and `streaming_device_cas(word,
//   expected, desired)`, a relaxed device-scope compare-and-swap, storing
//   `desired` only where the word holds `expected` and returning the word
//   before it (a landing launch's poll).
// - Record tables: `STREAMING_RECORDS_READ` and `STREAMING_RECORDS`, the
//   handles a record copy's source and destination records are addressed
//   from (below).
// - Group memory, one array of int words per call: `STREAMING_GROUP_PARAM`,
//   the parameter a function takes it by, and `STREAMING_GROUP(i)`, its
//   word `i`; the caller passes its group words (a pointer to them, or on
//   GLSL their first index in the shared view); inside a protocol function
//   the parameter is `streaming_group`, which passes it on;
//   `STREAMING_GROUP_ADD(i, v)`, a relaxed atomic add of `v` to its word `i`.
// - The entry's words where a body's `SEISMIC_*` macros read them:
//   `STREAMING_WORDS_PARAM` and `STREAMING_WORDS_ARG`, each with its leading
//   comma (empty where the macros read them globally).
// - `streaming_system_fence()`; the group barriers `streaming_barrier()`
//   (group memory), `streaming_device_barrier()` (group and device memory)
//   and `streaming_publish_barrier()` (after which thread 0's host-shared
//   stores follow every thread's earlier ones).
// - `streaming_load_acquire(word)`, an acquire load of a host-shared word
//   (CUDA, Metal: system scope; Vulkan: device scope), and one 64-bit
//   access of a u64 at an even word, `streaming_load_u64_acquire(word)` and
//   `streaming_store_u64_release(word, v)` (an exchange slot's request and
//   answer, a map entry's steps: streaming_words.seismic).
// - `streaming_copy_bytes(from, to, bytes, lane, lanes)`, a copy of `bytes`
//   bytes from byte handle `from` to `to`, its units shared among `lanes`
//   lanes, the caller lane `lane`.
// Whether a wait spins in its kernel or waits at a park before its call is
// no shim's to say: every native's prefix states the backend's fact
// (`SEISMIC_PARKS_IN_KERNELS`), which the waits read.
// The shim includes the waits (`streaming/park.hpp`) after this file.
//
// A record copy (streaming/list_copy.hpp) addresses records through hooks
// its includer defines before including it: `STREAMING_COPY_FROM(table,
// record)` and `STREAMING_COPY_TO(table, record)`, the byte handle of record
// `record` of its source's and its destination's table (a record port's
// table, or a tensor's base), in the space the port's records lie in.
//
// Every loop counter of a protocol body is 32-bit (`u32`, or `int`), widened
// to u64 only where it addresses: NVIDIA's Vulkan compiler (GB10, RTX 3090)
// miscompiles a 64-bit loop counter that is multiplied, taking its parity
// as 1 (at K = 2, every choice of a row read its last).
//
// A protocol body takes its thread's index `tid` among the group's
// `threads`, its group words and, where its `SEISMIC_*` macros read the
// entry's words, those. Includers include the shim, then the bodies they
// call.

