# Writing Vulkan kernels

This applies to the Vulkan assets in `kernels/vulkan/`, the Vulkan native library
(`inference/seismic/lang/native-library/vulkan/`) and the prelude they share
(`inference/seismic/runtime/src/native/vulkan_prelude.glsl`).

Vulkan promises nothing about whether a valid shader compiles, how long the compile takes, how much
memory the driver needs, or how faithfully the driver's compiler keeps the module's meaning. What we
control is the module we hand the driver. Every rule below names the failure it comes from: a
compile failure or blowup measured across Mesa's Intel, AMD and NVK compilers, AMD's proprietary
compiler and NVIDIA's proprietary driver, or a wrong answer a driver gave us.

A rule is either **enforced**, so formation refuses a kernel that breaks it, or **authored**, so you
and your reviewer are responsible for it. Passing the enforced checks says nothing about speed.

## Enforced at formation

`formation::compile` (`inference/seismic/backends/vulkan/src/formation.rs`) refuses a module that
breaks the emitted contract (`seal::contract`). It checks the cleaned SPIR-V on every formation: in
tests, in offline coverage and on the device. The error names the function and the construct.

- **No 64-bit integer division or remainder**, scalar or vector, signed or unsigned. Mesa's Intel
  compiler expands each into about 2,300 instructions, copied by every unrolled loop around it:
  `dense_expand_gemm` had 191k instructions and 345 KB of scratch, and Xe2 refused it (#159).
  - Dividing by a literal still divides at run time. Only a whole expression of constants folds.
- **No constant table in per-invocation memory.** glslang copies a `const` array indexed at run
  time into Function storage at every use, and passing it to a function copies it again; drivers
  put it in local memory. Moving the codebooks to a buffer turned Intel DG2's refusals of the
  routed kernels into compiles, and made iq3_s GEMV at one row 7x faster (1489 to 201 µs, GB10).
  Lookup data lives in the device's tables buffer (below).
  - Still legal: vectors, zero arrays, splats (compared by value), mutable working arrays, and
    constant arrays indexed only by constants (they fold away).
- **No float converted to a 64-bit integer**, scalar or vector. NVIDIA's compiler (580, GB10)
  converted the float 2^32 to 2^33 - 1 (`8f4cee7cf2`). Convert to 32 bits, or decode the float's
  bits (`shape_rows`' `shape_mass`).
- **No cooperative-matrix constant read by two functions.** AMD's LLPC compiler faults inside
  `vkCreateComputePipelines` on it (`inference/seismic/hardware/rdna3-vulkan.md`). Take every
  zero accumulator from `matrix_zero()` (`lib/core/matrix.glsl`).

Numerical admission and speed are judged only by native tuning (α), against the default
configuration. The default must form on every floor device. A failing tuning candidate is only
excluded, so excluding candidates never repairs a bad default.

## Authored rules

### Integer coordinates and addresses

- Keep addresses and byte offsets wide (`uint64_t`), and tile-local coordinates in 32 bits. Prove
  that every intermediate product, sum and ceil-division fits before converting: a valid final
  offset does not prove an intermediate 32-bit product safe.
- Don't split a wide flattened index back into coordinates with division (#159's cause). The
  prelude's division-free coordinates cover the common shapes:
  - `seismic_walk_grid` / `seismic_walk_begin`: a grid-stride loop over a flattened extent of up to
    four axes, its coordinates carried rather than divided;
  - `seismic_split`: a 32-bit value (an invocation, workgroup or i32 index) split by any extent;
  - `seismic_split_group`: item `local` of group `group` split by an extent without forming the
    wide index;
  - `seismic_divide` / `seismic_quotient`: a wide value by a divisor below 2^16, in 32-bit
    arithmetic (`REQUIRE_DIVISOR_BELOW_2_16` refuses a constant divisor outside it);
  - `seismic_narrow_divide`: a wide value by a wide divisor when the quotient is proved below 2^32.
- Write a 64-bit conditional subtraction (a carry, a restoring-division step) as a select,
  `x -= x >= m ? m : 0ul`, never as a branch around the subtraction: NVIDIA 580's compiler crashed
  on the branch form in `attention_prefill`, `attention_append_k8v4` and `import_dense`.
- Never cast a wide value to 32 bits just to get past the check. If the arithmetic really is wide,
  derive and prove an algorithm over the operand domain.

### Numerics

- Divide and take square roots by the prelude's `seismic_div_rn` and `seismic_sqrt_rn`, never a
  bare `/`, and use its `seismic_fma_rn` and its rounding and conversion helpers. Preserve rounding, infinities and NaNs,
  signed zero and denormals. No fast-math, no reassociation, no new tolerance gate, and no change
  to the order of a floating-point reduction.
- GLSL `fma` is not guaranteed fused (#173: an Intel driver rounded it twice). Only `OpFmaKHR` is,
  and formation binds it where the device has `VK_KHR_shader_fma`.
- Never convert a float to an integer type it may not fit, even in an arm a select discards. The
  conversion is undefined, and a compiler may carry it into the selected value: GB10's did, in
  `shape_rows`' mass (`uint(2^32)` beside the constant `1ul << 32`), and it showed only once
  unrelated code changed the compiled form (`31f5e854aa`, then again `9302245455`). Bound the value
  first, or convert only the in-range alternative.

### Lookup data and working state

- Grid points, codebook values and conversion lookup tables are read from the device's tables
  buffer through the bindings every program's prefix declares (`GRIDS_TABLE_<ID>`,
  `CODEBOOKS_TABLE_<ID>`, `SEISMIC_REPACK_TABLE_<ID>`; generated by `seismic-lang`'s build as
  `source::VULKAN_TABLES_LAYOUT`). A launch that stages them copies them from that buffer into
  shared memory (`packets_stage_tables`). Never write lookup data as a `const` array.
- A thread's working state is tied to its algorithmic tile. When a problem dimension grows (tile
  width, head group size, a model dimension), spread the extra work across threads or iterate over
  it; never multiply the arrays every thread keeps live. Grouped decode's `result[G][E]` grew with
  the query group size until Intel spilled past its scratch limit; distributing the heads across
  slices compiled on DG2, TGL and Xe2.
- Keep expensive helpers (division, `exp`, activations) out of loops unrolled once per output
  element: each unrolled copy duplicates the helper.

### Unrolling and recipe selection

- `[[unroll]]` needs a representation reason, such as constant indexing of a small intrinsic
  fragment. It is not the default for loops over model dimensions or for epilogues copied per
  output. Unroll to the configured extent, never to a maximum behind a guard.
- `[[unroll]]` and `[[dont_unroll]]` are hints; neither guarantees resources on any driver.
- Choose invariant recipes (paired or plain, activation kind) at the launch or entry boundary, not
  inside the region they would multiply: short convolution's joint Q6 recipe was refused on DG2,
  while each recipe alone compiled in about 16 s.
- Keep code a constant makes dead out of the driver's input with preprocessor groups: drivers
  compile it at full cost (RADV took 9.8 s and 1.96 GB for a GEMM staging every weight kind, 0.5 s
  and 213 MB for the one it used).

### Subgroups and communication

- Use the shared logical-subgroup helpers; kernels assume 32 logical lanes. For every shuffle,
  reduction or broadcast, establish which lanes take part, which source lanes are valid,
  convergence, and memory scope and ordering. A full-subgroup launch flag does not make a
  divergent branch uniform.
- Setting only `requiredSubgroupSize = 16` is not a valid change: another lane mapping must change
  coverage and communication together.

## Not rules

These were measured and rejected:

- caps on instruction count, private bytes per lane, registers or compile time: 964 pipelines that
  compiled on DG2 were larger than the smallest refusal, and compiling pipelines held up to
  18,432 private bytes per lane;
- blanket `[[dont_unroll]]`: decode still failed on DG2, and Xe2 decode that compiled was refused;
- rolling only the epilogue: short convolution, which compiled, was then refused on DG2;
- per-vendor subgroup switches, and fixed `SLICES` or group-size values.

Vulkan does not promise that lanes reconverge after a loop they leave at different iterations
(no kernel requests maximal reconvergence). No driver has given a wrong answer for it, so it is
not a rule; `shape_rows`, whose answers are compared exactly with the reference, combines its
lanes through shared atomics there.

## Checking a change

- **Contract:** `cargo test -p seismic-vulkan --lib` runs the contract's probes. Forming your entry
  runs the contract on it, in any test that forms it (`tests/priming_formation.rs`,
  `tests/vulkan_coverage.rs`).
- **Speed:** a successful compile is not a speed measurement. Changes to shared primitives or
  layouts need an `mrig ab kernel` comparison, with `--weights` where speed depends on the values a
  kernel reads (see the root `AGENTS.md`).
