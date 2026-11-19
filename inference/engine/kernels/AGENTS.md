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
- **No 64-bit conditional subtraction written as a branch** (a carry, a restoring-division step:
  `if (x >= m) x -= m;`). Write it as a select, `x -= x >= m ? m : 0ul`. NVIDIA 580's compiler
  crashed on the branch form in `attention_prefill`, `attention_append_k8v4` and `import_dense`.
- **`seismic_sqrt_rn`'s digit steps stay straight-line.** As a constant-count loop of 64-bit
  selects, NVIDIA's 580 drivers computed them wrongly (GB10) or hung (Ada); see
  `docs/compatibility.md`, "Driver defects". The numerics oracle on each guards it.
- **The execution model** (`docs/compatibility.md`, "Execution model"), checked on an inlined SSA
  copy of each module (`uniformity.rs`). "Subgroups and communication" below says how to write
  kernels that meet it. A branch on a constant (a static extent, an element kind, a constant
  passed in a struct through a call) is checked only along the way it takes; one on a
  specialization constant along both. A refusal names each kind once, with its count, its
  first instruction and where what it refuses comes from (the built-in load, racing read or
  subgroup operation, or the loop or selection and its condition): start there.
- **A path through an optional buffer is guarded by what makes the buffer present** (a static
  dimension, a constant argument), so it folds away where the buffer is absent. An absent
  buffer's address is 0, and an address made of constants may reach any memory, so a read the
  check cannot rule out (one kept unreached only by a runtime invariant) reads whatever the
  launch writes: `attention_prefill_vector`'s picks path was refused that way.

`seismic-build` also refuses, in kernel sources (not the prelude), every `gl_Subgroup*` built-in,
`gl_NumSubgroups`, and every subgroup operation that spans the hardware subgroup: on 64-lane
hardware it spans both logical subgroups. Only `subgroupShuffleXor` (masks below 32),
`subgroupShuffleUp`/`Down`, `subgroupBarrier` and the subgroup memory barriers stay. Everything
else goes through the prelude's logical-subgroup helpers.

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
- Never cast a wide value to 32 bits just to get past the check. If the arithmetic really is wide,
  derive and prove an algorithm over the operand domain.

### Numerics

- Accuracy is judged by the precision system (`docs/precision.md`), not by bit identity. A form
  that rounds differently (another reduction order, a hardware square root, an unfused
  multiply-add) is a candidate that native tuning admits at α or excludes. No kernel carries its
  own tolerance, exact-reference requirement or fixed-order rule.
- Non-finite values must match exactly under admission, so keep infinities and NaNs: no
  fast-math flags. The prelude's correctly rounded helpers (`seismic_div_rn`, `seismic_sqrt_rn`)
  exist for results that must not depend on the driver's rounding; they are slower than the
  hardware operations, so use them where that dependence matters, not by default.
- GLSL `fma` is not guaranteed fused (#173: an Intel driver rounded it twice). Only `OpFmaKHR` is,
  and formation binds it where the device has `VK_KHR_shader_fma`. Don't write an algorithm whose
  correctness, rather than its accuracy, depends on fusion.
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

Vulkan kernels may assume only Vulkan's guarantees plus the execution model's assumptions
(`docs/compatibility.md`, "Execution model"): a selection entered with subgroup-uniform control
flow rejoins at its merge, and a workgroup barrier is reached by every invocation together.
Subgroups do not run in lockstep, a lane is not the local id modulo 32, and no subgroup is
partial. Formation enforces every rule in this list; the next list is yours.

- Run subgroup operations (shuffles, reductions, votes, broadcasts, cooperative-matrix loads,
  stores and multiplies) only in uniform control: never inside a selection whose condition
  differs between lanes, and never after a lane-varying loop or a lane-dependent exit unless a
  workgroup `barrier()` comes between. A loop is lane-varying when lanes leave or continue it at
  different iterations; an exit (`return`, `break`, `continue`) is lane-dependent when its
  condition differs between lanes. A lane-strided staging loop, then `barrier()`, then subgroup
  reductions is accepted. A `subgroupBarrier()` does not end a lane-varying loop's divergence,
  and no barrier undoes a lane-dependent exit.
- Never put `barrier()` or `subgroupBarrier()` inside divergent control, after a lane-dependent
  exit, or (for `subgroupBarrier()`) after a lane-varying loop.
- A helper's `return` rejoins at the call, as a selection rejoins at its merge.
- To steer control flow by a vote of every lane, use the hardware-wide
  `seismic_uniform_all`/`seismic_uniform_any`: their results are uniform. The logical-subgroup
  helpers are clustered on 64-lane hardware, so their results are divergent.
- Where the two logical subgroups of a 64-lane hardware one may hold different work (a row, a
  head, a key range each), steer the hardware subgroup by `seismic_steer(live, p)` of what each
  holds, and let a logical subgroup without work run the step on inputs that exist and store
  nothing. It is the whole condition of an `if`, `while` or `for` (`bool live;` declared just
  before), so a steered loop exits at its header, never by a `break` or `continue` in its body;
  inside the region it guards (or after `if (!seismic_steer(live, p)) return`), read `live`, never
  `p` or a recomputation of it. On 32-lane hardware `live` is the literal true and the vote is
  removed before the module ships, so the 32-lane driver compiles exactly the plain loop or
  branch: a predicate threaded on, or an exit in a loop's body, it keeps and pays for (k8v4's
  keywise walk ran 9 to 24 times slower on NVIDIA). Broadcast a value by
  `seismic_hardware_first`.
- Number by `SEISMIC_THREAD` (`SEISMIC_SUBGROUP * 32 + SEISMIC_LANE`) whatever reaches a
  subgroup operation or a memory read in lane order; the local or global id
  (`gl_LocalInvocationIndex`, `gl_LocalInvocationID`, `gl_GlobalInvocationID`) is fine elsewhere.
  Vulkan does not say which invocations share a subgroup or their lane order, so a value derived
  from the local or global id may not reach a subgroup operation, and may not meet a value
  derived from the lane order (a lane or subgroup id, a per-lane subgroup result, or a
  whole-subgroup result that depends on which subgroup combined it, such as a subgroup sum of
  loads at `SEISMIC_THREAD`-strided or subgroup-indexed addresses, or a value carried out of a
  loop that subgroups leave at different iterations) in one value or one store, including
  through the control a store to workgroup or device memory runs under, and where invocations
  that took different paths meet. A store to a local variable or a call's parameter keeps
  what is stored, under any control: passing a struct to a helper inside a divergent branch
  does not make its members divergent. A
  whole-subgroup reduction of values that differ only by lane within the subgroup (a lane id, a
  load at a lane's address) is the same in every full subgroup, so it stays uniform. Shared
  memory is judged per class (all workgroup memory; all memory the launch writes), not per
  array, but for the bytes a load at a constant offset of workgroup memory reads: a store
  races it only where it may touch those bytes (any store at a variable offset may), and not
  where a branch the whole workgroup takes alike parts the store's way from the load's
  within the barrier interval. So a
  tile stored by the local id and read by `SEISMIC_THREAD` mixes them too, and a
  buffer the launch writes, read at one address, carries what its own workgroup stored before
  or beside the read (what another workgroup stored, only its race); a
  local array of structs keeps each member apart, joined across its elements. A kernel that
  reads no subgroups has no lane order and may use the local id throughout.
- Steer control, and give operands SPIR-V requires uniform, only values every lane of the
  hardware subgroup reads alike: push constants, the argument words, the tables buffer, a buffer
  the launch only reads (a `&` parameter, a slab or record port's address table, or a result or
  scratch an earlier launch wrote that this launch stores nothing through), or workgroup
  memory read where no store races it in its barrier interval (the stretch between two
  workgroup barriers; shared memory, above). A store through an address read from a port's table
  (`slab_out_row`, `slab_region`) writes only that port's bytes; reading them back in the
  same launch is reading memory it writes. A buffer the launch
  writes is never read alike: other workgroups may store to it at any time. To steer by one of
  its words, have one invocation load it into workgroup memory, then `barrier()`.
- Broadcast and read-invocation only from a uniform lane, and give cooperative-matrix loads and
  stores a uniform pointer, layout and stride. A shuffle from a uniform lane gives a uniform value.
- A kernel that reads its subgroups (a subgroup operation or built-in, `SEISMIC_LANE`,
  `SEISMIC_SUBGROUP` and `SEISMIC_THREAD` included) declares workgroup X as a multiple of 64; it
  is refused on every device otherwise. One that reads none takes any X.

Authored:

- Write a strided loop with a uniform trip count, so every lane runs every iteration, where a
  subgroup operation follows it with no workgroup barrier between:
  `for (base = 0; base < n; base += step) { i = base + lane_index; v = i < n ? load(min(i, n - 1)) : identity; }`.
- Use the prelude's logical-subgroup helpers; kernels assume 32 logical lanes. For every shuffle,
  reduction or broadcast, establish which source lanes are valid and the memory scope and
  ordering. Keep `subgroupShuffleXor` masks below 32 and discard the edge lanes of
  `subgroupShuffleUp`/`Down`.
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

## Checking a change

- **Contract:** `cargo test -p seismic-vulkan --lib` runs the contract's probes. Forming your entry
  runs the contract on it, in any test that forms it (`tests/priming_formation.rs`,
  `tests/vulkan_coverage.rs`).
- **Speed:** a successful compile is not a speed measurement. Changes to shared primitives or
  layouts need an `mrig ab kernel` comparison, with `--weights` where speed depends on the values a
  kernel reads (see the root `AGENTS.md`).
