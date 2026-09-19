# Metal realizations of the matmul contract. Every body is target code: loop bounds and
# addresses use the valid extents of the supplied tiles and tile-coordinate arithmetic, and
# the 8x8 atom is fixed native geometry. A `where` clause constrains the slice capacity; each
# body authors its own scalar tails, so it is correct for every visit up to that capacity.
# Aligned interiors use native matrix accumulation; tails keep the scalar FMA chain.

# Both operands bf16: 8x8 bfloat atoms with f32 accumulation over the aligned interior; b is
# loaded transposed so the atom sees [K, N]. Tails on every axis are finished element-wise.
lower matmul[M, N, K](a: tile[M, K] bf16, b: tile[N, K] bf16, inout into: tile[M, N] f32)
    for metal where M >= 8 and N >= 8 and K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..rows / 8:
        for j in 0..cols / 8:
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, into, i * 8, j * 8)
            for k in 0..depth / 8:
                var left = metal.simdgroup_matrix(bf16)
                var right = metal.simdgroup_matrix(bf16)
                metal.simdgroup_load(left, a, i * 8, k * 8)
                metal.simdgroup_load_t(right, b, j * 8, k * 8)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, into, i * 8, j * 8)
    for i, j in owned(into):
        if i >= 8 * (rows / 8) or j >= 8 * (cols / 8):
            for k in 0..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])
        else:
            for k in 8 * (depth / 8)..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# f32 left operand (already in threadgroup memory or registers): float atoms read it directly;
# each 8-row block of b is staged as f32 once, covering the whole K extent.
lower matmul[M, N, K](a: tile[M, K] f32, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where M >= 8 and N >= 8 and K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..rows / 8:
        for j in 0..cols / 8:
            var right_rows = tile[8, K] f32
            for r, q in owned(right_rows):
                right_rows[r, q] = f32(b[j * 8 + r, q])
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, into, i * 8, j * 8)
            for k in 0..depth / 8:
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, a, i * 8, k * 8)
                metal.simdgroup_load_t(right, right_rows, 0, k * 8)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, into, i * 8, j * 8)
    for i, j in owned(into):
        if i >= 8 * (rows / 8) or j >= 8 * (cols / 8):
            for k in 0..depth:
                into[i, j] = fma(a[i, k], f32(b[j, k]), into[i, j])
        else:
            for k in 8 * (depth / 8)..depth:
                into[i, j] = fma(a[i, k], f32(b[j, k]), into[i, j])

# Any operand types: each 8x8 operand block is staged as f32 in threadgroup memory, then
# float atoms accumulate. Operands are consumed at their declared precision.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where M >= 8 and N >= 8 and K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..rows / 8:
        for j in 0..cols / 8:
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, into, i * 8, j * 8)
            for k in 0..depth / 8:
                var left_block = tile[8, 8] f32
                var right_block = tile[8, 8] f32
                for r, q in owned(left_block):
                    left_block[r, q] = f32(a[i * 8 + r, k * 8 + q])
                for r, q in owned(right_block):
                    right_block[r, q] = f32(b[j * 8 + r, k * 8 + q])
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, left_block, 0, 0)
                metal.simdgroup_load_t(right, right_block, 0, 0)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, into, i * 8, j * 8)
    for i, j in owned(into):
        if i >= 8 * (rows / 8) or j >= 8 * (cols / 8):
            for k in 0..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])
        else:
            for k in 8 * (depth / 8)..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# Stage a left row block once for all output fragments. K is the authored contraction
# window of the enclosing traversal, so this strategy composes with bounded windows.
# Widening preserves the declared operand values.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where M >= 8 and N >= 8 and K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..rows / 8:
        var left_rows = tile[8, K] f32
        for r, q in owned(left_rows):
            left_rows[r, q] = f32(a[i * 8 + r, q])
        for j in 0..cols / 8:
            var right_rows = tile[8, K] f32
            for r, q in owned(right_rows):
                right_rows[r, q] = f32(b[j * 8 + r, q])
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, into, i * 8, j * 8)
            for k in 0..depth / 8:
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, left_rows, 0, k * 8)
                metal.simdgroup_load_t(right, right_rows, 0, k * 8)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, into, i * 8, j * 8)
    for i, j in owned(into):
        if i >= 8 * (rows / 8) or j >= 8 * (cols / 8):
            for k in 0..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])
        else:
            for k in 8 * (depth / 8)..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# Stage a right row block once across activation row blocks. This is the
# complementary reuse order: the prepared right rows survive every left row block.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where M >= 8 and N >= 8 and K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for j in 0..cols / 8:
        var right_rows = tile[8, K] f32
        for r, q in owned(right_rows):
            right_rows[r, q] = f32(b[j * 8 + r, q])
        for i in 0..rows / 8:
            var left_rows = tile[8, K] f32
            for r, q in owned(left_rows):
                left_rows[r, q] = f32(a[i * 8 + r, q])
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, into, i * 8, j * 8)
            for k in 0..depth / 8:
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, left_rows, 0, k * 8)
                metal.simdgroup_load_t(right, right_rows, 0, k * 8)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, into, i * 8, j * 8)
    for i, j in owned(into):
        if i >= 8 * (rows / 8) or j >= 8 * (cols / 8):
            for k in 0..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])
        else:
            for k in 8 * (depth / 8)..depth:
                into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# Complete output fragments admit matrix ownership even when M or N is below
# eight. Padding belongs only to unpublished rows/columns. K is never padded:
# complete 8-wide contractions use the existing F32 matrix cover and the exact
# remaining source leaves keep their scalar FMA order.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..(rows + 7) / 8:
        for j in 0..(cols + 7) / 8:
            var state = tile[8, 8] f32
            for r, s in owned(state):
                if i * 8 + r < rows and j * 8 + s < cols:
                    state[r, s] = into[i * 8 + r, j * 8 + s]
                else:
                    state[r, s] = 0.0
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, state, 0, 0)
            for k in 0..depth / 8:
                var left_block = tile[8, 8] f32
                var right_block = tile[8, 8] f32
                for r, q in owned(left_block):
                    if i * 8 + r < rows:
                        left_block[r, q] = f32(a[i * 8 + r, k * 8 + q])
                    else:
                        left_block[r, q] = 0.0
                for r, q in owned(right_block):
                    if j * 8 + r < cols:
                        right_block[r, q] = f32(b[j * 8 + r, k * 8 + q])
                    else:
                        right_block[r, q] = 0.0
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, left_block, 0, 0)
                metal.simdgroup_load_t(right, right_block, 0, 0)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, state, 0, 0)
            for r, s in owned(state):
                if i * 8 + r < rows and j * 8 + s < cols:
                    into[i * 8 + r, j * 8 + s] = state[r, s]
    for i, j in owned(into):
        for k in 8 * (depth / 8)..depth:
            into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# A bounded window can stage its left rows once across all output fragments,
# including partial fragments. The enclosing authored traversal supplies K.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..(rows + 7) / 8:
        var left_rows = tile[8, K] f32
        for r, q in owned(left_rows):
            if i * 8 + r < rows:
                left_rows[r, q] = f32(a[i * 8 + r, q])
            else:
                left_rows[r, q] = 0.0
        for j in 0..(cols + 7) / 8:
            var right_rows = tile[8, K] f32
            for r, q in owned(right_rows):
                if j * 8 + r < cols:
                    right_rows[r, q] = f32(b[j * 8 + r, q])
                else:
                    right_rows[r, q] = 0.0
            var state = tile[8, 8] f32
            for r, s in owned(state):
                if i * 8 + r < rows and j * 8 + s < cols:
                    state[r, s] = into[i * 8 + r, j * 8 + s]
                else:
                    state[r, s] = 0.0
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, state, 0, 0)
            for k in 0..depth / 8:
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, left_rows, 0, k * 8)
                metal.simdgroup_load_t(right, right_rows, 0, k * 8)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, state, 0, 0)
            for r, s in owned(state):
                if i * 8 + r < rows and j * 8 + s < cols:
                    into[i * 8 + r, j * 8 + s] = state[r, s]
    for i, j in owned(into):
        for k in 8 * (depth / 8)..depth:
            into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# Complementary row reuse: the prepared right rows survive all activation-row
# fragments. This remains a source strategy with ordinary storage choices.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for j in 0..(cols + 7) / 8:
        var right_rows = tile[8, K] f32
        for r, q in owned(right_rows):
            if j * 8 + r < cols:
                right_rows[r, q] = f32(b[j * 8 + r, q])
            else:
                right_rows[r, q] = 0.0
        for i in 0..(rows + 7) / 8:
            var left_rows = tile[8, K] f32
            for r, q in owned(left_rows):
                if i * 8 + r < rows:
                    left_rows[r, q] = f32(a[i * 8 + r, q])
                else:
                    left_rows[r, q] = 0.0
            var state = tile[8, 8] f32
            for r, s in owned(state):
                if i * 8 + r < rows and j * 8 + s < cols:
                    state[r, s] = into[i * 8 + r, j * 8 + s]
                else:
                    state[r, s] = 0.0
            var c = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c, state, 0, 0)
            for k in 0..depth / 8:
                var left = metal.simdgroup_matrix(f32)
                var right = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left, left_rows, 0, k * 8)
                metal.simdgroup_load_t(right, right_rows, 0, k * 8)
                metal.simdgroup_multiply_accumulate(c, left, right, c)
            metal.simdgroup_store(c, state, 0, 0)
            for r, s in owned(state):
                if i * 8 + r < rows and j * 8 + s < cols:
                    into[i * 8 + r, j * 8 + s] = state[r, s]
    for i, j in owned(into):
        for k in 8 * (depth / 8)..depth:
            into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# Retain a 2x2 rectangle of output fragments through the contraction. Each
# prepared activation/weight fragment feeds two products before reuse of its
# bounded staging tile. This cover neither pads K nor changes operand precision.
lower matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout into: tile[M, N] f32)
    for metal where K >= 8:
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..(rows + 15) / 16:
        for j in 0..(cols + 15) / 16:
            var state00 = tile[8, 8] f32
            for r, s in owned(state00):
                if i * 16 + 0 + r < rows and j * 16 + 0 + s < cols:
                    state00[r, s] = into[i * 16 + 0 + r, j * 16 + 0 + s]
                else:
                    state00[r, s] = 0.0
            var c00 = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c00, state00, 0, 0)
            var state01 = tile[8, 8] f32
            for r, s in owned(state01):
                if i * 16 + 0 + r < rows and j * 16 + 8 + s < cols:
                    state01[r, s] = into[i * 16 + 0 + r, j * 16 + 8 + s]
                else:
                    state01[r, s] = 0.0
            var c01 = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c01, state01, 0, 0)
            var state10 = tile[8, 8] f32
            for r, s in owned(state10):
                if i * 16 + 8 + r < rows and j * 16 + 0 + s < cols:
                    state10[r, s] = into[i * 16 + 8 + r, j * 16 + 0 + s]
                else:
                    state10[r, s] = 0.0
            var c10 = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c10, state10, 0, 0)
            var state11 = tile[8, 8] f32
            for r, s in owned(state11):
                if i * 16 + 8 + r < rows and j * 16 + 8 + s < cols:
                    state11[r, s] = into[i * 16 + 8 + r, j * 16 + 8 + s]
                else:
                    state11[r, s] = 0.0
            var c11 = metal.simdgroup_matrix(f32)
            metal.simdgroup_load(c11, state11, 0, 0)
            for k in 0..depth / 8:
                var left_block0 = tile[8, 8] f32
                for r, q in owned(left_block0):
                    if i * 16 + 0 + r < rows:
                        left_block0[r, q] = f32(a[i * 16 + 0 + r, k * 8 + q])
                    else:
                        left_block0[r, q] = 0.0
                var left0 = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left0, left_block0, 0, 0)
                var left_block1 = tile[8, 8] f32
                for r, q in owned(left_block1):
                    if i * 16 + 8 + r < rows:
                        left_block1[r, q] = f32(a[i * 16 + 8 + r, k * 8 + q])
                    else:
                        left_block1[r, q] = 0.0
                var left1 = metal.simdgroup_matrix(f32)
                metal.simdgroup_load(left1, left_block1, 0, 0)
                var right_block0 = tile[8, 8] f32
                for r, q in owned(right_block0):
                    if j * 16 + 0 + r < cols:
                        right_block0[r, q] = f32(b[j * 16 + 0 + r, k * 8 + q])
                    else:
                        right_block0[r, q] = 0.0
                var right0 = metal.simdgroup_matrix(f32)
                metal.simdgroup_load_t(right0, right_block0, 0, 0)
                var right_block1 = tile[8, 8] f32
                for r, q in owned(right_block1):
                    if j * 16 + 8 + r < cols:
                        right_block1[r, q] = f32(b[j * 16 + 8 + r, k * 8 + q])
                    else:
                        right_block1[r, q] = 0.0
                var right1 = metal.simdgroup_matrix(f32)
                metal.simdgroup_load_t(right1, right_block1, 0, 0)
                metal.simdgroup_multiply_accumulate(c00, left0, right0, c00)
                metal.simdgroup_multiply_accumulate(c01, left0, right1, c01)
                metal.simdgroup_multiply_accumulate(c10, left1, right0, c10)
                metal.simdgroup_multiply_accumulate(c11, left1, right1, c11)
            metal.simdgroup_store(c00, state00, 0, 0)
            for r, s in owned(state00):
                if i * 16 + 0 + r < rows and j * 16 + 0 + s < cols:
                    into[i * 16 + 0 + r, j * 16 + 0 + s] = state00[r, s]
            metal.simdgroup_store(c01, state01, 0, 0)
            for r, s in owned(state01):
                if i * 16 + 0 + r < rows and j * 16 + 8 + s < cols:
                    into[i * 16 + 0 + r, j * 16 + 8 + s] = state01[r, s]
            metal.simdgroup_store(c10, state10, 0, 0)
            for r, s in owned(state10):
                if i * 16 + 8 + r < rows and j * 16 + 0 + s < cols:
                    into[i * 16 + 8 + r, j * 16 + 0 + s] = state10[r, s]
            metal.simdgroup_store(c11, state11, 0, 0)
            for r, s in owned(state11):
                if i * 16 + 8 + r < rows and j * 16 + 8 + s < cols:
                    into[i * 16 + 8 + r, j * 16 + 8 + s] = state11[r, s]
    for i, j in owned(into):
        for k in 8 * (depth / 8)..depth:
            into[i, j] = fma(f32(a[i, k]), f32(b[j, k]), into[i, j])

# The portable fold remains a candidate on Metal, including for shapes and packed tails
# that no native body above admits.
lower matmul for metal = portable

# Packet vector contraction of `matmul_any_order` for a bf16 row against q4g64 weights. The
# cooperating subgroup owns at most eight outputs of one row (`into` stays below one subgroup of
# elements, so every lane visits every output). Of each 512-column chunk a lane takes its own
# 16-column packet: the activation packet is loaded once for all owned outputs, the two code
# words of each output are consumed as nibbles, and the group's coefficients are applied
# once per packet in the factored form the contract admits. One subgroup sum per output
# combines the lane partials. Columns past the last whole chunk are folded by lane zero.
lower matmul_any_order[M, N, K](a: tile[M, K] bf16, b: tile[N, K] q4g64, inout into: tile[M, N] f32)
    for metal where M <= 1 and N <= 8 and K >= 512:
    let lane = metal.lane_index()
    let rows = valid(into, 0)
    let cols = valid(into, 1)
    let depth = valid(a, 1)
    for i in 0..rows:
        var partial = tile[N] f32
        for j in owned(partial):
            partial[j] = 0.0
        for chunk in 0..depth / 512:
            let first = chunk * 512 + lane * 16
            var values = tile[16] f32
            for q in owned(values):
                values[q] = f32(a[i, first + q])
            var total = f32(0.0)
            for q in 0..16:
                total = total + values[q]
            for j in 0..cols:
                let low = b.words[j, first / 8]
                let high = b.words[j, first / 8 + 1]
                var d0 = f32(0.0)
                var d1 = f32(0.0)
                var d2 = f32(0.0)
                var d3 = f32(0.0)
                for q in 0..4:
                    d0 = d0 + values[2 * q] * f32((low >> u32(8 * q)) & 15)
                    d1 = d1 + values[2 * q + 1] * f32((low >> u32(8 * q + 4)) & 15)
                    d2 = d2 + values[8 + 2 * q] * f32((high >> u32(8 * q)) & 15)
                    d3 = d3 + values[8 + 2 * q + 1] * f32((high >> u32(8 * q + 4)) & 15)
                partial[j] = partial[j] + (((d0 + d1) + (d2 + d3)) * f32(b.scale[j, first / 64]) + total * f32(b.bias[j, first / 64]))
        if lane == 0:
            for j in 0..cols:
                for k in 512 * (depth / 512)..depth:
                    partial[j] = partial[j] + f32(a[i, k]) * f32(b[j, k])
        for j in 0..cols:
            into[i, j] = into[i, j] + metal.simd_sum(partial[j])

lower matmul_any_order for metal = portable
