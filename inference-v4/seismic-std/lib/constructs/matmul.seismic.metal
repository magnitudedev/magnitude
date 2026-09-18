# Both operands bf16: 8x8 bfloat atoms with f32 accumulation over the aligned interior; B is
# loaded transposed so the atom sees [K, N]. Tails on every axis are finished element-wise.
lower matmul[M, N, K](A: tile[M, K] bf16, B: tile[N, K] bf16, acc: tile[M, N] f32):
  for i in range(M / 8):
    for j in range(N / 8):
      c = simdgroup_matrix(f32)
      simdgroup_load(c, acc, i * 8, j * 8)
      for k in range(K / 8):
        a = simdgroup_matrix(bf16)
        b = simdgroup_matrix(bf16)
        simdgroup_load(a, A, i * 8, k * 8)
        simdgroup_load_t(b, B, j * 8, k * 8)
        simdgroup_multiply_accumulate(c, a, b, c)
      simdgroup_store(c, acc, i * 8, j * 8)
  for i, j in owned(acc):
    if i >= 8 * (M / 8) or j >= 8 * (N / 8):
      for k in range(K):
        acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])
    else:
      for k in range(8 * (K / 8), K):
        acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# f32 left operand (already in threadgroup memory or registers): float atoms read it directly;
# each 8-row block of B is staged as f32 once, covering the whole K extent.
lower matmul[M, N, K](A: tile[M, K] f32, B: tile[N, K] U, acc: tile[M, N] f32):
  for i in range(M / 8):
    for j in range(N / 8):
      Bs = tile[8, K] f32
      for r, q in owned(Bs): Bs[r, q] = B[j * 8 + r, q]
      c = simdgroup_matrix(f32)
      simdgroup_load(c, acc, i * 8, j * 8)
      for k in range(K / 8):
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, A, i * 8, k * 8)
        simdgroup_load_t(b, Bs, 0, k * 8)
        simdgroup_multiply_accumulate(c, a, b, c)
      simdgroup_store(c, acc, i * 8, j * 8)
  for i, j in owned(acc):
    if i >= 8 * (M / 8) or j >= 8 * (N / 8):
      for k in range(K):
        acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])
    else:
      for k in range(8 * (K / 8), K):
        acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# Any operand types: each 8x8 operand block is staged as f32 in threadgroup memory, then
# float atoms accumulate. Operands are consumed at their declared precision.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for i in range(M / 8):
    for j in range(N / 8):
      c = simdgroup_matrix(f32)
      simdgroup_load(c, acc, i * 8, j * 8)
      for k in range(K / 8):
        As = tile[8, 8] f32
        Bs = tile[8, 8] f32
        for r, q in owned(As): As[r, q] = A[i * 8 + r, k * 8 + q]
        for r, q in owned(Bs): Bs[r, q] = B[j * 8 + r, k * 8 + q]
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, As, 0, 0)
        simdgroup_load_t(b, Bs, 0, 0)
        simdgroup_multiply_accumulate(c, a, b, c)
      simdgroup_store(c, acc, i * 8, j * 8)
  for i, j in owned(acc):
    if i >= 8 * (M / 8) or j >= 8 * (N / 8):
      for k in range(K):
        acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])
    else:
      for k in range(8 * (K / 8), K):
        acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# The portable fold retains source-selected contiguous partial chains and merge
# trees for participant ownership selection, including arbitrary packed tails.
lower matmul: portable
