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

# Stage a left row block once for all output fragments. K is the enclosing
# compiler-selected contraction extent, so this strategy composes with bounded
# pieces. Widening preserves the declared operand values; packed decoding can
# independently use the representation-owned packet producer.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for i in range(M / 8):
    As = tile[8, K] f32
    for r, q in owned(As): As[r, q] = A[i * 8 + r, q]
    for j in range(N / 8):
      Bs = tile[8, K] f32
      for r, q in owned(Bs): Bs[r, q] = B[j * 8 + r, q]
      c = simdgroup_matrix(f32)
      simdgroup_load(c, acc, i * 8, j * 8)
      for k in range(K / 8):
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, As, 0, k * 8)
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

# Stage a right row block once across activation row blocks. This is the
# complementary reuse order: its independent serial regions can fuse across
# calls, and common left preparation remains visible to value sharing.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for j in range(N / 8):
    Bs = tile[8, K] f32
    for r, q in owned(Bs): Bs[r, q] = B[j * 8 + r, q]
    for i in range(M / 8):
      As = tile[8, K] f32
      for r, q in owned(As): As[r, q] = A[i * 8 + r, q]
      c = simdgroup_matrix(f32)
      simdgroup_load(c, acc, i * 8, j * 8)
      for k in range(K / 8):
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, As, 0, k * 8)
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

# Couple the rectangle's output state in one retained contraction. Participant
# ownership can distribute contiguous K chains while reusing each activation
# leaf across columns; the source's original per-output cover remains admitted.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  matmul_rectangle(A, B, acc)

# Complete output fragments admit matrix ownership even when M or N is below
# eight. Padding belongs only to unpublished rows/columns. K is never padded:
# complete 8-wide contractions use the existing F32 matrix cover and the exact
# remaining source leaves keep their scalar FMA order.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for i in range((M + 7) / 8):
    for j in range((N + 7) / 8):
      Cs = tile[8, 8] f32
      for r, s in owned(Cs):
        if i * 8 + r < M and j * 8 + s < N:
          Cs[r, s] = acc[i * 8 + r, j * 8 + s]
        else:
          Cs[r, s] = 0.0
      c = simdgroup_matrix(f32)
      simdgroup_load(c, Cs, 0, 0)
      for k in range(K / 8):
        As = tile[8, 8] f32
        Bs = tile[8, 8] f32
        for r, q in owned(As):
          if i * 8 + r < M: As[r, q] = A[i * 8 + r, k * 8 + q]
          else: As[r, q] = 0.0
        for r, q in owned(Bs):
          if j * 8 + r < N: Bs[r, q] = B[j * 8 + r, k * 8 + q]
          else: Bs[r, q] = 0.0
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, As, 0, 0)
        simdgroup_load_t(b, Bs, 0, 0)
        simdgroup_multiply_accumulate(c, a, b, c)
      simdgroup_store(c, Cs, 0, 0)
      for r, s in owned(Cs):
        if i * 8 + r < M and j * 8 + s < N:
          acc[i * 8 + r, j * 8 + s] = Cs[r, s]
  for i, j in owned(acc):
    for k in range(8 * (K / 8), K):
      acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# A bounded piece can stage its left rows once across all output fragments,
# including partial fragments. The enclosing stream choice supplies K.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for i in range((M + 7) / 8):
    As = tile[8, K] f32
    for r, q in owned(As):
      if i * 8 + r < M: As[r, q] = A[i * 8 + r, q]
      else: As[r, q] = 0.0
    for j in range((N + 7) / 8):
      Bs = tile[8, K] f32
      for r, q in owned(Bs):
        if j * 8 + r < N: Bs[r, q] = B[j * 8 + r, q]
        else: Bs[r, q] = 0.0
      Cs = tile[8, 8] f32
      for r, s in owned(Cs):
        if i * 8 + r < M and j * 8 + s < N: Cs[r, s] = acc[i * 8 + r, j * 8 + s]
        else: Cs[r, s] = 0.0
      c = simdgroup_matrix(f32)
      simdgroup_load(c, Cs, 0, 0)
      for k in range(K / 8):
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, As, 0, k * 8)
        simdgroup_load_t(b, Bs, 0, k * 8)
        simdgroup_multiply_accumulate(c, a, b, c)
      simdgroup_store(c, Cs, 0, 0)
      for r, s in owned(Cs):
        if i * 8 + r < M and j * 8 + s < N: acc[i * 8 + r, j * 8 + s] = Cs[r, s]
  for i, j in owned(acc):
    for k in range(8 * (K / 8), K): acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# Complementary row reuse: the prepared right rows survive all activation-row
# fragments. This remains a source strategy with ordinary storage choices.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for j in range((N + 7) / 8):
    Bs = tile[8, K] f32
    for r, q in owned(Bs):
      if j * 8 + r < N: Bs[r, q] = B[j * 8 + r, q]
      else: Bs[r, q] = 0.0
    for i in range((M + 7) / 8):
      As = tile[8, K] f32
      for r, q in owned(As):
        if i * 8 + r < M: As[r, q] = A[i * 8 + r, q]
        else: As[r, q] = 0.0
      Cs = tile[8, 8] f32
      for r, s in owned(Cs):
        if i * 8 + r < M and j * 8 + s < N: Cs[r, s] = acc[i * 8 + r, j * 8 + s]
        else: Cs[r, s] = 0.0
      c = simdgroup_matrix(f32)
      simdgroup_load(c, Cs, 0, 0)
      for k in range(K / 8):
        a = simdgroup_matrix(f32)
        b = simdgroup_matrix(f32)
        simdgroup_load(a, As, 0, k * 8)
        simdgroup_load_t(b, Bs, 0, k * 8)
        simdgroup_multiply_accumulate(c, a, b, c)
      simdgroup_store(c, Cs, 0, 0)
      for r, s in owned(Cs):
        if i * 8 + r < M and j * 8 + s < N: acc[i * 8 + r, j * 8 + s] = Cs[r, s]
  for i, j in owned(acc):
    for k in range(8 * (K / 8), K): acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# Retain a 2x2 rectangle of output fragments through the contraction. Each
# prepared activation/weight fragment feeds two products before reuse of its
# bounded staging tile. This cover neither pads K nor changes operand precision.
lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for i in range((M + 15) / 16):
    for j in range((N + 15) / 16):
      C00 = tile[8, 8] f32
      for r, s in owned(C00):
        if i * 16 + 0 + r < M and j * 16 + 0 + s < N: C00[r, s] = acc[i * 16 + 0 + r, j * 16 + 0 + s]
        else: C00[r, s] = 0.0
      c00 = simdgroup_matrix(f32)
      simdgroup_load(c00, C00, 0, 0)
      C01 = tile[8, 8] f32
      for r, s in owned(C01):
        if i * 16 + 0 + r < M and j * 16 + 8 + s < N: C01[r, s] = acc[i * 16 + 0 + r, j * 16 + 8 + s]
        else: C01[r, s] = 0.0
      c01 = simdgroup_matrix(f32)
      simdgroup_load(c01, C01, 0, 0)
      C10 = tile[8, 8] f32
      for r, s in owned(C10):
        if i * 16 + 8 + r < M and j * 16 + 0 + s < N: C10[r, s] = acc[i * 16 + 8 + r, j * 16 + 0 + s]
        else: C10[r, s] = 0.0
      c10 = simdgroup_matrix(f32)
      simdgroup_load(c10, C10, 0, 0)
      C11 = tile[8, 8] f32
      for r, s in owned(C11):
        if i * 16 + 8 + r < M and j * 16 + 8 + s < N: C11[r, s] = acc[i * 16 + 8 + r, j * 16 + 8 + s]
        else: C11[r, s] = 0.0
      c11 = simdgroup_matrix(f32)
      simdgroup_load(c11, C11, 0, 0)
      for k in range(K / 8):
        A0 = tile[8, 8] f32
        for r, q in owned(A0):
          if i * 16 + 0 + r < M: A0[r, q] = A[i * 16 + 0 + r, k * 8 + q]
          else: A0[r, q] = 0.0
        a0 = simdgroup_matrix(f32)
        simdgroup_load(a0, A0, 0, 0)
        A1 = tile[8, 8] f32
        for r, q in owned(A1):
          if i * 16 + 8 + r < M: A1[r, q] = A[i * 16 + 8 + r, k * 8 + q]
          else: A1[r, q] = 0.0
        a1 = simdgroup_matrix(f32)
        simdgroup_load(a1, A1, 0, 0)
        B0 = tile[8, 8] f32
        for r, q in owned(B0):
          if j * 16 + 0 + r < N: B0[r, q] = B[j * 16 + 0 + r, k * 8 + q]
          else: B0[r, q] = 0.0
        b0 = simdgroup_matrix(f32)
        simdgroup_load_t(b0, B0, 0, 0)
        B1 = tile[8, 8] f32
        for r, q in owned(B1):
          if j * 16 + 8 + r < N: B1[r, q] = B[j * 16 + 8 + r, k * 8 + q]
          else: B1[r, q] = 0.0
        b1 = simdgroup_matrix(f32)
        simdgroup_load_t(b1, B1, 0, 0)
        simdgroup_multiply_accumulate(c00, a0, b0, c00)
        simdgroup_multiply_accumulate(c01, a0, b1, c01)
        simdgroup_multiply_accumulate(c10, a1, b0, c10)
        simdgroup_multiply_accumulate(c11, a1, b1, c11)
      simdgroup_store(c00, C00, 0, 0)
      for r, s in owned(C00):
        if i * 16 + 0 + r < M and j * 16 + 0 + s < N: acc[i * 16 + 0 + r, j * 16 + 0 + s] = C00[r, s]
      simdgroup_store(c01, C01, 0, 0)
      for r, s in owned(C01):
        if i * 16 + 0 + r < M and j * 16 + 8 + s < N: acc[i * 16 + 0 + r, j * 16 + 8 + s] = C01[r, s]
      simdgroup_store(c10, C10, 0, 0)
      for r, s in owned(C10):
        if i * 16 + 8 + r < M and j * 16 + 0 + s < N: acc[i * 16 + 8 + r, j * 16 + 0 + s] = C10[r, s]
      simdgroup_store(c11, C11, 0, 0)
      for r, s in owned(C11):
        if i * 16 + 8 + r < M and j * 16 + 8 + s < N: acc[i * 16 + 8 + r, j * 16 + 8 + s] = C11[r, s]
  for i, j in owned(acc):
    for k in range(8 * (K / 8), K): acc[i, j] = fma(A[i, k], B[j, k], acc[i, j])

# The portable fold retains source-selected contiguous partial chains and merge
# trees for participant ownership selection, including arbitrary packed tails.
lower matmul: portable
