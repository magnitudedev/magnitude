# CUDA graph capture stalled during Windows GPU inference validation. Keep the
# Windows pack on the verified ordinary execution path until the engine changes.
set(GGML_CUDA_GRAPHS OFF CACHE BOOL "Use CUDA graphs" FORCE)
