"""Contiguous borrowed-buffer views preserve their physical origin."""

import tilelang.language as T


def reshape_buffer(source, shape):
    # Keep the public view's size validation, but retain the original origin.
    # T.view currently shares data while dropping a symbolic element offset.
    view = T.view(source, shape=shape)
    return T.decl_buffer(
        shape,
        source.dtype,
        data=view.data,
        elem_offset=source.elem_offset,
        scope=source.scope(),
    )


def rebase_buffer(source):
    """Inside a kernel, resolve a borrowed origin before vectorized indexing."""
    # make_tensor materializes a mutable pointer. Express that access explicitly:
    # a read-only address currently lowers to const T* assigned to void* on CUDA.
    return T.make_tensor(T.access_ptr(source, "rw"), source.shape, source.dtype)
