// Address a component region in a slab-stored tensor. Each table entry is a
// device address already offset to that component's region.
namespace slab {
template <typename T>
inline device T *region(device const ulong *table, ulong slab_index) {
    return reinterpret_cast<device T *>(table[slab_index]);
}

template <typename T>
inline device T *row(device const ulong *table, ulong index, ulong rows_per_slab,
    ulong row_elements) {
    return region<T>(table, index / rows_per_slab)
        + (index % rows_per_slab) * row_elements;
}
} // namespace slab
