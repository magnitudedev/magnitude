// Address a component region in a slab-stored tensor, or a record of a
// record port. Each table entry is a device address already offset to that
// component's region, or to that record.
//
// A port's records lie in an address space (`space`): `device`, or, for a
// record parameter the generated prefix marks host-written
// (`<PORT>_HOST_WRITTEN`: records the host lands while work that reads them
// runs), `volatile coherent(system) device`, whose loads read what the host
// last wrote where plain loads within one command buffer may not
// (`BackendName::plain_loads_read_host_writes`). `SLAB_SPACE(P)` is port P's.
// `as<T>(p)` reads p's bytes as T in p's space.

#include <seismic/element.h>

// `coherent(system)`.
#pragma METAL internals : enable

namespace slab {

template <bool HOST_WRITTEN>
struct space {
    template <typename T> using pointer = device T *;
};
template <>
struct space<true> {
    template <typename T> using pointer = volatile coherent(system) device T *;
};

#define SLAB_SPACE(P) slab::space<ELEMENT_HAS(P, _HOST_WRITTEN)>

// The pointer to T in the space and constness of byte pointer P.
template <typename T, typename P> struct rebound;
template <typename T> struct rebound<T, device uchar *> { typedef device T *type; };
template <typename T> struct rebound<T, device const uchar *> { typedef device const T *type; };
template <typename T> struct rebound<T, volatile coherent(system) device uchar *> {
    typedef volatile coherent(system) device T *type;
};
template <typename T> struct rebound<T, volatile coherent(system) device const uchar *> {
    typedef volatile coherent(system) device const T *type;
};

template <typename T, typename P>
inline typename rebound<T, P>::type as(P bytes) {
    return reinterpret_cast<typename rebound<T, P>::type>(bytes);
}

template <typename T, typename S = space<false>>
inline typename S::template pointer<T> region(device const ulong *table, ulong slab_index) {
    return reinterpret_cast<typename S::template pointer<T>>(table[slab_index]);
}

template <typename T>
inline device T *row(device const ulong *table, ulong index, ulong rows_per_slab,
    ulong row_elements) {
    return region<T>(table, index / rows_per_slab)
        + (index % rows_per_slab) * row_elements;
}
} // namespace slab
