// Seismic native library (Metal): `Out` over memory other threadgroups or the host read while a
// kernel runs. `#include <seismic/coherent.h>`; `coherent(...)` address qualifiers need internals.
#pragma METAL internals : enable
#include <seismic/element.h>

namespace element {

// `CoherentOut` is
// `coherent(device)` (grid-wide phases), `SystemOut` is `volatile coherent(system)` (host-mapped).
// Writes are the owner's, as `Out`'s.
template <typename E> struct CoherentOut {
    coherent(device) device Cell<E> *cells;
    explicit CoherentOut(coherent(device) device Cell<E> *cells) : cells(cells) {}
    float at(ulong index) const { return E::load(cells[index].stored); }
    void put(ulong index, float value) const { cells[index].stored = E::store(value); }
};

template <typename E> struct SystemOut {
    volatile coherent(system) device Cell<E> *cells;
    explicit SystemOut(volatile coherent(system) device Cell<E> *cells) : cells(cells) {}
    void put(ulong index, float value) const { cells[index].stored = E::store(value); }
};

// The same tensor through `coherent(device)` memory (reads and writes other threadgroups see).
template <typename E> CoherentOut<E> coherent_device(Out<E> out) {
    return CoherentOut<E>(reinterpret_cast<coherent(device) device Cell<E> *>(out.cells));
}

// The same tensor through host-visible `volatile coherent(system)` memory.
template <typename E> SystemOut<E> system(Out<E> out) {
    return SystemOut<E>(reinterpret_cast<volatile coherent(system) device Cell<E> *>(out.cells));
}
} // namespace element
