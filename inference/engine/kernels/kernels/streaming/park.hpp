// Expert streaming's waits, on every backend, over the backend's shim
// (streaming/contract.hpp), which includes it: the exchange's one wait rule
// (streaming_words.seismic, `slot_answered` in routed.seismic). Each reads
// host-shared words by acquire loads (the backend's
// `streaming_load_u64_acquire`), so what the host wrote before the answer
// is visible to every read after it. How a backend waits is its one fact
// (`SEISMIC_PARKS_IN_KERNELS`, which every native's prefix states from
// `BackendName::parks_in_kernels`):
// - Metal and CUDA (1): the wait spins in its kernel until the rule holds;
//   their devices see host writes while work runs.
// - Vulkan (0): the call waits at a park before it, a submission boundary
//   the host opens once the rule holds (the Vulkan spec makes no host write
//   visible to running work), so the wait reads its slot once.
// Every request is answered while the host lives (a stuck host is the host
// watchdog's to abort). A wait that spins also ends, as on its failure,
// once its slot's SLOT_PULSE, which the host advances while it lives, has
// stood still for PULSE_SPINS spins (streaming_words.seismic): a host that
// is gone leaves no kernel spinning. A wait that ends so stores its step as
// the slot's SLOT_EXPIRED, the one word the device writes for the host to
// learn it, and the host interrupts exactly that step. A wait answered at
// its first look reads no pulse; one that spins reads it once every
// STREAMING_PULSE_READ spins, by the acquire load its answer takes, and
// counts the spins between in full. A live host's late pulse (the host's
// loaded) can end a wait too: the host then interrupts the step, which is
// re-run from its accepted state (`exchange::Pulse`).

#define STREAMING_PULSE_READ 64

#if SEISMIC_PARKS_IN_KERNELS
// Spin `spins` of a spinning wait of step `q` on the slot at `slot` (its
// words `stride` apart), by the wait's one thread: whether its host is
// gone, the slot's SLOT_PULSE read every STREAMING_PULSE_READ spins (from
// spin 0, which only reads it) and still for PULSE_SPINS of them, `pulse`
// and `still` the wait's own record of it. A wait it ends has stored its
// step as the slot's SLOT_EXPIRED. Every spinning wait's give-up is this
// one (`streaming_await_slot`, `streaming_land_await`).
STREAMING_FN bool streaming_host_gone(STREAMING_SYSTEM_WORDS slot, u64 stride, u64 q, u32 spins,
    STREAMING_THREAD int &pulse, STREAMING_THREAD u32 &still) {
    if (spins % STREAMING_PULSE_READ != 0)
        return false;
    const int now = streaming_load_acquire(STREAMING_AT(slot, SEISMIC_CONST_SLOT_PULSE * stride));
    if (spins == 0 || now != pulse) {
        pulse = now;
        still = 0;
        return false;
    }
    if ((still += STREAMING_PULSE_READ) < u32(SEISMIC_CONST_PULSE_SPINS))
        return false;
    streaming_store_u64_release(STREAMING_AT(slot, SEISMIC_CONST_SLOT_EXPIRED * stride), q);
    return true;
}
#endif

// Step `q`'s wait on the slot at `slot` (its first word, its words `stride`
// apart): until its request is not q or its answer is q or q | SLOT_FAILED,
// or a spinning wait's host is gone (`streaming_host_gone`). True when the
// slot holds q's answer: neither its failure nor no request, nor a wait
// whose host is gone. One thread calls it.
STREAMING_FN bool streaming_await_slot(STREAMING_SYSTEM_WORDS slot, u64 stride, u64 q) {
    const u64 failed = q | (u64(u32(SEISMIC_CONST_SLOT_FAILED)) << 32);
#if SEISMIC_PARKS_IN_KERNELS
    int pulse = 0;
    u32 still = 0;
#endif
    for (u32 spins = 0;; ++spins) {
        const u64 answer = streaming_load_u64_acquire(STREAMING_AT(slot, SEISMIC_CONST_SLOT_ANSWER * stride));
        if (answer == q || answer == failed
            || streaming_load_u64_acquire(STREAMING_AT(slot, SEISMIC_CONST_SLOT_REQUEST * stride)) != q)
            return answer == q;
#if SEISMIC_PARKS_IN_KERNELS
        if (streaming_host_gone(slot, stride, q, spins, pulse, still))
            return false;
#else
        return false;
#endif
    }
}

// Store step `q` as the request of the slot at `slot` (its words `stride`
// apart): the slot's SLOT_VIOLATED cleared (a violation stored for an earlier
// step is never read as a later step's), then one 64-bit release store,
// after every earlier store of the calling thread's.
STREAMING_FN void streaming_request_slot(STREAMING_SYSTEM_WORDS slot, u64 stride, u64 q) {
    streaming_store(STREAMING_AT(slot, SEISMIC_CONST_SLOT_VIOLATED * stride), 0);
    streaming_store_u64_release(STREAMING_AT(slot, SEISMIC_CONST_SLOT_REQUEST * stride), q);
}

// The step a step pair holds (STEP_LOW, STEP_HIGH at `step`, `stride` words
// apart), the device's own word.
STREAMING_FN u64 streaming_step(STREAMING_DEVICE_READ step, u64 stride) {
    return (u64(u32(streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_HIGH * stride)))) << 32) |
           u64(u32(streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * stride))));
}

// The record a map's row holds for step `q` (streaming_words.seismic's map,
// `held` in routed.seismic): `row` the row's first word, `stride` its words'
// stride (1: a row's words are contiguous, so each step is one aligned u64).
// UNTIL is read first as one 64-bit acquire load, then FROM, then the
// record: the host stores the record, then FROM, then UNTIL last
// (`experts::map`), so a step that reads an entry's new UNTIL reads its new
// FROM and record, and one that reads the old UNTIL reads a range its step
// is past or a record that holds its expert.
STREAMING_FN int streaming_held(STREAMING_SYSTEM_READ row, u64 stride, u64 q) {
    const u64 until = streaming_load_u64_acquire(STREAMING_AT(row, SEISMIC_CONST_MAP_UNTIL * stride));
    const u64 from = streaming_load_u64_acquire(STREAMING_AT(row, SEISMIC_CONST_MAP_FROM * stride));
    if (q < from || q >= until)
        return -1;
    return streaming_load_acquire(STREAMING_AT(row, SEISMIC_CONST_MAP_RECORD * stride));
}
