// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`): E is the GGUF source, U the resident (representation, layout).
// The entry sees every weight as a [B, N, K] view.
//
// The conversion is data: its word form (`seismic_lang::repack::WordTable`,
// whose docs give the table's format), emitted as SEISMIC_CONVERSION_E_U_*,
// gives every 32-bit word of a source packet's record as the OR of pieces,
// runs of source bits or bits of values its postfix ops compute. The record
// is U's packet form (`packet`), or the packet's storage group's share of
// each row plane (row layouts). `repack::unit_record` and `repack::fixed_record`
// (`<seismic/repack.cuh>`) execute it over this kernel's readers of source packets; the rest of this
// kernel loads, and places records in U's layout.
//
// Each block, one warp, places one row tile: 16 rows x 32 columns, a lane
// per row (the other 16 lanes help stage and store), so the warp runs one
// stream of record words (only their data differ). Every group is a
// multiple of 32 columns, so a tile is one unit of the code planes
// (`WordSpan`): its words are its class's representative's, their runs moved
// by the unit's displacements. The tile at a group's first column also
// places the group's other record words. Stored-but-unoccupied bytes (mma16
// padding rows and groups, plane alignment tails, packet gaps) are written
// as zero, so the result is fully defined.

#if !defined(SEISMIC_ELEMENT_E_KIND_EXTERNAL) || !defined(SEISMIC_ELEMENT_U_KIND_PACKED)
#error "repack_weight converts an external GGUF source into packed resident storage"
#endif

#if !defined(SEISMIC_CONVERSION_E_U)
#error "repack_weight binding is not a registered GGUF-to-resident conversion"
#endif

#if !defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET) && !defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16) && \
    !defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
#error "repack_weight writes packet, rows16 or mma16 storage"
#endif

#include <seismic/repack.cuh>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define GROUP ((u32)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
#define TILE_ROWS 16u
#define LANES 32u
#define TILE_COLUMNS 32u
#define UNITS (GROUP / TILE_COLUMNS)
static_assert(GROUP % TILE_COLUMNS == 0, "a tile lies in one group");

// A unit's code words: the codes', then the high codes' (`WordSpan`).
#define CODES_WORDS SEISMIC_CONVERSION_E_U_CODES_UNIT_WORDS
#define HIGH_WORDS SEISMIC_CONVERSION_E_U_HIGH_UNIT_WORDS
#define UNIT_WORDS (CODES_WORDS + HIGH_WORDS)

// Staged units a lane loads before storing any.
#define STAGE_BATCH 16u

// The conversion's word form, decoded only through `<seismic/repack.cuh>`.
__device__ constexpr u32 conversion[] = {SEISMIC_CONVERSION_E_U_TABLE};
__device__ constexpr repack::WordForm form = {
    conversion,
    SEISMIC_CONVERSION_E_U_STACK,
    SEISMIC_CONVERSION_E_U_DIRECTORY,
    SEISMIC_CONVERSION_E_U_CLASSES,
};

// ---- layouts' fixed words --------------------------------------------------
//
// The record words a group's first tile places besides its unit's: at most
// two ranges, their source bits the table's own.

struct Words {
    u32 first, count;
};

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)
#define CODES_WORD SEISMIC_CONVERSION_E_U_CODES_FIRST
#define CODES_END (CODES_WORD + UNITS * CODES_WORDS)
static_assert(CODES_END * 4 <= SEISMIC_ELEMENT_U_PACKET_SIZE, "the codes are whole words of a packet");
// The words before the codes, and those after the group's.
__device__ constexpr Words fixed_words(u32 range) {
    return range == 0 ? Words{0, CODES_WORD} : Words{CODES_END, SEISMIC_CONVERSION_E_U_WORDS - CODES_END};
}
#else
#define ROLE_WORD(role) SEISMIC_CONVERSION_E_U##_##role##_WORD
#define ROLE_BYTES(role) SEISMIC_CONVERSION_E_U##_##role##_BYTES
#define ROLE_WORDS(role) ((ROLE_BYTES(role) + 3u) / 4u)
#if defined(SEISMIC_ELEMENT_U_PLANE_RECORDS)
// The codes role is a group's entries (its units' words), then its factors.
#define ENTRIES_BYTES (GROUP * SEISMIC_ELEMENT_U_PLANE_RECORDS_CODE_BITS / 8u)
#define FACTOR_BYTES (ROLE_BYTES(CODES) - ENTRIES_BYTES)
#define FACTORS_WORD (ROLE_WORD(CODES) + ENTRIES_BYTES / 4u)
static_assert(SEISMIC_CONVERSION_E_U_CODES_FIRST + UNITS * CODES_WORDS == FACTORS_WORD,
              "a records share is its units' entries, then its factors");
__device__ constexpr Words fixed_words(u32 range) {
    return range == 0 ? Words{FACTORS_WORD, (FACTOR_BYTES + 3) / 4} : Words{0, 0};
}
#else
// The group planes' shares.
__device__ constexpr Words fixed_words(u32 range) {
    return range == 0 ? Words{ROLE_WORD(SCALES), ROLE_WORDS(SCALES)} : Words{ROLE_WORD(SUPERS), ROLE_WORDS(SUPERS)};
}
#endif
#endif

// ---- tile plans -------------------------------------------------------------
//
// A tile's words (its unit's, and at a group's first tile the fixed words)
// read some of its rows' source bytes, which the warp stages, coalesced, as
// the conversion's tile plans lay them out (`SEISMIC_CONVERSION_E_U_PLAN_*`,
// `_RUN_*`, `_FIXED_POSITIONS`, `_STAGE_WORDS`, whose ABI docs give them):
// plan C for a unit of class C, plan FIXED_PLAN for the fixed words. The
// words read their runs there (`Reader`, `FixedReader`).

#define FIXED_PLAN ((u32)SEISMIC_CONVERSION_E_U_CLASS_COUNT)
// A staged row's words, padded to odd (`odd_words`).
#define STAGE_WORDS odd_words((u32)SEISMIC_CONVERSION_E_U_STAGE_WORDS)

// A segment: its part, first (representative) byte, length, staged
// position and copy unit.
struct Segment {
    u32 part, first, length, position, unit;
};

__device__ constexpr Segment segments[] = {SEISMIC_CONVERSION_E_U_PLAN_SEGMENTS};
__device__ constexpr u32 plan_segments[] = {SEISMIC_CONVERSION_E_U_PLAN_SEGMENT_FIRST};
__device__ constexpr u32 run_positions[] = {SEISMIC_CONVERSION_E_U_RUN_POSITIONS};
__device__ constexpr u32 class_runs[] = {SEISMIC_CONVERSION_E_U_RUN_FIRST};
__device__ constexpr u32 fixed_positions[] = {SEISMIC_CONVERSION_E_U_FIXED_POSITIONS};

// The units a lane stages of a segment of `count` units a row: the tile's
// rows' units dealt to the lanes in turn.
__host__ __device__ constexpr u32 lane_units(u32 count) { return (TILE_ROWS * count + LANES - 1) / LANES; }

// The units a lane stages of segment `g`.
__device__ constexpr u32 segment_units(u32 g) { return lane_units(segments[g].length / segments[g].unit); }

// The units a lane stages of plan P.
__device__ constexpr u32 plan_units(u32 p) {
    u32 units = 0;
    for (u32 g = plan_segments[p]; g < plan_segments[p + 1]; ++g) units += segment_units(g);
    return units;
}

// Staged unit U of plan P: its segment, and its index within the segment.
struct StagedUnit {
    u32 segment, within;
};
__device__ constexpr StagedUnit staged_unit(u32 p, u32 u) {
    u32 g = plan_segments[p];
    for (; u >= segment_units(g); ++g) u -= segment_units(g);
    return {g, u};
}

// `words` words padded to an odd number (each lane's row of a shared array of
// rows starts in its own bank).
__host__ __device__ constexpr u32 odd_words(u32 words) { return words % 2 == 0 ? words + 1 : words; }

// ---- the executor -----------------------------------------------------------

// LENGTH (1..32) bits of a staged row `row` from staged bit `at`: from the
// staged words holding them (one load where `at` is a constant within a
// word).
template <u32 LENGTH> __device__ __forceinline__ u32 staged_bits(const u32 *row, u32 at) {
    const u32 word = at / 32, shift = at % 32;
    const u32 high = shift + LENGTH > 32 ? row[word + 1] : 0u;
    return __funnelshift_r(row[word], high, shift) & repack::low_bits(LENGTH);
}

// LENGTH (1..32) bits of the BYTES staged bytes of `row` from byte `byte`,
// from bit `shift` of the first.
template <u32 BYTES, u32 LENGTH> __device__ __forceinline__ u32 staged_bytes(const u32 *row, u32 byte, u32 shift) {
    const u8 *bytes = reinterpret_cast<const u8 *>(row) + byte;
    if constexpr (BYTES <= 4) {
        u32 value = 0;
#pragma unroll
        for (u32 at = 0; at < BYTES; ++at) value |= u32(bytes[at]) << (8 * at);
        return (value >> shift) & repack::low_bits(LENGTH);
    } else {
        u64 value = 0;
#pragma unroll
        for (u32 at = 0; at < BYTES; ++at) value |= u64(bytes[at]) << (8 * at);
        return u32(value >> shift) & repack::low_bits(LENGTH);
    }
}

// One row's staged source bytes of a unit of class C, as plan C stages them:
// the runs its representative's block visits, each at its staged bits (the
// K-th run's), a bit-moved slot's shifted by the unit's bit displacement,
// and a member's moved by its `delta` bits more.
template <u32 C> struct Reader {
    const u32 *row;
    u32 displacements;

    __host__ __device__ static constexpr repack::WordForm form() { return ::form; }

    template <u32 AT, u32 K> __device__ __forceinline__ u32 run(int delta) const {
        constexpr u32 piece = repack::table_word(form(), AT), length = repack::piece_length(piece);
        constexpr u32 bytes = (repack::piece_first(piece) % 8 + length + 7) / 8;
        constexpr u32 from = run_positions[class_runs[C] + K];
        constexpr u32 entry = repack::table_word(form(), repack::class_slot_map(form(), C) + K);
        const u32 at = u32(int(from) + delta);
        if constexpr (repack::slot_still(entry) || repack::slot_whole(entry)) {
            // Unmoved, or moved by whole bytes: the representative's shift.
            return staged_bits<length>(row, at);
        } else {
            // Moved by bits too: the member's bytes, shifted by the bit
            // displacement more (it stays within them).
            return staged_bytes<bytes, length>(
                row, at / 8, at % 8 + repack::displacement_bits(form(), displacements, repack::slot_of(entry)));
        }
    }
};

// One row's staged source bytes of the fixed words, as plan FIXED_PLAN
// stages them: each run at its bytes' staged positions, a member's moved by
// its `delta` bits.
struct FixedReader {
    const u32 *row;

    __host__ __device__ static constexpr repack::WordForm form() { return ::form; }

    template <u32 AT, u32 K> __device__ __forceinline__ u32 run(int delta) const {
        constexpr u32 piece = repack::table_word(form(), AT), bit = repack::piece_first(piece);
        constexpr u32 length = repack::piece_length(piece);
        constexpr u32 from = 8 * fixed_positions[bit / 8] + bit % 8;
        return staged_bits<length>(row, u32(int(from) + delta));
    }
};

// ---- a warp's tile ----------------------------------------------------------

// Source packet `packet` of row `n` of matrix `b`.
__device__ __forceinline__ const u8 *source_packet(const u8 *source, const seismic_words_t &seismic_words_value, u64 b,
                                                   u64 n, u64 packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

// UNIT bytes of `words` from byte AT, at `out + AT`.
template <u32 UNIT, u32 AT> __device__ __forceinline__ void store_unit(u8 *out, const u32 *words) {
    constexpr u32 word = AT / 4;
    if constexpr (UNIT == 16) {
        *reinterpret_cast<uint4 *>(out + AT) = make_uint4(words[word], words[word + 1], words[word + 2], words[word + 3]);
    } else if constexpr (UNIT == 8) {
        *reinterpret_cast<uint2 *>(out + AT) = make_uint2(words[word], words[word + 1]);
    } else if constexpr (UNIT == 4) {
        *reinterpret_cast<u32 *>(out + AT) = words[word];
    } else if constexpr (UNIT == 2) {
        *reinterpret_cast<u16 *>(out + AT) = u16(words[word] >> (AT % 4 * 8));
    } else {
        out[AT] = u8(words[word] >> (AT % 4 * 8));
    }
}

// Stores unit I of UNIT bytes of `words` at `out`.
template <u32 UNIT> struct StoreUnit {
    u8 *out;
    const u32 *words;
    template <typename I> __device__ __forceinline__ void operator()(I) const {
        store_unit<UNIT, I::value * UNIT>(out, words);
    }
};

// BYTES bytes of `words` (little-endian) at `out`, aligned to ALIGN, in the
// widest stores both allow.
template <u32 BYTES, u32 ALIGN = repack::alignment(BYTES)> __device__ __forceinline__ void store(u8 *out, const u32 *words) {
    constexpr u32 unit = repack::alignment(BYTES) < ALIGN ? repack::alignment(BYTES) : ALIGN;
    repack::each<0, BYTES / unit>(StoreUnit<unit>{out, words});
}

// The widest copy unit, to 4 bytes, that offsets and lengths `bytes`
// allow.
template <typename... Bytes> __host__ __device__ constexpr u32 copy_unit(Bytes... bytes) {
    u32 unit = 4;
    ((unit = repack::alignment(bytes) < unit ? repack::alignment(bytes) : unit), ...);
    return unit;
}

// UNIT bytes from `from` to `to`, both aligned to UNIT.
template <u32 UNIT> __device__ __forceinline__ void copy(u8 *to, const u8 *from) {
    if constexpr (UNIT == 4) *reinterpret_cast<u32 *>(to) = *reinterpret_cast<const u32 *>(from);
    else if constexpr (UNIT == 2) *reinterpret_cast<u16 *>(to) = *reinterpret_cast<const u16 *>(from);
    else *to = *from;
}

// A warp's tile: rows row0.. of matrix b at source packet `packet`
// (`occupied` when the tile holds source columns), unit `unit` of the
// group, staging in `stage`.
struct Tile {
    const u8 *source;
    const seismic_words_t &seismic_words_value;
    u64 b, row0;
    u32 packet, unit, lane;
    bool occupied;
    u32 *stage;

    // Whether row `r` of the tile holds a source packet.
    __device__ __forceinline__ bool real(u32 r) const { return r < TILE_ROWS && occupied && row0 + r < SEISMIC_DIM_N; }

    // The table word of the unit's displacements (`unit_displacements`).
    __device__ __forceinline__ u32 displacements() const { return repack::unit_displacements(form, unit); }

    // Lane `lane`'s staged unit U of plan P: at = lane + 32 i of the tile's
    // rows' units of its segment, row at / units, unit at % units; its source
    // and staged addresses, and whether it is one of a real row.
    template <u32 P, u32 U> __device__ __forceinline__ bool locate(const u8 *&from, u8 *&to) const {
        constexpr StagedUnit staged = staged_unit(P, U);
        constexpr Segment segment = segments[staged.segment];
        constexpr u32 count = segment.length / segment.unit, i = staged.within;
        const u32 at = lane + LANES * i, r = at / count, k = at % count;
        const int moved = segment.part == 0 ? 0 : repack::displacement_bytes(form, displacements(), segment.part - 1);
        from = source_packet(source, seismic_words_value, b, row0, packet) +
               r * (u64(SEISMIC_SOURCE_STRIDE_1) * SEISMIC_ELEMENT_E_PACKET_SIZE) + moved + segment.first +
               k * segment.unit;
        to = reinterpret_cast<u8 *>(stage) + r * 4 * STAGE_WORDS + segment.position + k * segment.unit;
        return real(r);
    }

    template <u32 P, u32 U> __device__ __forceinline__ void load(u32 &value) const {
        constexpr u32 size = segments[staged_unit(P, U).segment].unit;
        const u8 *from;
        u8 *to;
        if (locate<P, U>(from, to)) {
            if constexpr (size == 4) value = *reinterpret_cast<const u32 *>(from);
            else if constexpr (size == 2) value = *reinterpret_cast<const u16 *>(from);
            else value = *from;
        }
    }

    template <u32 P, u32 U> __device__ __forceinline__ void store(u32 value) const {
        constexpr u32 size = segments[staged_unit(P, U).segment].unit;
        const u8 *from;
        u8 *to;
        if (locate<P, U>(from, to)) {
            if constexpr (size == 4) *reinterpret_cast<u32 *>(to) = value;
            else if constexpr (size == 2) *reinterpret_cast<u16 *>(to) = u16(value);
            else *to = u8(value);
        }
    }

    // Loads staged unit U0 + I into `values[I]`.
    template <u32 P, u32 U0> struct StagedLoad {
        const Tile &tile;
        u32 *values;
        template <typename I> __device__ __forceinline__ void operator()(I) const {
            tile.template load<P, U0 + I::value>(values[I::value]);
        }
    };

    // Stores `values[I]` as staged unit U0 + I.
    template <u32 P, u32 U0> struct StagedStore {
        const Tile &tile;
        const u32 *values;
        template <typename I> __device__ __forceinline__ void operator()(I) const {
            tile.template store<P, U0 + I::value>(values[I::value]);
        }
    };

    // Staged units [U0, U0 + COUNT): loaded, then stored, so a batch's loads
    // are in flight together (STAGE_BATCH bounds the registers holding them).
    template <u32 P, u32 U0, u32 COUNT> __device__ __forceinline__ void stage_batch() const {
        u32 values[COUNT];
        repack::each<0, COUNT>(StagedLoad<P, U0>{*this, values});
        repack::each<0, COUNT>(StagedStore<P, U0>{*this, values});
    }

    // Stages batch I of plan P's units.
    template <u32 P> struct StageBatch {
        const Tile &tile;
        template <typename I> __device__ __forceinline__ void operator()(I) const {
            constexpr u32 total = plan_units(P), u0 = I::value * STAGE_BATCH;
            tile.template stage_batch<P, u0, (total - u0 < STAGE_BATCH ? total - u0 : STAGE_BATCH)>();
        }
    };

    // Plan P's bytes of every real row staged; row `lane`'s.
    template <u32 P> __device__ __forceinline__ const u32 *stage_rows() const {
        constexpr u32 total = plan_units(P);
        repack::each<0, (total + STAGE_BATCH - 1) / STAGE_BATCH>(StageBatch<P>{*this});
        __syncwarp();
        // (Lanes past the tile's rows read a row's, and place nothing.)
        return stage + lane % TILE_ROWS * STAGE_WORDS;
    }

    // Words of row `lane` (zero where its row is not real), staged by their
    // own plan: the UNIT_WORDS of the tile's unit, of class C, its class's
    // representative's.
    template <u32 C> __device__ __forceinline__ void unit_record(u32 *out) const {
        const Reader<C> r{stage_rows<C>(), displacements()};
        const bool here = real(lane);
        u32 words[UNIT_WORDS];
        repack::unit_record<Reader<C>, repack::class_representative(form, C), UNIT_WORDS>(r, words);
        for (u32 i = 0; i < UNIT_WORDS; ++i) out[i] = here ? words[i] : 0u;
    }

    // ... and at a group's first tile, after its unit's, the fixed words'
    // ranges one after the other (FIXED_WORDS).
    __device__ __forceinline__ void fixed_record(u32 *out) const {
        // The stage is the warp's: every lane is done with what it held.
        __syncwarp();
        const FixedReader r{stage_rows<FIXED_PLAN>()};
        const bool here = real(lane);
        constexpr Words first = fixed_words(0), second = fixed_words(1);
        u32 values[first.count + second.count + 1];
        repack::fixed_record<FixedReader, UNITS, first.first, first.count>(r, values);
        repack::fixed_record<FixedReader, UNITS, second.first, second.count>(r, values + first.count);
        for (u32 i = 0; i < first.count + second.count; ++i) out[i] = here ? values[i] : 0u;
    }
};

#define FIXED_WORDS (fixed_words(0).count + fixed_words(1).count)

// Places the tile's unit (`Place::place<C>`, C its class), then at a group's
// first tile the group's fixed words (`Place::place_fixed`).
template <typename Place> struct Unit {
    const Place &place;
    template <u32 C> __device__ __forceinline__ void operator()(repack::Index<C>) const {
        place.template place<C>();
    }
};
template <typename Place> __device__ __forceinline__ void place_tile(const Place &place, u32 unit) {
    repack::dispatch<0, SEISMIC_CONVERSION_E_U_CLASS_COUNT>(repack::unit_class(form, unit), Unit<Place>{place});
    if (unit == 0) place.place_fixed();
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

// ---- packet: the record is the packet form ---------------------------------

__device__ __forceinline__ u8 *destination_packet(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                                  u64 n, u64 packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

// A tile's record words are staged per row before being stored, in the
// source stage's memory: the unit's words, then (at the group's first tile)
// the fixed words.
#define RECORD_STRIDE odd_words(UNIT_WORDS + FIXED_WORDS)

// Rows' words computed, then staged, then stored coalesced into the rows'
// destination packets.
struct PacketPlace {
    const Tile &tile;
    u8 *out;

    // Packet bytes [low, low + BYTES) (`low` aligned to ALIGN), staged from
    // record byte SLOT on, of every real row into its destination packet
    // (rows SEISMIC_RESULT_0_STRIDE_1 packets apart).
    template <u32 SLOT, u32 BYTES, u32 ALIGN> __device__ __forceinline__ void store_bytes(u32 low) const {
        const seismic_words_t &seismic_words_value = tile.seismic_words_value;
        if constexpr (BYTES > 0) {
            constexpr u32 unit = copy_unit(ALIGN, BYTES, SLOT);
            constexpr u32 units = BYTES / unit;
            const u64 row_bytes = u64(SEISMIC_RESULT_0_STRIDE_1) * SEISMIC_ELEMENT_U_PACKET_SIZE;
#pragma unroll 4
            for (u32 at = tile.lane; at < TILE_ROWS * units; at += LANES) {
                const u32 r = at / units, k = at % units;
                if (tile.real(r))
                    copy<unit>(out + r * row_bytes + low + k * unit,
                               reinterpret_cast<const u8 *>(tile.stage + r * RECORD_STRIDE) + SLOT + k * unit);
            }
        }
    }

    // Row `lane`'s COUNT words, staged from record word SLOT on.
    template <u32 SLOT, u32 COUNT> __device__ __forceinline__ void stage_record(const u32 *words) const {
        __syncwarp();
        if (tile.lane < TILE_ROWS)
            for (u32 word = 0; word < COUNT; ++word) tile.stage[tile.lane * RECORD_STRIDE + SLOT + word] = words[word];
        __syncwarp();
    }

    template <u32 C> __device__ __forceinline__ void place() const {
        u32 words[UNIT_WORDS];
        tile.unit_record<C>(words);
        stage_record<0, UNIT_WORDS>(words);
        store_bytes<0, 4 * CODES_WORDS, copy_unit(SEISMIC_ELEMENT_U_PACKET_SIZE, 4 * CODES_WORD, 4 * CODES_WORDS)>(
            4 * (CODES_WORD + tile.unit * CODES_WORDS));
    }

    __device__ __forceinline__ void place_fixed() const {
        u32 words[FIXED_WORDS + 1];
        tile.fixed_record(words);
        stage_record<UNIT_WORDS, FIXED_WORDS>(words);
        constexpr Words before = fixed_words(0), after = fixed_words(1);
        constexpr u32 end = 4 * (after.first + after.count) < SEISMIC_ELEMENT_U_PACKET_SIZE
                                ? 4 * (after.first + after.count)
                                : SEISMIC_ELEMENT_U_PACKET_SIZE;
        store_bytes<4 * UNIT_WORDS, 4 * before.count, copy_unit(SEISMIC_ELEMENT_U_PACKET_SIZE)>(0);
        store_bytes<4 * (UNIT_WORDS + before.count), (end > 4 * after.first ? end - 4 * after.first : 0),
                    copy_unit(SEISMIC_ELEMENT_U_PACKET_SIZE, 4 * after.first)>(4 * after.first);
    }
};

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    // The warp's staged source bytes, then its record words.
    __shared__ u32 stage[TILE_ROWS * (STAGE_WORDS > RECORD_STRIDE ? STAGE_WORDS : RECORD_STRIDE)];
    const u32 lane = threadIdx.x;
    const u64 b = blockIdx.z;
    const u64 row0 = u64(blockIdx.y) * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    if (column0 >= packets * GROUP || row0 >= SEISMIC_DIM_N) return;
    const u32 packet = column0 / GROUP;
    const Tile tile{source, seismic_words_value, b, row0, packet, column0 % GROUP / TILE_COLUMNS, lane, true,
                    stage};
    place_tile(PacketPlace{tile, destination_packet(destination, seismic_words_value, b, row0, packet)}, tile.unit);
}

#else

// ---- rows16 / mma16: the record is the row planes' shares -------------------

// Start of stored row `n` of matrix `b`.
__device__ __forceinline__ u8 *row_base(u8 *destination, const seismic_words_t &seismic_words_value, u64 b, u64 n) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1) *
                             SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
}

// Rows of an mma16 tile: a row tile's.
#define MMA_ROWS ((u32)SEISMIC_ELEMENT_U_TILE_ROWS)
static_assert(TILE_ROWS % MMA_ROWS == 0, "a row tile holds whole mma16 tiles");

#if defined(SEISMIC_ELEMENT_U_PLANE_RECORDS)

// ---- mma16 records ----------------------------------------------------------
//
// An mma16 tile's 16 rows are one sequence (`PackedRowLayout::record_byte`):
// every stored group's entries, then every group's factors, then zero to the
// tile's end. Entries are 16 bits, in GEMV lane order: entry `e` of row `r`
// of a group, q = e / 8, at byte q / 2 * 512 + (4 (r % 8) + e % 4) * 16 +
// (2 (q % 2) + r / 8) * 4 + 2 (e % 8 / 4); a group's factors are
// [r % 8][r / 8][factor bytes]. A unit's words are two entries each.

#define ENTRY_BITS 16u
#define ENTRY_BYTES ((u64)SEISMIC_ELEMENT_U_PLANE_RECORDS_ENTRY_BYTES)
static_assert(ENTRY_BYTES == MMA_ROWS * ENTRIES_BYTES, "a tile's group entries are its rows' entries");
static_assert(HIGH_WORDS == 0, "a records unit is its entries");

// Entry `e` of row `r` of a group (`value`) at the group's entries.
__device__ __forceinline__ void place_entry(u8 *entries, u32 r, u32 e, u32 value) {
    const u32 q = e / 8;
    *reinterpret_cast<u16 *>(entries + q / 2 * 512 + (4 * (r % 8) + e % 4) * 16 + (2 * (q % 2) + r / 8) * 4 +
                             2 * (e % 8 / 4)) = u16(value);
}

struct RecordsPlace {
    const Tile &tile;
    u8 *destination;
    u64 rows;

    // Row `lane`'s mma16 tile.
    __device__ __forceinline__ u8 *mma() const {
        return row_base(destination, tile.seismic_words_value, tile.b, tile.row0 + tile.lane - tile.lane % MMA_ROWS);
    }

    template <u32 C> __device__ __forceinline__ void place() const {
        u32 words[UNIT_WORDS];
        tile.unit_record<C>(words);
        if (tile.lane >= rows) return;
        const u32 r = tile.lane % MMA_ROWS;
        u8 *entries = mma() + tile.packet * ENTRY_BYTES;
        for (u32 word = 0; word < UNIT_WORDS; ++word) {
            const u32 e = 2 * (tile.unit * UNIT_WORDS + word);
            place_entry(entries, r, e, words[word]);
            place_entry(entries, r, e + 1, words[word] >> ENTRY_BITS);
        }
    }

    __device__ __forceinline__ void place_fixed() const {
        const seismic_words_t &seismic_words_value = tile.seismic_words_value;
        u32 words[FIXED_WORDS + 1];
        tile.fixed_record(words);
        if (tile.lane >= rows) return;
        const u32 r = tile.lane % MMA_ROWS;
        store<FACTOR_BYTES>(mma() + SEISMIC_RESULT_0_ROW_GROUPS * ENTRY_BYTES + tile.packet * (MMA_ROWS * FACTOR_BYTES) +
                                (2 * (r % 8) + r / 8) * FACTOR_BYTES,
                            words);
    }
};

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    __shared__ u32 stage[TILE_ROWS * STAGE_WORDS];
    const u32 lane = threadIdx.x;
    const u64 b = blockIdx.z;
    const u64 row0 = u64(blockIdx.y) * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const u64 stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
    // Stored rows: whole mma16 tiles, rows beyond N zero.
    const u64 stored_rows = (SEISMIC_DIM_N + MMA_ROWS - 1) / MMA_ROWS * MMA_ROWS;
    if (column0 >= stored_columns || row0 >= stored_rows) return;
    const u64 rows = min(u64(TILE_ROWS), stored_rows - row0);
    const Tile tile{source, seismic_words_value, b, row0, column0 / GROUP, column0 % GROUP / TILE_COLUMNS, lane,
                    column0 < packets * GROUP, stage};
    place_tile(RecordsPlace{tile, destination, rows}, tile.unit);

    // Each mma16 tile's bytes past its last factors.
    if (column0 + TILE_COLUMNS == stored_columns) {
        for (u64 tile_row = 0; tile_row < rows; tile_row += MMA_ROWS) {
            u8 *mma = row_base(destination, seismic_words_value, b, row0 + tile_row);
            for (u64 byte = SEISMIC_RESULT_0_ROW_GROUPS * MMA_ROWS * ROLE_BYTES(CODES) + lane;
                 byte < MMA_ROWS * SEISMIC_RESULT_0_ROW_STRIDE_BYTES; byte += LANES)
                mma[byte] = 0;
        }
    }
}

#else

// ---- rows16 / mma16 planes --------------------------------------------------

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
// Destination of byte `position` of an mma16 tile's code-plane sequence V:
// the tile's rows' plane payloads (`row_bytes` each) concatenated in row
// order.
__device__ __forceinline__ u8 *tile_plane_byte(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                               u64 tile_row0, u64 position, u64 row_bytes, u64 offset) {
    return row_base(destination, seismic_words_value, b, tile_row0 + position / row_bytes) + offset +
           position % row_bytes;
}
#endif

struct RowsPlace {
    const Tile &tile;
    u8 *destination;
    u64 rows;
    // The tile's first column of the row (its code planes' units are the
    // row's 32 columns at a time).
    u32 column0;
    // The code roles' words of the warp's rows (mma16 fragments).
    u32 *staged;

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // A code role, BITS a column, whose words of the warp's rows are staged
    // from word AT of each row: of each of the warp's mma16 tiles, this
    // lane's fragment slots of the tile's two k16 steps
    // (`PackedRowLayout::code_bit`): slot `j` of a step holds the code of
    // (row lane/4 + 8 (j % 2), column 2 (lane % 4) + 8 ((j % 4) / 2) + j / 4)
    // at bit BITS j.
    template <u32 BITS, u32 AT> __device__ __forceinline__ void place_fragments(u64 offset, u64 row_bytes) const {
        if constexpr (BITS != 0) {
            static_assert(32 % BITS == 0, "a code lies in one staged word");
            const u32 lane = tile.lane;
            const u32 block = column0 / 64, step0 = (column0 % 64) / 16;
            for (u32 tile_row = 0; tile_row < rows; tile_row += MMA_ROWS) {
                u32 value[(2 * BITS + 3) / 4] = {};
#pragma unroll
                for (u32 half_step = 0; half_step < 2; ++half_step)
#pragma unroll
                    for (u32 slot = 0; slot < 8; ++slot) {
                        const u32 row = tile_row + lane / 4 + 8 * (slot % 2);
                        const u32 column = 16 * half_step + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
                        const u32 code =
                            (staged[row * UNIT_WORDS + AT + column * BITS / 32] >> (column * BITS % 32)) &
                            repack::low_bits(BITS);
                        const u32 bit = (8 * half_step + slot) * BITS;
                        value[bit / 32] |= code << (bit % 32);
                    }
                store<2 * BITS>(tile_plane_byte(destination, tile.seismic_words_value, tile.b,
                                                tile.row0 + tile_row,
                                                (u64(block) * 32 + lane) * 4 * BITS + BITS * step0, row_bytes, offset),
                                value);
            }
        }
    }
#endif

    template <u32 C> __device__ __forceinline__ void place() const {
        const seismic_words_t &seismic_words_value = tile.seismic_words_value;
        const u32 lane = tile.lane;
        u32 words[UNIT_WORDS];
        tile.unit_record<C>(words);
        u8 *row = row_base(destination, seismic_words_value, tile.b, tile.row0 + lane);
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16) && defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
        // Grid entries stay whole: an mma16 tile's sequence is
        // [k-block][row][k-block bytes]; row `r`'s 32 columns (steps `step0`,
        // `step0 + 1`) are half of its k-block bytes.
        if (lane < rows) {
            const u32 r = lane % MMA_ROWS;
            const u32 block = column0 / 64, step0 = (column0 % 64) / 16;
            store<4 * CODES_WORDS>(tile_plane_byte(destination, seismic_words_value, tile.b, tile.row0 + lane - r,
                                                   (u64(block) * MMA_ROWS + r) * 8 * CODES_WORDS +
                                                       step0 / 2 * 4 * CODES_WORDS,
                                                   SEISMIC_RESULT_0_ROW_CODES_BYTES, SEISMIC_RESULT_0_ROW_CODES),
                                   words);
        }
#elif defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
        // The code roles' words of the warp's rows staged, then placed.
        if (lane < rows)
            for (u32 word = 0; word < UNIT_WORDS; ++word) staged[lane * UNIT_WORDS + word] = words[word];
        __syncwarp();
        place_fragments<CODES_WORDS, 0>(SEISMIC_RESULT_0_ROW_CODES, SEISMIC_RESULT_0_ROW_CODES_BYTES);
        place_fragments<HIGH_WORDS, CODES_WORDS>(SEISMIC_RESULT_0_ROW_HIGH, SEISMIC_RESULT_0_ROW_HIGH_BYTES);
#else
        // The tile's 32 columns of a row are one unit of each code plane.
        if (lane < rows) {
            if constexpr (CODES_WORDS != 0)
                store<4 * CODES_WORDS>(row + SEISMIC_RESULT_0_ROW_CODES + column0 / TILE_COLUMNS * 4 * CODES_WORDS, words);
            if constexpr (HIGH_WORDS != 0)
                store<4 * HIGH_WORDS>(row + SEISMIC_RESULT_0_ROW_HIGH + column0 / TILE_COLUMNS * 4 * HIGH_WORDS,
                                      words + CODES_WORDS);
        }
#endif
    }

    // Group planes, at the group's first tile: a row's groups are
    // contiguous.
    __device__ __forceinline__ void place_fixed() const {
        const seismic_words_t &seismic_words_value = tile.seismic_words_value;
        const u32 lane = tile.lane;
        u32 words[FIXED_WORDS + 1];
        tile.fixed_record(words);
        u8 *row = row_base(destination, seismic_words_value, tile.b, tile.row0 + lane);
        if (lane < rows) {
            if constexpr (ROLE_BYTES(SCALES) != 0)
                store<ROLE_BYTES(SCALES)>(row + SEISMIC_RESULT_0_ROW_SCALES + tile.packet * ROLE_BYTES(SCALES), words);
            if constexpr (ROLE_BYTES(SUPERS) != 0)
                store<ROLE_BYTES(SUPERS)>(row + SEISMIC_RESULT_0_ROW_SUPERS + tile.packet * ROLE_BYTES(SUPERS),
                                          words + ROLE_WORDS(SCALES));
        }
    }
};

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    __shared__ u32 stage[TILE_ROWS * STAGE_WORDS];
    const u32 lane = threadIdx.x;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16) && !defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
    __shared__ u32 staged[TILE_ROWS * UNIT_WORDS];
#else
    u32 *const staged = nullptr;
#endif
    const u64 b = blockIdx.z;
    const u64 row0 = u64(blockIdx.y) * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const u64 stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // Stored rows: whole mma16 tiles, rows beyond N zero.
    const u64 stored_rows = (SEISMIC_DIM_N + MMA_ROWS - 1) / MMA_ROWS * MMA_ROWS;
#else
    const u64 stored_rows = SEISMIC_DIM_N;
#endif
    if (column0 >= stored_columns || row0 >= stored_rows) return;
    const u64 rows = min(u64(TILE_ROWS), stored_rows - row0);
    const Tile tile{source, seismic_words_value, b, row0, column0 / GROUP, column0 % GROUP / TILE_COLUMNS, lane,
                    column0 < packets * GROUP, stage};
    place_tile(RowsPlace{tile, destination, rows, column0, staged}, tile.unit);

    // Alignment tails of every plane, from its end to the next plane's start
    // (or the row's end), at the row's last stored tile.
    if (column0 + TILE_COLUMNS == stored_columns && lane < rows) {
        u8 *row = row_base(destination, seismic_words_value, b, row0 + lane);
        const u64 offsets[4] = {SEISMIC_RESULT_0_ROW_CODES, SEISMIC_RESULT_0_ROW_HIGH, SEISMIC_RESULT_0_ROW_SCALES,
                                SEISMIC_RESULT_0_ROW_SUPERS};
        const u64 lengths[4] = {SEISMIC_RESULT_0_ROW_CODES_BYTES, SEISMIC_RESULT_0_ROW_HIGH_BYTES,
                                SEISMIC_RESULT_0_ROW_SCALES_BYTES, SEISMIC_RESULT_0_ROW_SUPERS_BYTES};
        for (u32 role = 0; role < 4; ++role) {
            if (lengths[role] == 0) continue;
            const u64 end = offsets[role] + lengths[role];
            u64 next = SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
            for (u32 other = 0; other < 4; ++other)
                if (lengths[other] != 0 && offsets[other] >= end) next = min(next, offsets[other]);
            for (u64 byte = end; byte < next; ++byte) row[byte] = 0;
        }
    }
}

#endif

#endif
