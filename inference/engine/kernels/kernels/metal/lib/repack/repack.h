// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`), as templates over a binding `B`: a type naming the source (E,
// a GGUF format) and resident (U: representation, layout, planes) elements
// as constant expressions, and the conversion's geometry (N, K, the source
// and destination strides and row planes) as functions of the entry's
// words. A binding is generated per entry and per conversion
// (lib/repack/bind.h for an entry of E, U, N, K and `source`;
// lib/streaming/bind_{gate,up,down}.h for the three conversions of the
// streamed gate/up kernel), so one kernel holds any number of conversions.
// Every view is [B, N, K].
//
// `repack_tile` converts one row tile (16 rows x 32 columns) with 32 threads.
// Every format's group (256, 64 or 32) is a multiple of 32, so a tile covers
// a whole 32-value packet or part of a larger one; the tile at a group's
// first column also writes that group's coefficients. Code stores are 4-,
// 8- or 16-byte vectors. Stored-but-unoccupied bytes (mma16 padding rows and
// groups, plane alignment tails, packet gaps) are written as zero, so the
// result is fully defined.
//
// The source is read through an addressing policy `A`: `Strided<B>` reads
// the bound source tensor; an includer that stages the source first
// (lib/streaming/window.h) addresses its copy. A matrix spans
// `tiles_x` x `tiles_y` tiles.

namespace repack {

enum class Source : uint {
    other, gguf_q3_k, gguf_iq3_s, gguf_q4_k, gguf_q5_k, gguf_q6_k, gguf_q8_0, gguf_iq4_nl, gguf_iq4_xs,
    gguf_q4_0, gguf_mxfp4, gguf_q5_0, gguf_q5_1, gguf_nvfp4, gguf_iq4_nl_block, gguf_iq2_xs, gguf_iq3_xxs
};
enum class Resident : uint { other, q4k, q5k, q6k, q8g32s, iq4g32, q4g32s, mxfp4g32, q5g32s, q5g32, nvfp4g16, iq4g32s, iq2xs, iq3xxs };
enum class Layout : uint { other, packet, rows16, rows32, mma16 };

// The registered conversions' code families.
enum class Family : uint { none, kquant, q6k, q3k, iq3s, q8, iq4nl, iq4, block32, nvfp4, grid };

constant constexpr uint TILE_ROWS = 16u;
constant constexpr uint TILE_COLUMNS = 32u;

// ---- GGUF source packet readers ------------------------------------------

// ggml's `iq3s_grid`: byte j of point i is magnitude j (odd, 1..15).
constant uint iq3s_grid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

// The conversion's format facts, from its binding.
template <class B> struct Format {
    static constexpr Family family() {
        return B::source() == Source::gguf_q3_k && B::resident() == Resident::q6k       ? Family::q3k
             : B::source() == Source::gguf_iq3_s && B::resident() == Resident::q6k      ? Family::iq3s
             : B::source() == Source::gguf_q4_k && B::resident() == Resident::q4k       ? Family::kquant
             : B::source() == Source::gguf_q5_k && B::resident() == Resident::q5k       ? Family::kquant
             : B::source() == Source::gguf_q6_k && B::resident() == Resident::q6k       ? Family::q6k
             : B::source() == Source::gguf_q8_0 && B::resident() == Resident::q8g32s    ? Family::q8
             : B::source() == Source::gguf_iq4_nl && B::resident() == Resident::iq4g32  ? Family::iq4nl
             : B::source() == Source::gguf_iq4_xs && B::resident() == Resident::iq4g32  ? Family::iq4
             : B::source() == Source::gguf_q4_0 && B::resident() == Resident::q4g32s    ? Family::block32
             : B::source() == Source::gguf_iq4_nl_block && B::resident() == Resident::iq4g32s ? Family::block32
             : B::source() == Source::gguf_iq2_xs && B::resident() == Resident::iq2xs   ? Family::grid
             : B::source() == Source::gguf_iq3_xxs && B::resident() == Resident::iq3xxs ? Family::grid
             : B::source() == Source::gguf_mxfp4 && B::resident() == Resident::mxfp4g32 ? Family::block32
             : B::source() == Source::gguf_q5_0 && B::resident() == Resident::q5g32s    ? Family::block32
             : B::source() == Source::gguf_q5_1 && B::resident() == Resident::q5g32     ? Family::block32
             : B::source() == Source::gguf_nvfp4 && B::resident() == Resident::nvfp4g16 ? Family::nvfp4
                                                                                        : Family::none;
    }
    static constexpr uint code_bits() {
        return B::source() == Source::gguf_q5_k || B::source() == Source::gguf_q5_0 ||
                       B::source() == Source::gguf_q5_1
                   ? 5u
               : family() == Family::q3k || family() == Family::iq3s || family() == Family::q6k ? 6u
               : family() == Family::q8                                                         ? 8u
               : B::source() == Source::gguf_iq2_xs                                             ? 2u
               : B::source() == Source::gguf_iq3_xxs                                            ? 3u
                                                                                                : 4u;
    }
    // Leading scale bytes of a 32-value block (block_q4_0: d | qs[16];
    // one block_iq4_nl: d | qs[16]; block_mxfp4: e | qs[16]; block_q5_0:
    // d | qh[4] | qs[16]; block_q5_1: d | m | qh[4] | qs[16]; block_nvfp4:
    // d[4] | qs[32]).
    static constexpr uint scale_bytes() {
        return B::source() == Source::gguf_q4_0 || B::source() == Source::gguf_iq4_nl_block ? 2u
             : B::source() == Source::gguf_mxfp4 ? 1u
             : B::source() == Source::gguf_q5_0 ? 2u
             : B::source() == Source::gguf_q5_1 ? 4u
             : B::source() == Source::gguf_nvfp4 ? 4u
                                                 : 0u;
    }
    // The offset of a 32-value block's qs.
    static constexpr uint qs() {
        return B::source() == Source::gguf_q4_0 || B::source() == Source::gguf_iq4_nl_block ? 2u
             : B::source() == Source::gguf_mxfp4 ? 1u
             : B::source() == Source::gguf_q5_0 ? 6u
             : B::source() == Source::gguf_q5_1 ? 8u
                                                 : 0u;
    }
    static constexpr bool high_bit() {
        return B::source() == Source::gguf_q5_0 || B::source() == Source::gguf_q5_1;
    }
    static constexpr bool e2m1() {
        return B::source() == Source::gguf_mxfp4 || B::source() == Source::gguf_nvfp4;
    }
    // Codes with a packed locals plane and a supers plane.
    static constexpr bool locals() {
        return family() == Family::kquant || family() == Family::q6k || family() == Family::q3k ||
               family() == Family::iq3s || family() == Family::grid;
    }
    // The power of two a vector-code block's d is scaled by: iq2_xs d / 8,
    // iq3_xxs d / 4.
    static constexpr float grid_factor() {
        return B::source() == Source::gguf_iq2_xs ? 0.125f : 0.25f;
    }
    static constexpr uint group() {
        return uint(B::group());
    }
    // Rows of one stored row tile: `rows32` interleaves every plane across
    // 32 rows; `rows16` rows stand alone.
    static constexpr ulong stored_tile() {
        return B::layout() == Layout::rows32 ? 32ul : 1ul;
    }
    static ulong tiles_x(constant ulong *seismic_words) {
        return (ulong(B::k(seismic_words)) + 255) / 256 * 8;
    }
    static ulong tiles_y(constant ulong *seismic_words) {
        return (ulong(B::n(seismic_words)) + 31) / 32 * 2;
    }
};

// Resident vector-code entry `e` (values 8e .. 8e + 7) of a source block: the
// grid indices, then the seven signs (registry `gguf_iq2_xs`,
// `gguf_iq3_xxs`).
template <class B, typename In> inline uint grid_entry(In in, uint e) {
    if constexpr (B::source() == Source::gguf_iq3_xxs) {
        uint block = e / 4, group = e % 4;
        uint signs = uint(in[66 + 4 * block]) | (uint(in[67 + 4 * block]) << 8) | (uint(in[68 + 4 * block]) << 16) |
                     (uint(in[69 + 4 * block]) << 24);
        return uint(in[2 + 8 * block + 2 * group]) | (uint(in[3 + 8 * block + 2 * group]) << 8) |
               (((signs >> (7 * group)) & 127u) << 16);
    } else {
        return uint(in[2 + 2 * e]) | (uint(in[3 + 2 * e]) << 8);
    }
}

// Raw code of position `p` (0 <= p < group) of one source packet; for vector
// codes, the position's `code_bits` slice of its entry (an entry is its eight
// positions' slices).
template <class B, typename In> inline uint source_code(In in, uint p) {
    typedef Format<B> F;
    static_assert(B::external() && B::packed(), "repack converts an external GGUF source into packed storage");
    static_assert(F::family() != Family::none, "repack binding is not a registered GGUF-to-resident conversion");
    static_assert(B::layout() != Layout::other, "repack writes packet, rows16, mma16 or rows32 storage");
    if constexpr (F::family() == Family::grid) {
        return (grid_entry<B>(in, p / 8) >> (F::code_bits() * (p % 8))) & ((1u << F::code_bits()) - 1u);
    } else if constexpr (F::family() == Family::kquant && F::code_bits() == 4u) {
        return (in[16 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
    } else if constexpr (F::family() == Family::kquant) {
        uint low = (in[48 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
        uint high = (in[16 + p % 32] >> (p / 32)) & 1u;
        return low | (high << 4);
    } else if constexpr (F::family() == Family::q6k) {
        uint low = (in[(p / 128) * 64 + p % 64] >> ((p % 128 / 64) * 4)) & 15u;
        uint high = (in[128 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
        return low | (high << 4);
    } else if constexpr (F::family() == Family::q3k) {
        // q + 32 with q = low + 4 high - 4 (the registry's exact q6k re-basing).
        uint low = (in[32 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
        uint high = (in[p % 32] >> (p / 32)) & 1u;
        return (low | (high << 2)) + 28u;
    } else if constexpr (F::family() == Family::iq3s) {
        // 32 plus the signed grid magnitude (the registry's exact q6k re-basing).
        uint block = p / 32, group = p % 32 / 8, slot = p % 8;
        uint point = uint(in[2 + 8 * block + 2 * group + slot / 4]) |
                     (((uint(in[66 + block]) >> (2 * group + slot / 4)) & 1u) << 8);
        uint magnitude = (iq3s_grid[point] >> (8 * (slot % 4))) & 255u;
        return ((in[74 + 4 * block + group] >> slot) & 1u) != 0 ? 32u - magnitude : 32u + magnitude;
    } else if constexpr (F::family() == Family::q8) {
        return uint(in[2 + p]);
    } else if constexpr (F::family() == Family::iq4nl) {
        return (in[(p / 32) * 18 + 2 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
    } else if constexpr (F::family() == Family::block32 || F::family() == Family::nvfp4) {
        uint code;
        if constexpr (F::family() == Family::block32) {
            // The low (p < 16) or high nibble of qs[p % 16], with bit p of qh.
            code = (in[F::qs() + p % 16] >> ((p / 16) * 4)) & 15u;
        } else {
            // Sixteen-value sub-block s = p / 16: the low (p % 16 < 8) or
            // high nibble of qs[8 s + p % 8].
            code = (in[4 + 8 * (p / 16) + p % 8] >> ((p % 16 / 8) * 4)) & 15u;
        }
        if constexpr (F::high_bit())
            code |= ((uint(in[F::scale_bytes() + p / 8]) >> (p % 8)) & 1u) << 4;
        if constexpr (F::e2m1()) {
            // E2M1 -0 (code 8) becomes ggml's +0 (code 0).
            code = code == 8u ? 0u : code;
        }
        return code;
    } else {
        return (in[8 + (p / 32) * 16 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
    }
}

// Six-bit local coefficient `field` (0 scale, 1 minimum) of K-quant
// sub-block `j`.
template <typename In> inline uint kquant_local(In in, uint j, uint field) {
    uint index = j % 4;
    uint low = uint(in[4 + field * 4 + index]);
    uint high = uint(in[12 + index]);
    return j < 4 ? (low & 63u) : (((high >> (4 * field)) & 15u) | ((low >> 6) << 4));
}

// Word `w` of the packed local coefficients: 16 six-bit fields, field
// `2j + f` of sub-block j at bit 12j + 6f (the registry `coefficients` /
// `scales` plane).
template <typename In> inline uint kquant_locals_word(In in, uint w) {
    uint value = 0;
    for (uint field = 0; field < 16; ++field) {
        int shift = int(field * 6) - int(w * 32);
        if (shift <= -6 || shift >= 32) continue;
        uint local = kquant_local(in, field / 2, field % 2);
        value |= shift >= 0 ? local << uint(shift) : local >> uint(-shift);
    }
    return value;
}

// Resident f32 scale of IQ4_XS sub-block `j` (32 values).
template <typename In> inline float iq4_scale(In in, uint j) {
    float base = float(as_type<half>(ushort(uint(in[0]) | (uint(in[1]) << 8))));
    uint low = (uint(in[4 + j / 2]) >> ((j % 2) * 4)) & 15u;
    uint high = ((uint(in[2]) | (uint(in[3]) << 8)) >> (2 * j)) & 3u;
    return base * float(int(low | (high << 4)) - 32);
}

// The per-group coefficient planes of one packet: `scales` (the packed
// locals) and `supers` (the per-group factors), in registry byte order.
template <class B, typename In> inline void write_scales(In in, device uchar *out) {
    typedef Format<B> F;
    if constexpr (F::family() == Family::kquant) {
        for (uint w = 0; w < 3; ++w)
            reinterpret_cast<device uint *>(out)[w] = kquant_locals_word(in, w);
    } else if constexpr (F::family() == Family::q6k) {
        for (uint w = 0; w < 4; ++w)
            reinterpret_cast<device uint *>(out)[w] = uint(in[192 + 4 * w]) | (uint(in[193 + 4 * w]) << 8) |
                                                       (uint(in[194 + 4 * w]) << 16) | (uint(in[195 + 4 * w]) << 24);
    } else if constexpr (F::family() == Family::q3k) {
        // Six-bit scale of sub-block j less 32, as a two's complement byte.
        for (uint j = 0; j < 16; ++j) {
            uint low = (uint(in[96 + j % 8]) >> (4 * (j / 8))) & 15u;
            uint high = (uint(in[104 + j % 4]) >> (2 * (j / 4))) & 3u;
            out[j] = uchar(int(low | (high << 4)) - 32);
        }
    } else if constexpr (F::family() == Family::iq3s) {
        // Both halves of 32-value sub-block b take 1 + 2 s, s its four-bit scale.
        for (uint j = 0; j < 16; ++j) out[j] = uchar(1u + 2u * ((uint(in[106 + j / 4]) >> (4 * (j / 2 % 2))) & 15u));
    } else if constexpr (F::family() == Family::grid && B::source() == Source::gguf_iq3_xxs) {
        // Sub-block b's scale is the top nibble of its u32, nibble b of the plane.
        for (uint j = 0; j < 4; ++j) out[j] = uchar((in[69 + 8 * j] >> 4) | (in[73 + 8 * j] & 0xf0u));
    } else if constexpr (F::family() == Family::grid) {
        for (uint j = 0; j < 8; ++j) out[j] = in[66 + j];
    }
}

template <class B, typename In> inline void write_supers(In in, device uchar *out) {
    typedef Format<B> F;
    if constexpr (F::family() == Family::kquant) {
        reinterpret_cast<device uint *>(out)[0] =
            uint(in[0]) | (uint(in[1]) << 8) | (uint(in[2]) << 16) | (uint(in[3]) << 24);
    } else if constexpr (F::family() == Family::q6k) {
        reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[208]) | (uint(in[209]) << 8));
    } else if constexpr (F::family() == Family::q3k) {
        reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[108]) | (uint(in[109]) << 8));
    } else if constexpr (F::family() == Family::iq3s || F::family() == Family::q8) {
        reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[0]) | (uint(in[1]) << 8));
    } else if constexpr (F::family() == Family::grid) {
        // d widened and scaled by a power of two, exactly.
        reinterpret_cast<device float *>(out)[0] =
            float(as_type<half>(ushort(uint(in[0]) | (uint(in[1]) << 8)))) * F::grid_factor();
    } else if constexpr (F::family() == Family::iq4nl) {
        // Each block's f16 scale, widened exactly.
        for (uint j = 0; j < 8; ++j)
            reinterpret_cast<device float *>(out)[j] =
                float(as_type<half>(ushort(uint(in[18 * j]) | (uint(in[18 * j + 1]) << 8))));
    } else if constexpr (F::scale_bytes() != 0) {
        // The block's leading scale fields (d, d | m, e or d[4]), bit for bit.
        for (uint i = 0; i < F::scale_bytes(); ++i) out[i] = in[i];
    } else {
        for (uint j = 0; j < 8; ++j)
            reinterpret_cast<device float *>(out)[j] = iq4_scale(in, j);
    }
}

// Eight consecutive `bits`-wide slices (code >> shift) of codes
// `first .. first + 8`, packed little-endian.
template <class B, typename In> inline uint pack_eight(In in, uint first, uint shift, uint bits) {
    uint value = 0;
    for (uint i = 0; i < 8; ++i)
        value |= ((source_code<B>(in, first + i) >> shift) & ((1u << bits) - 1u)) << (i * bits);
    return value;
}

// Word `word` of the codes of positions `first ..`, `code_bits` each, packed
// little-endian (a code may straddle two words).
template <class B, typename In> inline uint code_word(In in, uint first, uint word) {
    constexpr uint bits = Format<B>::code_bits();
    uint value = 0;
    for (uint code = (word * 32) / bits; code <= (word * 32 + 31) / bits; ++code) {
        int shift = int(code * bits) - int(word * 32);
        uint raw = source_code<B>(in, first + code);
        value |= shift >= 0 ? raw << uint(shift) : raw >> uint(-shift);
    }
    return value;
}

// Four consecutive eight-bit codes `first .. first + 4`.
template <class B, typename In> inline uint pack_four_bytes(In in, uint first) {
    uint value = 0;
    for (uint i = 0; i < 4; ++i) value |= source_code<B>(in, first + i) << (8 * i);
    return value;
}

inline void zero_bytes(device uchar *out, ulong count) {
    for (ulong i = 0; i < count; ++i) out[i] = 0;
}

// Addressing of the bound source tensor: packet `packet` of row `n` of
// matrix `b`.
template <class B> struct Strided {
    template <typename In> static In packet(In source, constant ulong *seismic_words, ulong b, ulong n, ulong packet) {
        return source + (b * B::source_stride_0(seismic_words) + n * B::source_stride_1(seismic_words) +
                         packet * B::source_stride_2(seismic_words)) *
                            B::source_packet_size();
    }
};

// ---- packet: planes interleaved per packet --------------------------------

template <class B>
inline device uchar *destination_packet(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                        ulong packet) {
    return destination + (b * B::out_stride_0(seismic_words) + n * B::out_stride_1(seismic_words) +
                          packet * B::out_stride_2(seismic_words)) * B::packet_size();
}

template <class B> inline bool packet_plane_byte(uint byte) {
    bool inside = false;
    for (uint plane = 0; plane < B::plane_count(); ++plane)
        inside = inside || (byte >= B::plane_offset(plane) &&
                            byte < B::plane_offset(plane) + B::plane_bytes_per_group(plane));
    return inside;
}

template <class B, class A, typename In>
inline void repack_packet_tile(In source, device uchar *destination, constant ulong *seismic_words, ulong b,
                               uint2 tile, uint lane) {
    typedef Format<B> F;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong dim_n = B::n(seismic_words);
    const ulong packets = (ulong(B::k(seismic_words)) + F::group() - 1) / F::group();
    if (column0 >= packets * F::group()) return;
    const uint packet = column0 / F::group();
    const uint within = column0 % F::group();
    // The tile's codes are `bits` words of the packed `words` plane per row.
    const uint code_words = F::code_bits();
    for (uint item = lane; item < TILE_ROWS * code_words; item += TILE_COLUMNS) {
        ulong n = row0 + item / code_words;
        if (n >= dim_n) continue;
        In in = A::packet(source, seismic_words, b, n, packet);
        uint word = within / 32 * code_words + item % code_words;
        uint value = code_word<B>(in, 0, word);
        device uchar *out = destination_packet<B>(destination, seismic_words, b, n, packet);
        if constexpr (B::packet_size() % 4 == 0) {
            reinterpret_cast<device uint *>(out + B::plane_offset(0))[word] = value;
        } else {
            // Byte-aligned packets (the E8M0 and UE4M3 scale planes): bytewise.
            for (uint byte = 0; byte < 4; ++byte)
                out[B::plane_offset(0) + 4 * word + byte] = uchar(value >> (8 * byte));
        }
    }
    if (within != 0 || lane >= TILE_ROWS || row0 + lane >= dim_n) return;
    const ulong n = row0 + lane;
    In in = A::packet(source, seismic_words, b, n, packet);
    device uchar *out = destination_packet<B>(destination, seismic_words, b, n, packet);
    if constexpr (F::locals()) {
        write_scales<B>(in, out + B::plane_offset(1));
        write_supers<B>(in, out + B::plane_offset(2));
    } else {
        write_supers<B>(in, out + B::plane_offset(1));
    }
    for (uint byte = 0; byte < B::packet_size(); ++byte)
        if (!packet_plane_byte<B>(byte)) out[byte] = 0;
}

// ---- rows16 / rows32 / mma16: row planes ------------------------------------

// Start of stored row `n` of matrix `b`.
template <class B>
inline device uchar *row_base(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n) {
    return destination + (b * B::out_stride_0(seismic_words) + n * B::out_stride_1(seismic_words)) *
                             B::out_row_stride_bytes(seismic_words);
}

// Unit `unit` (of `unit_bytes` bytes) of stored row `n` in the code plane at
// row offset `offset`: units follow one another along a `rows16` row; in
// `rows32` the tile's rows of one unit are adjacent
// (`PackedRowLayout::code_bit`).
template <class B>
inline device uchar *plane_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                ulong offset, ulong unit_bytes, ulong unit) {
    constexpr ulong stored = Format<B>::stored_tile();
    return row_base<B>(destination, seismic_words, b, n - n % stored) + offset * stored +
           (unit * stored + n % stored) * unit_bytes;
}

// Storage group `group` (of `group_bytes` bytes) of stored row `n` in the
// group plane at row offset `offset` (`row_bytes` a row): a row's groups are
// contiguous in every layout, and a `rows32` tile holds its rows' shares of
// the plane one after another (`PackedRowLayout::packet_bit`).
template <class B>
inline device uchar *plane_group(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                 ulong offset, ulong row_bytes, ulong group_bytes, ulong group) {
    constexpr ulong stored = Format<B>::stored_tile();
    return row_base<B>(destination, seismic_words, b, n - n % stored) + offset * stored +
           n % stored * row_bytes + group * group_bytes;
}

// The share of stored row `n` in the `length`-bytes-a-row gap that starts at
// row offset `offset` (a plane's alignment tail).
template <class B>
inline device uchar *plane_gap(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                               ulong offset, ulong length) {
    constexpr ulong stored = Format<B>::stored_tile();
    return row_base<B>(destination, seismic_words, b, n - n % stored) + offset * stored + (n % stored) * length;
}

// Code of fragment slot `slot` of k16 step `step` (within k-block `block`)
// owned by `lane`: (row lane/4 + 8 (slot % 2), column 64 block + 16 step +
// 2 (lane % 4) + 8 ((slot % 4) / 2) + slot / 4). Zero beyond the matrix's
// rows or the source packets (tile padding).
template <class B, class A, typename In>
inline uint fragment_code(In source, constant ulong *seismic_words, ulong b, ulong row0, ulong packets, uint block,
                          uint step, uint lane, uint slot) {
    typedef Format<B> F;
    ulong n = row0 + lane / 4 + 8 * (slot % 2);
    ulong column = ulong(block) * 64 + step * 16 + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
    if (n >= ulong(B::n(seismic_words)) || column >= packets * F::group()) return 0;
    return source_code<B>(A::packet(source, seismic_words, b, n, column / F::group()), uint(column % F::group()));
}

// Eight `bits`-wide slices of one lane's slots of one step.
template <class B, class A, typename In>
inline uint fragment_slice(In source, constant ulong *seismic_words, ulong b, ulong row0, ulong packets, uint block,
                           uint step, uint lane, uint shift, uint bits) {
    uint value = 0;
    for (uint slot = 0; slot < 8; ++slot)
        value |= ((fragment_code<B, A>(source, seismic_words, b, row0, packets, block, step, lane, slot) >> shift) &
                  ((1u << bits) - 1u)) << (slot * bits);
    return value;
}

// Destination of byte `position` of a tile's code-plane sequence V: the
// rows' plane payloads (`row_bytes` each) concatenated in row order.
template <class B>
inline device uchar *tile_plane_byte(device uchar *destination, constant ulong *seismic_words, ulong b, ulong row0,
                                     ulong position, ulong row_bytes, ulong offset) {
    return row_base<B>(destination, seismic_words, b, row0 + position / row_bytes) + offset + position % row_bytes;
}

template <class B, class A, typename In>
inline void repack_row_tile(In source, device uchar *destination, constant ulong *seismic_words, ulong b, uint2 tile,
                            uint lane) {
    typedef Format<B> F;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong dim_n = B::n(seismic_words);
    const ulong packets = (ulong(B::k(seismic_words)) + F::group() - 1) / F::group();
    const ulong stored_columns = ulong(B::out_row_groups(seismic_words)) * F::group();
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * F::group();
    const uint within = column0 % F::group();
    ulong rows;
    if constexpr (B::layout() == Layout::mma16) {
        // Every row of a tile is stored; rows beyond N are zero padding.
        if (row0 >= (dim_n + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS) return;
        rows = TILE_ROWS;
    } else if constexpr (B::layout() == Layout::rows32) {
        // Every row of a 32-row tile is stored; rows beyond N are zero padding.
        rows = TILE_ROWS;
    } else {
        if (row0 >= dim_n) return;
        rows = min(ulong(TILE_ROWS), dim_n - row0);
    }

    if constexpr (B::layout() == Layout::rows16 || B::layout() == Layout::rows32) {
        // Code planes: the 32 codes of this tile's columns of one row are one
        // unit of each code plane, the code of column c at bit `(c % 32) * bits`.
        if (lane < rows) {
            const ulong n = row0 + lane;
            const bool stored = n < dim_n;
            const ulong unit = column0 / TILE_COLUMNS;
            In in = A::packet(source, seismic_words, b, stored ? n : 0ul, column0 / F::group());
            if constexpr (B::grids()) {
                // The unit's four grid entries, `code_bits` words.
                device uint *grids = reinterpret_cast<device uint *>(plane_unit<B>(destination, seismic_words, b,
                    n, B::out_grids_row_offset(seismic_words), 4ul * F::code_bits(), unit));
                for (uint word = 0; word < F::code_bits(); ++word)
                    grids[word] = stored ? code_word<B>(in, within, word) : 0u;
            } else if constexpr (B::codes()) {
                device uint4 *codes = reinterpret_cast<device uint4 *>(plane_unit<B>(destination, seismic_words, b,
                    n, B::out_codes_row_offset(seismic_words), 32ul, unit));
                codes[0] = stored ? uint4(pack_four_bytes<B>(in, within), pack_four_bytes<B>(in, within + 4),
                                          pack_four_bytes<B>(in, within + 8), pack_four_bytes<B>(in, within + 12))
                                  : uint4(0u);
                codes[1] = stored ? uint4(pack_four_bytes<B>(in, within + 16), pack_four_bytes<B>(in, within + 20),
                                          pack_four_bytes<B>(in, within + 24), pack_four_bytes<B>(in, within + 28))
                                  : uint4(0u);
            } else {
                *reinterpret_cast<device uint4 *>(plane_unit<B>(destination, seismic_words, b, n,
                    B::out_codes_lo_row_offset(seismic_words), 16ul, unit)) =
                    stored ? uint4(pack_eight<B>(in, within, 0, 4), pack_eight<B>(in, within + 8, 0, 4),
                                   pack_eight<B>(in, within + 16, 0, 4), pack_eight<B>(in, within + 24, 0, 4))
                           : uint4(0u);
                if constexpr (B::codes_hi() && B::codes_hi_code_bits() == 1) {
                    *reinterpret_cast<device uint *>(plane_unit<B>(destination, seismic_words, b, n,
                        B::out_codes_hi_row_offset(seismic_words), 4ul, unit)) =
                        stored ? pack_eight<B>(in, within, 4, 1) | (pack_eight<B>(in, within + 8, 4, 1) << 8) |
                                     (pack_eight<B>(in, within + 16, 4, 1) << 16) |
                                     (pack_eight<B>(in, within + 24, 4, 1) << 24)
                               : 0u;
                } else if constexpr (B::codes_hi()) {
                    *reinterpret_cast<device uint2 *>(plane_unit<B>(destination, seismic_words, b, n,
                        B::out_codes_hi_row_offset(seismic_words), 8ul, unit)) =
                        stored ? uint2(pack_eight<B>(in, within, 4, 2) | (pack_eight<B>(in, within + 8, 4, 2) << 16),
                                       pack_eight<B>(in, within + 16, 4, 2) |
                                           (pack_eight<B>(in, within + 24, 4, 2) << 16))
                               : uint2(0u);
                }
            }
        }
    } else {
        // mma16: lane l owns, per 64-column k-block, 4 * bits bytes of each
        // code plane's tile sequence V (registry `PackedRowLayout::code_bit`);
        // this tile is k16 steps `step0` and `step0 + 1` of k-block `block`.
        const uint block = column0 / 64;
        const uint step0 = (column0 % 64) / 16;
        if constexpr (B::grids()) {
            // Grid entries stay whole: the tile sequence is [k-block][row][k-block
            // bytes]; row `lane`'s 32 columns (steps `step0`, `step0 + 1`) are
            // half of its k-block bytes.
            if (lane < TILE_ROWS) {
                const ulong n = row0 + lane;
                const bool stored = n < dim_n;
                In in = A::packet(source, seismic_words, b, stored ? n : 0ul, column0 / F::group());
                const ulong position = (ulong(block) * TILE_ROWS + lane) * 8 * F::code_bits() +
                                       step0 / 2 * 4 * F::code_bits();
                for (uint word = 0; word < F::code_bits(); ++word)
                    *reinterpret_cast<device uint *>(tile_plane_byte<B>(destination, seismic_words, b, row0,
                        position + 4 * word, B::out_grids_bytes_per_row(seismic_words),
                        B::out_grids_row_offset(seismic_words))) = stored ? code_word<B>(in, within, word) : 0u;
            }
        } else if constexpr (B::codes()) {
            uint4 value;
            for (uint half_step = 0; half_step < 2; ++half_step) {
                uint low = 0, high = 0;
                for (uint slot = 0; slot < 4; ++slot) {
                    low |= fragment_code<B, A>(source, seismic_words, b, row0, packets, block, step0 + half_step,
                                               lane, slot)
                           << (8 * slot);
                    high |= fragment_code<B, A>(source, seismic_words, b, row0, packets, block, step0 + half_step,
                                                lane, slot + 4)
                            << (8 * slot);
                }
                value[2 * half_step] = low;
                value[2 * half_step + 1] = high;
            }
            ulong position = (ulong(block) * 32 + lane) * 32 + 8 * step0;
            *reinterpret_cast<device uint4 *>(tile_plane_byte<B>(destination, seismic_words, b, row0, position,
                                                                 B::out_codes_bytes_per_row(seismic_words),
                                                                 B::out_codes_row_offset(seismic_words))) = value;
        } else {
            {
                ulong position = (ulong(block) * 32 + lane) * 16 + 4 * step0;
                *reinterpret_cast<device uint2 *>(tile_plane_byte<B>(destination, seismic_words, b, row0, position,
                                                                     B::out_codes_lo_bytes_per_row(seismic_words),
                                                                     B::out_codes_lo_row_offset(seismic_words))) =
                    uint2(fragment_slice<B, A>(source, seismic_words, b, row0, packets, block, step0, lane, 0, 4),
                          fragment_slice<B, A>(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 0, 4));
            }
            if constexpr (B::codes_hi() && B::codes_hi_code_bits() == 1) {
                ulong position = (ulong(block) * 32 + lane) * 4 + step0;
                *reinterpret_cast<device ushort *>(tile_plane_byte<B>(destination, seismic_words, b, row0, position,
                                                                      B::out_codes_hi_bytes_per_row(seismic_words),
                                                                      B::out_codes_hi_row_offset(seismic_words))) =
                    ushort(fragment_slice<B, A>(source, seismic_words, b, row0, packets, block, step0, lane, 4, 1) |
                           (fragment_slice<B, A>(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 4,
                                                 1)
                            << 8));
            } else if constexpr (B::codes_hi()) {
                ulong position = (ulong(block) * 32 + lane) * 8 + 2 * step0;
                *reinterpret_cast<device uint *>(tile_plane_byte<B>(destination, seismic_words, b, row0, position,
                                                                    B::out_codes_hi_bytes_per_row(seismic_words),
                                                                    B::out_codes_hi_row_offset(seismic_words))) =
                    fragment_slice<B, A>(source, seismic_words, b, row0, packets, block, step0, lane, 4, 2) |
                    (fragment_slice<B, A>(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 4, 2)
                     << 16);
            }
        }
    }

    // Coefficient planes, once per group (at its first tile), per row.
    if (within == 0 && lane < rows) {
        const ulong n = row0 + lane;
        const ulong group = column0 / F::group();
        const bool real = n < dim_n && occupied;
        if constexpr (B::scales()) {
            device uchar *scales = plane_group<B>(destination, seismic_words, b, n,
                B::out_scales_row_offset(seismic_words), B::out_scales_bytes_per_row(seismic_words),
                B::scales_bytes_per_group(), group);
            if (real) write_scales<B>(A::packet(source, seismic_words, b, n, group), scales);
            else zero_bytes(scales, B::scales_bytes_per_group());
        }
        device uchar *supers = plane_group<B>(destination, seismic_words, b, n,
            B::out_supers_row_offset(seismic_words), B::out_supers_bytes_per_row(seismic_words),
            B::supers_bytes_per_group(), group);
        if (real) write_supers<B>(A::packet(source, seismic_words, b, n, group), supers);
        else zero_bytes(supers, B::supers_bytes_per_group());
    }

    // Alignment tails of the coefficient planes and of the row, at the row's
    // last stored tile. Low code planes are whole multiples of 16 bytes; a
    // high-bit plane of 32-value groups may not be.
    if (column0 + TILE_COLUMNS == stored_columns && lane < rows) {
        const ulong n = row0 + lane;
        const ulong supers_offset = B::out_supers_row_offset(seismic_words);
        if constexpr (B::codes_hi()) {
            const ulong after_high = B::scales() ? ulong(B::out_scales_row_offset(seismic_words)) : supers_offset;
            const ulong high_end =
                B::out_codes_hi_row_offset(seismic_words) + B::out_codes_hi_bytes_per_row(seismic_words);
            zero_bytes(plane_gap<B>(destination, seismic_words, b, n, high_end, after_high - high_end),
                       after_high - high_end);
        }
        if constexpr (B::scales()) {
            const ulong scales_end =
                B::out_scales_row_offset(seismic_words) + B::out_scales_bytes_per_row(seismic_words);
            zero_bytes(plane_gap<B>(destination, seismic_words, b, n, scales_end, supers_offset - scales_end),
                       supers_offset - scales_end);
        }
        const ulong supers_end = supers_offset + B::out_supers_bytes_per_row(seismic_words);
        const ulong row_bytes = B::out_row_stride_bytes(seismic_words);
        zero_bytes(plane_gap<B>(destination, seismic_words, b, n, supers_end, row_bytes - supers_end),
                   row_bytes - supers_end);
    }
}

// Convert row tile `tile` (columns x, rows y) of matrix `b`, reading the
// source through `A`; `lane` is the thread's index in its 32-thread group.
// A tile outside the matrix's stored extent does nothing.
template <class B, class A = Strided<B>, typename In>
inline void repack_tile(In source, device uchar *destination, constant ulong *seismic_words, ulong b, uint2 tile,
                        uint lane) {
    if constexpr (B::layout() == Layout::packet)
        repack_packet_tile<B, A>(source, destination, seismic_words, b, tile, lane);
    else
        repack_row_tile<B, A>(source, destination, seismic_words, b, tile, lane);
}

} // namespace repack
