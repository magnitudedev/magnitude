// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`), one kernel per (source format, target layout) selected by the
// element macros: E is the GGUF source, U the resident (representation,
// layout). The entry sees every weight as a [B, N, K] view.
//
// Each threadgroup (32 threads) converts one row tile: 16 rows x 32 columns.
// Every format's group (256, 64 or 32) is a multiple of 32, so a tile covers
// a whole 32-value packet or part of a larger one; the tile at a group's
// first column also writes that group's coefficients. Code stores are 4-,
// 8- or 16-byte vectors. Stored-but-unoccupied bytes (mma16 padding rows and
// groups, plane alignment tails, packet gaps) are written as zero, so the
// result is fully defined.

#if !defined(SEISMIC_ELEMENT_E_KIND_EXTERNAL) || !defined(SEISMIC_ELEMENT_U_KIND_PACKED)
#error "repack_weight converts an external GGUF source into packed resident storage"
#endif

#if !defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET) && !defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16) && \
    !defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
#error "repack_weight writes packet, rows16 or mma16 storage"
#endif

#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q3_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q6K)
#define REPACK_Q3K 1
#define REPACK_CODE_BITS 6u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ3_S) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q6K)
#define REPACK_IQ3S 1
#define REPACK_CODE_BITS 6u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q4K)
#define REPACK_KQUANT 1
#define REPACK_CODE_BITS 4u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q5_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q5K)
#define REPACK_KQUANT 1
#define REPACK_CODE_BITS 5u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q6_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q6K)
#define REPACK_Q6K 1
#define REPACK_CODE_BITS 6u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q8_0) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q8G32S)
#define REPACK_Q8 1
#define REPACK_CODE_BITS 8u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_NL) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_IQ4G32)
#define REPACK_IQ4NL 1
#define REPACK_CODE_BITS 4u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_XS) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_IQ4G32)
#define REPACK_IQ4 1
#define REPACK_CODE_BITS 4u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_0) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q4G32S)
// block_q4_0: d (f16) | qs[16].
#define REPACK_BLOCK32 1
#define REPACK_SCALE_BYTES 2u
#define REPACK_QS 2u
#define REPACK_CODE_BITS 4u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_MXFP4) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_MXFP4G32)
// block_mxfp4: e (E8M0) | qs[16].
#define REPACK_BLOCK32 1
#define REPACK_E2M1 1
#define REPACK_SCALE_BYTES 1u
#define REPACK_QS 1u
#define REPACK_CODE_BITS 4u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q5_0) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q5G32S)
// block_q5_0: d (f16) | qh[4] | qs[16].
#define REPACK_BLOCK32 1
#define REPACK_SCALE_BYTES 2u
#define REPACK_HIGH_BIT 1
#define REPACK_QS 6u
#define REPACK_CODE_BITS 5u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q5_1) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q5G32)
// block_q5_1: d | m (f16) | qh[4] | qs[16].
#define REPACK_BLOCK32 1
#define REPACK_SCALE_BYTES 4u
#define REPACK_HIGH_BIT 1
#define REPACK_QS 8u
#define REPACK_CODE_BITS 5u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_NVFP4) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_NVFP4G16)
// block_nvfp4: d[4] (UE4M3) | qs[32].
#define REPACK_NVFP4 1
#define REPACK_E2M1 1
#define REPACK_SCALE_BYTES 4u
#define REPACK_CODE_BITS 4u
#else
#error "repack_weight binding is not a registered GGUF-to-resident conversion"
#endif

#define GROUP ((uint)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
#define TILE_ROWS 16u
#define TILE_COLUMNS 32u

// ---- GGUF source packet readers ------------------------------------------

#if defined(REPACK_IQ3S)
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
#endif

// Raw code of position `p` (0 <= p < GROUP) of one source packet.
inline uint source_code(device const uchar *in, uint p) {
#if defined(REPACK_KQUANT) && REPACK_CODE_BITS == 4u
    return (in[16 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
#elif defined(REPACK_KQUANT)
    uint low = (in[48 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
    uint high = (in[16 + p % 32] >> (p / 32)) & 1u;
    return low | (high << 4);
#elif defined(REPACK_Q6K)
    uint low = (in[(p / 128) * 64 + p % 64] >> ((p % 128 / 64) * 4)) & 15u;
    uint high = (in[128 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
    return low | (high << 4);
#elif defined(REPACK_Q3K)
    // q + 32 with q = low + 4 high - 4 (the registry's exact q6k re-basing).
    uint low = (in[32 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
    uint high = (in[p % 32] >> (p / 32)) & 1u;
    return (low | (high << 2)) + 28u;
#elif defined(REPACK_IQ3S)
    // 32 plus the signed grid magnitude (the registry's exact q6k re-basing).
    uint block = p / 32, group = p % 32 / 8, slot = p % 8;
    uint point = uint(in[2 + 8 * block + 2 * group + slot / 4]) |
                 (((uint(in[66 + block]) >> (2 * group + slot / 4)) & 1u) << 8);
    uint magnitude = (iq3s_grid[point] >> (8 * (slot % 4))) & 255u;
    return ((in[74 + 4 * block + group] >> slot) & 1u) != 0 ? 32u - magnitude : 32u + magnitude;
#elif defined(REPACK_Q8)
    return uint(in[2 + p]);
#elif defined(REPACK_IQ4NL)
    return (in[(p / 32) * 18 + 2 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
#elif defined(REPACK_BLOCK32) || defined(REPACK_NVFP4)
#if defined(REPACK_BLOCK32)
    // The low (p < 16) or high nibble of qs[p % 16], with bit p of qh.
    uint code = (in[REPACK_QS + p % 16] >> ((p / 16) * 4)) & 15u;
#else
    // Sixteen-value sub-block s = p / 16: the low (p % 16 < 8) or high
    // nibble of qs[8 s + p % 8].
    uint code = (in[4 + 8 * (p / 16) + p % 8] >> ((p % 16 / 8) * 4)) & 15u;
#endif
#if defined(REPACK_HIGH_BIT)
    code |= ((uint(in[REPACK_SCALE_BYTES + p / 8]) >> (p % 8)) & 1u) << 4;
#endif
#if defined(REPACK_E2M1)
    // E2M1 -0 (code 8) becomes ggml's +0 (code 0).
    code = code == 8u ? 0u : code;
#endif
    return code;
#else
    return (in[8 + (p / 32) * 16 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
#endif
}

#if defined(REPACK_KQUANT)
// Six-bit local coefficient `field` (0 scale, 1 minimum) of sub-block `j`.
inline uint kquant_local(device const uchar *in, uint j, uint field) {
    uint index = j % 4;
    uint low = uint(in[4 + field * 4 + index]);
    uint high = uint(in[12 + index]);
    return j < 4 ? (low & 63u) : (((high >> (4 * field)) & 15u) | ((low >> 6) << 4));
}

// Word `w` of the packed local coefficients: 16 six-bit fields, field
// `2j + f` of sub-block j at bit 12j + 6f (the registry `coefficients` /
// `scales` plane).
inline uint kquant_locals_word(device const uchar *in, uint w) {
    uint value = 0;
    for (uint field = 0; field < 16; ++field) {
        int shift = int(field * 6) - int(w * 32);
        if (shift <= -6 || shift >= 32) continue;
        uint local = kquant_local(in, field / 2, field % 2);
        value |= shift >= 0 ? local << uint(shift) : local >> uint(-shift);
    }
    return value;
}
#endif

#if defined(REPACK_IQ4)
// Resident f32 scale of IQ4_XS sub-block `j` (32 values).
inline float iq4_scale(device const uchar *in, uint j) {
    float base = float(as_type<half>(ushort(uint(in[0]) | (uint(in[1]) << 8))));
    uint low = (uint(in[4 + j / 2]) >> ((j % 2) * 4)) & 15u;
    uint high = ((uint(in[2]) | (uint(in[3]) << 8)) >> (2 * j)) & 3u;
    return base * float(int(low | (high << 4)) - 32);
}
#endif

// The per-group coefficient planes of one packet: `scales` (the packed
// locals) and `supers` (the per-group factors), in registry byte order.
inline void write_scales(device const uchar *in, device uchar *out) {
#if defined(REPACK_KQUANT)
    for (uint w = 0; w < 3; ++w)
        reinterpret_cast<device uint *>(out)[w] = kquant_locals_word(in, w);
#elif defined(REPACK_Q6K)
    for (uint w = 0; w < 4; ++w)
        reinterpret_cast<device uint *>(out)[w] = uint(in[192 + 4 * w]) | (uint(in[193 + 4 * w]) << 8) |
                                                   (uint(in[194 + 4 * w]) << 16) | (uint(in[195 + 4 * w]) << 24);
#elif defined(REPACK_Q3K)
    // Six-bit scale of sub-block j less 32, as a two's complement byte.
    for (uint j = 0; j < 16; ++j) {
        uint low = (uint(in[96 + j % 8]) >> (4 * (j / 8))) & 15u;
        uint high = (uint(in[104 + j % 4]) >> (2 * (j / 4))) & 3u;
        out[j] = uchar(int(low | (high << 4)) - 32);
    }
#elif defined(REPACK_IQ3S)
    // Both halves of 32-value sub-block b take 1 + 2 s, s its four-bit scale.
    for (uint j = 0; j < 16; ++j) out[j] = uchar(1u + 2u * ((uint(in[106 + j / 4]) >> (4 * (j / 2 % 2))) & 15u));
#endif
}

inline void write_supers(device const uchar *in, device uchar *out) {
#if defined(REPACK_KQUANT)
    reinterpret_cast<device uint *>(out)[0] =
        uint(in[0]) | (uint(in[1]) << 8) | (uint(in[2]) << 16) | (uint(in[3]) << 24);
#elif defined(REPACK_Q6K)
    reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[208]) | (uint(in[209]) << 8));
#elif defined(REPACK_Q3K)
    reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[108]) | (uint(in[109]) << 8));
#elif defined(REPACK_IQ3S)
    reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[0]) | (uint(in[1]) << 8));
#elif defined(REPACK_Q8)
    reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[0]) | (uint(in[1]) << 8));
#elif defined(REPACK_IQ4NL)
    // Each block's f16 scale, widened exactly.
    for (uint j = 0; j < 8; ++j)
        reinterpret_cast<device float *>(out)[j] =
            float(as_type<half>(ushort(uint(in[18 * j]) | (uint(in[18 * j + 1]) << 8))));
#elif defined(REPACK_SCALE_BYTES)
    // The block's leading scale fields (d, d | m, e or d[4]), bit for bit.
    for (uint i = 0; i < REPACK_SCALE_BYTES; ++i) out[i] = in[i];
#else
    for (uint j = 0; j < 8; ++j)
        reinterpret_cast<device float *>(out)[j] = iq4_scale(in, j);
#endif
}

// Eight consecutive `bits`-wide slices (code >> shift) of codes
// `first .. first + 8`, packed little-endian.
inline uint pack_eight(device const uchar *in, uint first, uint shift, uint bits) {
    uint value = 0;
    for (uint i = 0; i < 8; ++i)
        value |= ((source_code(in, first + i) >> shift) & ((1u << bits) - 1u)) << (i * bits);
    return value;
}

// Four consecutive eight-bit codes `first .. first + 4`.
inline uint pack_four_bytes(device const uchar *in, uint first) {
    uint value = 0;
    for (uint i = 0; i < 4; ++i) value |= source_code(in, first + i) << (8 * i);
    return value;
}

inline void zero_bytes(device uchar *out, ulong count) {
    for (ulong i = 0; i < count; ++i) out[i] = 0;
}

// Source packet `packet` of row `n` of matrix `b`.
inline device const uchar *source_packet(device const uchar *source, constant ulong *seismic_words,
                                         ulong b, ulong n, ulong packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

// ---- packet: planes interleaved per packet --------------------------------

inline device uchar *destination_packet(device uchar *destination, constant ulong *seismic_words,
                                        ulong b, ulong n, ulong packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

inline bool packet_plane_byte(uint byte) {
    bool inside = byte >= SEISMIC_ELEMENT_U_PLANE_0_OFFSET &&
                  byte < SEISMIC_ELEMENT_U_PLANE_0_OFFSET + SEISMIC_ELEMENT_U_PLANE_0_BYTES_PER_GROUP;
    inside = inside || (byte >= SEISMIC_ELEMENT_U_PLANE_1_OFFSET &&
                        byte < SEISMIC_ELEMENT_U_PLANE_1_OFFSET + SEISMIC_ELEMENT_U_PLANE_1_BYTES_PER_GROUP);
#if SEISMIC_ELEMENT_U_PLANE_COUNT > 2
    inside = inside || (byte >= SEISMIC_ELEMENT_U_PLANE_2_OFFSET &&
                        byte < SEISMIC_ELEMENT_U_PLANE_2_OFFSET + SEISMIC_ELEMENT_U_PLANE_2_BYTES_PER_GROUP);
#endif
#if SEISMIC_ELEMENT_U_PLANE_COUNT > 3
    inside = inside || (byte >= SEISMIC_ELEMENT_U_PLANE_3_OFFSET &&
                        byte < SEISMIC_ELEMENT_U_PLANE_3_OFFSET + SEISMIC_ELEMENT_U_PLANE_3_BYTES_PER_GROUP);
#endif
    return inside;
}

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    const uint lane = thread_position.x;
    const ulong b = tile.z;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    if (column0 >= packets * GROUP) return;
    const uint packet = column0 / GROUP;
    const uint within = column0 % GROUP;
    // The tile's codes are `bits` words of the packed `words` plane per row.
    const uint code_words = REPACK_CODE_BITS;
    for (uint item = lane; item < TILE_ROWS * code_words; item += TILE_COLUMNS) {
        ulong n = row0 + item / code_words;
        if (n >= SEISMIC_DIM_N) continue;
        device const uchar *in = source_packet(source, seismic_words, b, n, packet);
        uint word = within / 32 * code_words + item % code_words;
        uint first = (word * 32) / REPACK_CODE_BITS;
        uint last = (word * 32 + 31) / REPACK_CODE_BITS;
        uint value = 0;
        for (uint code = first; code <= last; ++code) {
            int shift = int(code * REPACK_CODE_BITS) - int(word * 32);
            uint raw = source_code(in, code);
            value |= shift >= 0 ? raw << uint(shift) : raw >> uint(-shift);
        }
        device uchar *out = destination_packet(destination, seismic_words, b, n, packet);
#if SEISMIC_ELEMENT_U_PACKET_SIZE % 4 == 0
        reinterpret_cast<device uint *>(out + SEISMIC_ELEMENT_U_PLANE_0_OFFSET)[word] = value;
#else
        // Byte-aligned packets (the E8M0 and UE4M3 scale planes): bytewise.
        for (uint byte = 0; byte < 4; ++byte)
            out[SEISMIC_ELEMENT_U_PLANE_0_OFFSET + 4 * word + byte] = uchar(value >> (8 * byte));
#endif
    }
    if (within != 0 || lane >= TILE_ROWS || row0 + lane >= SEISMIC_DIM_N) return;
    const ulong n = row0 + lane;
    device const uchar *in = source_packet(source, seismic_words, b, n, packet);
    device uchar *out = destination_packet(destination, seismic_words, b, n, packet);
#if defined(REPACK_KQUANT) || defined(REPACK_Q6K) || defined(REPACK_Q3K) || defined(REPACK_IQ3S)
    write_scales(in, out + SEISMIC_ELEMENT_U_PLANE_1_OFFSET);
    write_supers(in, out + SEISMIC_ELEMENT_U_PLANE_2_OFFSET);
#else
    write_supers(in, out + SEISMIC_ELEMENT_U_PLANE_1_OFFSET);
#endif
    for (uint byte = 0; byte < SEISMIC_ELEMENT_U_PACKET_SIZE; ++byte)
        if (!packet_plane_byte(byte)) out[byte] = 0;
}

#else

// ---- rows16 / mma16: row planes ---------------------------------------------

// Start of stored row `n` of matrix `b`.
inline device uchar *row_base(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1) *
                             SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
// Code of fragment slot `slot` of k16 step `step` (within k-block `block`)
// owned by `lane`: (row lane/4 + 8 (slot % 2), column 64 block + 16 step +
// 2 (lane % 4) + 8 ((slot % 4) / 2) + slot / 4). Zero beyond the matrix's
// rows or the source packets (tile padding).
inline uint fragment_code(device const uchar *source, constant ulong *seismic_words, ulong b, ulong row0,
                          ulong packets, uint block, uint step, uint lane, uint slot) {
    ulong n = row0 + lane / 4 + 8 * (slot % 2);
    ulong column = ulong(block) * 64 + step * 16 + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
    if (n >= SEISMIC_DIM_N || column >= packets * GROUP) return 0;
    return source_code(source_packet(source, seismic_words, b, n, column / GROUP), uint(column % GROUP));
}

// Eight `bits`-wide slices of one lane's slots of one step.
inline uint fragment_slice(device const uchar *source, constant ulong *seismic_words, ulong b, ulong row0,
                           ulong packets, uint block, uint step, uint lane, uint shift, uint bits) {
    uint value = 0;
    for (uint slot = 0; slot < 8; ++slot)
        value |= ((fragment_code(source, seismic_words, b, row0, packets, block, step, lane, slot) >> shift) &
                  ((1u << bits) - 1u)) << (slot * bits);
    return value;
}

// Destination of byte `position` of a tile's code-plane sequence V: the
// rows' plane payloads (`row_bytes` each) concatenated in row order.
inline device uchar *tile_plane_byte(device uchar *destination, constant ulong *seismic_words, ulong b,
                                     ulong row0, ulong position, ulong row_bytes, ulong offset) {
    return row_base(destination, seismic_words, b, row0 + position / row_bytes) + offset + position % row_bytes;
}
#endif

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    const uint lane = thread_position.x;
    const ulong b = tile.z;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const ulong stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * GROUP;
    const uint within = column0 % GROUP;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // Every row of a tile is stored; rows beyond N are zero padding.
    const ulong rows = TILE_ROWS;
#else
    const ulong rows = min(ulong(TILE_ROWS), SEISMIC_DIM_N - row0);
#endif

#if defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16)
    // Code planes: the code of column c sits at bit `c * bits` of the plane.
    if (lane < rows) {
        const ulong n = row0 + lane;
        device const uchar *in = source_packet(source, seismic_words, b, n, column0 / GROUP);
        device uchar *row = row_base(destination, seismic_words, b, n);
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES)
        device uint4 *codes = reinterpret_cast<device uint4 *>(row + SEISMIC_RESULT_0_PLANE_CODES_ROW_OFFSET + column0);
        codes[0] = uint4(pack_four_bytes(in, within), pack_four_bytes(in, within + 4),
                         pack_four_bytes(in, within + 8), pack_four_bytes(in, within + 12));
        codes[1] = uint4(pack_four_bytes(in, within + 16), pack_four_bytes(in, within + 20),
                         pack_four_bytes(in, within + 24), pack_four_bytes(in, within + 28));
#else
        *reinterpret_cast<device uint4 *>(row + SEISMIC_RESULT_0_PLANE_CODES_LO_ROW_OFFSET + column0 / 2) =
            uint4(pack_eight(in, within, 0, 4), pack_eight(in, within + 8, 0, 4),
                  pack_eight(in, within + 16, 0, 4), pack_eight(in, within + 24, 0, 4));
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI) && SEISMIC_ELEMENT_U_PLANE_CODES_HI_CODE_BITS == 1
        *reinterpret_cast<device uint *>(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + column0 / 8) =
            pack_eight(in, within, 4, 1) | (pack_eight(in, within + 8, 4, 1) << 8) |
            (pack_eight(in, within + 16, 4, 1) << 16) | (pack_eight(in, within + 24, 4, 1) << 24);
#elif defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
        *reinterpret_cast<device uint2 *>(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + column0 / 4) =
            uint2(pack_eight(in, within, 4, 2) | (pack_eight(in, within + 8, 4, 2) << 16),
                  pack_eight(in, within + 16, 4, 2) | (pack_eight(in, within + 24, 4, 2) << 16));
#endif
#endif
    }
#else
    // mma16: lane l owns, per 64-column k-block, 4 * bits bytes of each code
    // plane's tile sequence V (registry `PackedRowLayout::code_bit`); this
    // tile is k16 steps `step0` and `step0 + 1` of k-block `block`.
    {
        const uint block = column0 / 64;
        const uint step0 = (column0 % 64) / 16;
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES)
        {
            uint4 value;
            for (uint half_step = 0; half_step < 2; ++half_step) {
                uint low = 0, high = 0;
                for (uint slot = 0; slot < 4; ++slot) {
                    low |= fragment_code(source, seismic_words, b, row0, packets, block, step0 + half_step, lane, slot)
                           << (8 * slot);
                    high |= fragment_code(source, seismic_words, b, row0, packets, block, step0 + half_step, lane,
                                          slot + 4) << (8 * slot);
                }
                value[2 * half_step] = low;
                value[2 * half_step + 1] = high;
            }
            ulong position = (ulong(block) * 32 + lane) * 32 + 8 * step0;
            *reinterpret_cast<device uint4 *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                              SEISMIC_RESULT_0_PLANE_CODES_BYTES_PER_ROW,
                                                              SEISMIC_RESULT_0_PLANE_CODES_ROW_OFFSET)) = value;
        }
#else
        {
            ulong position = (ulong(block) * 32 + lane) * 16 + 4 * step0;
            *reinterpret_cast<device uint2 *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                              SEISMIC_RESULT_0_PLANE_CODES_LO_BYTES_PER_ROW,
                                                              SEISMIC_RESULT_0_PLANE_CODES_LO_ROW_OFFSET)) =
                uint2(fragment_slice(source, seismic_words, b, row0, packets, block, step0, lane, 0, 4),
                      fragment_slice(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 0, 4));
        }
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI) && SEISMIC_ELEMENT_U_PLANE_CODES_HI_CODE_BITS == 1
        {
            ulong position = (ulong(block) * 32 + lane) * 4 + step0;
            *reinterpret_cast<device ushort *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                               SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                                                               SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET)) =
                ushort(fragment_slice(source, seismic_words, b, row0, packets, block, step0, lane, 4, 1) |
                       (fragment_slice(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 4, 1) << 8));
        }
#elif defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
        {
            ulong position = (ulong(block) * 32 + lane) * 8 + 2 * step0;
            *reinterpret_cast<device uint *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                             SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                                                             SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET)) =
                fragment_slice(source, seismic_words, b, row0, packets, block, step0, lane, 4, 2) |
                (fragment_slice(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 4, 2) << 16);
        }
#endif
#endif
    }
#endif

    // Coefficient planes, once per group (at its first tile), per row.
    if (within == 0 && lane < rows) {
        const ulong n = row0 + lane;
        device uchar *row = row_base(destination, seismic_words, b, n);
        const ulong group = column0 / GROUP;
        const bool real = n < SEISMIC_DIM_N && occupied;
#if defined(SEISMIC_ELEMENT_U_PLANE_SCALES)
        device uchar *scales =
            row + SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET + group * SEISMIC_ELEMENT_U_PLANE_SCALES_BYTES_PER_GROUP;
        if (real) write_scales(source_packet(source, seismic_words, b, n, group), scales);
        else zero_bytes(scales, SEISMIC_ELEMENT_U_PLANE_SCALES_BYTES_PER_GROUP);
#endif
        device uchar *supers =
            row + SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET + group * SEISMIC_ELEMENT_U_PLANE_SUPERS_BYTES_PER_GROUP;
        if (real) write_supers(source_packet(source, seismic_words, b, n, group), supers);
        else zero_bytes(supers, SEISMIC_ELEMENT_U_PLANE_SUPERS_BYTES_PER_GROUP);
    }

    // Alignment tails of the coefficient planes and of the row, at the row's
    // last stored tile. Low code planes are whole multiples of 16 bytes; a
    // high-bit plane of 32-value groups may not be.
    if (column0 + TILE_COLUMNS == stored_columns && lane < rows) {
        device uchar *row = row_base(destination, seismic_words, b, row0 + lane);
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
#if defined(SEISMIC_ELEMENT_U_PLANE_SCALES)
        const ulong after_high = SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET;
#else
        const ulong after_high = SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET;
#endif
        zero_bytes(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                   after_high - SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET -
                       SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW);
#endif
#if defined(SEISMIC_ELEMENT_U_PLANE_SCALES)
        zero_bytes(row + SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET + SEISMIC_RESULT_0_PLANE_SCALES_BYTES_PER_ROW,
                   SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET - SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET -
                       SEISMIC_RESULT_0_PLANE_SCALES_BYTES_PER_ROW);
#endif
        zero_bytes(row + SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET + SEISMIC_RESULT_0_PLANE_SUPERS_BYTES_PER_ROW,
                   SEISMIC_RESULT_0_ROW_STRIDE_BYTES - SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET -
                       SEISMIC_RESULT_0_PLANE_SUPERS_BYTES_PER_ROW);
    }
}

#endif
