// The down conversion of `routed_bank_fill` (routed.seismic): SD -> EDW,
// `staged_down` -> `expert_down`, an [H, F] matrix.
// Generated with its siblings; build.rs checks that it defines every member
// lib/repack/repack.h and lib/streaming/window.h read.

#include "../repack/repack.h"

struct DownBinding {
    // The source element.
    static constexpr repack::Source source() {
#if defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q3_K)
        return repack::Source::gguf_q3_k;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_IQ3_S)
        return repack::Source::gguf_iq3_s;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q4_K)
        return repack::Source::gguf_q4_k;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q5_K)
        return repack::Source::gguf_q5_k;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q6_K)
        return repack::Source::gguf_q6_k;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q8_0)
        return repack::Source::gguf_q8_0;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_IQ4_NL)
        return repack::Source::gguf_iq4_nl;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_IQ4_XS)
        return repack::Source::gguf_iq4_xs;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q4_0)
        return repack::Source::gguf_q4_0;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_MXFP4)
        return repack::Source::gguf_mxfp4;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q5_0)
        return repack::Source::gguf_q5_0;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_Q5_1)
        return repack::Source::gguf_q5_1;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_NVFP4)
        return repack::Source::gguf_nvfp4;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_IQ4_NL_BLOCK)
        return repack::Source::gguf_iq4_nl_block;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_IQ2_XS)
        return repack::Source::gguf_iq2_xs;
#elif defined(SEISMIC_ELEMENT_SD_REPRESENTATION_GGUF_IQ3_XXS)
        return repack::Source::gguf_iq3_xxs;
#else
        return repack::Source::other;
#endif
    }
    static constexpr bool external() {
#if defined(SEISMIC_ELEMENT_SD_KIND_EXTERNAL)
        return true;
#else
        return false;
#endif
    }
    static constexpr uint group() {
#if defined(SEISMIC_ELEMENT_SD_LOGICAL_GROUP)
        return uint(SEISMIC_ELEMENT_SD_LOGICAL_GROUP);
#else
        return 0;
#endif
    }
    static constexpr uint source_packet_size() {
#if defined(SEISMIC_ELEMENT_SD_PACKET_SIZE)
        return uint(SEISMIC_ELEMENT_SD_PACKET_SIZE);
#else
        return 0;
#endif
    }
    // The resident element.
    static constexpr repack::Resident resident() {
#if defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q4K)
        return repack::Resident::q4k;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q5K)
        return repack::Resident::q5k;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q6K)
        return repack::Resident::q6k;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q8G32S)
        return repack::Resident::q8g32s;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_IQ4G32)
        return repack::Resident::iq4g32;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q4G32S)
        return repack::Resident::q4g32s;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_MXFP4G32)
        return repack::Resident::mxfp4g32;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q5G32S)
        return repack::Resident::q5g32s;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_Q5G32)
        return repack::Resident::q5g32;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_NVFP4G16)
        return repack::Resident::nvfp4g16;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_IQ4G32S)
        return repack::Resident::iq4g32s;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_IQ2XS)
        return repack::Resident::iq2xs;
#elif defined(SEISMIC_ELEMENT_EDW_REPRESENTATION_IQ3XXS)
        return repack::Resident::iq3xxs;
#else
        return repack::Resident::other;
#endif
    }
    static constexpr bool packed() {
#if defined(SEISMIC_ELEMENT_EDW_KIND_PACKED)
        return true;
#else
        return false;
#endif
    }
    static constexpr repack::Layout layout() {
#if defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET)
        return repack::Layout::packet;
#elif defined(SEISMIC_ELEMENT_EDW_LAYOUT_ROWS16)
        return repack::Layout::rows16;
#elif defined(SEISMIC_ELEMENT_EDW_LAYOUT_ROWS32)
        return repack::Layout::rows32;
#elif defined(SEISMIC_ELEMENT_EDW_LAYOUT_MMA16)
        return repack::Layout::mma16;
#else
        return repack::Layout::other;
#endif
    }
    static constexpr uint packet_size() {
#if defined(SEISMIC_ELEMENT_EDW_PACKET_SIZE)
        return uint(SEISMIC_ELEMENT_EDW_PACKET_SIZE);
#else
        return 0;
#endif
    }
    static constexpr uint plane_count() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_COUNT)
        return uint(SEISMIC_ELEMENT_EDW_PLANE_COUNT);
#else
        return 0;
#endif
    }
    static constexpr uint plane_offset(uint plane) {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_0_OFFSET)
        if (plane == 0) return uint(SEISMIC_ELEMENT_EDW_PLANE_0_OFFSET);
#endif
#if defined(SEISMIC_ELEMENT_EDW_PLANE_1_OFFSET)
        if (plane == 1) return uint(SEISMIC_ELEMENT_EDW_PLANE_1_OFFSET);
#endif
#if defined(SEISMIC_ELEMENT_EDW_PLANE_2_OFFSET)
        if (plane == 2) return uint(SEISMIC_ELEMENT_EDW_PLANE_2_OFFSET);
#endif
#if defined(SEISMIC_ELEMENT_EDW_PLANE_3_OFFSET)
        if (plane == 3) return uint(SEISMIC_ELEMENT_EDW_PLANE_3_OFFSET);
#endif
        return 0u;
    }
    static constexpr uint plane_bytes_per_group(uint plane) {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_0_BYTES_PER_GROUP)
        if (plane == 0) return uint(SEISMIC_ELEMENT_EDW_PLANE_0_BYTES_PER_GROUP);
#endif
#if defined(SEISMIC_ELEMENT_EDW_PLANE_1_BYTES_PER_GROUP)
        if (plane == 1) return uint(SEISMIC_ELEMENT_EDW_PLANE_1_BYTES_PER_GROUP);
#endif
#if defined(SEISMIC_ELEMENT_EDW_PLANE_2_BYTES_PER_GROUP)
        if (plane == 2) return uint(SEISMIC_ELEMENT_EDW_PLANE_2_BYTES_PER_GROUP);
#endif
#if defined(SEISMIC_ELEMENT_EDW_PLANE_3_BYTES_PER_GROUP)
        if (plane == 3) return uint(SEISMIC_ELEMENT_EDW_PLANE_3_BYTES_PER_GROUP);
#endif
        return 0u;
    }
    static constexpr bool codes() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_CODES)
        return true;
#else
        return false;
#endif
    }
    static constexpr bool grids() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_GRIDS)
        return true;
#else
        return false;
#endif
    }
    static constexpr bool codes_hi() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_CODES_HI)
        return true;
#else
        return false;
#endif
    }
    static constexpr uint codes_hi_code_bits() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_CODES_HI_CODE_BITS)
        return uint(SEISMIC_ELEMENT_EDW_PLANE_CODES_HI_CODE_BITS);
#else
        return 0;
#endif
    }
    static constexpr bool scales() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_SCALES)
        return true;
#else
        return false;
#endif
    }
    static constexpr uint scales_bytes_per_group() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_SCALES_BYTES_PER_GROUP)
        return uint(SEISMIC_ELEMENT_EDW_PLANE_SCALES_BYTES_PER_GROUP);
#else
        return 0;
#endif
    }
    static constexpr uint supers_bytes_per_group() {
#if defined(SEISMIC_ELEMENT_EDW_PLANE_SUPERS_BYTES_PER_GROUP)
        return uint(SEISMIC_ELEMENT_EDW_PLANE_SUPERS_BYTES_PER_GROUP);
#else
        return 0;
#endif
    }
    // The conversion's geometry.
    static ulong n(constant ulong *seismic_words) {
        return ulong(SEISMIC_DIM_H);
    }
    static ulong k(constant ulong *seismic_words) {
        return ulong(SEISMIC_DIM_F);
    }
    static ulong source_stride_0(constant ulong *seismic_words) {
        return ulong(SEISMIC_STAGED_DOWN_STRIDE_0);
    }
    static ulong source_stride_1(constant ulong *seismic_words) {
        return ulong(SEISMIC_STAGED_DOWN_STRIDE_1);
    }
    static ulong source_stride_2(constant ulong *seismic_words) {
        return ulong(SEISMIC_STAGED_DOWN_STRIDE_2);
    }
    static ulong out_stride_0(constant ulong *seismic_words) {
        return ulong(SEISMIC_EXPERT_DOWN_STRIDE_0);
    }
    static ulong out_stride_1(constant ulong *seismic_words) {
        return ulong(SEISMIC_EXPERT_DOWN_STRIDE_1);
    }
    static ulong out_stride_2(constant ulong *seismic_words) {
        return ulong(SEISMIC_EXPERT_DOWN_STRIDE_2);
    }
    static ulong out_row_stride_bytes(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET)
        return ulong(SEISMIC_EXPERT_DOWN_ROW_STRIDE_BYTES);
#else
        return 0ul;
#endif
    }
    static ulong out_row_groups(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET)
        return ulong(SEISMIC_EXPERT_DOWN_ROW_GROUPS);
#else
        return 0ul;
#endif
    }
    static ulong out_codes_row_offset(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_CODES)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_CODES_ROW_OFFSET);
#else
        return 0ul;
#endif
    }
    static ulong out_codes_bytes_per_row(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_CODES)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_CODES_BYTES_PER_ROW);
#else
        return 0ul;
#endif
    }
    static ulong out_codes_lo_row_offset(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_CODES_LO)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_CODES_LO_ROW_OFFSET);
#else
        return 0ul;
#endif
    }
    static ulong out_codes_lo_bytes_per_row(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_CODES_LO)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_CODES_LO_BYTES_PER_ROW);
#else
        return 0ul;
#endif
    }
    static ulong out_grids_row_offset(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_GRIDS)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_GRIDS_ROW_OFFSET);
#else
        return 0ul;
#endif
    }
    static ulong out_grids_bytes_per_row(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_GRIDS)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_GRIDS_BYTES_PER_ROW);
#else
        return 0ul;
#endif
    }
    static ulong out_codes_hi_row_offset(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_CODES_HI)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_CODES_HI_ROW_OFFSET);
#else
        return 0ul;
#endif
    }
    static ulong out_codes_hi_bytes_per_row(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_CODES_HI)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_CODES_HI_BYTES_PER_ROW);
#else
        return 0ul;
#endif
    }
    static ulong out_scales_row_offset(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_SCALES)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_SCALES_ROW_OFFSET);
#else
        return 0ul;
#endif
    }
    static ulong out_scales_bytes_per_row(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET) && defined(SEISMIC_ELEMENT_EDW_PLANE_SCALES)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_SCALES_BYTES_PER_ROW);
#else
        return 0ul;
#endif
    }
    static ulong out_supers_row_offset(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_SUPERS_ROW_OFFSET);
#else
        return 0ul;
#endif
    }
    static ulong out_supers_bytes_per_row(constant ulong *seismic_words) {
#if !defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET)
        return ulong(SEISMIC_EXPERT_DOWN_PLANE_SUPERS_BYTES_PER_ROW);
#else
        return 0ul;
#endif
    }
};
