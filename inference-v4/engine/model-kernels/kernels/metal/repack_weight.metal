inline uint read_bits(device const uchar *source, uint bit, uint width) {
    uint value = 0;
    for (uint offset = 0; offset < width; ++offset)
        value |= uint((source[(bit + offset) >> 3] >> ((bit + offset) & 7)) & 1) << offset;
    return value;
}

inline void write_bits(device uchar *destination, uint bit, uint width, uint value) {
    for (uint offset = 0; offset < width; ++offset) {
        uint target = bit + offset;
        uchar mask = uchar(1u << (target & 7));
        uchar encoded = uchar(((value >> offset) & 1u) << (target & 7));
        destination[target >> 3] = uchar((destination[target >> 3] & ~mask) | encoded);
    }
}

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong logical = ulong(raw_index);
    if (logical >= SEISMIC_DIM_N || logical % SEISMIC_ELEMENT_E_LOGICAL_GROUP != 0) return;
    ulong packet = logical / SEISMIC_ELEMENT_E_LOGICAL_GROUP;
    device const uchar *input = source + packet * SEISMIC_ELEMENT_E_PACKET_SIZE;
    device uchar *output = destination + packet * SEISMIC_ELEMENT_U_PACKET_SIZE;

#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q8_0) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q8G32S)
    for (uint byte = 0; byte < 32; ++byte)
        output[SEISMIC_ELEMENT_U_PLANE_0_OFFSET + byte] = input[2 + byte];
    output[SEISMIC_ELEMENT_U_PLANE_1_OFFSET] = input[0];
    output[SEISMIC_ELEMENT_U_PLANE_1_OFFSET + 1] = input[1];
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q4K)
    for (uint position = 0; position < 256; ++position) {
        uint source_byte = 16 + (position / 64) * 32 + position % 32;
        uint code = (input[source_byte] >> ((position % 64 / 32) * 4)) & 15;
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_0_OFFSET, position * 4, 4, code);
    }
    for (uint group = 0; group < 8; ++group) {
        uint index = group % 4;
        uint scale_low = uint(input[4 + index]);
        uint bias_low = uint(input[8 + index]);
        uint high = uint(input[12 + index]);
        uint scale = group < 4 ? scale_low & 63u
            : (high & 15u) | ((scale_low >> 6) << 4);
        uint bias = group < 4 ? bias_low & 63u
            : (high >> 4) | ((bias_low >> 6) << 4);
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_1_OFFSET, group * 12, 6, scale);
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_1_OFFSET, group * 12 + 6, 6, bias);
    }
    output[SEISMIC_ELEMENT_U_PLANE_2_OFFSET] = input[0];
    output[SEISMIC_ELEMENT_U_PLANE_2_OFFSET + 1] = input[1];
    output[SEISMIC_ELEMENT_U_PLANE_3_OFFSET] = input[2];
    output[SEISMIC_ELEMENT_U_PLANE_3_OFFSET + 1] = input[3];
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q5_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q5K)
    for (uint position = 0; position < 256; ++position) {
        uint low_byte = 48 + (position / 64) * 32 + position % 32;
        uint low = (input[low_byte] >> ((position % 64 / 32) * 4)) & 15;
        uint high = (input[16 + position % 32] >> (position / 32)) & 1;
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_0_OFFSET, position * 5, 5, low | (high << 4));
    }
    for (uint group = 0; group < 8; ++group) {
        uint index = group % 4;
        uint scale_low = uint(input[4 + index]);
        uint bias_low = uint(input[8 + index]);
        uint high = uint(input[12 + index]);
        uint scale = group < 4 ? scale_low & 63u
            : (high & 15u) | ((scale_low >> 6) << 4);
        uint bias = group < 4 ? bias_low & 63u
            : (high >> 4) | ((bias_low >> 6) << 4);
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_1_OFFSET, group * 12, 6, scale);
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_1_OFFSET, group * 12 + 6, 6, bias);
    }
    output[SEISMIC_ELEMENT_U_PLANE_2_OFFSET] = input[0];
    output[SEISMIC_ELEMENT_U_PLANE_2_OFFSET + 1] = input[1];
    output[SEISMIC_ELEMENT_U_PLANE_3_OFFSET] = input[2];
    output[SEISMIC_ELEMENT_U_PLANE_3_OFFSET + 1] = input[3];
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q6_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q6K)
    for (uint position = 0; position < 256; ++position) {
        uint low_byte = (position / 128) * 64 + position % 64;
        uint low = (input[low_byte] >> ((position % 128 / 64) * 4)) & 15;
        uint high_byte = 128 + (position / 128) * 32 + position % 32;
        uint high = (input[high_byte] >> ((position % 128 / 32) * 2)) & 3;
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_0_OFFSET, position * 6, 6, low | (high << 4));
    }
    for (uint byte = 0; byte < 16; ++byte)
        output[SEISMIC_ELEMENT_U_PLANE_1_OFFSET + byte] = input[192 + byte];
    output[SEISMIC_ELEMENT_U_PLANE_2_OFFSET] = input[208];
    output[SEISMIC_ELEMENT_U_PLANE_2_OFFSET + 1] = input[209];
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_XS) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_IQ4G32)
    for (uint position = 0; position < 256; ++position) {
        uint source_byte = 8 + (position / 32) * 16 + position % 16;
        uint code = (input[source_byte] >> ((position % 32 / 16) * 4)) & 15;
        write_bits(output + SEISMIC_ELEMENT_U_PLANE_0_OFFSET, position * 4, 4, code);
    }
    float base = float(as_type<half>(ushort(read_bits(input, 0, 16))));
    for (uint group = 0; group < 8; ++group) {
        uint low = read_bits(input, (4 + group / 2) * 8 + (group % 2) * 4, 4);
        uint high = read_bits(input, 16 + 2 * group, 2);
        float scale = base * float(int(low | (high << 4)) - 32);
        reinterpret_cast<device float *>(output + SEISMIC_ELEMENT_U_PLANE_1_OFFSET)[group] = scale;
    }
#else
#error "repack_weight binding is not a qualified GGUF-to-resident pair"
#endif
}
