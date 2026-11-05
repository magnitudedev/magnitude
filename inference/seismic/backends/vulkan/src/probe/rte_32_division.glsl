// `rte_32_division`: fp32 add, multiply and divide under `RoundingModeRTE 32`
// (the sealed environment with it declared). Buffer 0: [count][count pairs
// a, b][count triples a + b, a * b, a / b], as bits.
void main() {
    probe_u32 data = probe_u32(PROBE_ARGUMENT(0));
    const uint count = data.v[0];
    const uint i = gl_GlobalInvocationID.x;
    if (i >= count)
        return;
    const float a = uintBitsToFloat(data.v[1u + 2u * i]);
    const float b = uintBitsToFloat(data.v[2u + 2u * i]);
    const uint result = 1u + 2u * count + 3u * i;
    data.v[result] = floatBitsToUint(a + b);
    data.v[result + 1u] = floatBitsToUint(a * b);
    data.v[result + 2u] = floatBitsToUint(a / b);
}
