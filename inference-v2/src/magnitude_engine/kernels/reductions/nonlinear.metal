// Native scalar boundaries used by the ordered gated-delta equation.
// The exponential flavor is a numerical choice, independent of launch geometry.
template <typename T, bool PRECISE>
T magnitude_native_sigmoid(float x) {
    T exponential = T(PRECISE ? metal::precise::exp(metal::abs(x))
                             : metal::exp(metal::abs(x)));
    T denominator = T(1.0f + float(exponential));
    T reciprocal = T(1.0f / float(denominator));
    return x < 0.0f ? reciprocal : T(1.0f - float(reciprocal));
}

template <typename T>
float magnitude_decay(T input, T bias, float log_rate) {
    T value = T(float(input) + float(bias));
    T maximum = metal::max(value, T(0));
    T minimum = metal::min(value, T(0));
    T softplus = maximum + log1p(metal::exp(minimum - maximum));
    return metal::precise::exp(-metal::precise::exp(log_rate) * softplus);
}
