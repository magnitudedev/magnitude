template<typename T, uint SLOTS, bool SHARED, typename Scores, typename SharedScore>
struct RouteStep {
    Scores scores;
    SharedScore shared_score;
    using State = float;
    using Result = MagnitudeFragment<T, 4>;
    float initial() const { return 0.0f; }
    float step(float state, T value, uint row, uint slot) const {
        constexpr uint TOPK = SLOTS - SHARED;
        T score = SHARED && slot == TOPK ? shared_score[row] : scores[row * TOPK + slot];
        T contribution = T(value * score);
        return float(T(state + float(contribution)));
    }
    T finish(float state) const { return T(state); }
};
