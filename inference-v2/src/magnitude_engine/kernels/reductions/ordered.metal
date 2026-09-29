// Complete each producer before an ordered state update; no slot reassociation.
template<uint SLOTS, typename Producer, typename Reducer>
inline typename Reducer::Result magnitude_ordered_fragments(
    uint row, uint first, uint lane, const thread Producer& producer, const thread Reducer& reducer) {
    typename Reducer::State state[4];
    for (uint c = 0; c < 4; ++c) state[c] = reducer.initial();
    for (uint slot = 0; slot < SLOTS; ++slot) {
        auto tile = producer(row * SLOTS + slot, first, lane);
        for (uint c = 0; c < 4; ++c) state[c] = reducer.step(state[c], tile.values[c], row, slot);
    }
    typename Reducer::Result result;
    for (uint c = 0; c < 4; ++c) result.values[c] = reducer.finish(state[c]);
    return result;
}
