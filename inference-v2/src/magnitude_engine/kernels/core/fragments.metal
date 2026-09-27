// All lanes own replicas of every completed value. Only the selected lane stores.
template<typename T, uint ITEMS>
struct MagnitudeFragment { T values[ITEMS]; };
