uint d = thread_position_in_grid.x;      // element in [0, D)
if (d >= D) return;
uint b = thread_position_in_grid.y;      // token row
constexpr int KW = (D * BITS) / 32;
constexpr int G = D / GROUP;
int token = int(tok[b]);
uint row = uint(token < 0 ? token + VOCAB : token);
const device uint* wrow = weight + row * KW;
uint bp = d * BITS;
uint wi = bp >> 5, sh = bp & 31u;
uint v = wrow[wi] >> sh;
if (sh + BITS > 32u) v |= wrow[wi + 1] << (32u - sh);
uint qv = v & ((1u << BITS) - 1u);
T sc = scales[row * G + d / GROUP];
T bi = biases[row * G + d / GROUP];
out[b * D + d] = T(metal::fma(float(sc), float(qv), float(bi)));
