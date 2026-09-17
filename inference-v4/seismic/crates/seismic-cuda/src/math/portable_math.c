/* Offline generation only. Customer execution embeds the generated PTX and does
   not invoke a C compiler or depend on a CUDA toolkit. See THIRD_PARTY.md. */
#include "libm.h"
double floor(double x) { return __builtin_floor(x); }
#include "musl-1.2.5/scalbn.c"
#include "musl-1.2.5/__rem_pio2_large.c"
#include "musl-1.2.5/__rem_pio2f.c"
#include "musl-1.2.5/__sindf.c"
#include "musl-1.2.5/__cosdf.c"
#include "musl-1.2.5/sinf.c"
#include "musl-1.2.5/cosf.c"
#include "musl-1.2.5/logf_data.c"
#include "musl-1.2.5/logf.c"
