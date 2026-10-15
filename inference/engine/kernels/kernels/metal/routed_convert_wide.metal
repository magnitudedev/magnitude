// `routed_convert_wide`: a scratch fill round's convert, one threadgroup per
// row tile of every listed record (lib/streaming/convert.h).

#define CONVERT_KERNEL routed_convert_wide
#include "lib/streaming/convert.h"
