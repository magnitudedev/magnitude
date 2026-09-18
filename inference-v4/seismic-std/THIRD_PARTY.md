# Third-party numerical code

`lib/kernels/gelu.seismic.portable` adapts the error-function rational
approximations in rust-lang/libm 0.2.16, `src/math/erff.rs`, originating from
FreeBSD msun `s_erff.c`. The adaptation expresses the work as portable Seismic
and uses a bounded arithmetic truncation for the tail exponent split.

Source: https://docs.rs/crate/libm/0.2.16/source/src/math/erff.rs

Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
