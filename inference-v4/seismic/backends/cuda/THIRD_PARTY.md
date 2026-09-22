# CUDA scalar math provenance

`src/math/exp.ptx` adapts `expf` from rust-lang/libm 0.2.16, originally
FreeBSD msun / SunPro. Its bounded power-of-two scaling follows libm's generic
`scalbn` algorithm. Integer exponent range is established by the exponential's
input guards; redundant general scalbn cases are omitted. Floating-point exception
flags are not part of Seismic's value contract. No fast-math contraction or flush
to zero is permitted in this implementation.

Source: https://github.com/rust-lang/compiler-builtins/tree/dfd2203a4d6110820ad7bb65cafe1bf331a03a3d/libm/src/math

/* origin: FreeBSD /usr/src/lib/msun/src/e_expf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


rust-lang/libm as a whole is available for use under the MIT license:

------------------------------------------------------------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------


## Logarithm and trigonometry

`src/math/portable_math.ptx` implements Seismic's scalar log/sin/cos backend
primitives from the unchanged vendored musl 1.2.5 sources. This is backend primitive
implementation, not an alternate execution path for model tensor kernels.

- Release: https://musl.libc.org/releases/musl-1.2.5.tar.gz
- Archive SHA-256: `a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4`
- Per-file identities: `src/math/musl-1.2.5/source.json`.
- Original notices remain in every source; the complete release license is retained
  in `src/math/musl-1.2.5/COPYRIGHT`. Trigonometric functions originate in SunPro /
  FreeBSD msun. The logarithm is the Arm 2017–2018 MIT implementation.
- Offline generator: `src/math/generate.py`, Clang 18.1.3 NVPTX, explicit PTX 7.0,
  SM80, no contraction, no fast math. Exact flags, adapter and output hashes are in
  `src/math/generation.json`. The generator verifies immutable source hashes and
  rejects external dependencies, flush-to-zero and contraction in its output.
- Adapter headers supply target types, bit casts and scalar exception results; host
  errno/fenv flags are outside Seismic's value contract. Round-to-nearest is required.
  Large-argument trigonometric reduction is preserved. No reduced argument domain or
  device approximate sin/cos/log instruction substitutes for the portable operation.
- Customers execute embedded PTX through the CUDA driver; neither Clang nor a CUDA
  toolkit is invoked or required at customer compile/run time.

Numerical qualification is empirical across recorded independent-reference samples;
it is not an exhaustive proof over all binary32 inputs. Physical math instruction,
lookup-table and local-stack consumption still needs backend accounting mapping.
