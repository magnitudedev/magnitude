//! The Seismic CPU library, `seismic::cpu`: what every CPU native kernel
//! author builds on. It is the CPU counterpart of the GPU native libraries
//! (`<seismic/element.h>`, `<seismic/packets.h>`): universal, with no model
//! knowledge.
//!
//! - [`isa`]: instruction-set tiers, their token types and detection;
//! - [`element`]: the dense element types and their conversions;
//! - [`tensor`]: typed views of the tensors a kernel receives;
//! - [`reduce`], [`math`]: defined-order reductions and scalar functions;
//! - [`weights`]: weight representations, their row geometry and the bodies
//!   of their components;
//! - [`components`]: the compiled component instances and their resolution;
//! - [`repack`]: exact conversions of external weights into resident rows.
//!
//! Everything a kernel calls is `#[inline(always)]` and compiles with the
//! features of the generated tier boundary it inlines into; only component
//! calls cross a function boundary.

pub mod components;
pub mod element;
pub mod isa;
pub mod math;
pub mod quant;
pub mod reduce;
pub mod repack;
pub mod tensor;
pub mod weights;

pub use components::WeightKernels;
pub use element::{Bf16, Dense, Element, F16, F32, I32, U32};
pub use isa::{Isa, Tier};
pub use repack::{External, Packed};
pub use tensor::{Scalars, Scratch, Tensor};
pub use weights::RowGeometry;

#[cfg(target_arch = "aarch64")]
pub use isa::Neon;
#[cfg(target_arch = "x86_64")]
pub use isa::{X86V4Vnni, X86V2, X86V3, X86V4};

/// The version of this library, part of every CPU implementation digest.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+cpu-library-3");

/// A bound weight operand: a tensor of a weight representation whose rows
/// are read through its resolved components.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a> {
    base: *const u8,
    rows: usize,
    k: usize,
    geometry: RowGeometry,
    kernels: &'static WeightKernels,
    life: std::marker::PhantomData<&'a u8>,
}

unsafe impl Send for Weights<'_> {}
unsafe impl Sync for Weights<'_> {}

impl<'a> Weights<'a> {
    /// A weight operand of `rows` rows of `k` values at `base`;
    /// `dense_stride` is the tensor's row stride in bytes (dense rows only).
    ///
    /// # Safety
    /// `base` addresses `rows` rows of the resolved representation, valid
    /// for `'a`.
    pub unsafe fn from_raw(
        base: *const u8,
        rows: usize,
        k: usize,
        dense_stride: usize,
        kernels: &'static WeightKernels,
    ) -> Self {
        Self {
            base,
            rows,
            k,
            geometry: kernels
                .geometry(k, dense_stride)
                .with_base(base, rows.max(1)),
            kernels,
            life: std::marker::PhantomData,
        }
    }

    /// The weight operand of a bound tensor: its rows are the positions of
    /// every axis before the last, its values the last axis.
    ///
    /// # Safety
    /// `base`, `extents` and `strides` (in elements) describe a tensor of the
    /// resolved representation valid for `'a`.
    pub unsafe fn from_tensor(
        base: *const u8,
        extents: &[u64],
        strides: &[u64],
        kernels: &'static WeightKernels,
    ) -> Self {
        let rank = extents.len();
        assert!(rank >= 1, "a weight operand has a value axis");
        let k = extents[rank - 1] as usize;
        let rows = extents[..rank - 1].iter().product::<u64>() as usize;
        let dense_stride = if kernels.dense_bytes == 0 {
            0
        } else {
            assert_eq!(
                strides[rank - 1],
                1,
                "a dense weight operand has contiguous rows"
            );
            for axis in 0..rank.saturating_sub(2) {
                assert_eq!(
                    strides[axis],
                    strides[axis + 1] * extents[axis + 1],
                    "a dense weight operand's rows are evenly spaced"
                );
            }
            if rank >= 2 {
                strides[rank - 2] as usize * kernels.dense_bytes
            } else {
                k * kernels.dense_bytes
            }
        };
        let mut view = unsafe { Self::from_raw(base, rows.max(1), k, dense_stride, kernels) };
        view.geometry.matrix_rows = if rank >= 2 {
            extents[rank - 2] as usize
        } else {
            1
        };
        view
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Values per row.
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn representation(&self) -> &'static str {
        self.kernels.representation
    }

    /// The `k` values of row `row`.
    #[inline(always)]
    pub fn decode_row(&self, row: usize, out: &mut [f32]) {
        assert!(row < self.rows, "row {row} of {}", self.rows);
        // SAFETY: the row lies inside the operand (`from_raw`).
        unsafe {
            self.kernels.decode(
                self.base.add(row * self.geometry.stride),
                &self.geometry,
                self.k,
                out,
            )
        }
    }

    /// `out[r] = (row first + r) · x` for a block of `out.len()` rows (1, 2,
    /// 4 or 8): one component call across the whole row.
    #[inline(always)]
    pub fn dot(&self, first: usize, x: &[f32], out: &mut [f32]) {
        assert!(
            first + out.len() <= self.rows,
            "rows {first}..{} of {}",
            first + out.len(),
            self.rows
        );
        assert_eq!(x.len(), self.k, "activation length");
        // SAFETY: the block lies inside the operand (`from_raw`).
        unsafe {
            self.kernels.dot(
                self.base.add(first * self.geometry.stride),
                &self.geometry,
                x,
                out,
            )
        }
    }

    /// Four quantized activation rows by eight weight rows.
    #[inline(always)]
    pub fn gemm_q8(&self, first: usize, x: &[quant::Q8Block], out: &mut [f32; 32]) {
        assert!(first + 8 <= self.rows);
        // SAFETY: the eight rows lie inside the operand.
        unsafe {
            self.kernels.gemm_q8(
                self.base.add(first * self.geometry.stride),
                &self.geometry,
                x,
                self.k,
                out,
            )
        }
    }

    /// As [`Weights::dot`], against an activation row quantized into `x`
    /// (`quant::blocks(k)` blocks, see [`quant`]).
    #[inline(always)]
    pub fn dot_q8(&self, first: usize, x: &[quant::Q8Block], out: &mut [f32]) {
        assert!(
            first + out.len() <= self.rows,
            "rows {first}..{} of {}",
            first + out.len(),
            self.rows
        );
        // SAFETY: the block lies inside the operand (`from_raw`).
        unsafe {
            self.kernels.dot_q8(
                self.base.add(first * self.geometry.stride),
                &self.geometry,
                x,
                self.k,
                out,
            )
        }
    }
}
