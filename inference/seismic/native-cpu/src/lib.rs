//! The Seismic CPU library, `seismic::cpu`: what every CPU native kernel
//! author builds on. It is the CPU counterpart of the GPU native libraries
//! (`<seismic/element.h>`, `<seismic/packets.h>`): universal, with no model
//! knowledge.
//!
//! - [`isa`]: instruction-set tiers, their token types and detection;
//! - [`element`]: the dense element types and their conversions;
//! - [`tensor`]: typed views of the tensors a kernel receives;
//! - [`reduce`], [`math`]: defined-order reductions and scalar functions;
//! - [`weights`]: weight elements, their row geometry and the bodies of
//!   their components;
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
pub mod slab;
pub mod tensor;
pub mod weights;

pub use components::WeightKernels;
pub use element::{Bf16, Dense, Element, F16, F32, I32, U32, U64};
pub use isa::{Isa, Tier};
pub use repack::{External, Packed};
pub use tensor::{Scalars, Scratch, Tensor};
pub use weights::RowGeometry;

use seismic_lang::ids::RepresentationId;
use seismic_lang::registry;

/// Re-exports the tier tokens of this architecture, one per tier of
/// `seismic_cpu_tiers::tiers!`.
macro_rules! token_exports {
    ($($token:ident $name:ident $arch:tt [$($feature:tt),*];)*) => {
        $(
            #[cfg(target_arch = $arch)]
            pub use isa::$token;
        )*
    };
}

seismic_cpu_tiers::tiers!(token_exports);

/// The version of this library's kernel code, part of every CPU
/// implementation digest; bumped in `seismic-cpu-tiers`.
pub use seismic_cpu_tiers::VERSION;

/// A weight buffer's element as a call reads it: its storage element, that
/// element's resolved components (which read it in its own layout, a
/// device's too: `weights::Placement`), whether products over it take their
/// activation rows as NVFP4 operands (the registry's operand recipe,
/// `operands(W)`), and whether they take quantized activation rows under a
/// declaration's `INT8 = 1` (the quantized dot's admission,
/// `form(W, INT8)`).
#[derive(Clone, Copy, Debug)]
pub struct WeightOperand {
    kernels: &'static WeightKernels,
    storage: &'static registry::RepresentationInfo,
    nvfp4_operands: bool,
    quantized: bool,
}

impl WeightOperand {
    /// The weight operand of element `element` on `tier`, or `None` when
    /// `element` is no weight element the components read.
    pub fn of(tier: Tier, element: RepresentationId) -> Option<Self> {
        let storage = registry::representation_info(registry::storage_of(element));
        Some(Self {
            kernels: components::resolve(tier, storage.id)?,
            storage,
            nvfp4_operands: registry::representation_info(element).operands
                == registry::OperandRecipe::Nvfp4Rows,
            quantized: seismic_lang::forms::cpu_int8_of(element),
        })
    }

    /// Bytes of one value of dense rows; `None` for packed rows.
    pub fn dense_bytes(&self) -> Option<usize> {
        match self.storage.kind {
            registry::RepresentationKind::Dense(dtype) => Some(dtype.bytes() as usize),
            _ => None,
        }
    }

    /// Whether the operand multiplies NVFP4 operand rows (the weight's W4A4
    /// recipe).
    pub fn nvfp4_operands(&self) -> bool {
        self.nvfp4_operands
    }
}

/// A bound weight operand: a tensor of a weight representation whose rows
/// are read through its resolved components.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a> {
    base: *const u8,
    rows: usize,
    k: usize,
    geometry: RowGeometry,
    kernels: &'static WeightKernels,
    nvfp4_operands: bool,
    quantized: bool,
    /// The storage the components read.
    storage: &'static registry::RepresentationInfo,
    life: std::marker::PhantomData<&'a u8>,
}

unsafe impl Send for Weights<'_> {}
unsafe impl Sync for Weights<'_> {}

impl<'a> Weights<'a> {
    /// A weight operand of `rows` rows of `k` values at `base`;
    /// `dense_stride` is the tensor's row stride in bytes (dense rows only).
    ///
    /// # Safety
    /// `base` addresses `rows` rows of the resolved representation (in a
    /// tiled layout, its matrices whole tiles), valid for `'a`.
    pub unsafe fn from_raw(
        base: *const u8,
        rows: usize,
        k: usize,
        dense_stride: usize,
        operand: WeightOperand,
    ) -> Self {
        Self {
            base,
            rows,
            k,
            geometry: RowGeometry::of(operand.storage, k, dense_stride)
                .with_base(base, rows.max(1)),
            kernels: operand.kernels,
            nvfp4_operands: operand.nvfp4_operands,
            quantized: operand.quantized,
            storage: operand.storage,
            life: std::marker::PhantomData,
        }
    }

    /// The operand's columns `columns` widened to whole storage groups (a
    /// dense operand's exactly), and the first of them: their own operand of
    /// the same rows, read where they lie, whose reads touch no byte of any
    /// other column's storage group.
    pub fn columns(&self, columns: std::ops::Range<usize>) -> (usize, Weights<'a>) {
        assert!(
            columns.start < columns.end && columns.end <= self.k,
            "columns {columns:?} of {}",
            self.k
        );
        match &self.storage.kind {
            registry::RepresentationKind::PackedRows(layout) => {
                let group = layout.group() as usize;
                let first = columns.start / group * group;
                let end = columns.end.next_multiple_of(group).min(self.k);
                let mut geometry = self.geometry;
                geometry.first_group += first / group;
                (first, Weights { k: end - first, geometry, ..*self })
            }
            registry::RepresentationKind::Dense(dtype) => {
                // SAFETY: the columns lie in every row (`from_raw`).
                let base = unsafe { self.base.add(columns.start * dtype.bytes() as usize) };
                let view = Weights {
                    base,
                    k: columns.len(),
                    geometry: self.geometry.with_base(base, self.geometry.matrix_rows),
                    ..*self
                };
                (columns.start, view)
            }
            _ => panic!("`{}` is no weight operand", self.storage.name),
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
        operand: WeightOperand,
    ) -> Self {
        let rank = extents.len();
        assert!(rank >= 1, "a weight operand has a value axis");
        let k = extents[rank - 1] as usize;
        let rows = extents[..rank - 1].iter().product::<u64>() as usize;
        let dense_stride = if let Some(dense_bytes) = operand.dense_bytes() {
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
                strides[rank - 2] as usize * dense_bytes
            } else {
                k * dense_bytes
            }
        } else {
            0
        };
        let mut view = unsafe { Self::from_raw(base, rows.max(1), k, dense_stride, operand) };
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

    /// Whether a product over these weights takes its activation rows as
    /// NVFP4 operands (`WeightOperand`).
    pub fn nvfp4_operands(&self) -> bool {
        self.nvfp4_operands
    }

    /// Whether a product over these weights takes quantized activation rows
    /// under `INT8 = 1` (`WeightOperand`).
    pub fn quantized(&self) -> bool {
        self.quantized
    }

    /// Rows of the dot block a projection takes at a time
    /// (`weights::Placement::row_block`): an `mma16` tile, whose row pairs
    /// share their code bytes; eight rows of every other layout.
    pub fn row_block(&self) -> usize {
        self.geometry.placement.row_block()
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

    /// `out[r] = (row first + r) · x` for a block of `out.len()` rows (an
    /// entry of `components::ROW_BLOCKS`): one component call across the
    /// whole row.
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

    /// `out[r] = (row first + r) · x` for `out.len()` rows, every 16 a tile,
    /// against an activation row quantized into `x`: the sums
    /// [`Weights::dot_q8`] gives a row block (`Weights::row_block`) at a
    /// time, in one component call where the element has a whole-tile form
    /// (`mma16` tiles, `rows16` rows). `x` is one activation row.
    #[inline(always)]
    pub fn dot_q8_tiles(&self, first: usize, x: &[quant::Q8Block], out: &mut [f32]) {
        assert!(
            first + out.len() <= self.rows && out.len() % 16 == 0,
            "tiles of rows {first}..{} of {}",
            first + out.len(),
            self.rows
        );
        // SAFETY: the rows lie inside the operand (`from_raw`).
        let whole = unsafe { self.kernels.dot_q8_tiles(self.base.add(first * self.geometry.stride), &self.geometry, x, self.k, out) };
        if !whole {
            let block = self.row_block();
            for (index, out) in out.chunks_exact_mut(block).enumerate() {
                self.dot_q8(first + block * index, x, out);
            }
        }
    }

    /// As [`Weights::dot`] of a block of `R` rows from `first`, against each
    /// of the `m` activation rows quantized into `x` (`quant::blocks(k)`
    /// blocks each, row after row, see [`quant`]), activation-major:
    /// `out[a R + r] = (row first + r) · activation a` (`out.len() = m R`).
    /// Each weight block is decoded once for every group of up to four
    /// activation rows, and every activation row's sums are the ones it has
    /// alone.
    #[inline(always)]
    pub fn dot_q8(&self, first: usize, x: &[quant::Q8Block], out: &mut [f32]) {
        let activations = x.len() / quant::blocks(self.k);
        assert!(
            activations > 0 && first + out.len() / activations <= self.rows,
            "rows from {first} of {} against {activations} activation rows",
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

    /// Whether [`Weights::dot_q8`] takes any row count in one call, not
    /// blocks of [`Weights::row_block`] rows a call.
    pub fn spans(&self) -> bool {
        self.kernels.spans(&self.geometry)
    }

    /// Whether products over these weights take the tile form
    /// ([`Weights::tile_q8`]): packed integer codes.
    pub fn tiled(&self) -> bool {
        self.kernels.tiled()
    }

    /// `out[a n + r] = (row first + r) · activation a` for `n = out.len() /
    /// m` rows from `first` and the `m` activation rows quantized into the
    /// tiles `x` ([`quant::interleave`]): each weight block decoded once for
    /// every activation row, each output the sums [`Weights::dot_q8`] gives
    /// its row alone, bit for bit.
    #[inline(always)]
    pub fn tile_q8(&self, first: usize, x: &[quant::Q8Tile], m: usize, out: &mut [f32]) {
        let n = out.len() / m;
        assert!(m > 0 && n * m == out.len() && first + n <= self.rows, "rows {first}..{} of {} against {m} activation rows", first + n, self.rows);
        // SAFETY: the rows lie inside the operand (`from_raw`).
        unsafe {
            self.kernels.tile_q8(self.base.add(first * self.geometry.stride), &self.geometry, n, x, m, self.k, out)
        }
    }
}
