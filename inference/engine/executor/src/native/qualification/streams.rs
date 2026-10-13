//! The hyper-connection streams' entries (`stream_mix_down`, its exit form,
//! `stream_mix_up`, `repeat_rows`) and the n-gram input's (`ngram_rows`,
//! `stored_rows`, its key and value projections, `ngram_gate`,
//! `ngram_inject`, `selection_notice`), each run once at the model's
//! geometry on one row with weights chosen so the result is known.

use super::super::*;
use super::*;
use magnitude_family_contracts::{NGramHash, Operator, SublayerIndex, WeightKind, WeightScope};
use magnitude_kernels::{
    ngram_gate, ngram_inject, ngram_rows, repeat_rows, selection_notice, stored_rows,
    stream_mix_down, stream_mix_up,
};

/// Values agree with an expectation within this relative error: an entry
/// whose result passes through `exp` or `rsqrt` is not exact.
const TOLERANCE: f32 = 1.0e-3;

/// `tensor`'s F32 values within [`TOLERANCE`] of `expected`.
fn require_f32_near(
    tensor: &Tensor,
    expected: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    let actual = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    let near = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= TOLERANCE * expected.abs());
    if !near {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "semantic fixture result mismatch",
        ));
    }
    Ok(())
}

fn u64_tensor(
    device: &Device,
    values: &[u64],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::u64(), &[values.len() as u64], &bytes)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

fn view(
    tensor: &Tensor,
    extents: &[u64],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    tensor
        .reshape(extents)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

/// The recurrent controls of one slot of one row over a two-bank arena:
/// bank 0 is the zero seed the row reads, bank 1 its successor.
struct SlotControls {
    segments: Tensor,
    stop: Tensor,
    previous_bank: Tensor,
    previous_tape: Tensor,
    following_bank: Tensor,
}

impl SlotControls {
    fn new(device: &Device, entry: &'static str, label: &str) -> Result<Self, CatalogFailure> {
        let i32s =
            |extents: &[u64], values: &[i32]| semantic_i32(device, extents, values, entry, label);
        Ok(Self {
            segments: i32s(&[2, 2], &[0, 1, 1, 1])?,
            stop: i32s(&[1], &[1])?,
            previous_bank: i32s(&[1], &[0])?,
            previous_tape: i32s(&[1], &[0])?,
            following_bank: i32s(&[1], &[1])?,
        })
    }
}

impl<'a> QualificationView<'a> {
    /// The streams' entries of the decoder's mix sites, once: over one row
    /// of ones under a norm of ones and zero projections, a site's low-rank
    /// row is 0, its inject values `2 sigmoid(0) = 1`, each stream's inverse
    /// `rsqrt(1 + epsilon)`, and its partials `sigmoid(0) · inverse / S`;
    /// the widening copies its row into each stream.
    pub(super) fn qualify_streams(&self, device: &Device) -> Result<(), CatalogFailure> {
        let mut bindings = std::collections::HashSet::new();
        for (binding, kernels) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
            .filter_map(|(slot, block)| slot.streams().zip(block.streams.as_ref()))
        {
            if bindings.insert(binding) {
                qualify_stream_binding(device, binding, kernels)?;
            }
        }
        for block in self.programs.head.iter().flat_map(|head| &head.blocks) {
            if let Some((binding, kernels)) = block.binding.streams.zip(block.streams.as_ref()) {
                qualify_stream_binding(device, binding, kernels)?;
            }
        }
        Ok(())
    }

    /// Each n-gram input's entries, once per binding, on one row of a fresh
    /// sequence over a two-bank arena: `ngram_rows` forms the rows the
    /// host's rule (`NGramHash::rows`) names; `stored_rows` decodes a zero
    /// slot to zeros, which project to zero keys and values; the gate of a
    /// zero key is `sigmoid(0) = 1/2` with a zero value's inverse
    /// `rsqrt(epsilon)`; and `ngram_inject` of a zero value through zero
    /// taps leaves the streams unchanged.
    pub(super) fn qualify_ngram(&self, device: &Device) -> Result<(), CatalogFailure> {
        let mut qualified = std::collections::HashSet::new();
        for ((index, slot), (attested, block)) in
            self.plan.target().blocks().iter().enumerate().zip(
                self.programs
                    .target
                    .blocks
                    .iter()
                    .zip(&self.geometry.blocks),
            )
        {
            let (Some(binding), Some(kernels)) = (slot.ngram(), attested.ngram.as_ref()) else {
                continue;
            };
            let label = format!("{binding:?}");
            let Some((sublayer, hash)) =
                block
                    .sublayers
                    .iter()
                    .enumerate()
                    .find_map(|(sublayer, value)| match &value.op {
                        Operator::NGramInput(input) => Some((sublayer, &input.hash)),
                        _ => None,
                    })
            else {
                return Err(qualification_dynamic(
                    "target_ngram",
                    &label,
                    "an n-gram binding without its sublayer",
                ));
            };
            if !qualified.insert(binding) {
                continue;
            }
            let scope = WeightScope::TargetSublayer(SublayerIndex {
                block: u32::try_from(index)
                    .map_err(|_| qualification_dynamic("target_ngram", &label, "block index"))?,
                sublayer: u32::try_from(sublayer)
                    .map_err(|_| qualification_dynamic("target_ngram", &label, "sublayer index"))?,
            });
            self.qualify_ngram_input(device, binding, kernels, hash, scope, &label)?;
        }
        Ok(())
    }

    fn qualify_ngram_input(
        &self,
        device: &Device,
        binding: crate::NGramBinding,
        kernels: &NGramKernels,
        hash: &NGramHash,
        scope: WeightScope,
        label: &str,
    ) -> Result<(), CatalogFailure> {
        let entry = "target_ngram";
        const EPSILON: f32 = 1.0e-5;
        let (s, h) = (binding.streams, binding.hidden);
        let heads = binding.heads();
        let zeros =
            |element, extents: &[u64]| semantic_zeros(device, element, extents, entry, label);
        let controls = SlotControls::new(device, entry, label)?;
        // One tape row, the least a bank holds, in each window.
        let reach = binding.dilation * (binding.kernel - 1);
        let mut banks = seismic::SlabTensor::new(
            device,
            2,
            2,
            vec![
                seismic::SlabRegion {
                    element: Element::f32(),
                    row_shape: vec![reach + 1, s * h],
                },
                seismic::SlabRegion {
                    element: Element::u32(),
                    row_shape: vec![binding.context + 1, 1],
                },
            ],
        )
        .map_err(|error| qualification_dynamic(entry, label, error))?;
        banks
            .add_slab()
            .map_err(|error| qualification_dynamic(entry, label, error))?;
        let region = |index| {
            banks
                .logical_region(index)
                .map_err(|error| qualification_dynamic(entry, label, error))
        };
        let (mut window, mut token_window) = (region(0)?, region(1)?);

        // A fresh sequence's first token: every predecessor reads as reset.
        const TOKEN: i32 = 5;
        let tokens = semantic_i32(device, &[1, 2], &[TOKEN, 0], "ngram_rows", label)?;
        let reset = i32::try_from(hash.reset)
            .map_err(|_| qualification_dynamic("ngram_rows", label, "reset token exceeds i32"))?;
        let rows = kernels
            .rows
            .call(ngram_rows::Args {
                tokens: &tokens,
                multipliers: &u64_tensor(device, &hash.multipliers, "ngram_rows", label)?,
                sizes: &u64_tensor(device, &hash.sizes, "ngram_rows", label)?,
                offsets: &u64_tensor(device, &hash.offsets, "ngram_rows", label)?,
                segments: &controls.segments,
                stop: &controls.stop,
                previous_bank: &controls.previous_bank,
                previous_tape: &controls.previous_tape,
                following_bank: &controls.following_bank,
                window: &mut token_window,
                reset,
                slab_banks: 2,
            })
            .map_err(|error| qualification_dynamic("ngram_rows", label, error))?
            .value;
        let expected = hash
            .rows(&[], &[TOKEN as u64])
            .into_iter()
            .flat_map(|row| [row as i32, 0])
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        if rows
            .read_to_host()
            .map_err(|error| qualification_dynamic("ngram_rows", label, error))?
            != expected
        {
            return Err(qualification_dynamic(
                "ngram_rows",
                label,
                "table rows differ from the host's row rule",
            ));
        }

        // Every head reads slot 0 of a one-slot table held in slots.
        let slots = zeros(binding.table_source, &[1, binding.width])?;
        let decoded = kernels
            .stored
            .call(stored_rows::Args {
                slots: &slots,
                tokens: &zeros(Element::i32(), &[heads, 2])?,
                directory: &zeros(Element::i32(), &[0, 1])?,
            })
            .map_err(|error| qualification_dynamic("stored_rows", label, error))?
            .value;
        // A block of a zero scale decodes to signed zeros.
        if read_dense_values(&decoded, "stored_rows", label)?
            .iter()
            .any(|value| *value != 0.0)
        {
            return Err(qualification_dynamic(
                "stored_rows",
                label,
                "a zero-scale slot decoded to nonzero values",
            ));
        }
        let decoded = view(&decoded, &[1, binding.embedding()], "stored_rows", label)?;
        let project = |kernel: &NativeKernel<project_rows::Entry>,
                       kind: WeightKind,
                       element: Element,
                       outputs: u64| {
            let weight_plan = self
                .load
                .weights()
                .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                .ok_or_else(|| {
                    qualification_dynamic("project_rows", label, format!("no {kind:?} weight"))
                })?;
            let extent = weight_plan.scale_extent();
            let result = kernel
                .call(project_rows::Args {
                    source: &decoded,
                    weight_bound: &semantic_ones(
                        device,
                        Element::f32(),
                        &[weight_plan.bound],
                        "project_rows",
                        label,
                    )?,
                    weight: &zeros(element, &[outputs, binding.embedding()])?,
                    weight_scale: &semantic_ones(
                        device,
                        Element::f32(),
                        &[extent],
                        "project_rows",
                        label,
                    )?,
                })
                .map_err(|error| qualification_dynamic("project_rows", label, error))?
                .value;
            require_zero_result(&result, "project_rows", label)?;
            Ok::<_, CatalogFailure>(result)
        };
        let key = project(&kernels.key, WeightKind::NGramKey, binding.key, s * h)?;
        let value = project(&kernels.value, WeightKind::NGramValue, binding.value, h)?;

        let streams_values = vec![1.0_f32; (s * h) as usize];
        let streams = semantic_f32(device, &[1, s * h], &streams_values, entry, label)?;
        let norm = semantic_ones(device, binding.norm, &[s, h], entry, label)?;
        let scales = kernels
            .gate
            .call(ngram_gate::Args {
                hidden: &view(&streams, &[1, s, h], "ngram_gate", label)?,
                key: &view(&key, &[1, s, h], "ngram_gate", label)?,
                value: &value,
                key_norm: &norm,
                query_norm: &norm,
                epsilon: EPSILON,
            })
            .map_err(|error| qualification_dynamic("ngram_gate", label, error))?
            .value;
        let gates = std::iter::repeat_n(0.5, s as usize)
            .chain(std::iter::repeat_n(1.0 / EPSILON.sqrt(), s as usize))
            .collect::<Vec<_>>();
        require_f32_near(&scales, &gates, "ngram_gate", label)?;
        let injected = kernels
            .inject
            .call(ngram_inject::Args {
                hidden: &streams,
                scales: &scales,
                value: &value,
                norm: &norm,
                convolution: &zeros(Element::f32(), &[s * h, binding.kernel])?,
                segments: &controls.segments,
                stop: &controls.stop,
                previous_bank: &controls.previous_bank,
                previous_tape: &controls.previous_tape,
                following_bank: &controls.following_bank,
                window: &mut window,
                dilation: i32::try_from(binding.dilation).map_err(|_| {
                    qualification_dynamic("ngram_inject", label, "dilation exceeds i32")
                })?,
                slab_banks: 2,
            })
            .map_err(|error| qualification_dynamic("ngram_inject", label, error))?
            .value;
        require_f32_values(&injected, &streams_values, "ngram_inject", label)?;

        let mut notice = zeros(Element::i32(), &[1, 2])?;
        kernels
            .notice
            .call(selection_notice::Args {
                selection: &tokens,
                notice: &mut notice,
            })
            .map_err(|error| qualification_dynamic("selection_notice", label, error))?;
        if notice
            .read_to_host()
            .map_err(|error| qualification_dynamic("selection_notice", label, error))?
            != tokens
                .read_to_host()
                .map_err(|error| qualification_dynamic("selection_notice", label, error))?
        {
            return Err(qualification_dynamic(
                "selection_notice",
                label,
                "the notice differs from the selection",
            ));
        }
        Ok(())
    }
}

fn qualify_stream_binding(
    device: &Device,
    binding: crate::operators::streams::StreamsBinding,
    kernels: &StreamsKernels,
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let geometry = binding.geometry;
    let (h, s, r) = (geometry.hidden, geometry.streams, geometry.rank);
    let entry = "target_streams";
    const EPSILON: f32 = 1.0e-5;
    let streams = semantic_ones(device, Element::f32(), &[1, s * h], entry, &label)?;
    let hidden = view(&streams, &[1, s, h], entry, &label)?;
    let norm = semantic_ones(device, binding.norm, &[s, h], entry, &label)?;
    let down = semantic_zeros(device, binding.down, &[r, s * h], entry, &label)?;
    let inject = semantic_zeros(device, binding.inject, &[s, s * h], entry, &label)?;
    let up = semantic_zeros(device, binding.up, &[s * h, r], entry, &label)?;
    let inverse = 1.0 / (1.0 + EPSILON).sqrt();
    let inverses = vec![inverse; s as usize];
    let mixed = kernels
        .down
        .call(stream_mix_down::Args {
            hidden: &hidden,
            norm: &norm,
            down_weight: &down,
            inject_weight: &inject,
            epsilon: EPSILON,
        })
        .map_err(|error| qualification_dynamic("stream_mix_down", &label, error))?;
    require_zero_result(&mixed.r0, "stream_mix_down", &label)?;
    require_f32_near(&mixed.r1, &vec![1.0; s as usize], "stream_mix_down", &label)?;
    require_f32_near(&mixed.r2, &inverses, "stream_mix_down", &label)?;
    // The exit site injects into nothing: its inject weights are a
    // zero-row view of its down weights.
    let exit = kernels
        .exit
        .call(stream_mix_down::Args {
            hidden: &hidden,
            norm: &norm,
            down_weight: &down,
            inject_weight: &down
                .slice_leading(0, 0)
                .map_err(|error| qualification_dynamic("stream_mix_down", &label, error))?,
            epsilon: EPSILON,
        })
        .map_err(|error| qualification_dynamic("stream_mix_down", &label, error))?;
    require_zero_result(&exit.r0, "stream_mix_down", &label)?;
    require_f32_near(&exit.r2, &inverses, "stream_mix_down", &label)?;
    let partials = kernels
        .up
        .call(stream_mix_up::Args {
            low: &mixed.r0,
            up_weight: &up,
            hidden: &hidden,
            inverse: &mixed.r2,
            norm: &norm,
        })
        .map_err(|error| qualification_dynamic("stream_mix_up", &label, error))?
        .value;
    require_f32_near(
        &partials,
        &vec![0.5 * inverse / s as f32; (s * h) as usize],
        "stream_mix_up",
        &label,
    )?;
    let row_values = (0..h)
        .map(|index| (index % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    let row = semantic_f32(device, &[1, h], &row_values, "repeat_rows", &label)?;
    let widened = kernels
        .repeat
        .call(repeat_rows::Args {
            rows: &row,
            scales: &semantic_ones(device, Element::f32(), &[s], "repeat_rows", &label)?,
        })
        .map_err(|error| qualification_dynamic("repeat_rows", &label, error))?
        .value;
    let copies = (0..s)
        .flat_map(|_| row_values.iter().copied())
        .collect::<Vec<_>>();
    require_f32_values(&widened, &copies, "repeat_rows", &label)
}
