//! Whole-recipe readout qualification within the ordinary first-load allocation.
//! Packed is canonical. A dependent bounded recipe must confirm the same
//! selected tokens and a lower complete invocation cost; missing evidence is
//! a completed packed choice, never a request to continue on another load.
//!
//! The bounded recipe's cost depends on its rows: its later levels read only
//! the vocabulary rows the earlier bounds keep. Rows of noise score every
//! vocabulary row alike, so no bound separates them and every level reads the
//! whole head; that is the recipe's worst case and the hardest test of its
//! bounds, and no served row. The two recipes therefore agree on noise rows
//! and are timed on rows the head resolves (`RecipeCase::resolved_rows`).
//!
//! That timed witness is the bounded recipe's best case: a stored head row,
//! which the first bound separates from every other vocabulary row. It is
//! not a served row either. A qualification on rows the model itself
//! produces is not available here: Prepare forms each entry's program and
//! imports single weights, but places no model (no sealed graphs, resident
//! weights or state), so it cannot run a forward pass.
//!
//! The evidence that the selection this produces is right on served rows is
//! measured, not derived: readout device time per real decode step, bounded
//! over packed, on two models and two fixtures (an M4 Pro, 4,096 context):
//!   Qwen3.6-35B-A3B  prose-repeat 0.64, prose-continue 0.80
//!     (run 20261010T123257Z-a3b.diag-31f355)
//!   Nemotron 3.5 Lightning 30B-A3B Q4  prose-repeat 0.67, prose-continue 0.89
//!     (run 20261010T124501Z-a3b.diag-96f42c)
//! In whole decode steps three of the four cases are faster with the bounded
//! recipe and the worst (Nemotron, prose-continue) is break-even.

use super::{ComposedCensus, ComposedContender, PointShape, TuningInputs, TuningKey};
use crate::native::specialization::entry_statics;
use crate::native::{CatalogFailure, ProgressiveReadoutKernels, ProjectionKernels};
use crate::{KernelCache, TuningCacheKey};
use magnitude_family_contracts::{ProgressivePlane, WeightKind, WeightScope};
use magnitude_kernels::{
    readout_exact_rows, readout_features_rows, readout_head_rows, readout_planes_rows,
    readout_project_rows, readout_refine_rows, readout_top_rows,
};
use seismic::{
    Element, Entry, NativeKernel, NativeSpecialization, NativeTensorBatch, Shuffle, SustainedClock,
    Tally, Tensor, Verdict,
};
use std::{cell::RefCell, rc::Rc, time::Instant};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReadoutRecipe {
    pub scope: WeightScope,
    pub norm_scope: WeightScope,
    pub norm: Element,
    pub weight: Element,
    pub activation: Element,
    pub epsilon: f32,
    pub softcap: f32,
    pub prepared: bool,
    pub vocabulary: u64,
    pub hidden: u64,
    /// The readout's input form (`PT`, `NM`): partial rows summed, and the
    /// norm's extent (0 for the hyper-connection exit site's partials).
    pub parts: u64,
    pub norm_extent: u64,
    pub low_planes: u64,
    pub scale_extent: u64,
}

#[derive(Default)]
pub(crate) struct QualifiedRecipe {
    pub selected: bool,
    pub progressive: Option<ProgressiveReadoutKernels>,
    pub canonical_specification: Option<String>,
}

pub(crate) type RecipeState = Rc<RefCell<QualifiedRecipe>>;
pub(crate) type RecipePreparation = Rc<RefCell<Option<ProgressiveReadoutKernels>>>;

pub(crate) struct ReadoutContender<'a> {
    pub recipe: ReadoutRecipe,
    pub point: PointShape,
    pub key: TuningCacheKey,
    /// Where the completed choice is stored.
    pub cache: Option<&'a KernelCache>,
    /// Where a stored choice is read from: none for a fresh preparation.
    pub stored: Option<&'a KernelCache>,
    pub state: RecipeState,
    pub dependency: TuningKey,
    pub preparation: RecipePreparation,
    canonical_seconds: f64,
    candidate_seconds: f64,
    receipt: ReadoutChoiceReceipt,
}

#[derive(Default, serde::Serialize)]
struct ReadoutChoiceReceipt {
    selected: bool,
    reason: String,
    preparation_seconds: f64,
    canonical_specification: Option<String>,
    // Every pair is the complete submitted recipe, including dependent levels.
    pairs: Vec<(f64, f64)>,
}

fn fail(outcome: impl ToString) -> CatalogFailure {
    CatalogFailure::Qualification {
        entry: "readout_recipe",
        bindings: "packed/bounded".into(),
        outcome: outcome.to_string(),
    }
}

impl<'a> ReadoutContender<'a> {
    pub(crate) fn new(
        recipe: ReadoutRecipe,
        point: PointShape,
        key: TuningCacheKey,
        cache: Option<&'a KernelCache>,
        stored: Option<&'a KernelCache>,
        state: RecipeState,
        dependency: TuningKey,
        preparation: RecipePreparation,
    ) -> Self {
        Self {
            recipe,
            point,
            key,
            cache,
            stored,
            state,
            dependency,
            preparation,
            canonical_seconds: 0.0,
            candidate_seconds: 0.0,
            receipt: ReadoutChoiceReceipt {
                reason: "unmeasured within first-load allocation".into(),
                ..Default::default()
            },
        }
    }
}

/// Resolve the declaration's complete fixed configuration, including tuning
/// parameters. A composed comparison forms these defaults without searching
/// the component kernels independently.
fn declared_defaults<E: seismic::Entry>(
    device: &seismic::Device,
    statics: &NativeSpecialization,
) -> Result<NativeSpecialization, CatalogFailure> {
    seismic::generated::native_implementation::<E>(device)
        .map_err(fail)?
        .ok_or_else(|| fail("readout recipe has no native implementation"))?
        .default_specialization(statics)
        .map_err(fail)
}

impl ReadoutRecipe {
    /// The recipe's value of every static its kernels declare. The kernels
    /// read different inputs (hidden rows in the input form, or projected
    /// features), so each takes only the statics it declares.
    fn values(self) -> [(&'static str, u64); 8] {
        [
            ("V", self.vocabulary),
            ("D", self.hidden),
            ("PT", self.parts),
            ("NM", self.norm_extent),
            ("P", self.low_planes),
            ("WS", self.scale_extent),
            // The prepared projection publishes every vocabulary column of
            // implicit token rows.
            ("S", self.vocabulary),
            ("I", 0),
        ]
    }

    /// `E`'s statics in this recipe on `device`'s backend.
    fn statics<E: Entry>(
        self,
        device: &seismic::Device,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        entry_statics::<E>(device, "readout_recipe", &self.values())
    }

    /// The identity of the recipe's programs on `device`: its four bounded
    /// levels and its canonical projection, each at its own statics.
    pub(crate) fn digests(self, device: &seismic::Device) -> Result<[String; 5], CatalogFailure> {
        Ok([
            readout_top_rows::native_digest_with(
                device,
                readout_top_rows::Elements {
                    NW: self.norm,
                    A: self.activation,
                },
                &self.statics::<readout_top_rows::Entry>(device)?,
            )
            .map_err(fail)?,
            readout_refine_rows::native_digest_with(
                device,
                readout_refine_rows::Elements { A: self.activation },
                &self.statics::<readout_refine_rows::Entry>(device)?,
            )
            .map_err(fail)?,
            readout_exact_rows::native_digest_with(
                device,
                readout_exact_rows::Elements { A: self.activation },
                &self.statics::<readout_exact_rows::Entry>(device)?,
            )
            .map_err(fail)?,
            readout_planes_rows::native_digest_with(
                device,
                readout_planes_rows::Elements {
                    NW: self.norm,
                    A: self.activation,
                },
                &self.statics::<readout_planes_rows::Entry>(device)?,
            )
            .map_err(fail)?,
            if self.prepared {
                readout_project_rows::native_digest_with(
                    device,
                    readout_project_rows::Elements {
                        OW: self.weight,
                        A: self.activation,
                    },
                    &self.statics::<readout_project_rows::Entry>(device)?,
                )
                .map_err(fail)?
            } else {
                readout_head_rows::native_digest_with(
                    device,
                    readout_head_rows::Elements {
                        NW: self.norm,
                        OW: self.weight,
                        A: self.activation,
                    },
                    &self.statics::<readout_head_rows::Entry>(device)?,
                )
                .map_err(fail)?
            },
        ])
    }

    /// The recipe's feature reader, `readout_features_rows`.
    fn features(
        self,
        device: &seismic::Device,
    ) -> Result<NativeKernel<readout_features_rows::Entry>, CatalogFailure> {
        readout_features_rows::native_for_device_with(
            device,
            readout_features_rows::Elements {
                NW: self.norm,
                A: self.activation,
            },
            &self.statics::<readout_features_rows::Entry>(device)?,
        )
        .map_err(fail)
    }

    /// Row-independent fixed programs are formed once per exact physical
    /// candidate identity; row admission and confirmation remain separate.
    pub(crate) fn form_shared(
        self,
        device: &seismic::Device,
        preparation: &RecipePreparation,
    ) -> Result<QualifiedRecipe, CatalogFailure> {
        if let Some(progressive) = preparation.borrow().clone() {
            return Ok(QualifiedRecipe {
                progressive: Some(progressive),
                ..QualifiedRecipe::default()
            });
        }
        let formed = self.form(device)?;
        *preparation.borrow_mut() = formed.progressive.clone();
        Ok(formed)
    }

    /// Form the recipe's four programs, to the end.
    fn form(self, device: &seismic::Device) -> Result<QualifiedRecipe, CatalogFailure> {
        let top = readout_top_rows::native_for_device_with(
            device,
            readout_top_rows::Elements {
                NW: self.norm,
                A: self.activation,
            },
            &declared_defaults::<readout_top_rows::Entry>(
                device,
                &self.statics::<readout_top_rows::Entry>(device)?,
            )?,
        )
        .map_err(fail)?;
        let refine = readout_refine_rows::native_for_device_with(
            device,
            readout_refine_rows::Elements { A: self.activation },
            &declared_defaults::<readout_refine_rows::Entry>(
                device,
                &self.statics::<readout_refine_rows::Entry>(device)?,
            )?,
        )
        .map_err(fail)?;
        let exact = readout_exact_rows::native_for_device_with(
            device,
            readout_exact_rows::Elements { A: self.activation },
            &declared_defaults::<readout_exact_rows::Entry>(
                device,
                &self.statics::<readout_exact_rows::Entry>(device)?,
            )?,
        )
        .map_err(fail)?;
        let planes = readout_planes_rows::native_for_device_with(
            device,
            readout_planes_rows::Elements {
                NW: self.norm,
                A: self.activation,
            },
            &declared_defaults::<readout_planes_rows::Entry>(
                device,
                &self.statics::<readout_planes_rows::Entry>(device)?,
            )?,
        )
        .map_err(fail)?;
        let progressive = ProgressiveReadoutKernels {
            top,
            refine,
            exact,
            planes,
        };
        Ok(QualifiedRecipe {
            progressive: Some(progressive),
            ..QualifiedRecipe::default()
        })
    }

    fn canonical(
        self,
        device: &seismic::Device,
        specialization: &NativeSpecialization,
    ) -> Result<ProjectionKernels, CatalogFailure> {
        if self.prepared {
            readout_project_rows::native_for_device_with(
                device,
                readout_project_rows::Elements {
                    OW: self.weight,
                    A: self.activation,
                },
                specialization,
            )
            .map(ProjectionKernels::PreparedPacked)
            .map_err(fail)
        } else {
            readout_head_rows::native_for_device_with(
                device,
                readout_head_rows::Elements {
                    NW: self.norm,
                    OW: self.weight,
                    A: self.activation,
                },
                specialization,
            )
            .map(ProjectionKernels::RawPacked)
            .map_err(fail)
        }
    }
}

/// The input seeds a recipe's qualification compares the two recipes at
/// (the census measured at seed 1): one whose rows sample, one whose rows
/// are greedy.
const QUALIFICATION_SEEDS: [u64; 2] = [2, 3];

struct RecipeCase {
    hidden: Tensor,
    norm: Tensor,
    weight: Tensor,
    scale: Tensor,
    out_rows: Tensor,
    planes: [Tensor; 5],
    draws: Tensor,
    temperature: Tensor,
    mask: Tensor,
    constrained: Tensor,
    canonical: Tensor,
    coarse: Tensor,
    floor0: Tensor,
    features: Tensor,
    prepared_features: Tensor,
    length: Tensor,
    fine: Tensor,
    floor1: Tensor,
    exact: Tensor,
    token_rows: Tensor,
}

impl RecipeCase {
    fn new(
        inputs: &mut TuningInputs<'_, '_>,
        recipe: ReadoutRecipe,
        rows: u64,
        seed: u64,
    ) -> Result<Self, String> {
        let hidden =
            inputs.activation(Element::f32(), &[rows, recipe.parts * recipe.hidden], seed)?;
        let norm = if recipe.norm_extent == 0 {
            inputs.scratch(recipe.norm, &[0])?
        } else {
            inputs.shared_weight(recipe.norm_scope, WeightKind::OutputNorm)?
        };
        let weight = inputs
            .shared_weight(recipe.scope, WeightKind::Output)?
            .slice_leading(0, recipe.vocabulary)
            .map_err(|error| error.to_string())?;
        let mut views = Vec::new();
        for plane in ProgressivePlane::ALL {
            views.push(
                inputs
                    .shared_weight(recipe.scope, WeightKind::OutputPlane(plane))?
                    .slice_leading(0, recipe.vocabulary)
                    .map_err(|error| error.to_string())?,
            );
        }
        let planes = views
            .try_into()
            .map_err(|_| "readout candidate has incomplete planes")?;
        let draws = (0..rows * 6)
            .map(|i| {
                if i % 6 == 0 {
                    u32::from(seed % 2 == 0)
                } else {
                    (i as u32).wrapping_mul(0x9e37_79b9)
                }
            })
            .collect::<Vec<_>>();
        let words = recipe.vocabulary.div_ceil(32);
        let zeros = |element, shape: &[u64]| {
            Tensor::zeros(inputs.device, element, shape).map_err(|e| e.to_string())
        };
        Ok(Self {
            hidden,
            norm,
            weight,
            planes,
            scale: inputs.unit_scale(recipe.scale_extent)?,
            out_rows: inputs.every_row(rows)?,
            draws: inputs.u32s(&[rows, 6], &draws)?,
            temperature: inputs.f32s(&[rows], &vec![1.0; rows as usize])?,
            mask: inputs.u32s(&[rows, words], &vec![u32::MAX; (rows * words) as usize])?,
            constrained: inputs.i32s(&[rows], &vec![0; rows as usize])?,
            canonical: zeros(Element::f32(), &[rows, recipe.vocabulary])?,
            coarse: zeros(Element::f32(), &[rows, recipe.vocabulary])?,
            fine: zeros(Element::f32(), &[rows, recipe.vocabulary])?,
            exact: zeros(Element::f32(), &[rows, recipe.vocabulary])?,
            floor0: zeros(Element::f32(), &[rows])?,
            floor1: zeros(Element::f32(), &[rows])?,
            features: zeros(recipe.activation, &[rows, recipe.hidden])?,
            prepared_features: zeros(recipe.activation, &[rows, recipe.hidden])?,
            length: zeros(Element::f32(), &[rows])?,
            token_rows: inputs.i32s(&[rows, 0, recipe.vocabulary], &[])?,
        })
    }

    /// Hidden rows the head resolves: row `r` is the head's own stored row
    /// of one token (placed by `seed`), read back from the planes, so that
    /// token's logit stands clear of the others as a selected token's does.
    fn resolved_rows(
        &self,
        inputs: &TuningInputs<'_, '_>,
        recipe: ReadoutRecipe,
        rows: u64,
        seed: u64,
    ) -> Result<Tensor, String> {
        let [top, next, rest, scales, _] = &self.planes;
        let stored = |plane: &Tensor, token: u64| {
            plane
                .slice_leading(token, token + 1)
                .map_err(|error| error.to_string())?
                .read_to_host()
                .map_err(|error| error.to_string())
        };
        let columns = usize::try_from(recipe.hidden).map_err(|_| "hidden width exceeds usize")?;
        let bits = u8::try_from(2 * (recipe.low_planes + 1))
            .map_err(|_| "progressive code depth exceeds u8")?;
        // The row is the first of its partial rows; the others are zero.
        let partials = (recipe.parts - 1) * recipe.hidden;
        let mut values = Vec::new();
        for row in 0..rows {
            let token = u64::from(((seed * rows + row) as u32).wrapping_mul(0x9e37_79b9))
                % recipe.vocabulary;
            values.extend(crate::progressive::row_values(
                &stored(top, token)?,
                &stored(next, token)?,
                &stored(rest, token)?,
                &stored(scales, token)?,
                columns,
                bits,
            ));
            values.extend(std::iter::repeat_n(0.0, partials as usize));
        }
        inputs.f32s(&[rows, recipe.parts * recipe.hidden], &values)
    }

    /// Replace the case's hidden rows, and the features prepared from them.
    fn set_hidden(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        feature_kernel: &NativeKernel<readout_features_rows::Entry>,
        hidden: Tensor,
    ) -> Result<(), CatalogFailure> {
        self.hidden = hidden;
        if recipe.prepared {
            self.prepare_features(device, recipe, feature_kernel)?;
        }
        Ok(())
    }

    fn prepare_features(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        feature_kernel: &NativeKernel<readout_features_rows::Entry>,
    ) -> Result<(), CatalogFailure> {
        // The entry reads the rows as their `[M, PT, D]` parts.
        let rows = self.hidden.extents()[0];
        let partials = self
            .hidden
            .reshape(&[rows, recipe.parts, recipe.hidden])
            .map_err(fail)?;
        let mut batch = NativeTensorBatch::new(device);
        batch
            .push(
                feature_kernel,
                readout_features_rows::Args {
                    hidden: &partials,
                    norm: &self.norm,
                    out_rows: &self.out_rows,
                    epsilon: recipe.epsilon,
                },
                readout_features_rows::OutputArgs {
                    value: &mut self.prepared_features,
                },
            )
            .map_err(fail)?;
        batch.submit().map_err(fail)?.wait().map_err(fail)
    }

    /// Run the canonical packed recipe once; the seconds it took.
    fn packed(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        kernel: &ProjectionKernels,
    ) -> Result<f64, CatalogFailure> {
        let mut batch = NativeTensorBatch::new(device);
        match kernel {
            ProjectionKernels::RawPacked(kernel) => batch
                .push(
                    kernel,
                    readout_head_rows::Args {
                        hidden: &self.hidden,
                        norm: &self.norm,
                        weight: &self.weight,
                        weight_scale: &self.scale,
                        out_rows: &self.out_rows,
                        epsilon: recipe.epsilon,
                        softcap: recipe.softcap,
                    },
                    readout_head_rows::OutputArgs {
                        value: &mut self.canonical,
                    },
                )
                .map_err(fail)?,
            ProjectionKernels::PreparedPacked(kernel) => batch
                .push(
                    kernel,
                    readout_project_rows::Args {
                        features: &self.prepared_features,
                        weight: &self.weight,
                        token_rows: &self.token_rows,
                        weight_scale: &self.scale,
                        softcap: recipe.softcap,
                    },
                    readout_project_rows::OutputArgs {
                        value: &mut self.canonical,
                    },
                )
                .map_err(fail)?,
            _ => return Err(fail("canonical qualification recipe must be packed")),
        }
        let start = Instant::now();
        batch.submit().map_err(fail)?.wait().map_err(fail)?;
        Ok(start.elapsed().as_secs_f64())
    }

    /// Run the bounded progressive recipe once; the seconds it took.
    fn bounded(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        kernels: &ProgressiveReadoutKernels,
    ) -> Result<f64, CatalogFailure> {
        let [top, bit3, rest, scales, radius] = &self.planes;
        // The entry reads the rows as their `[M, PT, D]` parts.
        let rows = self.hidden.extents()[0];
        let partials = self
            .hidden
            .reshape(&[rows, recipe.parts, recipe.hidden])
            .map_err(fail)?;
        let mut batch = NativeTensorBatch::new(device);
        batch
            .push(
                &kernels.top,
                readout_top_rows::Args {
                    hidden: &partials,
                    norm: &self.norm,
                    top,
                    bit3,
                    rest,
                    scales,
                    radius,
                    out_rows: &self.out_rows,
                    draws: &self.draws,
                    temperature: &self.temperature,
                    mask: &self.mask,
                    constrained: &self.constrained,
                    epsilon: recipe.epsilon,
                    softcap: recipe.softcap,
                },
                readout_top_rows::OutputArgs {
                    r0: &mut self.coarse,
                    r1: &mut self.floor0,
                    r2: &mut self.features,
                    r3: &mut self.length,
                },
            )
            .map_err(fail)?;
        batch
            .push(
                &kernels.refine,
                readout_refine_rows::Args {
                    features: &self.features,
                    top,
                    bit3,
                    rest,
                    scales,
                    radius,
                    coarse: &self.coarse,
                    floor: &self.floor0,
                    length: &self.length,
                    draws: &self.draws,
                    temperature: &self.temperature,
                    mask: &self.mask,
                    constrained: &self.constrained,
                    softcap: recipe.softcap,
                },
                readout_refine_rows::OutputArgs {
                    r0: &mut self.fine,
                    r1: &mut self.floor1,
                },
            )
            .map_err(fail)?;
        batch
            .push(
                &kernels.exact,
                readout_exact_rows::Args {
                    features: &self.features,
                    top,
                    bit3,
                    rest,
                    scales,
                    radius,
                    fine: &self.fine,
                    floor: &self.floor1,
                    length: &self.length,
                    draws: &self.draws,
                    temperature: &self.temperature,
                    mask: &self.mask,
                    constrained: &self.constrained,
                    softcap: recipe.softcap,
                },
                readout_exact_rows::OutputArgs {
                    value: &mut self.exact,
                },
            )
            .map_err(fail)?;
        let start = Instant::now();
        batch.submit().map_err(fail)?.wait().map_err(fail)?;
        Ok(start.elapsed().as_secs_f64())
    }

    fn same_winners(&self, rows: u64, vocabulary: u64) -> Result<bool, CatalogFailure> {
        let read = |t: &Tensor| {
            t.read_to_host()
                .map(|bytes| {
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                })
                .map_err(fail)
        };
        let canonical = read(&self.canonical)?;
        let exact = read(&self.exact)?;
        let draws = self
            .draws
            .read_to_host()
            .map_err(fail)?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for row in 0..rows as usize {
            let winner = |values: &[f32]| {
                let mut best = None;
                let mut score = f32::NEG_INFINITY;
                for token in 0..vocabulary as usize {
                    let value = values[row * vocabulary as usize + token];
                    if value.is_nan() || value == f32::INFINITY {
                        return None;
                    }
                    let next = value + noise(token as u32, &draws[row * 6..row * 6 + 6]);
                    if next > score {
                        score = next;
                        best = Some(token);
                    }
                }
                best
            };
            let left = winner(&canonical);
            if left.is_none() || left != winner(&exact) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// The sampler's Philox4x32-10 score; comparing both recipes with the same
// token-indexed noise avoids qualifying greedy-only pruning for sampling.
fn noise(token: u32, draw: &[u32]) -> f32 {
    if draw[0] != 1 {
        return 0.0;
    }
    let mut c = [token, draw[3], draw[4], draw[5]];
    let mut key = [draw[1], draw[2]];
    for _ in 0..10 {
        let a = 3528531795u64 * c[0] as u64;
        let b = 3449720151u64 * c[2] as u64;
        c = [
            (b >> 32) as u32 ^ c[1] ^ key[0],
            b as u32,
            (a >> 32) as u32 ^ c[3] ^ key[1],
            a as u32,
        ];
        key[0] = key[0].wrapping_add(2654435769);
        key[1] = key[1].wrapping_add(3144134277);
    }
    let uniform = ((c[0] >> 9) as f32 + 0.5) * 0.00000011920928955078125;
    -(-uniform.ln()).ln()
}

impl<'a> ComposedContender<'a> for ReadoutContender<'a> {
    fn set_invocations(
        &mut self,
        invocations: &crate::programs::invocation_demand::InvocationCounts,
        objective: &super::Objective,
    ) {
        super::apply_invocation_counts(
            std::slice::from_mut(&mut self.point),
            invocations,
            objective,
        );
    }
    fn key(&self) -> String {
        self.key.as_str().to_owned()
    }
    fn dependencies(&self) -> Vec<TuningKey> {
        vec![self.dependency.clone()]
    }
    fn label(&self) -> &'static str {
        "readout_recipe"
    }
    fn cached(&self) -> Option<bool> {
        self.stored.and_then(|c| c.readout_choice(&self.key))
    }
    fn census(
        &mut self,
        inputs: &mut TuningInputs<'_, 'a>,
    ) -> Result<Option<ComposedCensus>, CatalogFailure> {
        let preparation = Instant::now();
        // Only statics outside a recipe kernel's domain exclude the recipe;
        // any other failure to form it is a fault, not a slower candidate.
        let formed = match self.recipe.form_shared(inputs.device, &self.preparation) {
            Ok(formed) => formed,
            Err(error @ CatalogFailure::KernelDomain { .. }) => {
                self.receipt.reason = format!("candidate outside its kernels' domain: {error:?}");
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        // Census prices the optional recipe with the native default. Admission
        // later compares against the dependency's finalized ordinary choice.
        let statics = self
            .dependency
            .2
            .iter()
            .fold(NativeSpecialization::new(), |spec, (name, value)| {
                spec.with_static(name.clone(), *value)
            });
        let default = if self.recipe.prepared {
            declared_defaults::<readout_project_rows::Entry>(inputs.device, &statics)?
        } else {
            declared_defaults::<readout_head_rows::Entry>(inputs.device, &statics)?
        };
        let packed = self.recipe.canonical(inputs.device, &default)?;
        let mut case = match RecipeCase::new(inputs, self.recipe, self.point.rows, 1) {
            Ok(case) => case,
            Err(error) => {
                self.receipt.reason = format!("candidate inputs not admitted: {error}");
                return Ok(None);
            }
        };
        let feature = self.recipe.features(inputs.device)?;
        if self.recipe.prepared {
            case.prepare_features(inputs.device, self.recipe, &feature)?;
        }
        let preparation = preparation.elapsed();
        self.receipt.preparation_seconds = preparation.as_secs_f64();
        let measured = Instant::now();
        self.canonical_seconds = case.packed(inputs.device, self.recipe, &packed)?;
        self.candidate_seconds = case.bounded(
            inputs.device,
            self.recipe,
            formed.progressive.as_ref().unwrap(),
        )?;
        self.receipt
            .pairs
            .push((self.canonical_seconds, self.candidate_seconds));
        self.receipt.reason =
            "censused; qualification not admitted within first-load allocation".into();
        *self.state.borrow_mut() = formed;
        let rounds = Tally::default().fewest(super::SEARCH_SETTINGS.error_rate);
        // The comparison's clock is established by its first raise: twice
        // over, readings of the packed run in doubling lengths from a
        // sample's, none shorter than one run (`seismic::SustainedClock`).
        let establishment = 2.
            * (0..rounds)
                .map(|stage| {
                    (super::MIN_SAMPLE_SECONDS * f64::from(1u32 << stage))
                        .max(self.canonical_seconds)
                })
                .sum::<f64>();
        Ok(Some(ComposedCensus {
            served: vec![self.point.clone()],
            measured: vec![self.point.clone()],
            seconds: vec![self.canonical_seconds],
            // The qualification prepares as the census did, establishes its
            // clock, and runs the pair the census ran, for each seed: twice
            // for agreement (noise rows, then the timed rows), and a lead and
            // a sample in each round it is admitted.
            qualification: preparation
                + std::time::Duration::from_secs_f64(establishment)
                + measured.elapsed() * (QUALIFICATION_SEEDS.len() as u32 * (2 + 2 * rounds)),
        }))
    }
    fn qualify(
        &mut self,
        inputs: &mut TuningInputs<'_, 'a>,
        before: std::time::Instant,
    ) -> Result<bool, CatalogFailure> {
        macro_rules! check_deadline {
            () => {
                if std::time::Instant::now() >= before {
                    self.receipt.reason = "first-load tuning allowance expired".into();
                    return Ok(false);
                }
            };
        }
        check_deadline!();
        let Some(specialization) = inputs.choice(&self.dependency).cloned() else {
            self.receipt.reason = "canonical dependency has no finalized choice".into();
            return Ok(false);
        };
        let packed = self.recipe.canonical(inputs.device, &specialization)?;
        check_deadline!();
        let canonical_specification = format!("{specialization:?}");
        self.receipt.canonical_specification = Some(canonical_specification.clone());
        self.state.borrow_mut().canonical_specification = Some(canonical_specification);
        let state = self.state.borrow();
        let Some(progressive) = &state.progressive else {
            self.receipt.reason = "candidate was not formed".into();
            return Ok(false);
        };
        let feature = self.recipe.features(inputs.device)?;
        let mut cases = Vec::new();
        for seed in QUALIFICATION_SEEDS {
            check_deadline!();
            let mut case = match RecipeCase::new(inputs, self.recipe, self.point.rows, seed) {
                Ok(case) => case,
                Err(error) => {
                    self.receipt.reason = format!("candidate inputs not admitted: {error}");
                    return Ok(false);
                }
            };
            if self.recipe.prepared {
                check_deadline!();
                case.prepare_features(inputs.device, self.recipe, &feature)?;
            }
            // Noise rows: every level keeps and reads the whole head, so
            // every bound is tested. Both recipes also write their outputs
            // here for the first time, outside the timed pairs.
            check_deadline!();
            case.packed(inputs.device, self.recipe, &packed)?;
            check_deadline!();
            case.bounded(inputs.device, self.recipe, progressive)?;
            check_deadline!();
            if !case.same_winners(self.point.rows, self.recipe.vocabulary)? {
                self.receipt.reason = "canonical and bounded selection disagree".into();
                return Ok(false);
            }
            let resolved = match case.resolved_rows(inputs, self.recipe, self.point.rows, seed) {
                Ok(resolved) => resolved,
                Err(error) => {
                    self.receipt.reason = format!("candidate inputs not admitted: {error}");
                    return Ok(false);
                }
            };
            check_deadline!();
            case.set_hidden(inputs.device, self.recipe, &feature, resolved)?;
            // The timed rows select the same tokens too. Checked here, before
            // any round: reading the logits back idles the device, and the
            // run that follows an idle device is the slower for it.
            check_deadline!();
            case.packed(inputs.device, self.recipe, &packed)?;
            check_deadline!();
            case.bounded(inputs.device, self.recipe, progressive)?;
            check_deadline!();
            if !case.same_winners(self.point.rows, self.recipe.vocabulary)? {
                self.receipt.reason = "canonical and bounded selection disagree".into();
                return Ok(false);
            }
            cases.push(case);
        }
        // The search's comparison (`seismic::Tally`), on whole
        // recipe runs: a round raises the device to a sustained clock, then
        // runs both recipes at every seed in random order, each behind a lead
        // of its own work, and is ahead when the
        // bounded recipe cost less over the seeds. The comparison is admitted
        // the fewest rounds in which it can resolve as a lead at the search's
        // error rate, and ends with its first round that is not ahead.
        let rate = super::SEARCH_SETTINGS.error_rate;
        let mut tally = Tally::default();
        let mut shuffle = Shuffle::default();
        // The comparison's own sustained clock, gauged with the work it
        // compares: back-to-back packed runs at the first seed, read as the
        // rounds read their samples.
        let mut clock = SustainedClock::new(super::MIN_SAMPLE_SECONDS, tally.fewest(rate));
        for _ in 0..tally.fewest(rate) {
            check_deadline!();
            let raised = clock.raise(|passes| {
                let mut seconds = 0.;
                for _ in 0..passes {
                    if std::time::Instant::now() >= before {
                        return Err(None);
                    }
                    seconds += cases[0]
                        .packed(inputs.device, self.recipe, &packed)
                        .map_err(Some)?;
                }
                Ok::<f64, Option<CatalogFailure>>(seconds)
            });
            match raised {
                Ok(_) => {}
                Err(Some(error)) => return Err(error),
                Err(None) => {
                    self.receipt.reason = "first-load tuning allowance expired".into();
                    return Ok(false);
                }
            }
            let mut difference = 0.;
            for case in &mut cases {
                check_deadline!();
                let mut order = [0, 1];
                shuffle.shuffle(&mut order);
                let mut seconds = [0.; 2];
                for recipe in order {
                    check_deadline!();
                    seconds[recipe] = if recipe == 0 {
                        case.packed(inputs.device, self.recipe, &packed)?;
                        check_deadline!();
                        case.packed(inputs.device, self.recipe, &packed)?
                    } else {
                        case.bounded(inputs.device, self.recipe, progressive)?;
                        check_deadline!();
                        case.bounded(inputs.device, self.recipe, progressive)?
                    };
                }
                self.receipt.pairs.push((seconds[0], seconds[1]));
                difference += seconds[1] / seconds[0] - 1.;
            }
            tally.observe(difference);
            if difference >= 0. {
                break;
            }
        }
        if tally.verdict(rate) != Some(Verdict::Leads) {
            self.receipt.reason = format!(
                "bounded whole recipe did not resolve as faster than canonical in {} rounds",
                tally.rounds()
            );
            return Ok(false);
        }
        self.receipt.reason = format!(
            "same selected tokens; bounded whole recipe resolved as faster in {} rounds",
            tally.rounds()
        );
        Ok(true)
    }
    fn conclude(&mut self, selected: bool) -> Result<(), CatalogFailure> {
        self.receipt.selected = selected;
        self.state.borrow_mut().selected = selected;
        if !selected {
            *self.state.borrow_mut() = QualifiedRecipe::default();
        }
        if let Some(cache) = self.cache {
            cache.store_readout_choice(&self.key, &self.receipt);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionPath, resident_layout};
    use seismic::{BackendName, DeviceCatalog};

    /// Every program of the recipe digests and forms on each backend with
    /// its kernels, in both input forms: a normed exit's one row, and a
    /// hyper-connection exit's partial rows without a norm. Its kernels read
    /// different inputs, so a static one of them declares and another does
    /// not reaches only the one.
    #[test]
    fn every_recipe_program_takes_its_own_statics() {
        let catalog = DeviceCatalog::discover().unwrap();
        for backend in [BackendName::Metal, BackendName::Cuda] {
            let Ok(device) = catalog.open_backend(backend) else {
                continue;
            };
            let weight =
                Element::stored("q8g32s", resident_layout(ExecutionPath::Native, backend)).unwrap();
            for (parts, norm_extent) in [(1, 2048), (4, 0)] {
                for prepared in [false, true] {
                    let recipe = ReadoutRecipe {
                        scope: WeightScope::Target,
                        norm_scope: WeightScope::Target,
                        norm: Element::bf16(),
                        weight,
                        activation: Element::bf16(),
                        epsilon: 1e-6,
                        softcap: 0.,
                        prepared,
                        vocabulary: 4096,
                        hidden: 2048,
                        parts,
                        norm_extent,
                        low_planes: 3,
                        scale_extent: 0,
                    };
                    let form =
                        format!("{backend:?} PT={parts} NM={norm_extent} prepared={prepared}");
                    recipe
                        .digests(&device)
                        .unwrap_or_else(|error| panic!("{form}: {error:?}"));
                    recipe
                        .form(&device)
                        .unwrap_or_else(|error| panic!("{form}: {error:?}"));
                    recipe
                        .features(&device)
                        .unwrap_or_else(|error| panic!("{form}: {error:?}"));
                }
            }
        }
    }
}
