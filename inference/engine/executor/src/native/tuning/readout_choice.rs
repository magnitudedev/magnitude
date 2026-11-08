//! Whole-recipe readout qualification within the ordinary first-load allocation.
//! Packed is canonical. A class's bounded levels serve it where they select
//! the same tokens as the canonical projection and cost less than the planes'
//! full pass, the other way a scope stored as planes reads the class
//! ([`reads`]). The census establishes both, before any
//! optional work: the selection on noise and resolved rows, the costs on the
//! resolved rows. The qualification, once its canonical is final, prices the
//! class again against it. Evidence a load's allowance did not complete is
//! recorded unsettled, so a later load completes it, and every load reads
//! what was measured: an open whose qualification did not run decides its
//! scope's stored form on its census, as an open that qualified does.
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
//!
//! A class's choice is not its scope's stored form. A scope is stored in one
//! form, and every reader's every class reads it: its certified classes
//! through the bounded levels, a proposing reader's classes through the first
//! view, every other class through the planes' full pass, or all of them
//! through the packed projection. So each class's qualification also times
//! its read through the planes beside the two recipes ([`RecipeCosts`]), and
//! the scope stores whichever form its readers' classes, weighed by the load's
//! demand, cost less in all ([`stores_planes`]). A class that cannot be
//! certified (one past the rows the levels serve on its backend) has no
//! bounded recipe, and a proposing class needs none ([`class_levels`]): each
//! is priced through the packed projection and its planes read alone.

use super::{ComposedCensus, ComposedContender, PointShape, TuningInputs, TuningKey};
use crate::native::specialization::{entry_statics, prepare_program};
use crate::native::{CatalogFailure, ProgressiveReadoutKernels, ProjectionKernels};
use crate::programs::graph::readout::Levels;
use crate::{KernelCache, TuningCacheKey};
use magnitude_family_contracts::{ProgressiveCodeDepth, ProgressivePlane, WeightKind, WeightScope};
use magnitude_kernels::{
    readout_exact_rows, readout_features_rows, readout_head_rows, readout_planes_rows,
    readout_project_rows, readout_refine_rows, readout_top_rows,
};
use seismic::measure::{Shuffle, SustainedClock, Tally};
use seismic::{
    BackendName, BoundEntry, BuildError, Element, Entry, NativeKernel, NativeSpecialization,
    NativeTensorBatch, Tensor,
};
use std::{cell::RefCell, collections::HashMap, rc::Rc, time::Instant};

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
    /// The progressive head's code depth (`ImportTransform::Progressive`).
    pub depth: ProgressiveCodeDepth,
    pub scale_extent: u64,
}

#[derive(Default)]
pub(crate) struct QualifiedRecipe {
    pub progressive: Option<ProgressiveReadoutKernels>,
    /// The class's measured costs, from its census on and its qualification
    /// once timed; kept whatever the class's choice.
    pub costs: Option<RecipeCosts>,
    /// Whether its bounded levels selected the canonical projection's tokens
    /// on every row its census and qualification compared.
    pub agreed: bool,
    /// The class's executions per decode step of the load's demand: each
    /// decode term's launches of it times the term's weight
    /// ([`super::workload::terms`]), whatever is measured of the terms'
    /// times, so that every open of the load weighs it alike. Prefill
    /// launches are not among them ([`ReadoutContender::serves`]).
    pub weight: f64,
}

/// One class's measured cost of a whole readout invocation in each form its
/// scope's projection can be read through, in seconds: the packed
/// projection, the bounded levels over the planes where the class has them
/// ([`class_levels`]), and its uncertified read of the planes (their full
/// pass, or a proposing class's first view). All are timed on the same rows
/// in the same comparison.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct RecipeCosts {
    pub packed: f64,
    pub bounded: Option<f64>,
    pub planes: f64,
}

/// What a class whose reader and rows allow `levels` ([`class_levels`])
/// reads through its scope's planes, on its evidence: its bounded levels
/// where they `agreed` with the canonical selection and its `costs` show
/// them cheaper than the full pass, the other way a scope stored as planes
/// reads it, whatever they cost against packed; the full pass where not.
/// A proposing or full-pass class reads what it allows.
pub(crate) fn reads(levels: Levels, costs: Option<RecipeCosts>, agreed: bool) -> Levels {
    let serve = agreed
        && costs.is_some_and(|costs| costs.bounded.is_some_and(|bounded| bounded < costs.planes));
    match levels {
        Levels::Certified if !serve => Levels::Full,
        levels => levels,
    }
}

impl QualifiedRecipe {
    /// What the class reads through its scope's planes when its reader and
    /// rows allow `levels` ([`reads`]).
    pub(crate) fn reads(&self, levels: Levels) -> Levels {
        reads(levels, self.costs, self.agreed)
    }
}

impl RecipeCosts {
    /// The class's cost in each stored form: through the planes, as it
    /// `reads` them ([`reads`]): its bounded levels, or its uncertified read
    /// (the full pass, or a proposing class's first view).
    pub(crate) fn stored(self, reads: Levels) -> FormCosts {
        let planes = match (reads, self.bounded) {
            (Levels::Full | Levels::View, _) => self.planes,
            (Levels::Certified, Some(bounded)) => bounded,
            (Levels::Certified, None) => {
                panic!("a certified class was timed through its bounded levels")
            }
        };
        FormCosts {
            packed: self.packed,
            planes,
        }
    }
}

/// A class's cost of a whole readout invocation when its scope is stored
/// packed, and when it is stored as planes, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FormCosts {
    pub packed: f64,
    pub planes: f64,
}

/// What one reader's class of a scope weighs in its stored form: its
/// executions per decode step of the load's demand
/// ([`QualifiedRecipe::weight`]) and its measured costs in each form, if any
/// were measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClassCost {
    pub weight: f64,
    pub costs: Option<FormCosts>,
}

/// What a class of `rows` selected rows reads through its scope's planes.
/// A proposing reader's selections are only ever verified, so its classes
/// read the first view wherever the levels serve their rows
/// ([`Levels::proposing`]). Any other class has bounded levels when its rows
/// are within those the levels serve on `backend`
/// ([`crate::programs::graph::readout::certified_rows`]), and reads the full
/// pass otherwise. Only a class with bounded levels certifies a selection.
pub(crate) fn class_levels(proposes: bool, rows: u64, backend: BackendName) -> Levels {
    if proposes {
        Levels::proposing(rows)
    } else if rows <= crate::programs::graph::readout::certified_rows(backend) {
        Levels::Certified
    } else {
        Levels::Full
    }
}

/// Whether a scope is stored as its planes: when every class its readers
/// demand was measured, and the planes cost the load less over all of them
/// than the packed projection does. Packed is canonical: a demanded class
/// without measured costs, a tie, and a scope nothing demands keep it.
pub(crate) fn stores_planes(classes: impl IntoIterator<Item = ClassCost>) -> bool {
    let (mut packed, mut planes) = (0., 0.);
    for class in classes {
        if class.weight <= 0. {
            continue;
        }
        let Some(costs) = class.costs else {
            return false;
        };
        packed += class.weight * costs.packed;
        planes += class.weight * costs.planes;
    }
    planes < packed
}

pub(crate) type RecipeState = Rc<RefCell<QualifiedRecipe>>;
pub(crate) type RecipePreparation = Rc<RefCell<Option<ProgressiveReadoutKernels>>>;

pub(crate) struct ReadoutContender<'a> {
    pub recipe: ReadoutRecipe,
    pub point: PointShape,
    /// The recipe's unit: its weights, programs, rows and canonical unit.
    pub key: TuningCacheKey,
    /// Where its comparison is recorded ([`Self::record`]), once the
    /// canonical unit's choice is final.
    record: Option<TuningCacheKey>,
    /// The record was another load's, taken as it stands.
    adopted: bool,
    /// Where the completed choice is stored.
    pub cache: Option<&'a KernelCache>,
    /// Where a stored choice is read from: none for a fresh preparation.
    pub stored: Option<&'a KernelCache>,
    pub state: RecipeState,
    pub dependency: TuningKey,
    pub preparation: RecipePreparation,
    /// What the class reads through the planes ([`class_levels`]): one
    /// without bounded levels is priced through the packed projection and
    /// that read alone.
    levels: Levels,
    /// The qualification ran out of the load's allowance: what it measured
    /// is kept unsettled, for a later load to complete.
    expired: bool,
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
    /// The class's costs as its scope's stored form weighs them.
    costs: Option<RecipeCosts>,
    /// Whether the bounded levels selected the canonical tokens on every
    /// compared row ([`QualifiedRecipe::agreed`]).
    agreed: bool,
    settled: bool,
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
        levels: Levels,
    ) -> Self {
        Self {
            recipe,
            point,
            key,
            record: None,
            adopted: false,
            cache,
            stored,
            state,
            dependency,
            preparation,
            levels,
            expired: false,
            canonical_seconds: 0.0,
            candidate_seconds: 0.0,
            receipt: ReadoutChoiceReceipt {
                reason: "unmeasured within first-load allocation".into(),
                ..Default::default()
            },
        }
    }

    /// Key the record of the comparison against `canonical`, the canonical
    /// unit's final choice, as it serves the compared rows
    /// ([`ReadoutRecipe::served_canonical`]): loads whose canonicals serve
    /// those rows alike share a record, and one whose canonical serves them
    /// otherwise compares against its own and keeps its own.
    fn record(
        &mut self,
        device: &seismic::Device,
        canonical: &NativeSpecialization,
    ) -> Result<&TuningCacheKey, CatalogFailure> {
        let served = self.recipe.served_canonical(device, canonical, self.point.rows)?;
        self.receipt.canonical_specification = Some(served.clone());
        let key = TuningCacheKey::of(&format!("{}\ncanonical {served}", self.key.as_str()));
        Ok(self.record.insert(key))
    }

    /// The settled choice stored under `record`, when one is. What is stored
    /// there is the class's evidence, the costs it measured and whether its
    /// bounded levels agreed, which price its scope's stored form: a settled
    /// record's always, an unsettled one's where this open measured nothing
    /// of its own.
    fn stored_choice(&mut self, record: &TuningCacheKey) -> Option<bool> {
        let cache = self.stored?;
        let selected = cache.readout_choice(record);
        let evidence = cache
            .readout_costs(record)
            .filter(|_| selected.is_some() || self.receipt.costs.is_none());
        if let Some(costs) = evidence {
            self.receipt.costs = Some(costs);
            self.receipt.agreed = cache.readout_agreed(record).unwrap_or(false);
            let mut state = self.state.borrow_mut();
            state.costs = self.receipt.costs;
            state.agreed = self.receipt.agreed;
        }
        selected
    }

    /// The deferred lookup of a load that searched the canonical unit
    /// itself: once its choice `canonical` is final, another load's settled
    /// comparison against a canonical that serves these rows alike is taken
    /// as it stands, and is not recorded again. None when this load is the
    /// first to compare against it.
    fn adopt(
        &mut self,
        device: &seismic::Device,
        canonical: &NativeSpecialization,
    ) -> Result<Option<bool>, CatalogFailure> {
        let record = self.record(device, canonical)?.clone();
        let stored = self.stored_choice(&record);
        self.adopted = stored.is_some();
        Ok(stored)
    }
}

/// Resolve the declaration's complete fixed configuration, including tuning
/// parameters, at the element bindings `entry` prepares (its `where` may
/// read them, as `form(OW, MATRIX)` does), as `device` offers it. A
/// composed comparison forms these defaults without searching the
/// component kernels independently.
fn declared_defaults<E: Entry>(
    device: &seismic::Device,
    entry: &BoundEntry<E>,
    statics: &NativeSpecialization,
) -> Result<NativeSpecialization, CatalogFailure> {
    entry
        .default_specialization(device, statics)
        .map_err(fail)?
        .ok_or_else(|| fail("readout recipe kernel admits no default at its statics"))
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
            ("P", u64::from(self.depth.rest_planes())),
            ("WS", self.scale_extent),
            // The prepared projection publishes every vocabulary column of
            // implicit token rows.
            ("S", self.vocabulary),
            ("I", 0),
        ]
    }

    /// `entry`'s statics in this recipe on `backend`.
    fn statics<E: Entry>(
        self,
        backend: BackendName,
        entry: &BoundEntry<E>,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        entry_statics(backend, entry, "readout_recipe", &self.values())
    }

    /// The bounded top level's entry.
    fn top_entry(self) -> BoundEntry<readout_top_rows::Entry> {
        readout_top_rows::native_entry_with(readout_top_rows::Elements {
            NW: self.norm,
            A: self.activation,
        })
    }

    /// The bounded refinement level's entry.
    fn refine_entry(self) -> BoundEntry<readout_refine_rows::Entry> {
        readout_refine_rows::native_entry_with(readout_refine_rows::Elements { A: self.activation })
    }

    /// The bounded exact level's entry.
    fn exact_entry(self) -> BoundEntry<readout_exact_rows::Entry> {
        readout_exact_rows::native_entry_with(readout_exact_rows::Elements { A: self.activation })
    }

    /// The bounded planes level's entry.
    fn planes_entry(self) -> BoundEntry<readout_planes_rows::Entry> {
        readout_planes_rows::native_entry_with(readout_planes_rows::Elements {
            NW: self.norm,
            A: self.activation,
        })
    }

    /// The feature preparation's entry.
    fn features_entry(self) -> BoundEntry<readout_features_rows::Entry> {
        readout_features_rows::native_entry_with(readout_features_rows::Elements {
            NW: self.norm,
            A: self.activation,
        })
    }

    /// The canonical projection's entry over raw hidden rows.
    fn head_entry(self) -> BoundEntry<readout_head_rows::Entry> {
        readout_head_rows::native_entry_with(readout_head_rows::Elements {
            NW: self.norm,
            OW: self.weight,
            A: self.activation,
        })
    }

    /// The canonical projection's entry over prepared features.
    fn project_entry(self) -> BoundEntry<readout_project_rows::Entry> {
        readout_project_rows::native_entry_with(readout_project_rows::Elements {
            OW: self.weight,
            A: self.activation,
        })
    }

    /// The identity of `entry`'s program in this recipe on `device`.
    fn digest<E: Entry>(
        self,
        device: &seismic::Device,
        entry: &BoundEntry<E>,
    ) -> Result<String, CatalogFailure> {
        entry
            .digest(device, &self.statics(device.backend(), entry)?)
            .map_err(fail)
    }

    /// The identity of the recipe's programs on `device`: its four bounded
    /// levels and its canonical projection, each at its own statics.
    pub(crate) fn digests(self, device: &seismic::Device) -> Result<[String; 5], CatalogFailure> {
        Ok([
            self.digest(device, &self.top_entry())?,
            self.digest(device, &self.refine_entry())?,
            self.digest(device, &self.exact_entry())?,
            self.digest(device, &self.planes_entry())?,
            match self.prepared {
                true => self.digest(device, &self.project_entry())?,
                false => self.digest(device, &self.head_entry())?,
            },
        ])
    }

    /// The recipe's feature reader, `readout_features_rows`.
    fn features(
        self,
        device: &seismic::Device,
    ) -> Result<NativeKernel<readout_features_rows::Entry>, CatalogFailure> {
        let entry = self.features_entry();
        let statics = self.statics(device.backend(), &entry)?;
        prepare_program(device, entry, &statics).map_err(fail)
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

    /// `entry` prepared at its declared defaults at this recipe's statics.
    fn at_defaults<E: Entry>(
        self,
        device: &seismic::Device,
        entry: BoundEntry<E>,
    ) -> Result<NativeKernel<E>, CatalogFailure> {
        let defaults = declared_defaults(device, &entry, &self.statics(device.backend(), &entry)?)?;
        prepare_program(device, entry, &defaults).map_err(fail)
    }

    /// Form the recipe's four programs, to the end.
    fn form(self, device: &seismic::Device) -> Result<QualifiedRecipe, CatalogFailure> {
        let progressive = ProgressiveReadoutKernels {
            top: self.at_defaults(device, self.top_entry())?,
            refine: self.at_defaults(device, self.refine_entry())?,
            exact: self.at_defaults(device, self.exact_entry())?,
            planes: self.at_defaults(device, self.planes_entry())?,
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
            prepare_program(device, self.project_entry(), specialization)
                .map(ProjectionKernels::PreparedPacked)
                .map_err(fail)
        } else {
            prepare_program(device, self.head_entry(), specialization)
                .map(ProjectionKernels::RawPacked)
                .map_err(fail)
        }
    }

    /// The canonical projection under `specialization` as it serves `rows`
    /// rows ([`seismic::served_key`]): what a paired comparison at `rows`
    /// measured the bounded recipe against. A choice that differs only in
    /// what other row counts run, such as the row count from which a
    /// batched launch serves, serves `rows` alike.
    pub(crate) fn served_canonical(
        self,
        device: &seismic::Device,
        specialization: &NativeSpecialization,
        rows: u64,
    ) -> Result<String, CatalogFailure> {
        if self.prepared {
            self.served(device, &self.project_entry(), specialization, rows)
        } else {
            self.served(device, &self.head_entry(), specialization, rows)
        }
    }

    fn served<E: Entry>(
        self,
        device: &seismic::Device,
        entry: &BoundEntry<E>,
        specialization: &NativeSpecialization,
        rows: u64,
    ) -> Result<String, CatalogFailure> {
        let implementation = entry
            .offered(device)
            .map_err(fail)?
            .ok_or_else(|| fail("the canonical projection is not offered on the device"))?;
        // The rows read (`M`) are the rows projected (`O`), as the paired
        // comparison runs them (`RecipeCase::new`).
        let dimensions = std::collections::BTreeMap::from([("M".into(), rows), ("O".into(), rows)]);
        let key = seismic::served_key(
            &implementation,
            &self.statics(device.backend(), entry)?,
            &dimensions,
            specialization,
        )
        .map_err(fail)?;
        Ok(format!("{key:?}"))
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
    /// The planes' full pass's logits.
    full: Tensor,
    token_rows: Tensor,
}

impl RecipeCase {
    fn new(
        inputs: &mut TuningInputs<'_, '_>,
        recipe: ReadoutRecipe,
        rows: u64,
        seed: u64,
    ) -> Result<Self, BuildError> {
        let hidden =
            inputs.activation(Element::f32(), &[rows, recipe.parts * recipe.hidden], seed)?;
        let norm = if recipe.norm_extent == 0 {
            inputs.scratch(recipe.norm, &[0])?
        } else {
            inputs.shared_weight(recipe.norm_scope, WeightKind::OutputNorm)?
        };
        let weight = inputs
            .shared_weight(recipe.scope, WeightKind::Output)?
            .slice_leading(0, recipe.vocabulary)?;
        let mut views = Vec::new();
        for plane in ProgressivePlane::ALL {
            views.push(
                inputs
                    .shared_weight(recipe.scope, WeightKind::OutputPlane(plane))?
                    .slice_leading(0, recipe.vocabulary)?,
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
            Tensor::zeros(inputs.device, element, shape).map_err(BuildError::from)
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
            full: zeros(Element::f32(), &[rows, recipe.vocabulary])?,
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
    ) -> Result<Tensor, BuildError> {
        let [top, next, rest, scales, _] = &self.planes;
        let stored = |plane: &Tensor, token: u64| -> Result<Vec<u8>, BuildError> {
            Ok(plane.slice_leading(token, token + 1)?.read_to_host()?)
        };
        let columns = usize::try_from(recipe.hidden).map_err(|_| "hidden width exceeds usize")?;
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
                recipe.depth,
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

    /// Push the first level, writing its view, floor, features and lengths.
    fn push_top(
        &mut self,
        batch: &mut NativeTensorBatch,
        recipe: ReadoutRecipe,
        kernels: &ProgressiveReadoutKernels,
    ) -> Result<(), CatalogFailure> {
        let [top, bit3, rest, scales, radius] = &self.planes;
        // The entry reads the rows as their `[M, PT, D]` parts.
        let rows = self.hidden.extents()[0];
        let partials = self
            .hidden
            .reshape(&[rows, recipe.parts, recipe.hidden])
            .map_err(fail)?;
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
        Ok(())
    }

    /// Run the bounded progressive recipe once; the seconds it took.
    fn bounded(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        kernels: &ProgressiveReadoutKernels,
    ) -> Result<f64, CatalogFailure> {
        let mut batch = NativeTensorBatch::new(device);
        self.push_top(&mut batch, recipe, kernels)?;
        let [top, bit3, rest, scales, radius] = &self.planes;
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

    /// Run a class's read of the planes without bounded levels once: a
    /// proposing class's first view, any other class's full pass (a class
    /// with bounded levels reads it when its qualification does not certify
    /// them); the seconds it took.
    fn uncertified(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        kernels: &ProgressiveReadoutKernels,
        levels: Levels,
    ) -> Result<f64, CatalogFailure> {
        match levels {
            Levels::View => self.view(device, recipe, kernels),
            Levels::Full | Levels::Certified => self.planes(device, recipe, kernels),
        }
    }

    /// Run the first level alone once, a proposing class's whole read of
    /// the planes (`Levels::View`); the seconds it took.
    fn view(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        kernels: &ProgressiveReadoutKernels,
    ) -> Result<f64, CatalogFailure> {
        let mut batch = NativeTensorBatch::new(device);
        self.push_top(&mut batch, recipe, kernels)?;
        let start = Instant::now();
        batch.submit().map_err(fail)?.wait().map_err(fail)?;
        Ok(start.elapsed().as_secs_f64())
    }

    /// Run the planes' full pass once, the readout of every class a scope
    /// stored as planes neither certifies nor proposes from; the seconds it
    /// took.
    fn planes(
        &mut self,
        device: &seismic::Device,
        recipe: ReadoutRecipe,
        kernels: &ProgressiveReadoutKernels,
    ) -> Result<f64, CatalogFailure> {
        let [top, bit3, rest, scales, _] = &self.planes;
        // The entry reads the rows as their `[M, PT, D]` parts.
        let rows = self.hidden.extents()[0];
        let partials = self
            .hidden
            .reshape(&[rows, recipe.parts, recipe.hidden])
            .map_err(fail)?;
        let mut batch = NativeTensorBatch::new(device);
        batch
            .push(
                &kernels.planes,
                readout_planes_rows::Args {
                    hidden: &partials,
                    norm: &self.norm,
                    top,
                    bit3,
                    rest,
                    scales,
                    out_rows: &self.out_rows,
                    epsilon: recipe.epsilon,
                    softcap: recipe.softcap,
                },
                readout_planes_rows::OutputArgs {
                    value: &mut self.full,
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
        self.state.borrow_mut().weight = self
            .point
            .terms
            .iter()
            .map(|(term, launches)| objective.terms[*term].weight * launches)
            .sum();
    }
    /// A readout projects in its graph's decode launches, at the rows of the
    /// class that serves them. In prefill a draft injects without projecting,
    /// and the target projects one row of a prompt: the demand states no
    /// prompts per generated token to weigh that row against decode steps,
    /// so it is not priced.
    fn serves(
        &self,
        launch: &crate::programs::invocation_demand::Launch,
        phase: super::workload::Phase,
    ) -> Option<u64> {
        let rows = self.point.rows;
        let decode = phase == super::workload::Phase::Decode;
        (decode && launch.rows > rows / 2 && launch.rows <= rows).then_some(rows)
    }
    fn key(&self) -> String {
        self.key.as_str().to_owned()
    }
    fn dependencies(&self) -> Vec<TuningKey> {
        vec![self.dependency.clone()]
    }
    fn pin_identity(&self) -> String {
        format!("{:?}/rows={}", self.recipe, self.point.rows)
    }
    fn label(&self) -> &'static str {
        "readout_recipe"
    }
    fn cached(
        &mut self,
        device: &seismic::Device,
        chosen: &HashMap<TuningKey, NativeSpecialization>,
    ) -> Result<Option<bool>, CatalogFailure> {
        let Some(canonical) = chosen.get(&self.dependency) else {
            return Ok(None);
        };
        let record = self.record(device, canonical)?.clone();
        Ok(self.stored_choice(&record))
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
        let statics = self.dependency.2.specialization();
        let default = if self.recipe.prepared {
            declared_defaults(inputs.device, &self.recipe.project_entry(), &statics)?
        } else {
            declared_defaults(inputs.device, &self.recipe.head_entry(), &statics)?
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
        let progressive = formed.progressive.as_ref().unwrap();
        // Each form writes its outputs here first, outside the timed runs.
        case.packed(inputs.device, self.recipe, &packed)?;
        case.uncertified(inputs.device, self.recipe, progressive, self.levels)?;
        if self.levels == Levels::Certified {
            // Noise rows test every bound; the rows the head resolves are
            // the ones the levels serve, and are timed.
            case.bounded(inputs.device, self.recipe, progressive)?;
            self.receipt.agreed = case.same_winners(self.point.rows, self.recipe.vocabulary)?;
            if self.receipt.agreed {
                let resolved = match case.resolved_rows(inputs, self.recipe, self.point.rows, 1) {
                    Ok(resolved) => resolved,
                    Err(error) => {
                        self.receipt.reason = format!("candidate inputs not admitted: {error}");
                        return Ok(None);
                    }
                };
                case.set_hidden(inputs.device, self.recipe, &feature, resolved)?;
                case.packed(inputs.device, self.recipe, &packed)?;
                case.bounded(inputs.device, self.recipe, progressive)?;
                self.receipt.agreed =
                    case.same_winners(self.point.rows, self.recipe.vocabulary)?;
            }
        }
        let measured = Instant::now();
        self.canonical_seconds = case.packed(inputs.device, self.recipe, &packed)?;
        let bounded = match self.levels {
            Levels::Certified => {
                self.candidate_seconds = case.bounded(inputs.device, self.recipe, progressive)?;
                self.receipt
                    .pairs
                    .push((self.canonical_seconds, self.candidate_seconds));
                Some(self.candidate_seconds)
            }
            Levels::Full | Levels::View => None,
        };
        let planes = case.uncertified(inputs.device, self.recipe, progressive, self.levels)?;
        // Until a qualification times the class against its final
        // canonical, its census prices it.
        self.receipt.costs = Some(RecipeCosts {
            packed: self.canonical_seconds,
            bounded,
            planes,
        });
        self.receipt.reason = match (self.levels, self.receipt.agreed) {
            (Levels::Certified, false) => "canonical and bounded selection disagree at the census".into(),
            _ => "censused; qualification not admitted within first-load allocation".into(),
        };
        {
            let mut state = self.state.borrow_mut();
            state.progressive = formed.progressive;
            state.costs = self.receipt.costs;
            state.agreed = self.receipt.agreed;
        }
        let rounds = Tally::default().fewest(super::SEARCH_SETTINGS.error_rate);
        // The comparison's clock is established by its first raise: twice
        // over, readings of the packed run in doubling lengths from a
        // sample's, none shorter than one run (`seismic::measure::SustainedClock`).
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
            // clock, and runs what the census ran, for each seed: twice for
            // agreement (noise rows, then the timed rows; the two recipes
            // alone, so this bounds it), and a lead and a sample in each
            // round it is admitted. A class without bounded levels runs less.
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
                    self.expired = true;
                    return Ok(false);
                }
            };
        }
        check_deadline!();
        let Some(specialization) = inputs.choice(&self.dependency).cloned() else {
            self.receipt.reason = "canonical dependency has no finalized choice".into();
            return Ok(false);
        };
        if let Some(selected) = self.adopt(inputs.device, &specialization)? {
            return Ok(selected);
        }
        let packed = self.recipe.canonical(inputs.device, &specialization)?;
        check_deadline!();
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
            if self.levels != Levels::Certified {
                // Both forms read every row alike, so any rows price them;
                // each writes its outputs here first, outside the rounds.
                check_deadline!();
                case.packed(inputs.device, self.recipe, &packed)?;
                check_deadline!();
                case.uncertified(inputs.device, self.recipe, progressive, self.levels)?;
                cases.push(case);
                continue;
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
                self.receipt.agreed = false;
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
                self.receipt.agreed = false;
                return Ok(false);
            }
            cases.push(case);
        }
        // The search's comparison (`seismic::measure::Tally`), on whole
        // recipe runs: a round raises the device to a sustained clock, then
        // runs both recipes and the planes' full pass at every seed in random
        // order, each behind a lead of its own work, and is ahead when the
        // bounded levels cost less over the seeds than the full pass, the
        // other way a scope stored as planes reads the class. Every round's
        // samples of all three price the class for its scope's stored form.
        // The comparison is admitted the fewest rounds in which it can
        // resolve as a lead at the search's error rate, and ends with its
        // first round that is not ahead. A class without bounded levels
        // compares nothing: its rounds, as many, price the packed projection
        // and its planes read alone.
        let rate = super::SEARCH_SETTINGS.error_rate;
        let mut tally = Tally::default();
        let mut shuffle = Shuffle::default();
        // The comparison's own sustained clock, gauged with the work it
        // compares: back-to-back packed runs at the first seed, read as the
        // rounds read their samples.
        let mut clock = SustainedClock::new(super::MIN_SAMPLE_SECONDS, tally.fewest(rate));
        // Each form's summed samples, and their count.
        let mut sums = [0.; 3];
        let mut samples = 0u32;
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
                    self.expired = true;
                    return Ok(false);
                }
            }
            let mut difference = 0.;
            let mut round = [0.; 3];
            for case in &mut cases {
                check_deadline!();
                let certified = self.levels == Levels::Certified;
                let mut order = match certified {
                    true => vec![0, 1, 2],
                    false => vec![0, 2],
                };
                shuffle.shuffle(&mut order);
                let mut seconds = [0.; 3];
                for form in order {
                    check_deadline!();
                    seconds[form] = match form {
                        0 => {
                            case.packed(inputs.device, self.recipe, &packed)?;
                            check_deadline!();
                            case.packed(inputs.device, self.recipe, &packed)?
                        }
                        1 => {
                            case.bounded(inputs.device, self.recipe, progressive)?;
                            check_deadline!();
                            case.bounded(inputs.device, self.recipe, progressive)?
                        }
                        _ => {
                            case.uncertified(inputs.device, self.recipe, progressive, self.levels)?;
                            check_deadline!();
                            case.uncertified(inputs.device, self.recipe, progressive, self.levels)?
                        }
                    };
                }
                if certified {
                    self.receipt.pairs.push((seconds[0], seconds[1]));
                    difference += seconds[1] / seconds[2] - 1.;
                }
                for (sum, seconds) in round.iter_mut().zip(seconds) {
                    *sum += seconds;
                }
            }
            // A whole round prices the class, in place of its census.
            for (sum, seconds) in sums.iter_mut().zip(round) {
                *sum += seconds;
            }
            samples += cases.len() as u32;
            let [packed, bounded, planes] = sums.map(|sum| sum / f64::from(samples));
            let certified = self.levels == Levels::Certified;
            self.receipt.costs = Some(RecipeCosts {
                packed,
                bounded: certified.then_some(bounded),
                planes,
            });
            if !certified {
                continue;
            }
            tally.observe(difference);
            if difference >= 0. {
                break;
            }
        }
        if self.levels != Levels::Certified {
            self.receipt.reason = format!(
                "priced through the packed projection and its {:?} planes read in {} rounds; no bounded levels at this class",
                self.levels,
                tally.fewest(rate)
            );
            return Ok(false);
        }
        // Every seed's selections agreed with the canonical's here; the
        // census's own compared rows stand beside them.
        let certified =
            reads(self.levels, self.receipt.costs, self.receipt.agreed) == Levels::Certified;
        self.receipt.reason = match (self.receipt.agreed, certified) {
            (false, _) => "canonical and bounded selection disagreed at the census".into(),
            (true, true) => format!(
                "same selected tokens; bounded levels cost less than the full pass over {} rounds",
                tally.rounds()
            ),
            (true, false) => format!(
                "same selected tokens; bounded levels cost no less than the full pass over {} rounds",
                tally.rounds()
            ),
        };
        Ok(certified)
    }
    fn finalize(
        &mut self,
        device: &seismic::Device,
        chosen: &HashMap<TuningKey, NativeSpecialization>,
    ) -> Result<(), CatalogFailure> {
        // Never qualified: its census is recorded, unsettled, against the
        // canonical's final choice, so the load reads it and a later load
        // qualifies it. Another load's settled comparison there stands.
        if self.record.is_none() {
            if let Some(canonical) = chosen.get(&self.dependency) {
                let record = self.record(device, canonical)?.clone();
                self.adopted = self.stored.and_then(|cache| cache.readout_choice(&record)).is_some();
            }
        }
        let certified = self.state.borrow().reads(self.levels) == Levels::Certified;
        self.conclude(certified, false)
    }
    fn conclude(&mut self, selected: bool, settled: bool) -> Result<(), CatalogFailure> {
        self.receipt.selected = selected;
        // What an expired qualification measured is evidence, not a choice:
        // a later load completes it.
        self.receipt.settled = settled && !self.expired;
        {
            let mut state = self.state.borrow_mut();
            state.costs = self.receipt.costs;
            state.agreed = self.receipt.agreed;
            if !selected {
                state.progressive = None;
            }
        }
        // A comparison is recorded against the canonical it compared with:
        // none is known before that unit's choice is final, and a record
        // taken from another load stands as it was.
        if let (Some(cache), Some(record), false) = (self.cache, &self.record, self.adopted) {
            cache.store_readout_choice(record, &self.receipt);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionPath, resident_layout};

    fn class(weight: f64, certified: bool, [packed, bounded, planes]: [f64; 3]) -> ClassCost {
        let reads = if certified { Levels::Certified } else { Levels::Full };
        ClassCost {
            weight,
            costs: Some(
                RecipeCosts {
                    packed,
                    bounded: Some(bounded),
                    planes,
                }
                .stored(reads),
            ),
        }
    }

    /// A drafting round of an MTP load as measured on an M4 Pro (ms): one
    /// plain step's 1-row class certified (bounded 1.62, packed 2.05), and
    /// the draft's 3 rows and the verification's 4, whose classes are not
    /// certified, reading the full pass at 2.91 against packed's 2.04.
    fn drafting_round(planes: f64) -> [ClassCost; 3] {
        [
            class(1., true, [2.05, 1.62, 2.91]),
            class(1., false, [2.04, 3.9, planes]),
            class(1., false, [2.04, 3.9, planes]),
        ]
    }

    /// The scope stores the form its readers' classes, weighed, cost less
    /// in all: not the form one reader's certified class prefers.
    #[test]
    fn a_scope_stores_the_form_its_classes_weigh_cheaper() {
        // 1.62 + 2 x 2.91 = 7.44 through the planes, 6.13 packed.
        assert!(!stores_planes(drafting_round(2.91)));
        // A full pass at packed's pace: 1.62 + 2 x 2.04 = 5.70.
        assert!(stores_planes(drafting_round(2.04)));
        // Weight decides: plain steps outnumbering the round's other reads
        // by enough (past 1.74 / 0.43, about 4 to 1) make the planes the
        // cheaper form even at 2.91.
        let mut weighed = drafting_round(2.91);
        weighed[0].weight = 5.;
        assert!(stores_planes(weighed));
        // A certified class alone, as on a load that does not draft.
        assert!(stores_planes([class(1., true, [2.05, 1.62, 2.91])]));
        // A full pass cheaper than packed stores the planes with nothing
        // certified.
        assert!(stores_planes([class(1., false, [2., 3., 1.5])]));
    }

    /// Equal totals keep the canonical packed projection, and the choice
    /// does not depend on the order the classes are met in.
    #[test]
    fn a_tie_keeps_the_packed_projection_in_any_order() {
        let tied = [class(1., true, [2., 1., 4.]), class(1., false, [2., 4., 3.])];
        assert!(!stores_planes(tied));
        assert!(!stores_planes(tied.into_iter().rev()));
        let mut round = drafting_round(2.04);
        assert!(stores_planes(round));
        round.reverse();
        assert!(stores_planes(round));
    }

    /// A demanded class without measured costs keeps the packed
    /// projection; a class nothing demands weighs nothing, measured or not,
    /// and a scope nothing demands keeps it too.
    #[test]
    fn missing_evidence_keeps_the_packed_projection() {
        let unmeasured = ClassCost {
            weight: 1.,
            costs: None,
        };
        let certified = class(1., true, [2.05, 1.62, 2.91]);
        assert!(!stores_planes([certified, unmeasured]));
        assert!(stores_planes([
            certified,
            ClassCost {
                weight: 0.,
                ..unmeasured
            },
            class(0., false, [1., 9., 9.]),
        ]));
        assert!(!stores_planes([class(0., true, [2., 1., 1.])]));
        assert!(!stores_planes(std::iter::empty()));
    }

    /// Bounded levels exist only for a target class at rows the levels serve
    /// on its backend; every target class past that bound reads the full
    /// pass. A proposing reader's classes read the first view at every row
    /// count the levels serve, on every backend, and the full pass beyond:
    /// each class without bounded levels is priced through the packed
    /// projection and that read alone, and certifies nothing.
    #[test]
    fn bounded_levels_exist_only_where_they_serve() {
        for (backend, served) in [(BackendName::Metal, 1), (BackendName::Cuda, 4)] {
            assert_eq!(class_levels(false, served, backend), Levels::Certified);
            assert_eq!(class_levels(false, served + 1, backend), Levels::Full);
            for rows in [1, served + 1, 8] {
                assert_eq!(class_levels(true, rows, backend), Levels::View);
            }
            assert_eq!(class_levels(true, 9, backend), Levels::Full);
        }
        assert_eq!(class_levels(false, 1, BackendName::Vulkan), Levels::Full);
        let priced = RecipeCosts {
            packed: 2.,
            bounded: None,
            planes: 3.,
        };
        assert_eq!(
            priced.stored(Levels::Full),
            FormCosts {
                packed: 2.,
                planes: 3.
            }
        );
    }

    /// A class certified without its bounded levels timed is an invariant
    /// violation, not a planes cost.
    #[test]
    #[should_panic(expected = "a certified class was timed through its bounded levels")]
    fn a_certified_class_has_its_bounded_cost() {
        let _ = RecipeCosts {
            packed: 2.,
            bounded: None,
            planes: 3.,
        }
        .stored(Levels::Certified);
    }

    /// A class runs in its graph's decode launches at its rows, never in a
    /// prefill launch (a draft injects without projecting there), and weighs
    /// its executions per decode step: each decode term's launches times the
    /// term's weight, whatever is measured of the terms' times.
    #[test]
    fn a_class_weighs_its_decode_executions_alone() {
        use super::super::workload::{Metric, Phase, Term};
        use crate::programs::invocation_demand::{InvocationCounts, Launch};
        let layout = resident_layout(ExecutionPath::Native, BackendName::Metal);
        let recipe = ReadoutRecipe {
            scope: WeightScope::Target,
            norm_scope: WeightScope::Target,
            norm: Element::bf16(),
            weight: Element::stored("q8g32s", layout).unwrap(),
            activation: Element::bf16(),
            epsilon: 1e-6,
            softcap: 0.,
            prepared: false,
            vocabulary: 4096,
            hidden: 2048,
            parts: 1,
            norm_extent: 2048,
            depth: ProgressiveCodeDepth::Eight,
            scale_extent: 0,
        };
        let state = RecipeState::default();
        let point = PointShape {
            label: "m4".into(),
            weight: 1.,
            terms: Vec::new(),
            rows: 4,
            sequences: 1,
            context: None,
            class: None,
        };
        let mut contender = ReadoutContender::new(
            recipe,
            point,
            TuningCacheKey::of("readout recipe unit"),
            None,
            None,
            state.clone(),
            ("readout_head_rows", "head".into(), super::super::FixedValues::default()),
            RecipePreparation::default(),
            Levels::Full,
        );
        let launch = |term, rows| Launch {
            term,
            rows,
            sequences: 1,
            history: 4096,
        };
        assert_eq!(contender.serves(&launch(0, 4), Phase::Prefill), None);
        assert_eq!(contender.serves(&launch(0, 512), Phase::Prefill), None);
        assert_eq!(contender.serves(&launch(1, 4), Phase::Decode), Some(4));
        assert_eq!(contender.serves(&launch(1, 3), Phase::Decode), Some(4));
        assert_eq!(contender.serves(&launch(1, 2), Phase::Decode), None);
        let term = |phase| Term {
            metric: Metric {
                phase,
                context: 4096,
                sequences: 1,
            },
            width: None,
            weight: 0.5,
        };
        let mut counts = InvocationCounts::default();
        counts.launch(launch(1, 4), 2.);
        for measured in [vec![0., 0.], vec![2.68, 0.029]] {
            let objective = super::super::Objective {
                terms: vec![term(Phase::Prefill), term(Phase::Decode)],
                measured,
            };
            contender.set_invocations(&counts, &objective);
            assert_eq!(state.borrow().weight, 1.);
        }
    }

    /// Every program of the recipe digests and forms on each backend with
    /// its kernels, in both input forms: a normed exit's one row, and a
    /// hyper-connection exit's partial rows without a norm. Its kernels read
    /// different inputs, so a static one of them declares and another does
    /// not reaches only the one.
    #[test]
    fn every_recipe_program_takes_its_own_statics() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let available = [BackendName::Metal, BackendName::Cuda]
            .into_iter()
            .filter(|&backend| crate::test_support::available(&catalog, backend));
        for backend in available {
            let device = crate::test_support::open_backend(&catalog, backend)
                .unwrap_or_else(|error| panic!("an available {backend:?} executor opens: {error}"));
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
                        depth: ProgressiveCodeDepth::Eight,
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
                    let default = if prepared {
                        let project = recipe.project_entry();
                        recipe
                            .statics(backend, &project)
                            .and_then(|statics| declared_defaults(&device, &project, &statics))
                    } else {
                        let head = recipe.head_entry();
                        recipe
                            .statics(backend, &head)
                            .and_then(|statics| declared_defaults(&device, &head, &statics))
                    }
                    .unwrap_or_else(|error| panic!("{form}: {error:?}"));
                    recipe
                        .canonical(&device, &default)
                        .unwrap_or_else(|error| panic!("{form}: {error:?}"));
                }
            }
        }
    }

    /// The recipe of the projection a Metal or CUDA test device reads, with
    /// its canonical head's declared defaults.
    fn recipe_on(device: &seismic::Device) -> (ReadoutRecipe, NativeSpecialization) {
        let backend = device.backend();
        let recipe = ReadoutRecipe {
            scope: WeightScope::Target,
            norm_scope: WeightScope::Target,
            norm: Element::bf16(),
            weight: Element::stored("q8g32s", resident_layout(ExecutionPath::Native, backend))
                .unwrap(),
            activation: Element::bf16(),
            epsilon: 1e-6,
            softcap: 0.,
            prepared: false,
            vocabulary: 4096,
            hidden: 2048,
            parts: 1,
            norm_extent: 2048,
            depth: ProgressiveCodeDepth::Eight,
            scale_extent: 0,
        };
        let head = recipe.head_entry();
        let defaults = recipe
            .statics(backend, &head)
            .and_then(|statics| declared_defaults(device, &head, &statics))
            .unwrap();
        (recipe, defaults)
    }

    /// The test devices that read progressive heads.
    fn devices() -> Vec<seismic::Device> {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        [BackendName::Metal, BackendName::Cuda]
            .into_iter()
            .filter(|&backend| crate::test_support::available(&catalog, backend))
            .map(|backend| {
                catalog
                    .open_backend(backend)
                    .unwrap_or_else(|error| panic!("an available {backend:?} executor opens: {error}"))
            })
            .collect()
    }

    /// A canonical head choice as it serves the compared rows: one that
    /// differs only in what other row counts run (on Metal the row count
    /// from which the batched launch serves, on CUDA the large-batch GEMM's
    /// rotation) serves one row as the compared choice does, and one that
    /// differs in what those rows run does not.
    #[test]
    fn a_canonical_is_compared_as_the_rows_serve_it() {
        for device in devices() {
            let backend = device.backend();
            let (recipe, compared) = recipe_on(&device);
            let (other, rows) = match backend {
                BackendName::Metal => (compared.clone().with_param("BATCH_FROM", 3), 4),
                _ => (compared.clone().with_launch_param(4, "ROTATE", 1), 65),
            };
            let served = |specialization: &NativeSpecialization, rows| {
                recipe.served_canonical(&device, specialization, rows).unwrap()
            };
            assert_eq!(served(&compared, 1), served(&other, 1), "{backend:?}");
            assert_ne!(served(&compared, rows), served(&other, rows), "{backend:?}");
        }
    }

    /// A plain open, then a drafting open whose canonical head serves one
    /// row otherwise (on Metal the one-row GEMV's mapping, on CUDA its
    /// K split, as two loads tuned over different row counts pick), each
    /// find no record of their own canonical, compare against it and
    /// record their winner beside the other's. A third open of either kind
    /// finds its winner and searches nothing, and a canonical that serves
    /// the row as the plain one does reads the plain open's record.
    #[test]
    fn each_canonical_keeps_its_own_record_across_opens() {
        for device in devices() {
            let backend = device.backend();
            let (recipe, plain) = recipe_on(&device);
            let drafting = match backend {
                BackendName::Metal => plain
                    .clone()
                    .with_launch_param(1, "ROWS", 2)
                    .with_launch_param(1, "SIMDGROUPS", 8),
                _ => plain.clone().with_launch_param(1, "KSPLIT", 2),
            };
            let alike = match backend {
                BackendName::Metal => plain.clone().with_param("BATCH_FROM", 3),
                _ => plain.clone().with_launch_param(4, "ROTATE", 1),
            };
            let root = std::env::temp_dir().join(format!(
                "magnitude-readout-records-{backend:?}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let cache =
                KernelCache::open(root.clone(), crate::kernel_cache::DEFAULT_KERNEL_CACHE_BYTES)
                    .unwrap();
            let dependency: TuningKey =
                ("readout_head_rows", "head".into(), super::super::FixedValues::default());
            let point = PointShape {
                label: "m1".into(),
                weight: 1.,
                terms: Vec::new(),
                rows: 1,
                sequences: 1,
                context: None,
                class: None,
            };
            let open = || {
                ReadoutContender::new(
                    recipe,
                    point.clone(),
                    TuningCacheKey::of("readout recipe unit"),
                    Some(&cache),
                    Some(&cache),
                    RecipeState::default(),
                    dependency.clone(),
                    RecipePreparation::default(),
                    Levels::Certified,
                )
            };
            let found = |contender: &mut ReadoutContender, canonical: &NativeSpecialization| {
                contender
                    .cached(&device, &HashMap::from([(dependency.clone(), canonical.clone())]))
                    .unwrap()
            };
            for canonical in [&plain, &drafting] {
                let mut first = open();
                assert_eq!(found(&mut first, canonical), None, "{backend:?}");
                first.conclude(true, true).unwrap();
            }
            for canonical in [&plain, &drafting, &alike] {
                assert_eq!(found(&mut open(), canonical), Some(true), "{backend:?}");
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    /// An open whose allowance ends before its recipe qualifies (its head
    /// was searched to the end, as after a key change) records its census
    /// unsettled against the head's final choice. The load after it reads
    /// that evidence, so it certifies the class and prices the scope as an
    /// open that qualified does; a later open still finds no settled
    /// choice and qualifies, and a qualification that runs out of the
    /// allowance settles nothing either.
    #[test]
    fn an_open_that_never_qualified_records_its_census_for_the_load() {
        for device in devices() {
            let backend = device.backend();
            let (recipe, plain) = recipe_on(&device);
            let root = std::env::temp_dir().join(format!(
                "magnitude-readout-census-{backend:?}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let cache =
                KernelCache::open(root.clone(), crate::kernel_cache::DEFAULT_KERNEL_CACHE_BYTES)
                    .unwrap();
            let dependency: TuningKey =
                ("readout_head_rows", "head".into(), super::super::FixedValues::default());
            let chosen = HashMap::from([(dependency.clone(), plain.clone())]);
            let point = PointShape {
                label: "m1".into(),
                weight: 1.,
                terms: Vec::new(),
                rows: 1,
                sequences: 1,
                context: None,
                class: None,
            };
            let open = |state: RecipeState| {
                ReadoutContender::new(
                    recipe,
                    point.clone(),
                    TuningCacheKey::of("readout recipe unit"),
                    Some(&cache),
                    Some(&cache),
                    state,
                    dependency.clone(),
                    RecipePreparation::default(),
                    Levels::Certified,
                )
            };
            // Its census: the levels agreed, and cost less than the full
            // pass (the M4 Pro's one-row class, seconds).
            let costs = RecipeCosts { packed: 2.19e-3, bounded: Some(1.63e-3), planes: 2.24e-3 };
            let mut first = open(RecipeState::default());
            assert_eq!(first.cached(&device, &HashMap::new()).unwrap(), None);
            first.conclude(false, false).unwrap();
            first.receipt.costs = Some(costs);
            first.receipt.agreed = true;
            {
                let mut state = first.state.borrow_mut();
                state.costs = Some(costs);
                state.agreed = true;
            }
            first.finalize(&device, &chosen).unwrap();
            // The load: no settled choice, the census's evidence.
            let state = RecipeState::default();
            assert_eq!(open(state.clone()).cached(&device, &chosen).unwrap(), None, "{backend:?}");
            assert_eq!(state.borrow().reads(Levels::Certified), Levels::Certified, "{backend:?}");
            assert_eq!(state.borrow().costs, Some(costs), "{backend:?}");
            // A qualification that expires records what it measured, unsettled.
            let mut expired = open(RecipeState::default());
            assert_eq!(expired.cached(&device, &chosen).unwrap(), None);
            expired.expired = true;
            expired.conclude(false, true).unwrap();
            assert_eq!(open(RecipeState::default()).cached(&device, &chosen).unwrap(), None);
            // One that completes settles it.
            let mut qualified = open(RecipeState::default());
            assert_eq!(qualified.cached(&device, &chosen).unwrap(), None);
            qualified.conclude(true, true).unwrap();
            assert_eq!(open(RecipeState::default()).cached(&device, &chosen).unwrap(), Some(true));
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    /// Bounded levels serve a class where they agreed and cost less than the
    /// full pass, whatever they cost against packed: a scope stored as
    /// planes reads the class one of those two ways. A class without
    /// evidence reads the full pass, and a proposing class its view.
    #[test]
    fn bounded_levels_serve_where_they_agree_and_beat_the_full_pass() {
        let costs = |bounded| Some(RecipeCosts { packed: 2.0, bounded: Some(bounded), planes: 3.0 });
        let certified = Levels::Certified;
        assert_eq!(reads(certified, costs(1.5), true), Levels::Certified);
        assert_eq!(reads(certified, costs(2.5), true), Levels::Certified, "slower than packed");
        assert_eq!(reads(certified, costs(3.5), true), Levels::Full);
        assert_eq!(reads(certified, costs(1.5), false), Levels::Full);
        assert_eq!(reads(certified, None, true), Levels::Full);
        assert_eq!(reads(Levels::View, costs(1.5), false), Levels::View);
        assert_eq!(reads(Levels::Full, costs(1.5), true), Levels::Full);
    }

    /// A first load that searches the canonical head itself knows no record
    /// while the head's choice is open: it stores nothing then, however its
    /// recipe's mandatory work concludes. Once the choice is final it takes
    /// another load's record for a canonical that serves the row alike as
    /// it stands, without recording it again, and with none it compares and
    /// records its own under that canonical, which a later open finds.
    #[test]
    fn a_load_that_searches_its_head_looks_its_record_up_once_the_head_is_final() {
        for device in devices() {
            let backend = device.backend();
            let (recipe, plain) = recipe_on(&device);
            let drafting = match backend {
                BackendName::Metal => plain
                    .clone()
                    .with_launch_param(1, "ROWS", 2)
                    .with_launch_param(1, "SIMDGROUPS", 8),
                _ => plain.clone().with_launch_param(1, "KSPLIT", 2),
            };
            let root = std::env::temp_dir().join(format!(
                "magnitude-readout-deferred-{backend:?}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let cache =
                KernelCache::open(root.clone(), crate::kernel_cache::DEFAULT_KERNEL_CACHE_BYTES)
                    .unwrap();
            // Every stored record, by name, with its bytes.
            let records = || {
                std::fs::read_dir(root.join("readout"))
                    .unwrap()
                    .map(|entry| {
                        let path = entry.unwrap().path();
                        (path.clone(), std::fs::read(path).unwrap())
                    })
                    .collect::<std::collections::BTreeMap<_, _>>()
            };
            let dependency: TuningKey =
                ("readout_head_rows", "head".into(), super::super::FixedValues::default());
            let point = PointShape {
                label: "m1".into(),
                weight: 1.,
                terms: Vec::new(),
                rows: 1,
                sequences: 1,
                context: None,
                class: None,
            };
            let open = || {
                ReadoutContender::new(
                    recipe,
                    point.clone(),
                    TuningCacheKey::of("readout recipe unit"),
                    Some(&cache),
                    Some(&cache),
                    RecipeState::default(),
                    dependency.clone(),
                    RecipePreparation::default(),
                    Levels::Certified,
                )
            };
            // Another load recorded its winner against the plain canonical.
            let mut earlier = open();
            let chosen = |canonical: &NativeSpecialization| {
                HashMap::from([(dependency.clone(), canonical.clone())])
            };
            assert_eq!(earlier.cached(&device, &chosen(&plain)).unwrap(), None);
            earlier.conclude(true, true).unwrap();
            let before = records();
            for (canonical, stored) in [(&plain, Some(true)), (&drafting, None)] {
                let mut first = open();
                // The head is still searched: no record is known, and the
                // recipe's mandatory conclusion stores nothing.
                assert_eq!(first.cached(&device, &HashMap::new()).unwrap(), None, "{backend:?}");
                first.conclude(false, false).unwrap();
                assert_eq!(records(), before, "{backend:?}: stored under a guess");
                // The head is final.
                assert_eq!(first.adopt(&device, canonical).unwrap(), stored, "{backend:?}");
                first.conclude(true, true).unwrap();
            }
            // The adopted record was not written again; the new one was.
            let after = records();
            assert_eq!(after.len(), before.len() + 1, "{backend:?}");
            assert!(before.iter().all(|(path, bytes)| after.get(path) == Some(bytes)), "{backend:?}");
            for canonical in [&plain, &drafting] {
                assert_eq!(open().cached(&device, &chosen(canonical)).unwrap(), Some(true));
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    /// The recipe is offered for Q8_0 output projections, and its packed
    /// head's Metal declaration admits the shared matrix GEMV only where the
    /// weight's form allows it (`form(OW, MATRIX)`): which configurations it
    /// has depends on the bound weight. Every program of the recipe resolves
    /// its statics and declared defaults at its own element bindings on each
    /// backend that reads progressive heads, without a device, at the heads
    /// of Qwen3.5-2B (248,320 x 2,048) and LFM3-1.7B (128,000 x 2,048), each
    /// tied to its Q8_0 token embedding.
    #[test]
    fn every_recipe_program_is_admitted_at_its_bindings() {
        fn admitted<E: Entry>(
            recipe: ReadoutRecipe,
            backend: BackendName,
            entry: BoundEntry<E>,
            form: &str,
        ) {
            let statics = recipe
                .statics(backend, &entry)
                .unwrap_or_else(|error| panic!("{} {form}: {error:?}", E::NAME));
            seismic::generated::native_implementation_at::<E>(backend, entry.elements())
                .unwrap_or_else(|error| panic!("{} {form}: {error:?}", E::NAME))
                .unwrap_or_else(|| panic!("{} {form}: no implementation", E::NAME))
                .device_independent()
                .unwrap_or_else(|| panic!("{} {form}: the declaration reads the device", E::NAME))
                .default_specialization(&statics)
                .unwrap_or_else(|error| panic!("{} {form}: {error:?}", E::NAME));
        }
        for backend in [BackendName::Metal, BackendName::Cuda] {
            let weight =
                Element::stored("q8g32s", resident_layout(ExecutionPath::Native, backend)).unwrap();
            for vocabulary in [248_320, 128_000] {
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
                        vocabulary,
                        hidden: 2048,
                        parts: 1,
                        norm_extent: 2048,
                        depth: ProgressiveCodeDepth::Eight,
                        scale_extent: 0,
                    };
                    let form = format!("{backend:?} V={vocabulary} prepared={prepared}");
                    admitted(recipe, backend, recipe.top_entry(), &form);
                    admitted(recipe, backend, recipe.refine_entry(), &form);
                    admitted(recipe, backend, recipe.exact_entry(), &form);
                    admitted(recipe, backend, recipe.planes_entry(), &form);
                    admitted(recipe, backend, recipe.features_entry(), &form);
                    if prepared {
                        admitted(recipe, backend, recipe.project_entry(), &form);
                    } else {
                        admitted(recipe, backend, recipe.head_entry(), &form);
                    }
                }
            }
        }
    }
}
