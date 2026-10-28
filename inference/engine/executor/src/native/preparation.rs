use super::specialization::{Specializer, Tuning};
use super::tuning::readout_choice::{
    QualifiedRecipe, ReadoutContender, ReadoutRecipe, RecipePreparation, RecipeState,
};
use super::tuning::streams::{StreamMixDownTuning, StreamMixUpTuning};
use super::tuning::{
    ModelInputs, TunedEntry, Tuner, TuningLimits, TuningWeights,
    attention::{
        AttentionDecodeK8V4Tuning, AttentionDecodeTuning, AttentionMix, AttentionOutputTuning,
        AttentionPrefillK8V4Tuning, AttentionPrefillTuning, AttentionProjectTuning,
        IndexerProjectTuning, PrefillListing, history_window,
    },
    cases::{DenseExpandTuning, DenseOutputTuning, DenseUpTuning},
    general_routed::{
        RoutedDownTuning, RoutedExpandDecodeTuning, RoutedExpertTilesTuning, RoutedGateUpTuning,
        RoutedGatedTilesTuning, RoutedScatterTuning, RoutedSelectTuning, RoutedUpTilesTuning,
        RoutedUpTuning, routing_shape,
    },
    post_norm::ProjectRowsTuning,
    readout::{
        DraftRowsTuning, HeadRowsTuning, PreparedProjectionTuning, SampleRowsTuning,
        SelectedRowsTuning, ShapeRowsTuning,
    },
    recurrent::{
        RecurrentChunkTuning, RecurrentOutputTuning, RecurrentProjectConvolvedTuning,
        RecurrentProjectTuning, RecurrentShape, RecurrentState, RecurrentStepConvolvedTuning,
        RecurrentStepTuning,
    },
    routed::{
        RoutedChoicesTuning, RoutedCombineTuning, RoutedExpandTuning, RoutedExpertsTuning,
        RoutedGroupTuning, RoutedOutputTuning, RoutedRouteSharedTuning, RoutedRouteTuning,
        RoutedShape,
    },
    short_conv::{ShortConvOutputTuning, ShortConvProjectTuning},
    state_space::{
        StateSpaceChunkTuning, StateSpaceOutputTuning, StateSpaceProjectTuning, StateSpaceState,
        StateSpaceStepTuning,
    },
};
use super::*;
use crate::operators::gated_delta::graph::StepForm;
use crate::operators::routed::fused_graph::DecodeForm;
use crate::operators::streams::SiteEntryBindings;
use crate::programs::invocation_demand::{GraphInvocationDemand, InvocationCounts};
use crate::{
    DenseBinding, GeneralRoutedBinding, ModelLoadPlan, ReadoutHead, ShortConvBinding,
    StateSpaceBinding, SublayerTail,
};
use magnitude_family_contracts::{
    ModelDefinition, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_kernels::{
    attention_append_dense, cluster_token_rows, conditioning_overlay, corrector_attention,
    corrector_factors, corrector_features, corrector_nodes, corrector_path, corrector_rows,
    draft_convolve_input,
    draft_convolve_residual, draft_gated_rows, draft_path_step, draft_top_k, feature_copy,
    feature_rows, import_dense, moe_tail, ngram_gate, ngram_inject, ngram_rows, packing_bound,
    per_layer_inputs, place_logits_rows, post_norm_residual, repack_weight, repeat_rows,
    selection_notice, stored_rows, tap_rows, widen_rows,
};
use seismic::KernelRequest;
use std::collections::HashSet;
use std::{cell::RefCell, rc::Rc};
#[derive(Default)]
struct ReadoutStates {
    choices: HashMap<String, RecipeState>,
    programs: HashMap<String, RecipePreparation>,
}
type ReadoutRecipeStates = Rc<RefCell<ReadoutStates>>;

/// The affine Metal matrix form can share one decoded tile across four
/// rows for these head geometries. Give that row band its own objective,
/// independently of the model or the configuration that tuning selects.
pub(super) fn separates_packed_attention_rows(
    backend: seismic::BackendName,
    group: u64,
    width: u64,
    max_rows: u64,
) -> bool {
    backend == seismic::BackendName::Metal
        && (1..=8).contains(&group)
        && matches!(width, 128 | 256)
        && max_rows >= 4
}

/// The phase-one catalog. Every handle is prepared before qualification and
/// retained for warm calls; this type has no API capable of preparing again.
#[derive(Debug)]
pub(super) struct NativePreparationCache {
    pub(super) owner: Tensor,
    pub(super) import: ImportKernels,
    pub(super) glue: GlueKernels,
    pub(super) target: TargetKernels,
    pub(super) head: Option<HeadKernels>,
    pub(super) shared_head: Option<SharedHeadKernels>,
    pub(super) draft: Option<DraftKernels>,
    pub(super) vision: Option<VisionKernels>,
    pub(super) tuned: Vec<TunedEntry>,
    /// Units prepared at their canonical defaults for want of a stored
    /// tuning record.
    pub(super) unprepared: usize,
    pub(super) readout_choices: Vec<crate::ReadoutClassChoice>,
}

/// Everything program preparation reads.
pub(super) struct PreparationInputs<'a> {
    pub device: &'a Device,
    pub plan: &'a ProgramPlan,
    pub load: &'a ModelLoadPlan,
    pub limits: TuningLimits,
    pub tuning: TuningContext<'a>,
    /// What a search is held to; a load, which never searches, has none.
    pub room: Option<super::tuning::TuningRoom>,
    pub selection_limits: crate::ResourceLimits,
    pub graph_demand: GraphInvocationDemand,
}

/// Prepare one entry that has no tuning case. Evaluates to `Option`: `None`
/// during a count or a listing.
macro_rules! fixed {
    ($spec:expr, $module:ident, $bindings:expr, $elements:expr) => {
        fixed!($spec, $module, $bindings, $elements, statics & [])
    };
    ($spec:expr, $module:ident, $bindings:expr, $elements:expr, statics $statics:expr) => {
        $spec.fixed(&$bindings, $statics, $module::native_entry_with($elements))?
    };
    ($spec:expr, $module:ident, $bindings:expr) => {
        $spec.fixed(&$bindings, &[], $module::native_entry())?
    };
}

/// The layers each binding is prepared for, in model order. Tuning cases
/// rotate through these layers' resident weights.
struct BindingLayers<K> {
    layers: HashMap<K, Vec<WeightScope>>,
}

impl<K: Copy + Eq + std::hash::Hash> BindingLayers<K> {
    fn of(bindings: impl IntoIterator<Item = (K, WeightScope)>) -> Self {
        let mut layers: HashMap<K, Vec<WeightScope>> = HashMap::new();
        for (binding, scope) in bindings {
            layers.entry(binding).or_default().push(scope);
        }
        Self { layers }
    }

    fn scopes(&self, binding: K) -> Vec<WeightScope> {
        self.layers[&binding].clone()
    }
}

/// Each distinct binding of `bindings` in order of first appearance, with
/// the layers it is prepared for in model order.
fn ordered_layers<K: Eq>(
    bindings: impl IntoIterator<Item = (K, WeightScope)>,
) -> Vec<(K, Vec<WeightScope>)> {
    let mut ordered: Vec<(K, Vec<WeightScope>)> = Vec::new();
    for (binding, scope) in bindings {
        match ordered.iter_mut().find(|(known, _)| *known == binding) {
            Some((_, scopes)) => scopes.push(scope),
            None => ordered.push((binding, vec![scope])),
        }
    }
    ordered
}

/// The scope of sublayer `sublayer` (0 the mixer, 1 the feed-forward) of
/// target block `index`.
fn sublayer_scope(index: usize, sublayer: u32) -> WeightScope {
    WeightScope::TargetSublayer(SublayerIndex {
        block: u32::try_from(index).expect("block count fits u32"),
        sublayer,
    })
}

fn head_scope(index: usize) -> WeightScope {
    WeightScope::HeadBlock(u32::try_from(index).expect("head block count fits u32"))
}

fn head_sublayer_scope(index: usize, sublayer: u32) -> WeightScope {
    WeightScope::HeadSublayer(SublayerIndex {
        block: u32::try_from(index).expect("head block count fits u32"),
        sublayer,
    })
}

impl NativePreparationCache {
    /// A load's preparation: one walk forms every entry at the choice its
    /// stored tuning record names, or at its canonical default where no
    /// record is stored, when the load serves defaults; a load that reports
    /// such units ends at the first with [`CatalogFailure::Unprepared`].
    /// Nothing is searched, measured or published.
    pub(super) fn prepare_programs(
        inputs: PreparationInputs<'_>,
        unprepared: super::tuning::UnpreparedUnits,
    ) -> Result<Self, CatalogFailure> {
        let PreparationInputs {
            device,
            plan,
            load,
            limits,
            tuning,
            room: _,
            selection_limits,
            graph_demand,
        } = inputs;
        let recipe_states = ReadoutRecipeStates::default();
        let mut spec = Specializer::new(device);
        let import = imports(&mut spec, plan)?;
        let owner = AttestedPrograms::owner(device)?;
        let epsilon = epsilon(tuning.definition)?;
        let invocations = invocation_demand(
            device,
            plan,
            load,
            limits,
            tuning,
            selection_limits,
            &graph_demand,
            epsilon,
            &recipe_states,
        )?;
        let (glue, target, head, shared_head, draft, vision, tuned, unprepared, readout_choices) = {
            let mut preparation = Preparation::new(
                plan,
                tuning.definition,
                limits,
                epsilon,
                recipe_states.clone(),
                Some(selection_limits),
                spec,
                Tuning::Tuner(Tuner::serve(
                    device,
                    tuning,
                    limits,
                    TuningWeights::new(device, load, tuning.weights, &import),
                    invocations,
                    unprepared,
                )),
            );
            preparation.graph_demand = Some(graph_demand);
            let glue = preparation.walk(plan)?;
            let Preparation {
                tuning,
                target,
                head,
                shared_head,
                draft,
                vision,
                readout_choices,
                ..
            } = preparation;
            let tuner = tuning.into_tuner();
            let unprepared = tuner.unprepared();
            (
                glue,
                target,
                head,
                shared_head,
                draft,
                vision,
                tuner.conclude(),
                unprepared,
                readout_choices,
            )
        };
        Ok(Self {
            readout_choices,
            owner,
            import,
            glue,
            target,
            head,
            shared_head,
            draft,
            vision,
            tuned,
            unprepared,
        })
    }

    /// The search: every unit without a stored tuning record is searched on
    /// the device within the tuning budget and its record published. A count
    /// walk finds those units and their launches, and the entries without
    /// tuning parameters; the mandatory phase forms and publishes each
    /// unit's canonical default and forms those entries; a census walk
    /// collects the units with optional work, which are then searched
    /// together within the tuning time left. Each walk's preparation ends
    /// with its walk: the tuner, which borrows the import kernels, passes on
    /// to the next. No walk prepares the choices for serving: a load does,
    /// from the records. Returns every unit's outcome.
    pub(super) fn search_programs(
        inputs: PreparationInputs<'_>,
    ) -> Result<Vec<TunedEntry>, CatalogFailure> {
        let PreparationInputs {
            device,
            plan,
            load,
            limits,
            tuning,
            room,
            selection_limits,
            graph_demand,
        } = inputs;
        let recipe_states = ReadoutRecipeStates::default();
        // The search imports the weights its units measure through the
        // import kernels.
        let import = imports(&mut Specializer::new(device), plan)?;
        let epsilon = epsilon(tuning.definition)?;
        let invocations = invocation_demand(
            device,
            plan,
            load,
            limits,
            tuning,
            selection_limits,
            &graph_demand,
            epsilon,
            &recipe_states,
        )?;
        // The count also collects every entry without tuning parameters:
        // the mandatory phase forms them with the units' canonical
        // programs, so a load after the search forms nothing the search
        // did not.
        let (counted, untuned) = {
            let mut count = Preparation::new(
                plan,
                tuning.definition,
                limits,
                epsilon,
                recipe_states.clone(),
                Some(selection_limits),
                Specializer::count_untuned(device),
                Tuning::Tuner(
                    Tuner::count(
                        device,
                        tuning,
                        limits,
                        TuningWeights::new(device, load, tuning.weights, &import),
                        invocations,
                    )
                    .held_to(room),
                ),
            );
            count.graph_demand = Some(graph_demand.clone());
            count.walk(plan)?;
            let Preparation { spec, tuning, .. } = count;
            (tuning.into_tuner(), spec.into_untuned())
        };
        let censused = {
            let mut census = Preparation::new(
                plan,
                tuning.definition,
                limits,
                epsilon,
                recipe_states.clone(),
                Some(selection_limits),
                Specializer::count(device),
                Tuning::Tuner(counted.census(untuned)?),
            );
            census.graph_demand = Some(graph_demand.clone());
            census.walk(plan)?;
            census.tuning.into_tuner()
        };
        let searched = censused.search()?;
        // A development survey measures the units it owns in the search's
        // process, after the search: one more walk that forms nothing and
        // reads every other unit's published choice. Without an installed
        // survey there is nothing to measure and no walk.
        #[cfg(feature = "tuning-survey")]
        let searched = if !super::tuning::survey::installed() {
            searched
        } else {
            let mut survey = Preparation::new(
                plan,
                tuning.definition,
                limits,
                epsilon,
                recipe_states,
                Some(selection_limits),
                Specializer::count(device),
                Tuning::Tuner(searched),
            );
            survey.graph_demand = Some(graph_demand);
            survey.walk(plan)?;
            survey.tuning.into_tuner()
        };
        // The tuner borrows the import kernels: it ends here, before them.
        let tuned = searched.conclude();
        Ok(tuned)
    }
}

/// The logical calls of every tuned entry, collected by a metadata-only walk
/// before any tuner reads a record or opens a search: the same operator walk
/// under the same settings, so every key it records is the key the tuner
/// looks up. Physical weight provenance is absent from this map's role and
/// frequency decisions.
#[allow(clippy::too_many_arguments)]
fn invocation_demand(
    device: &Device,
    plan: &ProgramPlan,
    load: &ModelLoadPlan,
    limits: TuningLimits,
    tuning: TuningContext<'_>,
    selection_limits: crate::ResourceLimits,
    graph_demand: &GraphInvocationDemand,
    epsilon: f32,
    recipe_states: &ReadoutRecipeStates,
) -> Result<super::specialization::InvocationDemand, CatalogFailure> {
    let mut demand = Preparation::new(
        plan,
        tuning.definition,
        limits,
        epsilon,
        recipe_states.clone(),
        Some(selection_limits),
        Specializer::count(device),
        Tuning::demand(
            ModelInputs {
                definition: tuning.definition,
                limits,
                load,
            },
            device,
            tuning.settings,
            super::tuning::Objective::new(graph_demand.terms.clone()),
        ),
    );
    demand.graph_demand = Some(graph_demand.clone());
    demand.walk(plan)?;
    Ok(demand.tuning.into_invocations())
}

/// The requests of the native kernels preparing `plan` on `backend` prepares,
/// named without a device: the weight-import entries and the entries the
/// program walk prepares, each once, in walk order.
pub(crate) fn kernel_requests(
    backend: BackendName,
    plan: &ProgramPlan,
    definition: &ModelDefinition,
    load: &ModelLoadPlan,
    limits: TuningLimits,
) -> Result<Vec<KernelRequest>, CatalogFailure> {
    let mut spec = Specializer::list(backend);
    imports(&mut spec, plan)?;
    let mut listing = Preparation::new(
        plan,
        definition,
        limits,
        epsilon(definition)?,
        ReadoutRecipeStates::default(),
        None,
        spec,
        Tuning::Listing(ModelInputs {
            definition,
            limits,
            load,
        }),
    );
    listing.walk(plan)?;
    let mut seen = HashSet::new();
    Ok(listing
        .spec
        .into_requests()
        .into_iter()
        .filter(|request| seen.insert(request.clone()))
        .collect())
}

/// Every admitted definition normalizes with one epsilon, which the
/// readout's final norm states.
fn epsilon(definition: &ModelDefinition) -> Result<f32, CatalogFailure> {
    crate::programs::graph::readout::readout_epsilon(&definition.decoder).map_err(|outcome| {
        CatalogFailure::Preparation {
            entry: "normalization",
            bindings: "decoder epsilon".into(),
            outcome,
        }
    })
}

/// Prepare the weight-import entries of `plan`'s import slots.
fn imports(
    spec: &mut Specializer<'_>,
    plan: &ProgramPlan,
) -> Result<ImportKernels, CatalogFailure> {
    let mut import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    for slot in plan.imports() {
        match *slot {
            ImportProgramSlot::Dense { source, resident } => {
                if import.import_dense.contains_key(&(source, resident)) {
                    continue;
                }
                let bindings = dense_binding_name(source, resident);
                if let Some(kernel) = fixed!(
                    spec,
                    import_dense,
                    bindings,
                    import_dense::Elements {
                        E: Element::dense(source),
                        U: Element::dense(resident),
                    }
                ) {
                    import.import_dense.insert((source, resident), kernel);
                }
            }
            ImportProgramSlot::Repack { source, resident } => {
                // A resident element with a packing bound makes it with its
                // own entry after every import into it.
                if crate::planning::packing_run(resident).is_some()
                    && !import.packing_bound.contains_key(&resident)
                {
                    if let Some(kernel) = fixed!(
                        spec,
                        packing_bound,
                        resident.name().to_owned(),
                        packing_bound::Elements { W: resident }
                    ) {
                        import.packing_bound.insert(resident, kernel);
                    }
                }
                if import.repack_weight.contains_key(&(source, resident)) {
                    continue;
                }
                let bindings = element_binding_name(source, resident);
                if let Some(kernel) = fixed!(
                    spec,
                    repack_weight,
                    bindings,
                    repack_weight::Elements {
                        E: source,
                        U: resident,
                    }
                ) {
                    import.repack_weight.insert((source, resident), kernel);
                }
            }
        }
    }
    Ok(import)
}

/// Which reader of a projection scope a readout recipe serves: where its
/// kernels are kept.
#[derive(Clone, Copy)]
enum ReadoutReader {
    Target(crate::ReadoutBinding),
    HeadBlock(crate::HeadBinding),
    /// A head's own projection, which nothing else reads.
    SharedHead,
}

/// A reader of a candidate scope that selected no class of its own. A scope
/// is stored in one form, so when another reader of the scope selects one
/// this reader reads the views too, through their full pass, and is given
/// their kernels.
struct ReadoutFollower {
    reader: ReadoutReader,
    scope: WeightScope,
    recipe: ReadoutRecipe,
    preparation: RecipePreparation,
}

struct Preparation<'a> {
    limits: TuningLimits,
    graph_demand: Option<GraphInvocationDemand>,
    spec: Specializer<'a>,
    tuning: Tuning<'a>,
    recipe_states: ReadoutRecipeStates,
    selection_limits: Option<crate::ResourceLimits>,
    readout_choices: Vec<crate::ReadoutClassChoice>,
    /// Readers of a candidate scope that selected no class of their own
    /// ([`Preparation::follow_readout_scopes`]).
    readout_followers: Vec<ReadoutFollower>,
    epsilon: f32,
    softcap: f32,
    /// Model width: the static dimension of the feature readout.
    hidden: u64,
    /// The target's vocabulary: the static dimension of token selection.
    vocabulary: u64,
    target: TargetKernels,
    head: Option<HeadKernels>,
    shared_head: Option<SharedHeadKernels>,
    draft: Option<DraftKernels>,
    vision: Option<VisionKernels>,
}

impl<'a> Preparation<'a> {
    fn new(
        plan: &ProgramPlan,
        definition: &ModelDefinition,
        limits: TuningLimits,
        epsilon: f32,
        recipe_states: ReadoutRecipeStates,
        selection_limits: Option<crate::ResourceLimits>,
        spec: Specializer<'a>,
        tuning: Tuning<'a>,
    ) -> Self {
        Self {
            limits,
            graph_demand: None,
            spec,
            tuning,
            recipe_states,
            selection_limits,
            readout_choices: Vec::new(),
            readout_followers: Vec::new(),
            epsilon,
            softcap: crate::programs::graph::readout::readout_softcap(&definition.decoder),
            hidden: definition.decoder.hidden,
            vocabulary: definition.decoder.vocabulary,
            target: TargetKernels::default(),
            head: plan.head().map(|_| HeadKernels::default()),
            shared_head: plan.shared_head().map(|_| SharedHeadKernels::default()),
            draft: plan.draft().map(|_| DraftKernels::default()),
            vision: plan.vision().map(|_| VisionKernels::default()),
        }
    }

    /// Register one complete physical recipe per served selected-row class.
    /// A cache hit forms retained configurations but never measures or searches.
    /// A reader that selects no class still gets the views' kernels when its
    /// scope is stored as them for another reader.
    fn qualified_readout(
        &mut self,
        reader: ReadoutReader,
        scope: WeightScope,
        norm_scope: WeightScope,
        norm: Element,
        weight: Element,
        activation: Element,
        softcap: f32,
        prepared: bool,
        row_bound: usize,
        canonical: Option<ProjectionKernels>,
    ) -> Result<Option<ProjectionKernels>, CatalogFailure> {
        if self
            .selection_limits
            .is_none_or(|limits| limits.exported_logits_rows != 0)
        {
            return Ok(None);
        }
        let Some(device) = self.tuning.device() else {
            return Ok(None);
        };
        let dependency = if prepared {
            self.tuning.dependency(&PreparedProjectionTuning {
                scope,
                kind: WeightKind::Output,
                weight,
                activation,
                columns: None,
                indexed: false,
                softcap,
            })?
        } else {
            self.tuning.dependency(&HeadRowsTuning {
                projection_scope: scope,
                norm_scope,
                norm,
                weight,
                activation,
                epsilon: self.epsilon,
                rows: None,
            })?
        };
        let Some(dependency) = dependency else {
            return Ok(None);
        };
        let model = self.tuning.model();
        let scale_extent = model
            .scale_extent(scope, WeightKind::Output)
            .map_err(|error| CatalogFailure::Preparation {
                entry: "readout_recipe",
                bindings: format!("{scope:?}"),
                outcome: error,
            })?;
        let candidates = model.load.readout_candidates();
        let Some(rest) = candidates.iter().find(|plan| {
            plan.role
                == magnitude_family_contracts::WeightRole {
                    scope,
                    kind: WeightKind::OutputPlane(
                        magnitude_family_contracts::ProgressivePlane::Rest,
                    ),
                }
        }) else {
            return Ok(None);
        };
        let packed = super::tuning::weight_plan(model.load, scope, WeightKind::Output).map_err(
            |outcome| CatalogFailure::Preparation {
                entry: "readout_recipe",
                bindings: format!("{scope:?}"),
                outcome,
            },
        )?;
        let [stored_vocabulary, hidden] = packed.shape[..] else {
            return Ok(None);
        };
        let vocabulary = stored_vocabulary.min(self.vocabulary);
        let Some(depth) = rest.progressive_depth() else {
            return Ok(None);
        };
        let (parts, norm_extent) = model.readout_form(norm_scope, hidden).map_err(|outcome| {
            CatalogFailure::Preparation {
                entry: "readout_recipe",
                bindings: format!("{scope:?}"),
                outcome,
            }
        })?;
        // A readout of unnormalized partials (`NM = 0`) has no norm.
        let norm_plan = (norm_extent > 0)
            .then(|| super::tuning::weight_plan(model.load, norm_scope, WeightKind::OutputNorm))
            .transpose()
            .map_err(|outcome| CatalogFailure::Preparation {
                entry: "readout_recipe",
                bindings: format!("{scope:?}"),
                outcome,
            })?;
        let recipe = ReadoutRecipe {
            scope,
            norm_scope,
            norm,
            weight,
            activation,
            epsilon: self.epsilon,
            softcap,
            prepared,
            vocabulary,
            hidden,
            parts,
            norm_extent,
            depth,
            scale_extent,
        };
        let [top, levels @ ..] = recipe.digests(device)?;
        let preparation_key = format!(
            "readout-physical-programs/device={:?}/top={top}/levels={:?}",
            device.tuning_identity(),
            &levels[..3],
        );
        let preparation = self
            .recipe_states
            .borrow_mut()
            .programs
            .entry(preparation_key)
            .or_insert_with(|| Rc::new(RefCell::new(None)))
            .clone();
        let digest = format!("{top}/{levels:?}");
        let norm_identity = norm_plan.map_or_else(
            || "no-norm".to_owned(),
            |plan| format!("{:?}", plan.storage_identity()),
        );
        let mut selected_rows = Vec::new();
        let mut selected_state = None;
        for rows in magnitude_batching::row_classes(row_bound) {
            let point = super::tuning::PointShape {
                label: format!("m{rows}"),
                weight: 1.0,
                terms: Vec::new(),
                rows: rows as u64,
                sequences: 1,
                context: None,
                class: None,
            };
            // The choice is a tuning result: it is read only under the
            // budget it was qualified within.
            let budget = self.tuning.settings().map(|settings| settings.budget);
            let key = crate::TuningCacheKey::of(&format!(
                "readout-recipe-v4/budget={budget:?}/device={:?}/{:?}/{}/{recipe:?}/epsilon={:08x}/cap={:08x}/rows={}/dependency={dependency:?}/{digest}",
                device.tuning_identity(),
                packed.storage_identity(),
                norm_identity,
                self.epsilon.to_bits(),
                softcap.to_bits(),
                point.rows
            ));
            let state = self
                .recipe_states
                .borrow_mut()
                .choices
                .entry(key.as_str().to_owned())
                .or_insert_with(|| Rc::new(RefCell::new(QualifiedRecipe::default())))
                .clone();
            let contender = ReadoutContender::new(
                recipe,
                point.clone(),
                key.clone(),
                self.tuning.cache(),
                self.tuning.stored_results(),
                state.clone(),
                dependency.clone(),
                preparation.clone(),
            );
            if self
                .tuning
                .register_composed(Box::new(contender), point.rows)?
                == Some(true)
            {
                let Some(specification) = self
                    .tuning
                    .choice(&dependency)
                    .map(|choice| format!("{choice:?}"))
                else {
                    continue;
                };
                let evidence = state.borrow().canonical_specification.clone().or_else(|| {
                    self.tuning
                        .stored_results()
                        .and_then(|cache| cache.readout_canonical_choice(&key))
                });
                // Cached evidence is consumed as-is. A mismatch falls back;
                // it never grants a new search or post-budget qualification.
                if evidence.as_ref() != Some(&specification) {
                    continue;
                }
                let needs_formation = state.borrow().progressive.is_none();
                if needs_formation {
                    *state.borrow_mut() = recipe.form_shared(device, &preparation)?;
                }
                {
                    let mut state = state.borrow_mut();
                    state.selected = true;
                    state.canonical_specification = Some(specification);
                }
                selected_rows.push(point.rows);
                selected_state = Some(state);
            }
        }
        let Some(state) = selected_state else {
            // The scope's readers are prepared in a fixed order: one that
            // selected before this reader is known here, one that selects
            // after it is met in `follow_readout_scopes`.
            if !self
                .readout_choices
                .iter()
                .any(|choice| choice.scope == scope)
            {
                self.readout_followers.push(ReadoutFollower {
                    reader,
                    scope,
                    recipe,
                    preparation,
                });
                return Ok(None);
            }
            let Some(packed) = canonical else {
                return Ok(None);
            };
            return Ok(Some(ProjectionKernels::Qualified {
                packed: Box::new(packed),
                progressive: recipe
                    .form_shared(device, &preparation)?
                    .progressive
                    .expect("a formed recipe retains bounded kernels"),
            }));
        };
        selected_rows.sort_unstable();
        selected_rows.dedup();
        if let Some(choice) = self.readout_choices.iter_mut().find(|choice| {
            choice.scope == scope && choice.norm_scope == norm_scope && choice.prepared == prepared
        }) {
            choice.rows.extend(&selected_rows);
            choice.rows.sort_unstable();
            choice.rows.dedup();
        } else {
            self.readout_choices.push(crate::ReadoutClassChoice {
                scope,
                norm_scope,
                prepared,
                rows: selected_rows,
            });
        }
        // A count or a listing forms no kernels, the canonical projection
        // among them: the class choice above is all such a pass records. A
        // search over a cache that already stores this recipe's choice
        // reaches here in its count.
        let Some(packed) = canonical else {
            return Ok(None);
        };
        let state = state.borrow();
        Ok(Some(ProjectionKernels::Qualified {
            packed: Box::new(packed),
            progressive: state
                .progressive
                .clone()
                .expect("completed recipe retains bounded kernels"),
        }))
    }

    /// Give the views' kernels to each reader that was prepared before
    /// another reader of its scope selected a class: the target's projection
    /// is read by the target readout and by an embedded head's blocks, and
    /// is stored in one form for all of them.
    fn follow_readout_scopes(&mut self) -> Result<(), CatalogFailure> {
        let Some(device) = self.tuning.device() else {
            return Ok(());
        };
        for follower in std::mem::take(&mut self.readout_followers) {
            if !self
                .readout_choices
                .iter()
                .any(|choice| choice.scope == follower.scope)
            {
                continue;
            }
            let progressive = follower
                .recipe
                .form_shared(device, &follower.preparation)?
                .progressive
                .expect("a formed recipe retains bounded kernels");
            match follower.reader {
                ReadoutReader::Target(binding) => {
                    if let Some(readout) = self.target.readout.get_mut(&binding) {
                        readout.head = ReadoutHeadKernels::Qualified {
                            packed: Box::new(readout.head.clone()),
                            progressive,
                        };
                    }
                }
                ReadoutReader::HeadBlock(binding) => {
                    if let Some(logits) = self
                        .head
                        .as_mut()
                        .and_then(|head| head.logits.get_mut(&binding))
                    {
                        *logits = ProjectionKernels::Qualified {
                            packed: Box::new(logits.clone()),
                            progressive,
                        };
                    }
                }
                ReadoutReader::SharedHead => {
                    unreachable!("a head's own projection has one reader")
                }
            }
        }
        Ok(())
    }

    /// Prepare every program entry, in a fixed order.
    fn walk(&mut self, plan: &ProgramPlan) -> Result<GlueKernels, CatalogFailure> {
        if let Some(demand) = &self.graph_demand {
            let mut selection = demand.target.clone();
            if plan.head().is_some() || plan.shared_head().is_some() {
                selection.add(&demand.head);
            }
            self.tuning.set_invocations(selection, true);
        }
        let glue = self.glue(plan)?;
        if let Some(demand) = &self.graph_demand {
            self.tuning.set_invocations(demand.target.clone(), true);
        }
        self.target(plan)?;
        self.head(plan)?;
        self.follow_readout_scopes()?;
        self.shared_head(plan)?;
        self.draft(plan)?;
        self.vision(plan)?;
        Ok(glue)
    }

    /// Token selection, the conditioning overlay and the state row copies.
    fn glue(&mut self, plan: &ProgramPlan) -> Result<GlueKernels, CatalogFailure> {
        let copies = plan.state().copies();
        let copy = |spec: &mut Specializer<'_>, element: Element, bindings: &'static str| {
            if !copies.contains(&element) {
                return Ok(None);
            }
            Ok::<_, CatalogFailure>(fixed!(
                spec,
                copy_rows,
                bindings,
                copy_rows::Elements { A: element }
            ))
        };
        Ok(GlueKernels {
            shape_rows: self.spec.tuned(
                &mut self.tuning,
                &ShapeRowsTuning {
                    vocabulary: self.vocabulary,
                },
            )?,
            sample_rows: self.spec.tuned(
                &mut self.tuning,
                &SampleRowsTuning {
                    vocabulary: self.vocabulary,
                },
            )?,
            conditioning_overlay: fixed!(self.spec, conditioning_overlay, "fixed"),
            selection_notice: match plan.notices_selection() {
                true => fixed!(self.spec, selection_notice, "fixed"),
                false => None,
            },
            feature_copy: fixed!(
                self.spec,
                feature_copy,
                "activation",
                feature_copy::Elements {
                    A: plan
                        .head()
                        .and_then(|head| head.blocks().first())
                        .map_or(plan.target().embedding().activation, |head| head
                            .conditioning)
                }
            ),
            copy_rows_f32: copy(&mut self.spec, Element::f32(), "A=f32")?,
            copy_rows_f16: copy(&mut self.spec, Element::f16(), "A=f16")?,
            copy_rows_bf16: copy(&mut self.spec, Element::bf16(), "A=bf16")?,
            copy_rows_u32: copy(&mut self.spec, Element::u32(), "A=u32")?,
            copy_rows_i32: copy(&mut self.spec, Element::i32(), "A=i32")?,
        })
    }

    fn target(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        if plan.head().is_some_and(|head| {
            head.blocks()
                .iter()
                .any(|head| head.conditioning == Element::f32())
        }) {
            self.target.residual_features = fixed!(
                self.spec,
                feature_rows,
                "residual",
                feature_rows::Elements { A: Element::f32() }
            );
        }
        let spec = &mut self.spec;
        let target = plan.target();
        let b = target.embedding();
        if let Some(kernel) = spec.fixed(
            &format!("{b:?}"),
            &[("D", self.hidden)],
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: b.table,
                A: b.activation,
            }),
        )? {
            self.target.embedding.insert(b, kernel);
        }
        // The decoder blocks of a draft head reading target history are
        // prepared with the target's, by binding: their layers are the
        // head's.
        let head_blocks = plan
            .shared_head()
            .map_or(&[][..], |head| head.blocks.as_slice());
        let mut mixer_demand = HashMap::new();
        let mut feed_demand = HashMap::new();
        if let Some(demand) = &self.graph_demand {
            for (blocks, calls) in [
                (target.blocks(), &demand.target),
                (head_blocks, &demand.head),
            ] {
                for block in blocks {
                    mixer_demand
                        .entry(block.mixer())
                        .or_insert_with(InvocationCounts::default)
                        .add(calls);
                    if let Some(feed) = block.feed_forward() {
                        feed_demand
                            .entry(feed)
                            .or_insert_with(InvocationCounts::default)
                            .add(calls);
                    }
                }
            }
        }
        let attention_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .map(|(index, block)| (block, sublayer_scope(index, 0)))
                .chain(
                    head_blocks
                        .iter()
                        .enumerate()
                        .map(|(index, block)| (block, head_sublayer_scope(index, 0))),
                )
                .filter_map(|(block, scope)| match block.mixer() {
                    MixerProgramSlot::Attention(binding) => Some((binding, scope)),
                    _ => None,
                }),
        );
        let recurrent_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::Recurrent(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let state_space_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::StateSpace(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let short_conv_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::ShortConv(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let dense_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .map(|(index, block)| (block, sublayer_scope(index, 1)))
                .chain(
                    head_blocks
                        .iter()
                        .enumerate()
                        .map(|(index, block)| (block, head_sublayer_scope(index, 1))),
                )
                .filter_map(|(block, scope)| match block.feed_forward()? {
                    FeedForwardProgramSlot::Dense(binding) => Some((binding, scope)),
                    FeedForwardProgramSlot::Routed(_)
                    | FeedForwardProgramSlot::GeneralRouted(_)
                    | FeedForwardProgramSlot::Parallel(_) => None,
                }),
        );
        let routed_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward()? {
                FeedForwardProgramSlot::Routed(binding) => {
                    Some((binding, sublayer_scope(index, 1)))
                }
                FeedForwardProgramSlot::Dense(_)
                | FeedForwardProgramSlot::GeneralRouted(_)
                | FeedForwardProgramSlot::Parallel(_) => None,
            },
        ));
        let general_routed_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .filter_map(|(index, block)| match block.feed_forward()? {
                    FeedForwardProgramSlot::GeneralRouted(binding) => {
                        Some((binding, sublayer_scope(index, 1)))
                    }
                    FeedForwardProgramSlot::Dense(_)
                    | FeedForwardProgramSlot::Routed(_)
                    | FeedForwardProgramSlot::Parallel(_) => None,
                }),
        );
        let parallel_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward()? {
                FeedForwardProgramSlot::Parallel(binding) => {
                    Some((binding, sublayer_scope(index, 1)))
                }
                FeedForwardProgramSlot::Dense(_)
                | FeedForwardProgramSlot::Routed(_)
                | FeedForwardProgramSlot::GeneralRouted(_) => None,
            },
        ));
        // Each binding is walked once: its case binds every layer it serves,
        // and a walk that tunes nothing (the count, the census) must not
        // meet it again at each of them, or its launches per step would be
        // counted once per layer.
        let mut mixers = HashSet::new();
        let mut feeds = HashSet::new();
        for block in target.blocks().iter().chain(head_blocks) {
            if let Some(calls) = mixer_demand.get(&block.mixer()) {
                self.tuning.set_invocations(calls.clone(), false);
            }
            match Some(block.mixer()).filter(|slot| mixers.insert(*slot)) {
                Some(MixerProgramSlot::Attention(b)) if !self.target.attention.contains_key(&b) => {
                    if let Some(kernels) = self.attention(b, attention_layers.scopes(b))? {
                        self.target.attention.insert(b, kernels);
                    }
                }
                Some(MixerProgramSlot::Recurrent(b)) if !self.target.recurrent.contains_key(&b) => {
                    if let Some(kernels) = self.recurrent(b, recurrent_layers.scopes(b))? {
                        self.target.recurrent.insert(b, kernels);
                    }
                }
                Some(MixerProgramSlot::StateSpace(b))
                    if !self.target.state_space.contains_key(&b) =>
                {
                    if let Some(kernels) = self.state_space(b, state_space_layers.scopes(b))? {
                        self.target.state_space.insert(b, kernels);
                    }
                }
                Some(MixerProgramSlot::ShortConv(b))
                    if !self.target.short_conv.contains_key(&b) =>
                {
                    if let Some(kernels) = self.short_conv(b, short_conv_layers.scopes(b))? {
                        self.target.short_conv.insert(b, kernels);
                    }
                }
                Some(
                    MixerProgramSlot::Attention(_)
                    | MixerProgramSlot::Recurrent(_)
                    | MixerProgramSlot::StateSpace(_)
                    | MixerProgramSlot::ShortConv(_),
                )
                | None => {}
            }
            if let Some(calls) = block.feed_forward().and_then(|feed| feed_demand.get(&feed)) {
                self.tuning.set_invocations(calls.clone(), false);
            }
            match block.feed_forward().filter(|slot| feeds.insert(*slot)) {
                Some(FeedForwardProgramSlot::Dense(b)) if !self.target.dense.contains_key(&b) => {
                    if let Some(kernels) = self.dense(b, dense_layers.scopes(b))? {
                        self.target.dense.insert(b, kernels);
                    }
                }
                Some(FeedForwardProgramSlot::Routed(b)) if !self.target.routed.contains_key(&b) => {
                    if let Some(kernels) = self.routed(b, routed_layers.scopes(b))? {
                        self.target.routed.insert(b, kernels);
                    }
                }
                Some(FeedForwardProgramSlot::GeneralRouted(b))
                    if !self.target.general_routed.contains_key(&b) =>
                {
                    if let Some(kernels) =
                        self.general_routed(b, general_routed_layers.scopes(b))?
                    {
                        self.target.general_routed.insert(b, kernels);
                    }
                }
                Some(FeedForwardProgramSlot::Parallel(b))
                    if !self.target.parallel.contains_key(&b) =>
                {
                    if let Some(kernels) = self.parallel(b, parallel_layers.scopes(b))? {
                        self.target.parallel.insert(b, kernels);
                    }
                }
                Some(
                    FeedForwardProgramSlot::Dense(_)
                    | FeedForwardProgramSlot::Routed(_)
                    | FeedForwardProgramSlot::GeneralRouted(_)
                    | FeedForwardProgramSlot::Parallel(_),
                )
                | None => {}
            }
        }
        if let Some(demand) = &self.graph_demand {
            self.tuning.set_invocations(demand.target.clone(), true);
        }
        let per_layer_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .filter_map(|(index, block)| Some((block.per_layer()?, sublayer_scope(index, 2)))),
        );
        let mut per_layers = HashSet::new();
        for b in target.blocks().iter().filter_map(|block| block.per_layer()) {
            if per_layers.insert(b) && !self.target.per_layer.contains_key(&b) {
                if let Some(kernels) = self.per_layer(b, per_layer_layers.scopes(b))? {
                    self.target.per_layer.insert(b, kernels);
                }
            }
        }
        if let Some(b) = target.per_layer() {
            if let Some(kernels) = self.per_layer_entry(b)? {
                self.target.per_layer_entry.insert(b, kernels);
            }
        }
        let ngram_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .filter_map(|(index, block)| Some((block.ngram()?, sublayer_scope(index, 2)))),
        );
        let mut ngrams = HashSet::new();
        for b in target.blocks().iter().filter_map(|block| block.ngram()) {
            if ngrams.insert(b) && !self.target.ngram.contains_key(&b) {
                if let Some(kernels) = self.ngram(b, ngram_layers.scopes(b))? {
                    self.target.ngram.insert(b, kernels);
                }
            }
        }
        let sites = target
            .blocks()
            .iter()
            .enumerate()
            .filter_map(|(index, block)| Some((index, block.streams()?)))
            .flat_map(|(index, binding)| {
                binding.entries().scoped([
                    sublayer_scope(index, 0),
                    sublayer_scope(index, 1),
                    WeightScope::Target,
                ])
            })
            .collect::<Vec<_>>();
        let mut streams = std::mem::take(&mut self.target.streams);
        self.streams(&mut streams, &sites)?;
        self.target.streams = streams;
        let b = target.readout();
        let bindings = format!("{b:?}");
        let features = self.target_features(&bindings, b.norm, b.activation)?;
        let weight = b.head.packed();
        let head = match b.head {
            ReadoutHead::Packed { weight, .. } | ReadoutHead::Qualified { weight, .. } => {
                let head = self.spec.tuned(
                    &mut self.tuning,
                    &HeadRowsTuning {
                        projection_scope: WeightScope::Target,
                        norm_scope: WeightScope::Target,
                        norm: b.norm,
                        weight,
                        activation: b.activation,
                        epsilon: self.epsilon,
                        rows: None,
                    },
                )?;
                let selected = self.spec.tuned(
                    &mut self.tuning,
                    &SelectedRowsTuning {
                        norm: b.norm,
                        weight,
                        activation: b.activation,
                        epsilon: self.epsilon,
                    },
                )?;
                head.zip(selected)
                    .map(|(head, selected)| ReadoutHeadKernels::Packed { head, selected })
            }
        };
        let qualified = self.qualified_readout(
            ReadoutReader::Target(b),
            WeightScope::Target,
            WeightScope::Target,
            b.norm,
            weight.element,
            b.activation,
            self.softcap,
            false,
            self.selection_limits
                .map_or(0, |limits| limits.max_selected_rows),
            head.as_ref().map(|head| match head {
                ReadoutHeadKernels::Packed { head, .. } => {
                    ProjectionKernels::RawPacked(head.clone())
                }
                _ => unreachable!("ordinary target readout is canonical packed"),
            }),
        )?;
        let head = match (head, qualified) {
            (
                Some(ReadoutHeadKernels::Packed { head, selected }),
                Some(ProjectionKernels::Qualified { progressive, .. }),
            ) => Some(ReadoutHeadKernels::Qualified {
                packed: Box::new(ReadoutHeadKernels::Packed { head, selected }),
                progressive,
            }),
            (head, _) => head,
        };
        if let (Some(features), Some(head)) = (features, head) {
            self.target
                .readout
                .insert(b, ReadoutKernels { features, head });
        }
        if let Some(b) = target.features() {
            if let Some(kernel) = self.target_features(&format!("{b:?}"), b.norm, b.activation)? {
                self.target.features.insert(b, kernel);
            }
        }
        if let Some(taps) = target.taps() {
            self.taps(taps)?;
        }
        Ok(())
    }

    /// A separate draft's target taps: the tap, the fusion projection of the
    /// taps (`project_rows` into F32) and the conditioning feature rows.
    fn taps(&mut self, taps: &crate::TapProgramPlan) -> Result<(), CatalogFailure> {
        let activation = taps.activation;
        let bindings = format!("A={}", activation.name());
        let tap = fixed!(
            self.spec,
            tap_rows,
            bindings,
            tap_rows::Elements { A: activation }
        );
        let fusion = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight: taps.fusion,
                activation,
                kind: WeightKind::DraftFusion,
                output: Element::f32(),
                scopes: vec![WeightScope::Draft],
            },
        )?;
        let features = fixed!(
            self.spec,
            feature_rows,
            bindings,
            feature_rows::Elements { A: activation }
        );
        if let (Some(tap), Some(fusion), Some(features)) = (tap, fusion, features) {
            self.target.taps = Some(TapKernels {
                tap,
                fusion,
                features,
            });
        }
        Ok(())
    }

    /// `readout_features_rows` over the target decoder's exit: the exit
    /// site's partial rows, or the final row under the output norm.
    fn target_features(
        &mut self,
        bindings: &str,
        norm: Element,
        activation: Element,
    ) -> Result<Option<NativeKernel<readout_features_rows::Entry>>, CatalogFailure> {
        let form = self
            .tuning
            .model()
            .readout_form(WeightScope::Target, self.hidden)
            .map_err(|outcome| CatalogFailure::Preparation {
                entry: "readout_features_rows",
                bindings: bindings.into(),
                outcome,
            })?;
        self.features_at(bindings, norm, activation, self.hidden, form)
    }

    /// `readout_features_rows` over rows `width` wide in the input form
    /// `(parts, norm extent)`.
    fn features_at(
        &mut self,
        bindings: &str,
        norm: Element,
        activation: Element,
        width: u64,
        (parts, norm_extent): (u64, u64),
    ) -> Result<Option<NativeKernel<readout_features_rows::Entry>>, CatalogFailure> {
        self.spec.fixed(
            bindings,
            &[("D", width), ("PT", parts), ("NM", norm_extent)],
            readout_features_rows::native_entry_with(readout_features_rows::Elements {
                NW: norm,
                A: activation,
            }),
        )
    }

    fn attention(
        &mut self,
        binding: AttentionBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<AttentionKernels>, CatalogFailure> {
        let AttentionBinding {
            shape,
            activation,
            output,
            ..
        } = binding;
        let project = self.spec.tuned(
            &mut self.tuning,
            &AttentionProjectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let window = history_window(&self.tuning.model(), &scopes).map_err(|outcome| {
            CatalogFailure::Preparation {
                entry: "attention_decode",
                bindings: format!("{shape:?}"),
                outcome,
            }
        })?;
        let mix = || AttentionMix {
            codec: binding.history,
            activation,
            shape,
            scopes: scopes.clone(),
            epsilon: self.epsilon,
            decode_rows: None,
            listing: PrefillListing::Every,
            window,
            selection: binding.selection,
        };
        // The history's prefill reads every visible row (a selected read's
        // classes without scoring too); its WorkList form is the selection's.
        let visible = || AttentionMix {
            selection: None,
            ..mix()
        };
        let history = match binding.history.encoding() {
            magnitude_state::HistoryEncoding::Dense => {
                // As for K8/V4, multi-row launches (draft blocks,
                // verification) get their own tuning identity, so their
                // grouped form is not displaced by the single-row one. CUDA
                // also changes history-tile row capacity with its warp count.
                let split_decode = matches!(
                    self.spec.backend(),
                    seismic::BackendName::Metal
                        | seismic::BackendName::Vulkan
                        | seismic::BackendName::Cuda
                ) && self.limits.max_rows >= 2;
                let mut single = mix();
                if split_decode {
                    single.decode_rows = Some(1..2);
                }
                let decode = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionDecodeTuning(single))?;
                let verify = if split_decode {
                    let mut selected = mix();
                    selected.decode_rows =
                        Some(2..crate::operators::attention::graph::DECODE_ROWS + 1);
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeTuning(selected))?
                } else {
                    None
                };
                let prefill = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionPrefillTuning(visible()))?;
                let append = fixed!(
                    self.spec,
                    attention_append_dense,
                    format!("{shape:?}"),
                    attention_append_dense::Elements {
                        A: binding.activation
                    },
                    statics & shape.append_statics()
                );
                decode
                    .zip(prefill)
                    .zip(append)
                    .map(
                        |((decode, prefill), append)| AttentionHistoryKernels::Dense {
                            append,
                            decode,
                            verify,
                            prefill,
                        },
                    )
            }
            magnitude_state::HistoryEncoding::Affine => {
                let split_decode = matches!(
                    self.spec.backend(),
                    seismic::BackendName::Metal
                        | seismic::BackendName::Vulkan
                        | seismic::BackendName::Cuda
                ) && self.limits.max_rows >= 2;
                let mut single = mix();
                // The base kernel serves only single-row decode when the
                // graph binds separate verification kernels. Pricing their
                // rows here as well would duplicate the same served work.
                if split_decode {
                    single.decode_rows = Some(1..2);
                }
                let decode = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(single))?;
                let packed_four_rows = split_decode
                    && separates_packed_attention_rows(
                        self.spec.backend(),
                        shape.group,
                        shape.width,
                        self.limits.max_rows,
                    );
                let verify = if split_decode {
                    let mut selected = mix();
                    selected.decode_rows = Some(
                        2..if packed_four_rows {
                            4
                        } else {
                            crate::operators::attention::graph::DECODE_ROWS + 1
                        },
                    );
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(selected))?
                } else {
                    None
                };
                // The packed form reuses each K/V tile across four Verify
                // rows. Give that workload its own tuning identity so
                // shorter and wider rows cannot displace its form.
                let verify_four = if packed_four_rows {
                    let mut selected = mix();
                    selected.decode_rows = Some(4..5);
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(selected))?
                } else {
                    None
                };
                let verify_eight = if packed_four_rows && self.limits.max_rows >= 5 {
                    let mut selected = mix();
                    selected.decode_rows =
                        Some(5..crate::operators::attention::graph::DECODE_ROWS + 1);
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(selected))?
                } else {
                    None
                };
                // Where the entry's forms differ by it, a launch that lists
                // the history row tiles its rows see and one that does not
                // are two kernels with their own admissible forms.
                // (`StateResourcePlan::lists_history_tiles`, of the opened
                // device's own fact; graph preparation checks they agree.)
                let lists = crate::operators::attention::graph::lists_history_tiles(binding)
                    && crate::planning::reads_decoded_history(
                        self.spec.backend(),
                        self.spec.forms_tensor_operations(),
                        shape.width,
                    );
                // A launch whose rows see at most one request's history
                // lists the tiles they see; the unlisted kernel serves the
                // others, or every launch where nothing lists.
                let capacity = super::tuning::request_history_rows(
                    &self.tuning.model(),
                    magnitude_state::KvCodec::AffineK8V4,
                    window,
                )
                .map_err(|outcome| CatalogFailure::Preparation {
                    entry: "attention_prefill_k8v4",
                    bindings: format!("{shape:?}"),
                    outcome,
                })?;
                let mut unlisted = visible();
                if lists {
                    unlisted.listing = PrefillListing::Unlisted { capacity };
                }
                let prefill = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionPrefillK8V4Tuning(unlisted))?;
                let prefill_listed = if lists {
                    let mut listed = visible();
                    listed.listing = PrefillListing::Listed { capacity };
                    self.spec
                        .tuned(&mut self.tuning, &AttentionPrefillK8V4Tuning(listed))?
                } else {
                    None
                };
                let mut append_statics = shape.append_statics();
                append_statics.extend(
                    crate::operators::attention::graph::affine_dimensions(binding.history)
                        .map_err(|error| CatalogFailure::Preparation {
                            entry: "attention_append_k8v4",
                            bindings: format!("{:?}", binding.history),
                            outcome: error.to_string(),
                        })?,
                );
                let append = fixed!(
                    self.spec,
                    attention_append_k8v4,
                    format!("{shape:?};codec={:?}", binding.history),
                    attention_append_k8v4::Elements {
                        A: binding.activation
                    },
                    statics & append_statics
                );
                decode
                    .zip(prefill)
                    .zip(append)
                    .map(
                        |((decode, prefill), append)| AttentionHistoryKernels::AffineK8V4 {
                            append,
                            decode,
                            verify,
                            verify_four,
                            verify_eight,
                            prefill,
                            prefill_listed,
                        },
                    )
            }
            magnitude_state::HistoryEncoding::Unsupported => {
                return Err(CatalogFailure::Preparation {
                    entry: "attention_decode",
                    bindings: format!("{shape:?}"),
                    outcome: "the native path has no rotated K4/V4 history entries".into(),
                });
            }
        };
        // A selected read's indexer: its projection, tuned over the layers'
        // indexer weights, and its selection at the attention's rotary.
        let selection = match binding.selection {
            None => Some(None),
            Some(selection) => {
                let project = self.spec.tuned(
                    &mut self.tuning,
                    &IndexerProjectTuning {
                        binding,
                        selection,
                        scopes: scopes.clone(),
                        epsilon: self.epsilon,
                    },
                )?;
                let select = fixed!(
                    self.spec,
                    indexer_select,
                    format!("{selection:?};P={}", shape.rotary_pairs),
                    indexer_select::Elements {
                        A: binding.activation
                    },
                    statics
                        & [
                            ("H", selection.shape.heads),
                            ("P", shape.rotary_pairs),
                            ("S", selection.shape.width - 2 * shape.rotary_pairs),
                            ("SB", selection.shape.block),
                        ]
                );
                let prefill = match binding.history.encoding() {
                    magnitude_state::HistoryEncoding::Affine => self
                        .spec
                        .tuned(&mut self.tuning, &AttentionPrefillK8V4Tuning(mix()))?
                        .map(SelectedPrefillKernel::AffineK8V4),
                    _ => self
                        .spec
                        .tuned(&mut self.tuning, &AttentionPrefillTuning(mix()))?
                        .map(SelectedPrefillKernel::Dense),
                };
                match (project, select, prefill) {
                    (Some(project), Some(select), Some(prefill)) => Some(Some(SelectionKernels {
                        project,
                        select,
                        prefill,
                    })),
                    _ => None,
                }
            }
        };
        let output = match binding.tail {
            SublayerTail::Residual => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &AttentionOutputTuning {
                        output,
                        activation,
                        shape,
                        scopes,
                    },
                )?
                .map(SublayerOutput::Residual),
            SublayerTail::PostNorm { norm, .. } => self
                .post_norm(
                    WeightKind::AttentionOutput,
                    output,
                    activation,
                    norm,
                    scopes,
                )?
                .map(SublayerOutput::PostNorm),
        };
        Ok(match (project, history, output, selection) {
            (Some(project), Some(history), Some(output), Some(selection)) => {
                Some(AttentionKernels {
                    codec: binding.history,
                    project,
                    history,
                    output,
                    selection,
                })
            }
            _ => None,
        })
    }

    /// The state-space entries (`operators::state_space`): the projection and
    /// output reuse the attention projection entries at this geometry.
    fn state_space(
        &mut self,
        binding: StateSpaceBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<StateSpaceKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &StateSpaceProjectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let state = || StateSpaceState {
            activation: binding.activation,
            shape: binding.shape,
            scopes: scopes.clone(),
        };
        let step = self
            .spec
            .tuned(&mut self.tuning, &StateSpaceStepTuning(state()))?;
        let chunk = self
            .spec
            .tuned(&mut self.tuning, &StateSpaceChunkTuning(state()))?;
        let shape = binding.shape;
        let gate = fixed!(
            self.spec,
            state_space_gate,
            format!("{binding:?}"),
            state_space_gate::Elements {
                A: binding.activation
            },
            statics
                & [
                    ("G", shape.norm_groups()),
                    ("U", shape.norm_heads),
                    ("P", shape.head_width)
                ]
        );
        let output = self.spec.tuned(
            &mut self.tuning,
            &StateSpaceOutputTuning { binding, scopes },
        )?;
        Ok(match (project, step, chunk, gate, output) {
            (Some(project), Some(step), Some(chunk), Some(gate), Some(output)) => {
                Some(StateSpaceKernels {
                    project,
                    step,
                    chunk,
                    gate,
                    output,
                })
            }
            _ => None,
        })
    }

    /// The short-convolution entries (`operators::short_conv`): the output
    /// reuses the attention output entry at one head of `channels`.
    fn short_conv(
        &mut self,
        binding: ShortConvBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<ShortConvKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &ShortConvProjectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let rows = fixed!(
            self.spec,
            short_conv_rows,
            format!("{binding:?}"),
            short_conv_rows::Elements {
                A: binding.activation
            }
        );
        let output = match binding.tail {
            SublayerTail::Residual => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &ShortConvOutputTuning {
                        binding,
                        scopes,
                    },
                )?
                .map(SublayerOutput::Residual),
            SublayerTail::PostNorm { norm, .. } => self
                .post_norm(
                    WeightKind::RecurrentOutput,
                    binding.output,
                    binding.activation,
                    norm,
                    scopes,
                )?
                .map(SublayerOutput::PostNorm),
        };
        Ok(match (project, rows, output) {
            (Some(project), Some(rows), Some(output)) => Some(ShortConvKernels {
                project,
                rows,
                output,
            }),
            _ => None,
        })
    }

    fn recurrent(
        &mut self,
        b: RecurrentBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<RecurrentKernels>, CatalogFailure> {
        let shape = RecurrentShape {
            key_heads: b.key_heads,
            value_heads: b.value_heads,
            width: b.width,
            convolution_width: b.convolution_width,
            scopes,
        };
        let state = || RecurrentState {
            recurrent_norm: b.recurrent_norm,
            activation: b.activation,
            shape: shape.clone(),
            epsilon: self.epsilon,
        };
        let form =
            StepForm::of(self.spec.backend()).map_err(|outcome| CatalogFailure::Preparation {
                entry: "gated_delta_step_convolved",
                bindings: format!("{b:?}"),
                outcome,
            })?;
        let project_tuning = || RecurrentProjectTuning {
            norm: b.norm,
            qkv: b.qkv,
            gate: b.gate,
            alpha: b.alpha,
            beta: b.beta,
            activation: b.activation,
            shape: shape.clone(),
            epsilon: self.epsilon,
        };
        let step = match form {
            StepForm::Step => self
                .spec
                .tuned(&mut self.tuning, &RecurrentStepTuning(state()))?
                .map(RecurrentStepKernels::Step),
            StepForm::Convolved => {
                let project = self.spec.tuned(
                    &mut self.tuning,
                    &RecurrentProjectConvolvedTuning(project_tuning()),
                )?;
                let step = self
                    .spec
                    .tuned(&mut self.tuning, &RecurrentStepConvolvedTuning(state()))?;
                project
                    .zip(step)
                    .map(|(project, step)| RecurrentStepKernels::Convolved { project, step })
            }
        };
        let chunk = self
            .spec
            .tuned(&mut self.tuning, &RecurrentChunkTuning(state()))?;
        let project = self.spec.tuned(&mut self.tuning, &project_tuning())?;
        let output = self.spec.tuned(
            &mut self.tuning,
            &RecurrentOutputTuning {
                output: b.output,
                activation: b.activation,
                shape,
            },
        )?;
        Ok(match (project, step, chunk, output) {
            (Some(project), Some(step), Some(chunk), Some(output)) => Some(RecurrentKernels {
                project,
                step,
                chunk,
                output,
            }),
            _ => None,
        })
    }

    fn dense(
        &mut self,
        binding: DenseBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<DenseKernels>, CatalogFailure> {
        let DenseBinding {
            features: _,
            norm,
            gate,
            up,
            down,
            activation,
            function,
            tail,
            // The tunings read their scale ports' extents from the plan.
            scales: _,
        } = binding;
        let expand = self.spec.tuned(
            &mut self.tuning,
            &DenseExpandTuning {
                norm,
                gate,
                up,
                activation,
                gate_kind: WeightKind::DenseGate,
                up_kind: WeightKind::DenseUp,
                function,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let output = match tail {
            SublayerTail::Residual => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down,
                        activation,
                        down_kind: WeightKind::DenseDown,
                        scopes,
                    },
                )?
                .map(SublayerOutput::Residual),
            SublayerTail::PostNorm { norm, .. } => self
                .post_norm(WeightKind::DenseDown, down, activation, norm, scopes)?
                .map(SublayerOutput::PostNorm),
        };
        Ok(match (expand, output) {
            (Some(expand), Some(output)) => Some(DenseKernels { expand, output }),
            _ => None,
        })
    }

    /// A per-layer input sublayer: its gate, then the post-norm tail of its
    /// projection back to the hidden width.
    fn per_layer(
        &mut self,
        binding: crate::PerLayerBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<PerLayerKernels>, CatalogFailure> {
        let SublayerTail::PostNorm { norm, .. } = binding.tail else {
            return Err(CatalogFailure::Preparation {
                entry: "per_layer_gate",
                bindings: format!("{binding:?}"),
                outcome: "a per-layer input sublayer ends in a post-norm tail".into(),
            });
        };
        let gate = self.spec.tuned(
            &mut self.tuning,
            &super::tuning::per_layer::PerLayerGateTuning {
                binding,
                scopes: scopes.clone(),
            },
        )?;
        let output = self.post_norm(
            WeightKind::PerLayerProjection,
            binding.projection,
            binding.activation,
            norm,
            scopes,
        )?;
        Ok(gate
            .zip(output)
            .map(|(gate, output)| PerLayerKernels { gate, output }))
    }

    /// An n-gram input sublayer; `scopes` are its sublayers' scopes.
    fn ngram(
        &mut self,
        binding: crate::NGramBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<NGramKernels>, CatalogFailure> {
        let projection = |kind| ProjectRowsTuning {
            weight: match kind {
                WeightKind::NGramKey => binding.key,
                _ => binding.value,
            },
            activation: binding.activation,
            kind,
            output: Element::f32(),
            scopes: scopes.clone(),
        };
        let key = self
            .spec
            .tuned(&mut self.tuning, &projection(WeightKind::NGramKey))?;
        let value = self
            .spec
            .tuned(&mut self.tuning, &projection(WeightKind::NGramValue))?;
        let rows = self.spec.fixed(
            "fixed",
            &[("C", binding.context), ("P", binding.heads_per_order)],
            ngram_rows::native_entry(),
        )?;
        let stored = self.spec.fixed(
            &format!(
                "E={},A={}",
                binding.table.name(),
                binding.activation.name()
            ),
            &[("W", binding.width)],
            stored_rows::native_entry_with(stored_rows::Elements {
                E: binding.table,
                A: binding.activation,
            }),
        )?;
        let norm = format!("NW={}", binding.norm.name());
        let gate = self.spec.fixed(
            &norm,
            &[("H", binding.hidden), ("S", binding.streams)],
            ngram_gate::native_entry_with(ngram_gate::Elements { NW: binding.norm }),
        )?;
        let inject = self.spec.fixed(
            &norm,
            &[
                ("H", binding.hidden),
                ("S", binding.streams),
                ("K", binding.kernel),
            ],
            ngram_inject::native_entry_with(ngram_inject::Elements { NW: binding.norm }),
        )?;
        Ok(match (rows, stored, key, value, gate, inject) {
            (Some(rows), Some(stored), Some(key), Some(value), Some(gate), Some(inject)) => {
                Some(NGramKernels {
                    rows,
                    stored,
                    key,
                    value,
                    gate,
                    inject,
                })
            }
            _ => None,
        })
    }

    /// The hyper-connection streams entries of mix sites `sites` (each
    /// site's entry bindings at its scope) that `prepared` lacks: each
    /// distinct `stream_mix_down` and `stream_mix_up` binding prepared once,
    /// tuned over every site that binds it in site order, and the widening
    /// of the embedded rows.
    fn streams(
        &mut self,
        prepared: &mut StreamsEntryKernels,
        sites: &[(SiteEntryBindings, WeightScope)],
    ) -> Result<(), CatalogFailure> {
        if sites.is_empty() {
            return Ok(());
        }
        let downs = ordered_layers(sites.iter().map(|(site, scope)| (site.down, *scope)));
        for (binding, scopes) in downs {
            if prepared.down.contains_key(&binding) {
                continue;
            }
            let kernel = self.spec.tuned(
                &mut self.tuning,
                &StreamMixDownTuning {
                    binding,
                    scopes,
                    epsilon: self.epsilon,
                },
            )?;
            if let Some(kernel) = kernel {
                prepared.down.insert(binding, kernel);
            }
        }
        let ups = ordered_layers(sites.iter().map(|(site, scope)| (site.up, *scope)));
        for (binding, scopes) in ups {
            if prepared.up.contains_key(&binding) {
                continue;
            }
            let kernel = self
                .spec
                .tuned(&mut self.tuning, &StreamMixUpTuning { binding, scopes })?;
            if let Some(kernel) = kernel {
                prepared.up.insert(binding, kernel);
            }
        }
        if prepared.repeat.is_none() {
            let spec = &mut self.spec;
            prepared.repeat = fixed!(spec, repeat_rows, "fixed");
        }
        Ok(())
    }

    /// The per-layer entry: the embedding's projection into F32 per-layer
    /// channels and their combination with the host-table rows.
    fn per_layer_entry(
        &mut self,
        binding: crate::PerLayerEntryBinding,
    ) -> Result<Option<PerLayerEntryKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight: binding.projection,
                activation: binding.activation,
                kind: WeightKind::PerLayerModelProjection,
                output: Element::f32(),
                scopes: vec![WeightScope::Target],
            },
        )?;
        let spec = &mut self.spec;
        let activation = binding
            .activation
            .dtype()
            .ok_or_else(|| CatalogFailure::Preparation {
                entry: "import_dense",
                bindings: format!("{binding:?}"),
                outcome: "the activation element is not dense".into(),
            })?;
        let round = fixed!(
            spec,
            import_dense,
            dense_binding_name(seismic::DType::F32, activation),
            import_dense::Elements {
                E: Element::f32(),
                U: binding.activation,
            }
        );
        let table = match (binding.table_source.dtype(), binding.table.dtype()) {
            (Some(source), Some(resident)) => fixed!(
                spec,
                import_dense,
                dense_binding_name(source, resident),
                import_dense::Elements {
                    E: binding.table_source,
                    U: binding.table,
                }
            )
            .map(TableConversion::Dense),
            _ => fixed!(
                spec,
                repack_weight,
                element_binding_name(binding.table_source, binding.table),
                repack_weight::Elements {
                    E: binding.table_source,
                    U: binding.table,
                }
            )
            .map(TableConversion::Repack),
        };
        let copy = fixed!(spec, conditioning_overlay, "fixed");
        let inputs = self.spec.fixed(
            &format!("TW={},NW={}", binding.table.name(), binding.norm.name()),
            &[("L", binding.layers), ("P", binding.width)],
            per_layer_inputs::native_entry_with(per_layer_inputs::Elements {
                TW: binding.table,
                NW: binding.norm,
            }),
        )?;
        Ok(match (round, project, table, inputs, copy) {
            (Some(round), Some(project), Some(table), Some(inputs), Some(copy)) => {
                Some(PerLayerEntryKernels {
                    round,
                    project,
                    table,
                    inputs,
                    copy,
                })
            }
            _ => None,
        })
    }

    /// A dense branch beside a routed branch; `scopes` are the parallel
    /// sublayers' scopes, whose branches hold the weights.
    fn parallel(
        &mut self,
        binding: crate::ParallelBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<ParallelKernels>, CatalogFailure> {
        let branch_scopes = |branch: usize| {
            scopes
                .iter()
                .map(|scope| match *scope {
                    WeightScope::TargetSublayer(sublayer) => {
                        Ok(crate::operators::parallel::DenseBesideRouted::scopes(sublayer)[branch])
                    }
                    other => Err(CatalogFailure::Preparation {
                        entry: "moe_tail",
                        bindings: format!("{binding:?}"),
                        outcome: format!("parallel branches in {other:?}"),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let (dense_scopes, routed_scopes) = (branch_scopes(0)?, branch_scopes(1)?);
        let dense = binding.dense;
        let expand = self.spec.tuned(
            &mut self.tuning,
            &DenseExpandTuning {
                norm: dense.norm,
                gate: dense.gate,
                up: dense.up,
                activation: dense.activation,
                gate_kind: WeightKind::DenseGate,
                up_kind: WeightKind::DenseUp,
                function: dense.function,
                scopes: dense_scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let down = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight: dense.down,
                activation: dense.activation,
                kind: WeightKind::DenseDown,
                output: Element::f32(),
                scopes: dense_scopes,
            },
        )?;
        let routed = self.general_routed(binding.routed, routed_scopes)?;
        let tail = fixed!(
            self.spec,
            moe_tail,
            format!("NW={}", binding.norm.name()),
            moe_tail::Elements { NW: binding.norm }
        );
        Ok(match (expand, down, routed, tail) {
            (Some(expand), Some(down), Some(routed), Some(tail)) => Some(ParallelKernels {
                expand,
                down,
                routed,
                tail,
            }),
            _ => None,
        })
    }

    /// A post-norm tail: the output projection `kind` into F32 rows, then the
    /// row op that normalizes them into the residual.
    fn post_norm(
        &mut self,
        kind: WeightKind,
        weight: Element,
        activation: Element,
        norm: Element,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<PostNormKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight,
                activation,
                kind,
                output: Element::f32(),
                scopes,
            },
        )?;
        let residual = fixed!(
            self.spec,
            post_norm_residual,
            format!("NW={}", norm.name()),
            post_norm_residual::Elements { NW: norm }
        );
        Ok(project
            .zip(residual)
            .map(|(project, residual)| PostNormKernels { project, residual }))
    }

    /// The general routed entries (`operators::routed`), with the shared
    /// expert and latent projections as dense entries over their own weight
    /// kinds.
    fn general_routed(
        &mut self,
        binding: GeneralRoutedBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<GeneralRoutedKernels>, CatalogFailure> {
        let shape = binding.shape;
        let activation = binding.activation;
        let epsilon = self.epsilon;
        let select = self.spec.tuned(
            &mut self.tuning,
            &RoutedSelectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon,
            },
        )?;
        let decode = || RoutedExpandDecodeTuning {
            binding,
            scopes: scopes.clone(),
        };
        let tiles = || RoutedExpertTilesTuning {
            binding,
            scopes: scopes.clone(),
        };
        let experts = if shape.experts_expansion.gated {
            let decode = self
                .spec
                .tuned(&mut self.tuning, &RoutedGateUpTuning(decode()))?;
            let grouped = self
                .spec
                .tuned(&mut self.tuning, &RoutedGatedTilesTuning(tiles()))?;
            decode
                .zip(grouped)
                .map(|(decode, grouped)| ExpertKernels::Gated { decode, grouped })
        } else {
            let decode = self
                .spec
                .tuned(&mut self.tuning, &RoutedUpTuning(decode()))?;
            let grouped = self
                .spec
                .tuned(&mut self.tuning, &RoutedUpTilesTuning(tiles()))?;
            decode
                .zip(grouped)
                .map(|(decode, grouped)| ExpertKernels::Plain { decode, grouped })
        };
        let down = self.spec.tuned(
            &mut self.tuning,
            &RoutedDownTuning {
                binding,
                scopes: scopes.clone(),
            },
        )?;
        let group = self.spec.tuned(
            &mut self.tuning,
            &RoutedGroupTuning {
                shape: routing_shape(&binding),
                layers: scopes.len(),
            },
        )?;
        let scatter = self.spec.tuned(
            &mut self.tuning,
            &RoutedScatterTuning {
                binding,
                layers: scopes.len(),
            },
        )?;
        let shared = match (shape.shared, binding.shared) {
            (Some((_, expansion)), Some((gate, up, down))) => {
                let expansion = match gate {
                    Some(gate) => self
                        .spec
                        .tuned(
                            &mut self.tuning,
                            &DenseExpandTuning {
                                norm: binding.norm,
                                gate,
                                up,
                                activation,
                                gate_kind: WeightKind::SharedGate,
                                up_kind: WeightKind::SharedUp,
                                function: expansion.activation,
                                scopes: scopes.clone(),
                                epsilon,
                            },
                        )?
                        .map(DenseExpansionKernel::Gated),
                    None => self
                        .spec
                        .tuned(
                            &mut self.tuning,
                            &DenseUpTuning {
                                norm: binding.norm,
                                up,
                                activation,
                                up_kind: WeightKind::SharedUp,
                                function: expansion.activation,
                                scopes: scopes.clone(),
                                epsilon,
                            },
                        )?
                        .map(DenseExpansionKernel::Plain),
                };
                let output = self.spec.tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down,
                        activation,
                        down_kind: WeightKind::SharedDown,
                        scopes: scopes.clone(),
                    },
                )?;
                Some(expansion.zip(output))
            }
            (None, None) => None,
            _ => {
                return Err(CatalogFailure::Preparation {
                    entry: "dense_output",
                    bindings: format!("{binding:?}"),
                    outcome: "the shared expert's shape and binding disagree".into(),
                });
            }
        };
        let latent = match binding.latent {
            Some((down, up)) => {
                let project = self.spec.tuned(
                    &mut self.tuning,
                    &ProjectRowsTuning {
                        weight: down,
                        activation,
                        kind: WeightKind::LatentDown,
                        output: activation,
                        scopes: scopes.clone(),
                    },
                )?;
                let output = self.spec.tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down: up,
                        activation,
                        down_kind: WeightKind::LatentUp,
                        scopes,
                    },
                )?;
                Some(project.zip(output))
            }
            None => None,
        };
        // `None` anywhere is a tuning count or a missing implementation.
        Ok(match (select, experts, down, group, scatter) {
            (Some(select), Some(experts), Some(down), Some(group), Some(scatter)) => {
                let shared = match shared {
                    Some(Some(shared)) => Some(shared),
                    Some(None) => return Ok(None),
                    None => None,
                };
                let latent = match latent {
                    Some(Some(latent)) => Some(latent),
                    Some(None) => return Ok(None),
                    None => None,
                };
                Some(GeneralRoutedKernels {
                    select,
                    experts,
                    down,
                    group,
                    scatter,
                    shared,
                    latent,
                })
            }
            _ => None,
        })
    }

    fn routed(
        &mut self,
        b: RoutedBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<RoutedKernels>, CatalogFailure> {
        let shape = RoutedShape {
            hidden: b.hidden,
            experts: b.experts,
            selected: b.selected,
            features: b.features,
            shared: b.shared,
        };
        let form =
            DecodeForm::of(self.spec.backend()).map_err(|outcome| CatalogFailure::Preparation {
                entry: "routed_route_shared",
                bindings: format!("{b:?}"),
                outcome,
            })?;
        let route_tuning = RoutedRouteTuning {
            norm: b.norm,
            router: b.router,
            activation: b.activation,
            shape,
            scopes: scopes.clone(),
            epsilon: self.epsilon,
            decode: form == DecodeForm::Expand,
        };
        let route = self.spec.tuned(&mut self.tuning, &route_tuning)?;
        let group = self.spec.tuned(
            &mut self.tuning,
            &RoutedGroupTuning {
                shape,
                layers: scopes.len(),
            },
        )?;
        let decode = match form {
            DecodeForm::Expand => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &RoutedExpandTuning {
                        expert_gate: b.expert_gate,
                        expert_up: b.expert_up,
                        shared_gate: b.shared_gate,
                        shared_up: b.shared_up,
                        activation: b.activation,
                        shape,
                        scopes: scopes.clone(),
                    },
                )?
                .map(RoutedDecodeKernels::Expand),
            DecodeForm::SharedRoute => {
                let route = self.spec.tuned(
                    &mut self.tuning,
                    &RoutedRouteSharedTuning {
                        route: route_tuning,
                        shared_gate: b.shared_gate,
                        shared_up: b.shared_up,
                    },
                )?;
                let choices = self.spec.tuned(
                    &mut self.tuning,
                    &RoutedChoicesTuning {
                        expert_gate: b.expert_gate,
                        expert_up: b.expert_up,
                        activation: b.activation,
                        shape,
                        scopes: scopes.clone(),
                    },
                )?;
                route
                    .zip(choices)
                    .map(|(route, choices)| RoutedDecodeKernels::SharedRoute { route, choices })
            }
        };
        let output = self.spec.tuned(
            &mut self.tuning,
            &RoutedOutputTuning {
                expert_down: b.expert_down,
                shared_down: b.shared_down,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
            },
        )?;
        let experts = self.spec.tuned(
            &mut self.tuning,
            &RoutedExpertsTuning {
                expert_gate: b.expert_gate,
                expert_up: b.expert_up,
                expert_down: b.expert_down,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
            },
        )?;
        let combine = self.spec.tuned(
            &mut self.tuning,
            &RoutedCombineTuning {
                shared_gate: b.shared_gate,
                shared_up: b.shared_up,
                shared_down: b.shared_down,
                activation: b.activation,
                shape,
                scopes,
            },
        )?;
        Ok(match (route, decode, output, group, experts, combine) {
            (
                Some(route),
                Some(decode),
                Some(output),
                Some(group),
                Some(experts),
                Some(combine),
            ) => Some(RoutedKernels {
                route,
                decode,
                output,
                group,
                experts,
                combine,
            }),
            _ => None,
        })
    }

    fn head(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        if let Some(demand) = &self.graph_demand {
            let mut calls = demand.head.clone();
            calls.add(&demand.head_prefix);
            self.tuning.set_invocations(calls, true);
        }
        let Some(head_plan) = plan.head() else {
            return Ok(());
        };
        let scoped = |scope: fn(usize) -> WeightScope| {
            BindingLayers::of(
                head_plan
                    .blocks()
                    .iter()
                    .enumerate()
                    .map(|(index, binding)| (*binding, scope(index))),
            )
        };
        let layers = scoped(head_scope);
        let attention_layers = scoped(|index| head_sublayer_scope(index, 0));
        let feed_forward_layers = scoped(|index| head_sublayer_scope(index, 1));
        let mut registered = HashSet::new();
        for (head_index, &b) in head_plan.blocks().iter().enumerate() {
            if !registered.insert(b)
                || self
                    .head
                    .as_ref()
                    .is_some_and(|head| head.input.contains_key(&b))
            {
                continue;
            }
            if let Some(demand) = &self.graph_demand {
                let mut calls = demand.head.clone();
                calls.add(&demand.head_prefix);
                self.tuning.set_invocations(calls, true);
            }
            let bindings = format!("{b:?}");
            let stream_kernels = match b.streams {
                Some(streams) => {
                    let mut prepared = StreamsEntryKernels::default();
                    let sites = streams
                        .entries()
                        .scoped([
                            head_sublayer_scope(head_index, 0),
                            head_sublayer_scope(head_index, 1),
                            head_scope(head_index),
                        ])
                        .collect::<Vec<_>>();
                    self.streams(&mut prepared, &sites)?;
                    prepared.kernels(streams)
                }
                None => None,
            };
            let copy = fixed!(
                self.spec,
                feature_copy,
                &bindings,
                feature_copy::Elements { A: b.conditioning }
            );
            let gather = fixed!(
                self.spec,
                feature_rows,
                &bindings,
                feature_rows::Elements { A: b.conditioning }
            );
            let input = self.spec.tuned(
                &mut self.tuning,
                &DraftRowsTuning {
                    conditioning: b.conditioning,
                    streams: b.streams.map_or(1, |s| s.geometry.streams),
                    embedding: b.embedding_table,
                    embedding_norm: b.embedding_norm,
                    hidden_norm: b.hidden_norm,
                    combine: b.combine,
                    activation: b.activation,
                    scopes: layers.scopes(b),
                    epsilon: self.epsilon,
                },
            )?;
            // Priming injects K/V only: it does not run attention or FFN.
            if let Some(demand) = &self.graph_demand {
                self.tuning.set_invocations(demand.head.clone(), true);
            }
            // The draft head's history is always dense (its binding says so).
            let attention = self.attention(b.attention, attention_layers.scopes(b))?;
            if let Some(demand) = &self.graph_demand {
                self.tuning
                    .set_invocations(demand.head_prefix.clone(), true);
            }
            let priming_project = self.spec.tuned(
                &mut self.tuning,
                &AttentionProjectTuning {
                    binding: AttentionBinding {
                        key_value_only: true,
                        ..b.attention
                    },
                    scopes: attention_layers.scopes(b),
                    epsilon: self.epsilon,
                },
            )?;
            if let Some(demand) = &self.graph_demand {
                self.tuning.set_invocations(demand.head.clone(), true);
            }
            let feed_forward = match b.feed_forward {
                FeedForwardProgramSlot::Dense(binding) => self
                    .dense(binding, feed_forward_layers.scopes(b))?
                    .map(AttestedFeedForward::Dense),
                FeedForwardProgramSlot::Routed(binding) => self
                    .routed(binding, feed_forward_layers.scopes(b))?
                    .map(AttestedFeedForward::Routed),
                // `operators::admit` keeps draft heads on the fused form.
                FeedForwardProgramSlot::GeneralRouted(_) | FeedForwardProgramSlot::Parallel(_) => {
                    return Err(CatalogFailure::Preparation {
                        entry: "routed_select",
                        bindings: format!("{b:?}"),
                        outcome: "draft heads run the fused routed form only".into(),
                    });
                }
            };
            if let Some(demand) = &self.graph_demand {
                let mut calls = InvocationCounts::default();
                for _ in &layers.scopes(b) {
                    calls.add(&demand.head);
                }
                self.tuning.set_invocations(calls, false);
            }
            let form = b
                .streams
                .map_or((1, self.hidden), |streams| (streams.geometry.streams, 0));
            let features =
                self.features_at(&bindings, b.output_norm, b.activation, self.hidden, form)?;
            let weight = b.projection.packed();
            let logits = match b.projection {
                ReadoutHead::Packed { weight, .. } | ReadoutHead::Qualified { weight, .. } => self
                    .spec
                    .tuned(
                        &mut self.tuning,
                        &PreparedProjectionTuning {
                            scope: WeightScope::Target,
                            kind: WeightKind::Output,
                            weight,
                            activation: b.activation,
                            columns: None,
                            indexed: false,
                            softcap: 0.0,
                        },
                    )?
                    .map(ProjectionKernels::PreparedPacked),
            };
            let qualified = self.qualified_readout(
                ReadoutReader::HeadBlock(b),
                WeightScope::Target,
                head_scope(head_index),
                b.output_norm,
                weight.element,
                b.activation,
                0.0,
                true,
                self.selection_limits.map_or(0, |limits| {
                    limits.max_launch_slots.min(limits.max_launch_rows)
                }),
                logits.clone(),
            )?;
            let logits = qualified.or(logits);
            let head = self
                .head
                .as_mut()
                .expect("a head plan creates the head group");
            if let (
                Some(input),
                Some(attention),
                Some(priming_project),
                Some(feed_forward),
                Some(features),
                Some(logits),
            ) = (
                input,
                attention,
                priming_project,
                feed_forward,
                features,
                logits,
            ) {
                if let Some(streams) = stream_kernels {
                    head.streams.insert(b, streams);
                }
                if let Some(copy) = copy {
                    head.copy.insert(b, copy);
                }
                if let Some(gather) = gather {
                    head.gather.insert(b, gather);
                }
                head.input.insert(b, input);
                head.attention.insert(b, attention);
                head.priming_project.insert(b, priming_project);
                match feed_forward {
                    AttestedFeedForward::Dense(dense) => {
                        head.dense.insert(b, dense);
                    }
                    AttestedFeedForward::Routed(routed) => {
                        head.routed.insert(b, routed);
                    }
                    AttestedFeedForward::GeneralRouted(_) | AttestedFeedForward::Parallel(_) => {
                        return Err(CatalogFailure::Preparation {
                            entry: "routed_select",
                            bindings: format!("{b:?}"),
                            outcome: "draft heads run the fused routed form only".into(),
                        });
                    }
                }
                head.features.insert(b, features);
                head.logits.insert(b, logits);
            }
        }
        // Token selection over the draft vocabulary.
        let vocabulary = self.vocabulary;
        let shape = self
            .spec
            .tuned(&mut self.tuning, &ShapeRowsTuning { vocabulary })?;
        let sample = self
            .spec
            .tuned(&mut self.tuning, &SampleRowsTuning { vocabulary })?;
        let head = self
            .head
            .as_mut()
            .expect("a head plan creates the head group");
        head.shape = shape;
        head.sample = sample;
        Ok(())
    }

    /// A draft head reading target history, apart from its decoder blocks
    /// (prepared with the target's): the join of its input halves, the
    /// projections into and out of its width, its normalized output and its
    /// own vocabulary table.
    fn shared_head(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        if let Some(demand) = &self.graph_demand {
            self.tuning.set_invocations(demand.head.clone(), true);
        }
        let Some(head) = plan.shared_head() else {
            return Ok(());
        };
        let activation = head.activation;
        let scopes = vec![head_scope(0)];
        let bindings = format!("A={}", activation.name());
        let join = fixed!(
            self.spec,
            tap_rows,
            bindings,
            tap_rows::Elements { A: activation }
        );
        // Both projections publish F32 rows: the residual the blocks read,
        // and the hidden row the next step joins.
        let project = |this: &mut Self, weight: Element, kind: WeightKind| {
            this.spec.tuned(
                &mut this.tuning,
                &ProjectRowsTuning {
                    weight,
                    activation,
                    kind,
                    output: Element::f32(),
                    scopes: scopes.clone(),
                },
            )
        };
        let input = project(self, head.input.element, WeightKind::HeadInputProjection)?;
        let chain = project(self, head.chain.element, WeightKind::HeadChainProjection)?;
        let features = self.features_at(
            &format!("NW={},A={}", head.output_norm.name(), activation.name()),
            head.output_norm,
            activation,
            head.width,
            (1, head.width),
        )?;
        let compact_columns = head.clustered.as_ref().map(|cluster| {
            cluster.selected_clusters * (cluster.token_order.len() as u64 / cluster.clusters)
        });
        let projection = match head.projection {
            ReadoutHead::Packed { weight, .. } | ReadoutHead::Qualified { weight, .. } => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &PreparedProjectionTuning {
                        scope: head_scope(0),
                        kind: WeightKind::Output,
                        weight,
                        activation,
                        columns: compact_columns,
                        indexed: head.clustered.is_some(),
                        softcap: 0.0,
                    },
                )?
                .map(ProjectionKernels::PreparedPacked),
        };
        let qualified = if head.clustered.is_none() {
            let weight = head.projection.packed();
            self.qualified_readout(
                ReadoutReader::SharedHead,
                head_scope(0),
                head_scope(0),
                head.output_norm,
                weight.element,
                activation,
                0.0,
                true,
                self.selection_limits.map_or(0, |limits| {
                    limits
                        .max_launch_slots
                        .min(limits.max_launch_rows)
                        .min(limits.max_drafting_slots)
                }),
                projection.clone(),
            )?
        } else {
            None
        };
        let projection = qualified.or(projection);
        let centroids = head
            .clustered
            .as_ref()
            .map(|cluster| {
                self.spec.tuned(
                    &mut self.tuning,
                    &PreparedProjectionTuning {
                        scope: head_scope(0),
                        kind: WeightKind::OutputCentroids,
                        weight: cluster.centroids.element,
                        activation,
                        columns: None,
                        indexed: false,
                        softcap: 0.0,
                    },
                )
            })
            .transpose()?
            .flatten();
        let (cluster_selection, cluster_rows, placement) = if head.clustered.is_some() {
            (
                fixed!(self.spec, draft_top_k, ""),
                fixed!(self.spec, cluster_token_rows, ""),
                fixed!(self.spec, place_logits_rows, ""),
            )
        } else {
            (None, None, None)
        };
        let kernels = self
            .shared_head
            .as_mut()
            .expect("a shared head plan creates its group");
        kernels.join = join;
        kernels.input = input;
        kernels.features = features;
        kernels.projection = projection;
        kernels.centroids = centroids;
        kernels.cluster_selection = cluster_selection;
        kernels.cluster_rows = cluster_rows;
        kernels.placement = placement;
        kernels.chain = chain;
        Ok(())
    }

    /// A separate draft: per distinct binding, its layers' attention (block
    /// and injection) and dense entries; the block embedding; the output
    /// norm and projection of the proposing rows; DSpark's chain entries.
    fn draft(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        if let Some(demand) = &self.graph_demand {
            self.tuning.set_invocations(demand.draft.clone(), true);
        }
        let Some(draft_plan) = plan.draft() else {
            return Ok(());
        };
        let draft_scope = |index: usize, sublayer: u32| {
            WeightScope::DraftSublayer(SublayerIndex {
                block: u32::try_from(index).expect("draft block count fits u32"),
                sublayer,
            })
        };
        let attention_layers = BindingLayers::of(draft_plan.blocks().iter().enumerate().flat_map(
            |(index, b)| {
                [
                    (b.attention, draft_scope(index, 0)),
                    (b.injection, draft_scope(index, 0)),
                ]
            },
        ));
        let dense_layers = BindingLayers::of(
            draft_plan
                .blocks()
                .iter()
                .enumerate()
                .map(|(index, b)| (b.feed_forward, draft_scope(index, 1))),
        );
        let mut registered_attention = HashSet::new();
        let mut registered_dense = HashSet::new();
        for b in draft_plan.blocks() {
            for attention in [b.attention, b.injection] {
                if !registered_attention.insert(attention)
                    || self.draft_kernels().attention.contains_key(&attention)
                {
                    continue;
                }
                if let Some(kernels) =
                    self.attention(attention, attention_layers.scopes(attention))?
                {
                    self.draft_kernels().attention.insert(attention, kernels);
                }
            }
            if registered_dense.insert(b.feed_forward)
                && !self.draft_kernels().dense.contains_key(&b.feed_forward)
            {
                if let Some(kernels) =
                    self.dense(b.feed_forward, dense_layers.scopes(b.feed_forward))?
                {
                    self.draft_kernels().dense.insert(b.feed_forward, kernels);
                }
            }
        }
        let b = draft_plan.embedding();
        let embedding = self.spec.fixed(
            &format!("{b:?}"),
            &[("D", self.hidden)],
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: b.table,
                A: b.activation,
            }),
        )?;
        let head = match draft_plan.projection() {
            ReadoutHead::Packed { weight, .. } | ReadoutHead::Qualified { weight, .. } => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &HeadRowsTuning {
                        projection_scope: draft_plan.projection_scope(),
                        norm_scope: WeightScope::Draft,
                        norm: draft_plan.output_norm(),
                        weight,
                        activation: draft_plan.activation(),
                        epsilon: self.epsilon,
                        rows: Some(self.vocabulary),
                    },
                )?
                .map(ProjectionKernels::RawPacked),
        };
        let shape = self.spec.tuned(
            &mut self.tuning,
            &ShapeRowsTuning {
                vocabulary: self.vocabulary,
            },
        )?;
        let sample = self.spec.tuned(
            &mut self.tuning,
            &SampleRowsTuning {
                vocabulary: self.vocabulary,
            },
        )?;
        let widen = fixed!(
            self.spec,
            widen_rows,
            format!("A={}", draft_plan.activation().name()),
            widen_rows::Elements {
                A: draft_plan.activation()
            }
        );
        let markov = match draft_plan.markov() {
            None => None,
            Some(markov) => {
                let activation = draft_plan.activation();
                let bindings = format!("{markov:?}");
                let rank = markov.rank;
                let embedding = self.spec.fixed(
                    &bindings,
                    &[("D", rank)],
                    embedding_rows::native_entry_with(embedding_rows::Elements {
                        EW: markov.embedding,
                        A: activation,
                    }),
                )?;
                let projection = self.spec.tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down: markov.projection,
                        activation,
                        down_kind: WeightKind::MarkovProjection,
                        scopes: vec![WeightScope::Draft],
                    },
                )?;
                match (embedding, projection) {
                    (Some(embedding), Some(projection)) => Some(MarkovKernels {
                        embedding,
                        projection,
                    }),
                    _ => None,
                }
            }
        };
        self.draft_stages(draft_plan)?;
        let draft = self.draft_kernels();
        draft.embedding = embedding;
        draft.head = head;
        draft.shape = shape;
        draft.sample = sample;
        draft.widen = widen;
        draft.markov = markov;
        Ok(())
    }

    /// The unfused draft stages: every projection (per kind, weight and
    /// published element, over every scope sharing it) and norm (per element
    /// and width) of `super::draft::stage_entries`, then a convolved
    /// backbone's convolution halves and gated product, DFlash2's top-k,
    /// codebook gathers and path step, and LiLiCorr's corrector entries.
    /// Leaves a stage absent during a tuning count.
    fn draft_stages(&mut self, plan: &crate::DraftProgramPlan) -> Result<(), CatalogFailure> {
        let activation = plan.activation();
        let (projections, norm_keys) = super::draft::stage_entries(plan);
        for ((kind, weight, output), scopes) in projections {
            if let Some(kernel) = self.spec.tuned(
                &mut self.tuning,
                &ProjectRowsTuning {
                    weight,
                    activation,
                    kind,
                    output,
                    scopes,
                },
            )? {
                self.draft_kernels()
                    .projections
                    .insert((kind, weight, output), kernel);
            }
        }
        for (norm, width) in norm_keys {
            if let Some(kernel) = self.features_at(
                &format!("draft NW={} D={width}", norm.name()),
                norm,
                activation,
                width,
                (1, width),
            )? {
                self.draft_kernels().norms.insert((norm, width), kernel);
            }
        }
        let bindings = format!("A={}", activation.name());
        if plan.convolutions().is_some() {
            let convolve_input = fixed!(
                self.spec,
                draft_convolve_input,
                bindings,
                draft_convolve_input::Elements { A: activation }
            );
            let convolve_residual = fixed!(self.spec, draft_convolve_residual, "");
            let gated = fixed!(
                self.spec,
                draft_gated_rows,
                bindings,
                draft_gated_rows::Elements { A: activation }
            );
            self.draft_kernels().convolved = match (convolve_input, convolve_residual, gated) {
                (Some(convolve_input), Some(convolve_residual), Some(gated)) => {
                    Some(ConvolvedKernels {
                        convolve_input,
                        convolve_residual,
                        gated,
                    })
                }
                _ => None,
            };
        }
        if let Some(selector) = plan.selector() {
            let top_k = fixed!(self.spec, draft_top_k, "");
            let path = fixed!(
                self.spec,
                draft_path_step,
                bindings,
                draft_path_step::Elements { A: activation }
            );
            let codebook = |spec: &mut Specializer, table: Element| {
                spec.fixed(
                    &format!("selector EW={}", table.name()),
                    &[("D", selector.rank)],
                    embedding_rows::native_entry_with(embedding_rows::Elements {
                        EW: table,
                        A: activation,
                    }),
                )
            };
            let predecessor = codebook(&mut self.spec, selector.predecessor)?;
            let successor = codebook(&mut self.spec, selector.successor)?;
            self.draft_kernels().selector = match (top_k, path, predecessor, successor) {
                (Some(top_k), Some(path), Some(predecessor), Some(successor)) => {
                    Some(SelectorKernels {
                        top_k,
                        predecessor,
                        successor,
                        path,
                    })
                }
                _ => None,
            };
        }
        if let Some(corrector) = plan.corrector() {
            let corrector = corrector.clone();
            let kernels = self.corrector(&corrector, activation)?;
            self.draft_kernels().corrector = kernels;
        }
        Ok(())
    }

    /// LiLiCorr's corrector entries beside its projections and norms, at its
    /// geometry. `None` during a tuning count.
    fn corrector(
        &mut self,
        binding: &crate::CorrectorBinding,
        activation: Element,
    ) -> Result<Option<CorrectorKernels>, CatalogFailure> {
        let (width, top_k) = (binding.width, binding.top_k);
        let slots = binding.slots;
        let nodes = slots * top_k;
        let bindings = format!("A={}", activation.name());
        let top = fixed!(self.spec, draft_top_k, "");
        let features = fixed!(
            self.spec,
            corrector_features,
            bindings,
            corrector_features::Elements { A: activation },
            statics & [("V", self.vocabulary), ("K", top_k), ("H", width)]
        );
        let mut rows = HashMap::new();
        let mut complete = true;
        for key in super::draft::corrector_rows_keys(binding, activation) {
            let (columns, residual, output) = key;
            match fixed!(
                self.spec,
                corrector_rows,
                format!("Y={}", output.name()),
                corrector_rows::Elements { Y: output },
                statics & [("N", columns), ("R", residual)]
            ) {
                Some(kernel) => {
                    rows.insert(key, kernel);
                }
                None => complete = false,
            }
        }
        let node_rows = self.spec.fixed(
            "",
            &[
                ("J", slots),
                ("K", top_k),
                ("H", width),
                ("TB", u64::from(binding.token.is_some())),
            ],
            corrector_nodes::native_entry(),
        )?;
        let attention = fixed!(
            self.spec,
            corrector_attention,
            bindings,
            corrector_attention::Elements { A: activation },
            statics & [("N", nodes), ("NH", binding.heads), ("W", width / binding.heads)]
        );
        let factors = fixed!(
            self.spec,
            corrector_factors,
            bindings,
            corrector_factors::Elements { A: activation },
            statics & [("N", nodes), ("H", width)]
        );
        let path = self.spec.fixed(
            "",
            &[("J", slots), ("K", top_k), ("F", binding.factor)],
            corrector_path::native_entry(),
        )?;
        Ok(
            match (complete, top, features, node_rows, attention, factors, path) {
                (
                    true,
                    Some(top_k),
                    Some(features),
                    Some(nodes),
                    Some(attention),
                    Some(factors),
                    Some(path),
                ) => Some(CorrectorKernels {
                    top_k,
                    features,
                    rows,
                    nodes,
                    attention,
                    factors,
                    path,
                }),
                _ => None,
            },
        )
    }

    fn draft_kernels(&mut self) -> &mut DraftKernels {
        self.draft
            .as_mut()
            .expect("a draft plan creates the draft group")
    }

    fn vision(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        if let Some(demand) = &self.graph_demand {
            self.tuning.set_invocations(demand.target.clone(), true);
        }
        let Some(vision_plan) = plan.vision() else {
            return Ok(());
        };
        let spec = &mut self.spec;
        let vision = self
            .vision
            .as_mut()
            .expect("a vision plan creates the vision group");
        for kernel in vision_plan.kernels() {
            let label = format!("{kernel:?}");
            let statics = kernel.statics.as_slice();
            macro_rules! element {
                ($module:ident, $name:literal) => {
                    super::vision::element(<$module::Entry as seismic::Entry>::NAME, kernel, $name)?
                };
            }
            macro_rules! prepare {
                ($module:ident, $map:ident, $elements:expr) => {{
                    let elements = $elements;
                    if let Some(native) = fixed!(spec, $module, label, elements, statics & statics)
                    {
                        vision.$map.insert(kernel.clone(), native);
                    }
                }};
            }
            match kernel.entry {
                VisionEntry::PatchStem => prepare!(
                    vision_patch_stem,
                    patch_stem,
                    vision_patch_stem::Elements {
                        W0: element!(vision_patch_stem, "W0"),
                        W1: element!(vision_patch_stem, "W1"),
                        B: element!(vision_patch_stem, "B"),
                        PE: element!(vision_patch_stem, "PE"),
                    }
                ),
                VisionEntry::Norm => prepare!(
                    vision_norm,
                    norm,
                    vision_norm::Elements {
                        NWE: element!(vision_norm, "NWE"),
                        NBE: element!(vision_norm, "NBE"),
                        Y: element!(vision_norm, "Y"),
                    }
                ),
                VisionEntry::Linear => prepare!(
                    vision_linear,
                    linear,
                    vision_linear::Elements {
                        A: element!(vision_linear, "A"),
                        W: element!(vision_linear, "W"),
                        B: element!(vision_linear, "B"),
                        Y: element!(vision_linear, "Y"),
                    }
                ),
                VisionEntry::Clamp => prepare!(
                    vision_clamp,
                    clamp,
                    vision_clamp::Elements {
                        A: element!(vision_clamp, "A"),
                    }
                ),
                VisionEntry::Attention => prepare!(
                    vision_attention,
                    attention,
                    vision_attention::Elements {
                        A: element!(vision_attention, "A"),
                    }
                ),
                VisionEntry::GatedConvolution => prepare!(
                    vision_gated_convolution,
                    gated_convolution,
                    vision_gated_convolution::Elements {
                        A: element!(vision_gated_convolution, "A"),
                        KE: element!(vision_gated_convolution, "KE"),
                    }
                ),
                VisionEntry::Pool => prepare!(
                    vision_pool,
                    pool,
                    vision_pool::Elements {
                        SB: element!(vision_pool, "SB"),
                        SS: element!(vision_pool, "SS"),
                    }
                ),
                VisionEntry::Position => prepare!(
                    vision_position,
                    position,
                    vision_position::Elements {
                        PE: element!(vision_position, "PE"),
                    }
                ),
                VisionEntry::PostNormResidual => prepare!(
                    post_norm_residual,
                    post_norm,
                    post_norm_residual::Elements {
                        NW: element!(post_norm_residual, "NW"),
                    }
                ),
            }
        }
        Ok(())
    }
}
