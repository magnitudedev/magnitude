//! The general routed feed-forward graph (`operators::routed`): the shared
//! expert's dense feed-forward onto the residual as the `base`, selection,
//! an optional latent down projection, the experts (decode form up to
//! [`DECODE_ROWS`](super::fused_graph::DECODE_ROWS) rows, grouped form beyond) summed onto the base, and an
//! optional latent up projection. Row tables, the selection bias a model
//! lacks and a latent sum's zero base are graph constants; a scale it lacks
//! is the absent operand. A split layer's front selects and projects down,
//! its experts are grouped in its back or its group's pass, and its back
//! forms the sum: the shared expert, the scatter and the latent up
//! projection run at the launch's own rows.

use seismic::LaunchEncoding;
use super::fused_graph::{TILE_ROWS, decodes};
use super::grouped::{ExpertsPass, Grouped};
use super::split::{Crossing, PassPorts, Split};
use super::streamed::{
    BankCombineEntry, BankForm, BankNumbers, BankRoute, LayerBank, DecodeReads, DecodeShape, RouteLocals,
    StreamedLayer, LandPorts,
};
use super::{GeneralRoutedForm, GeneralRoutedNumbers, SharedExpertNumbers};
use crate::operators::scalar;
use seismic::Argument;
use crate::planning::{BankPart, RecordTensors};
use crate::programs::graph::facts::Numbers;
use crate::native::{DenseExpansionKernel, ExpertKernels, GeneralRoutedKernels};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    PlannedWeights, WeightPart, WeightPort, WeightScale, packed_weight, scaled_weight, weight, weight_port,
};
use crate::{GeneralRoutedBinding, GeneralRoutedShape, ModelLoadPlan};
use magnitude_family_contracts::{GluLimit, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{
    dense_expand, dense_output, dense_up, project_rows, routed_down, routed_down_project, routed_down_sited, routed_experts,
    routed_experts_up, routed_gate_up, routed_group, routed_scatter, routed_select,
    routed_select_streamed, routed_up,
};
use seismic::{Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// What decides a general routed layer's graph beside its numbers: its
/// operator's form and how it reads its experts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GeneralRoutedTopology<R = Argument<f64>> {
    pub form: GeneralRoutedForm<R>,
    pub bank: BankForm,
}

impl<R> Numbers<R> for GeneralRoutedTopology<R> {
    type Of<E> = GeneralRoutedTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> GeneralRoutedTopology<E> {
        GeneralRoutedTopology { form: self.form.map(f), bank: self.bank }
    }
}

/// The numbers of a general routed layer: its operator's, and a streamed
/// layer's bank's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GeneralRoutedLayer<D = Dim> {
    pub shape: GeneralRoutedNumbers<D>,
    pub bank: Option<BankNumbers<D>>,
}

impl<D> Numbers<D> for GeneralRoutedLayer<D> {
    type Of<E> = GeneralRoutedLayer<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> GeneralRoutedLayer<E> {
        GeneralRoutedLayer { shape: self.shape.map(&mut *f), bank: self.bank.map(|bank| bank.map(&mut *f)) }
    }
}

/// The topology and numbers of the general routed layer of `scope` of
/// `shape`, whose graph reads its experts as `bank` does.
pub(crate) fn split<G: GraphDraft>(
    load: &ModelLoadPlan,
    scope: WeightScope,
    shape: &GeneralRoutedShape,
    bank: &LayerBank<'_, '_, G>,
) -> Result<(GeneralRoutedTopology<f64>, GeneralRoutedLayer<u64>), GraphError> {
    let (form, numbers) = shape.split();
    let (bank, bank_numbers) = super::streamed::split(load, scope, bank)?;
    Ok((GeneralRoutedTopology { form, bank }, GeneralRoutedLayer { shape: numbers, bank: bank_numbers }))
}

/// The experts' expansion entries.
pub(crate) enum ExpertEntries<'a, G: GraphDraft + 'a> {
    Gated {
        decode: G::Binding<'a, routed_gate_up::Entry>,
        grouped: G::Binding<'a, routed_experts::Entry>,
    },
    Plain {
        decode: G::Binding<'a, routed_up::Entry>,
        grouped: G::Binding<'a, routed_experts_up::Entry>,
    },
}

/// What the routed sum accumulates onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoutedSum {
    /// The residual: the sublayer's output is the next residual.
    Residual,
    /// Zeros: the output is a branch's own, which a tail combines
    /// (`moe_tail`).
    Branch,
}

/// A dense expansion entry.
pub(crate) enum ExpansionEntry<'a, G: GraphDraft + 'a> {
    Gated(G::Binding<'a, dense_expand::Entry>),
    Plain(G::Binding<'a, dense_up::Entry>),
}

pub(crate) struct GeneralRoutedGraphEntries<'a, G: GraphDraft + 'a> {
    pub select: G::Binding<'a, routed_select::Entry>,
    pub experts: ExpertEntries<'a, G>,
    pub down: G::Binding<'a, routed_down::Entry>,
    pub group: G::Binding<'a, routed_group::Entry>,
    pub scatter: G::Binding<'a, routed_scatter::Entry>,
    pub shared: Option<(ExpansionEntry<'a, G>, G::Binding<'a, dense_output::Entry>)>,
    pub latent: Option<(
        G::Binding<'a, project_rows::Entry>,
        G::Binding<'a, dense_output::Entry>,
    )>,
}

impl<'a, G: GraphDraft + 'a> Copy for GeneralRoutedGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for GeneralRoutedGraphEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, G: GraphDraft + 'a> Copy for ExpertEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for ExpertEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, G: GraphDraft + 'a> Copy for ExpansionEntry<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for ExpansionEntry<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a GeneralRoutedKernels> for GeneralRoutedGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a GeneralRoutedKernels) -> Self {
        Self {
            select: &kernels.select,
            experts: match &kernels.experts {
                ExpertKernels::Gated { decode, grouped } => {
                    ExpertEntries::Gated { decode, grouped }
                }
                ExpertKernels::Plain { decode, grouped } => {
                    ExpertEntries::Plain { decode, grouped }
                }
            },
            down: &kernels.down,
            group: &kernels.group,
            scatter: &kernels.scatter,
            shared: kernels.shared.as_ref().map(|(expansion, output)| {
                (
                    match expansion {
                        DenseExpansionKernel::Gated(kernel) => ExpansionEntry::Gated(kernel),
                        DenseExpansionKernel::Plain(kernel) => ExpansionEntry::Plain(kernel),
                    },
                    output,
                )
            }),
            latent: kernels.latent.as_ref().map(|(down, up)| (down, up)),
        }
    }
}

/// Entry element assignments from the program binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedGeneralRoutedEntries {
    select: [(&'static str, Element); 4],
    gate_up: [(&'static str, Element); 3],
    up: [(&'static str, Element); 2],
    experts: [(&'static str, Element); 4],
    experts_up: [(&'static str, Element); 3],
    down: [(&'static str, Element); 3],
    group: [(&'static str, Element); 0],
    scatter: [(&'static str, Element); 2],
    shared_expand: [(&'static str, Element); 4],
    shared_up: [(&'static str, Element); 3],
    shared_output: [(&'static str, Element); 2],
    latent_down: [(&'static str, Element); 3],
    latent_up: [(&'static str, Element); 2],
    binding: GeneralRoutedBinding,
}

/// The element the routed sum publishes in: activations for a latent sum
/// that feeds the latent up projection, else F32 onto the residual.
pub(crate) fn sum_element(binding: &GeneralRoutedBinding) -> Element {
    if binding.shape.latent {
        binding.activation
    } else {
        Element::f32()
    }
}

impl CheckedGeneralRoutedEntries {
    pub(crate) fn new(binding: GeneralRoutedBinding) -> Self {
        let a = binding.activation;
        // An absent weight's element is never read: its entry is not built.
        let gate = binding.expert_gate.unwrap_or(binding.expert_up);
        let (shared_gate, shared_up, shared_down) = binding
            .shared
            .map(|(gate, up, down)| (gate.unwrap_or(up), up, down))
            .unwrap_or((a, a, a));
        let (latent_down, latent_up) = binding.latent.unwrap_or((a, a));
        Self {
            select: [
                ("NW", binding.norm),
                ("RNW", binding.router_norm),
                ("RW", binding.router),
                ("A", a),
            ],
            gate_up: [("EGW", gate), ("EUW", binding.expert_up), ("A", a)],
            up: [("EUW", binding.expert_up), ("A", a)],
            experts: [
                ("EGW", gate),
                ("EUW", binding.expert_up),
                ("EDW", binding.expert_down),
                ("A", a),
            ],
            experts_up: [
                ("EUW", binding.expert_up),
                ("EDW", binding.expert_down),
                ("A", a),
            ],
            down: [
                ("EDW", binding.expert_down),
                ("A", a),
                ("R", sum_element(&binding)),
            ],
            group: [],
            scatter: [("A", a), ("R", sum_element(&binding))],
            shared_expand: [
                ("NW", binding.norm),
                ("GW", shared_gate),
                ("UW", shared_up),
                ("A", a),
            ],
            shared_up: [("NW", binding.norm), ("UW", shared_up), ("A", a)],
            shared_output: [("DW", shared_down), ("A", a)],
            latent_down: [("A", a), ("W", latent_down), ("Y", a)],
            latent_up: [("DW", latent_up), ("A", a)],
            binding,
        }
    }

    pub(crate) fn entries(&self) -> GeneralRoutedGraphEntries<'_, NativeGraphMetadata> {
        let shape = self.binding.shape;
        GeneralRoutedGraphEntries {
            select: &self.select,
            experts: if shape.experts_expansion.gated {
                ExpertEntries::Gated {
                    decode: &self.gate_up,
                    grouped: &self.experts,
                }
            } else {
                ExpertEntries::Plain {
                    decode: &self.up,
                    grouped: &self.experts_up,
                }
            },
            down: &self.down,
            group: &self.group,
            scatter: &self.scatter,
            shared: shape.shared.map(|(_, expansion)| {
                (
                    if expansion.gated {
                        ExpansionEntry::Gated(&self.shared_expand[..])
                    } else {
                        ExpansionEntry::Plain(&self.shared_up[..])
                    },
                    &self.shared_output[..],
                )
            }),
            latent: shape
                .latent
                .then_some((&self.latent_down[..], &self.latent_up[..])),
        }
    }
}

/// A general layer's selection weights: its input norm, its router's
/// (the input norm where it has none of its own), its router, its
/// selection bias (zeros where it has none), and the combine's per-expert
/// scales (ES), the stored output scales or the down weight's second-level
/// scales (plan admission refuses both at once), else absent.
struct Selection {
    norm: WorkflowTensor,
    router_norm: WorkflowTensor,
    router: WorkflowTensor,
    bias: WorkflowTensor,
    expert_scale: WeightScale,
}

impl Selection {
    fn of<G: GraphDraft>(
        graph: &mut G,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        form: &GeneralRoutedForm,
        shape: &GeneralRoutedNumbers,
    ) -> Result<Self, GraphError> {
        let mut weight = |graph: &mut G, kind| weight(graph, planned, scope, kind, weights);
        let norm = weight(graph, WeightKind::InputNorm)?;
        let router_norm = match form.router_norm {
            true => weight(graph, WeightKind::RouterInputNorm)?,
            false => norm.clone(),
        };
        let router = weight(graph, WeightKind::Router)?;
        let bias = match form.bias {
            true => weight(graph, WeightKind::RouterSelectionBias)?,
            false => {
                let constant = GraphConstant::f32_filled(graph, std::slice::from_ref(&shape.experts), 0.0)?;
                let tensor = constant.port().tensor().clone();
                constants.push(constant);
                tensor
            }
        };
        let stored = form
            .expert_scale
            .then(|| weight(graph, WeightKind::ExpertScale))
            .transpose()?;
        let role = WeightRole { scope, kind: WeightKind::ExpertDown };
        let down = WeightScale::per_expert(graph, planned, role, weights, constants)?;
        Ok(Self {
            norm,
            router_norm,
            router,
            bias,
            expert_scale: WeightScale {
                extent: shape.expert_scale_extent(form, down.extent),
                tensor: stored.unwrap_or(down.tensor),
            },
        })
    }

    /// The selection's dimensions over `rows` rows.
    fn dimensions(&self, shape: &GeneralRoutedNumbers, rows: &Dim) -> Vec<(&'static str, Dim)> {
        [shape.select_dimensions(rows).as_slice(), &[("ES", self.expert_scale.extent.clone())]].concat()
    }

    /// The selection of `residual`'s rows into `routes` and their `weights`:
    /// the normalized rows.
    #[allow(clippy::too_many_arguments)]
    fn select<'a, G: GraphDraft + 'a>(
        &self,
        graph: &mut G,
        entry: G::Binding<'a, routed_select::Entry>,
        residual: &WorkflowTensor,
        [routes, weights]: [&mut NativePort; 2],
        form: &GeneralRoutedForm,
        shape: &GeneralRoutedNumbers,
        rows: &Dim,
        epsilon: Argument<f32>,
    ) -> Result<WorkflowTensor, GraphError> {
        Ok(graph
            .enqueue(
                entry,
                &self.dimensions(shape, rows),
                routed_select::WorkflowArgs {
                    residual: residual.into(),
                    norm: (&self.norm).into(),
                    router_norm: (&self.router_norm).into(),
                    router: (&self.router).into(),
                    bias: (&self.bias).into(),
                    expert_scale: (&self.expert_scale.tensor).into(),
                    routes: routes.tensor_mut().into(),
                    weights: weights.tensor_mut().into(),
                    epsilon,
                    score: scalar(&shape.score, "a route's score code")?,
                    normalization: scalar(&shape.normalization, "a route's normalization code")?,
                    normalization_epsilon: form.normalization_epsilon.f32(),
                    scale: form.scale.f32(),
                },
            )?
            .value)
    }
}

/// A general layer's stacked expert weights (a streamed layer's are its
/// bank group's records, which its fills write, or in its resident form its
/// home records among them) and the per-expert scales
/// of their expansions, absent where they have none. Up-only experts have
/// no gate, nor its scale.
struct ExpertStacks {
    gate: Option<(NativePort, WeightScale)>,
    up: NativePort,
    down: NativePort,
    up_scale: WeightScale,
}

impl ExpertStacks {
    /// The layer's stacks, each its `part` of the planned stack
    /// (`LayerBank::stacks`), and their scales.
    fn of<G: GraphDraft>(
        graph: &mut G,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        form: &GeneralRoutedForm,
        part: WeightPart,
    ) -> Result<Self, GraphError> {
        let role = |kind| WeightRole { scope, kind };
        let stack = |graph: &mut G, weights: &mut Vec<(WeightPort, NativePort)>, kind| {
            weight_port(graph, planned, role(kind), part, weights)
        };
        let gate = match form.experts_expansion.gated {
            true => Some((
                stack(graph, weights, WeightKind::ExpertGate)?,
                WeightScale::per_expert(graph, planned, role(WeightKind::ExpertGate), weights, constants)?,
            )),
            false => None,
        };
        let up = stack(graph, weights, WeightKind::ExpertUp)?;
        let down = stack(graph, weights, WeightKind::ExpertDown)?;
        let up_scale = WeightScale::per_expert(graph, planned, role(WeightKind::ExpertUp), weights, constants)?;
        Ok(Self { gate, up, down, up_scale })
    }

    /// The stacks' tensors.
    fn tensors(&self) -> RecordTensors<WorkflowTensor> {
        RecordTensors { gate: self.gate.as_ref().map(|(gate, _)| gate), up: &self.up, down: &self.down }
            .map(|_, port| port.tensor().clone())
    }

    /// The stacks' ports.
    fn records(&mut self) -> RecordTensors<&mut NativePort> {
        RecordTensors { gate: self.gate.as_mut().map(|(gate, _)| gate), up: &mut self.up, down: &mut self.down }
    }
}

/// What the experts act on of the `normalized` rows: a latent operator's
/// projection of them down into the latent width, else the rows.
#[allow(clippy::too_many_arguments)]
fn expert_input<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: Option<G::Binding<'a, project_rows::Entry>>,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    normalized: WorkflowTensor,
    rows: &Dim,
    form: &GeneralRoutedForm,
    shape: &GeneralRoutedNumbers,
) -> Result<WorkflowTensor, GraphError> {
    let entry = match (entry, form.latent) {
        (Some(entry), true) => entry,
        (None, false) => return Ok(normalized),
        _ => return Err("latent entries disagree with its weights".into()),
    };
    let down = packed_weight(graph, planned, scope, WeightKind::LatentDown, weights, constants)?;
    Ok(graph
        .enqueue(
            entry,
            &[
                ("M", rows.clone()),
                ("G", Dim::from(1)),
                ("K", shape.hidden.clone()),
                ("N", shape.expert_hidden.clone()),
                ("WS", down.operands.scale.extent.clone()),
                ("WB", down.operands.bound.extent.clone()),
            ],
            project_rows::WorkflowArgs {
                source: (&normalized).into(),
                weight: (&down.weight).into(),
                weight_scale: (&down.operands.scale.tensor).into(),
                weight_bound: (&down.operands.bound.tensor).into(),
            },
        )?
        .value)
}

/// A routed sum's base over `rows` rows: the shared expert's output onto
/// the root (the residual, or a branch's zeros), and what the experts sum
/// onto: the base, or a latent operator's zeros in the latent width, whose
/// sum the latent up projection adds to the base (`finish`). The dense
/// projections' row table is a graph port only where one reads it (an
/// unread port is unbound).
struct SumBase {
    base: WorkflowTensor,
    onto: WorkflowTensor,
    row_table: Option<GraphConstant>,
    latent_zeros: Option<GraphConstant>,
}

impl SumBase {
    /// The base of `residual`'s rows; the shared expert reads `norm`, a
    /// whole layer's selection's, or else its own.
    #[allow(clippy::too_many_arguments)]
    fn of<'a, G: GraphDraft + 'a>(
        graph: &mut G,
        shared: Option<(ExpansionEntry<'a, G>, G::Binding<'a, dense_output::Entry>)>,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        residual: &WorkflowTensor,
        sum: RoutedSum,
        rows: &Dim,
        form: &GeneralRoutedForm,
        shape: &GeneralRoutedNumbers,
        epsilon: Argument<f32>,
        norm: Option<WorkflowTensor>,
    ) -> Result<Self, GraphError> {
        let row_table = (form.shared.is_some() || form.latent)
            .then(|| GraphConstant::row_identity(graph, rows, Some("M")))
            .transpose()?;
        let root = match sum {
            RoutedSum::Residual => residual.clone(),
            RoutedSum::Branch => {
                let zeros = GraphConstant::row_zeros(graph, Element::f32(), rows, &shape.hidden, Some("M"))?;
                let tensor = zeros.port().tensor().clone();
                constants.push(zeros);
                tensor
            }
        };
        let base = match (shared, form.shared, &shape.shared) {
            (Some((expansion, output)), Some(form), Some(SharedExpertNumbers { features, activation })) => {
                let activation = scalar(activation, "a shared expert's activation code")?;
                let norm = match norm {
                    Some(norm) => norm,
                    None => weight(graph, planned, scope, WeightKind::InputNorm, weights)?,
                };
                let table = row_table.as_ref().ok_or("general routed row table is absent")?.port().tensor().clone();
                let dimensions =
                    [("M", rows.clone()), ("O", rows.clone()), ("H", shape.hidden.clone()), ("F", features.clone())];
                let product = match (expansion, form.gated) {
                    (ExpansionEntry::Gated(entry), true) => {
                        let gate = packed_weight(graph, planned, scope, WeightKind::SharedGate, weights, constants)?;
                        let up = packed_weight(graph, planned, scope, WeightKind::SharedUp, weights, constants)?;
                        graph
                            .enqueue(
                                entry,
                                &[
                                    dimensions.as_slice(),
                                    &[
                                        ("GS", gate.operands.scale.extent.clone()),
                                        ("US", up.operands.scale.extent.clone()),
                                        ("GB", gate.operands.bound.extent.clone()),
                                        ("UB", up.operands.bound.extent.clone()),
                                        ("LB", Dim::from(GluLimit::word(form.limit))),
                                    ],
                                ]
                                .concat(),
                                dense_expand::WorkflowArgs {
                                    residual: residual.into(),
                                    norm: (&norm).into(),
                                    gate_weight: (&gate.weight).into(),
                                    up_weight: (&up.weight).into(),
                                    out_rows: (&table).into(),
                                    eps: epsilon.into(),
                                    activation,
                                    gate_scale: (&gate.operands.scale.tensor).into(),
                                    up_scale: (&up.operands.scale.tensor).into(),
                                    gate_bound: (&gate.operands.bound.tensor).into(),
                                    up_bound: (&up.operands.bound.tensor).into(),
                                    glu_limit: (&crate::operators::glu_limit(&table, form.limit)).into(),
                                },
                            )?
                            .value
                    }
                    (ExpansionEntry::Plain(entry), false) => {
                        let up = scaled_weight(graph, planned, scope, WeightKind::SharedUp, weights, constants)?;
                        graph
                            .enqueue(
                                entry,
                                &[dimensions.as_slice(), &[("US", up.scale.extent.clone())]].concat(),
                                dense_up::WorkflowArgs {
                                    residual: residual.into(),
                                    norm: (&norm).into(),
                                    up_weight: (&up.weight).into(),
                                    out_rows: (&table).into(),
                                    eps: epsilon.into(),
                                    activation,
                                    up_scale: (&up.scale.tensor).into(),
                                },
                            )?
                            .value
                    }
                    _ => return Err("shared expert entries disagree with its weights".into()),
                };
                let down = packed_weight(graph, planned, scope, WeightKind::SharedDown, weights, constants)?;
                graph
                    .enqueue(
                        output,
                        &[
                            dimensions.as_slice(),
                            &[("DS", down.operands.scale.extent.clone()), ("DB", down.operands.bound.extent.clone())],
                        ]
                        .concat(),
                        dense_output::WorkflowArgs {
                            residual: (&root).into(),
                            product: (&product).into(),
                            down_weight: (&down.weight).into(),
                            out_rows: (&table).into(),
                            down_scale: (&down.operands.scale.tensor).into(),
                            down_bound: (&down.operands.bound.tensor).into(),
                        },
                    )?
                    .value
            }
            (None, None, None) => root,
            _ => return Err("shared expert entries disagree with its weights".into()),
        };
        let latent_zeros = form
            .latent
            .then(|| GraphConstant::row_zeros(graph, Element::f32(), rows, &shape.expert_hidden, Some("M")))
            .transpose()?;
        let onto = match &latent_zeros {
            Some(zeros) => zeros.port().tensor().clone(),
            None => base.clone(),
        };
        Ok(Self { base, onto, row_table, latent_zeros })
    }

    /// The sum's output from the experts' `routed` sum: a latent operator's
    /// projected up onto the base, else the sum itself.
    #[allow(clippy::too_many_arguments)]
    fn finish<'a, G: GraphDraft + 'a>(
        self,
        graph: &mut G,
        entry: Option<G::Binding<'a, dense_output::Entry>>,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        routed: WorkflowTensor,
        rows: &Dim,
        shape: &GeneralRoutedNumbers,
    ) -> Result<WorkflowTensor, GraphError> {
        let output = match (entry, self.latent_zeros) {
            (Some(entry), Some(zeros)) => {
                let up = packed_weight(graph, planned, scope, WeightKind::LatentUp, weights, constants)?;
                let table = self.row_table.as_ref().ok_or("general routed row table is absent")?.port().tensor().clone();
                let output = graph
                    .enqueue(
                        entry,
                        &[
                            ("M", rows.clone()),
                            ("O", rows.clone()),
                            ("H", shape.hidden.clone()),
                            ("F", shape.expert_hidden.clone()),
                            ("DS", up.operands.scale.extent.clone()),
                            ("DB", up.operands.bound.extent.clone()),
                        ],
                        dense_output::WorkflowArgs {
                            residual: (&self.base).into(),
                            product: (&routed).into(),
                            down_weight: (&up.weight).into(),
                            out_rows: (&table).into(),
                            down_scale: (&up.operands.scale.tensor).into(),
                            down_bound: (&up.operands.bound.tensor).into(),
                        },
                    )?
                    .value;
                constants.push(zeros);
                output
            }
            (None, None) => routed,
            _ => return Err("latent entries disagree with its weights".into()),
        };
        constants.extend(self.row_table);
        Ok(output)
    }
}

/// A general layer's experts (gated `routed_experts` or up-only
/// `routed_experts_up`) over a grouping (`grouped`), with the `stacks`'
/// scales.
fn expand<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: ExpertEntries<'a, G>,
    stacks: &ExpertStacks,
    grouped: Grouped<'_, RecordTensors<WorkflowTensor>>,
    form: &GeneralRoutedForm,
    shape: &GeneralRoutedNumbers,
) -> Result<WorkflowTensor, GraphError> {
    let Grouped { rows, blocks, first, rows_read, input, order, block_experts, used, reads } = grouped;
    let [mapped, step_words] = reads.map_dimensions();
    let dimensions = [
        ("M", rows.clone()),
        ("H", shape.expert_hidden.clone()),
        ("E", shape.experts.clone()),
        ("F", shape.intermediate.clone()),
        ("B", blocks.clone()),
        ("T", Dim::from(TILE_ROWS)),
        mapped,
        step_words,
        ("R", reads.count.clone()),
    ];
    let activation = scalar(&shape.activation, "an expert activation code")?;
    let (map, step, records) = (&reads.map, &reads.step, &reads.records);
    let RecordTensors { gate, up, down } = &reads.stacks;
    Ok(match (entries, gate, &records.gate, &stacks.gate) {
        (ExpertEntries::Gated { grouped, .. }, Some(gate), Some(records_gate), Some((_, gate_scale))) => {
            graph
                .enqueue(
                    grouped,
                    &[
                        dimensions.as_slice(),
                        &[
                            ("GS", gate_scale.extent.clone()),
                            ("US", stacks.up_scale.extent.clone()),
                            ("LB", Dim::from(GluLimit::word(form.experts_expansion.limit))),
                        ],
                    ]
                    .concat(),
                    routed_experts::WorkflowArgs {
                        first: first.into(),
                        rows: rows_read.into(),
                        normalized: input.into(),
                        order: order.into(),
                        blocks: block_experts.into(),
                        used: used.into(),
                        expert_gate: gate.into(),
                        expert_up: up.into(),
                        expert_down: down.into(),
                        map: map.into(),
                        step: step.into(),
                        records_gate: records_gate.into(),
                        records_up: (&records.up).into(),
                        records_down: (&records.down).into(),
                        activation,
                        gate_scale: (&gate_scale.tensor).into(),
                        up_scale: (&stacks.up_scale.tensor).into(),
                        glu_limit: (&crate::operators::glu_limit(first, form.experts_expansion.limit)).into(),
                    },
                )?
                .value
        }
        (ExpertEntries::Plain { grouped, .. }, None, None, None) => {
            graph
                .enqueue(
                    grouped,
                    &[dimensions.as_slice(), &[("US", stacks.up_scale.extent.clone())]].concat(),
                    routed_experts_up::WorkflowArgs {
                        first: first.into(),
                        rows: rows_read.into(),
                        normalized: input.into(),
                        order: order.into(),
                        blocks: block_experts.into(),
                        used: used.into(),
                        expert_up: up.into(),
                        expert_down: down.into(),
                        map: map.into(),
                        step: step.into(),
                        records_up: (&records.up).into(),
                        records_down: (&records.down).into(),
                        up_scale: (&stacks.up_scale.tensor).into(),
                        activation,
                    },
                )?
                .value
        }
        _ => return Err("expert entries disagree with its weights".into()),
    })
}

/// The routed sum of `rows` rows of `residual` after their experts'
/// grouped `pass`: the experts' rows read at the positions the pass names
/// for them, weighted by the choices' `route_weights` (`SumBase::of` reads
/// `norm`).
#[allow(clippy::too_many_arguments)]
fn combine<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    shared: Option<(ExpansionEntry<'a, G>, G::Binding<'a, dense_output::Entry>)>,
    scatter: G::Binding<'a, routed_scatter::Entry>,
    latent_up: Option<G::Binding<'a, dense_output::Entry>>,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    sum: RoutedSum,
    pass: &ExpertsPass,
    route_weights: &WorkflowTensor,
    rows: &Dim,
    form: &GeneralRoutedForm,
    shape: &GeneralRoutedNumbers,
    epsilon: Argument<f32>,
    norm: Option<WorkflowTensor>,
) -> Result<WorkflowTensor, GraphError> {
    let base =
        SumBase::of(graph, shared, planned, scope, weights, constants, residual, sum, rows, form, shape, epsilon, norm)?;
    let routed = graph
        .enqueue(
            scatter,
            &[
                ("M", rows.clone()),
                ("H", shape.expert_hidden.clone()),
                ("K", shape.selected.clone()),
                ("B", pass.blocks.clone()),
                ("T", Dim::from(TILE_ROWS)),
            ],
            routed_scatter::WorkflowArgs {
                base: (&base.onto).into(),
                expert_output: (&pass.output).into(),
                inverse: (&pass.inverse).into(),
                weights: route_weights.into(),
            },
        )?
        .value;
    base.finish(graph, latent_up, planned, scope, weights, constants, routed, rows, shape)
}

/// The general routed feed-forward of one block over `rows` rows of
/// `residual`: the new residual rows (F32), and the sequence input a run of
/// a streamed layer writes. A prompt launch of a split layer is built as
/// its front (selection into its crossing, the experts' input projected
/// down where latent, and, where scratch-served, its request), or its back
/// after its merge set's pass, which grouped the set's rows (`split`);
/// a front's output is the residual it read. A layer the load streams takes
/// its `bank` entries: a bank-served launch selects, resolves and assigns
/// (`routed_select_streamed`), lands the experts its host reads
/// (`routed_land`) and requests the CPU's choices
/// (`routed_site_request`), then the decode entries read each choice where
/// its source names, a bank record or a tier record in place (`X` = the
/// experts, `E` = `BANK`), and the sum adds the CPU's answer. Its resident
/// form (`LayerBank::Home`) decodes as a resident layer does, over its whole
/// bank's home records.
#[allow(clippy::too_many_arguments)]
pub(crate) fn general_routed<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: GeneralRoutedGraphEntries<'a, G>,
    bank: LayerBank<'_, 'a, G>,
    topology: &GeneralRoutedTopology,
    numbers: &GeneralRoutedLayer,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    sum: RoutedSum,
    rows: &Dim,
    epsilon: Argument<f32>,
    split: Option<Split<'_>>,
) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
    let (form, shape) = (&topology.form, &numbers.shape);
    let part = bank.stacks();
    let mut streamed = StreamedLayer::of(planned, &topology.bank, numbers.bank.as_ref(), scope, bank)?;
    let decode = decodes(rows);
    let scratch = streamed.is_some() && !decode;
    let GeneralRoutedGraphEntries { select, experts, down, group, scatter, shared, latent } = entries;
    let (latent_down, latent_up) = (latent.map(|(down, _)| down), latent.map(|(_, up)| up));
    match (split, &mut streamed) {
        (None, _) if !scratch => {}
        (Some(Split::Front(crossing)), layer) if !decode => {
            let selection = Selection::of(graph, planned, scope, weights, constants, form, shape)?;
            let Crossing { routes, weights: route_weights, normalized: into, .. } = crossing;
            let normalized =
                selection.select(graph, select, residual, [routes, route_weights], form, shape, rows, epsilon)?;
            let input =
                expert_input(graph, latent_down, planned, scope, weights, constants, normalized, rows, form, shape)?;
            graph.store(&input, into)?;
            let sequence = match layer {
                Some(layer) => layer.request(graph, routes.tensor(), rows)?,
                None => None,
            };
            return Ok((residual.clone(), sequence));
        }
        // A back after its merge set's pass: its rows' experts are the pass's,
        // at the positions its rows of the merged positions name.
        (Some(Split::Combine(back, blocks)), _) if !decode => {
            let output = combine(
                graph, shared, scatter, latent_up, planned, scope, weights, constants, residual, sum, &back.pass(blocks),
                back.crossing.weights.tensor(), rows, form, shape, epsilon, None,
            )?;
            return Ok((output, None));
        }
        _ => return Err("a routed layer's graph disagrees with how its experts are served".into()),
    }
    // A whole layer.
    let selection = Selection::of(graph, planned, scope, weights, constants, form, shape)?;
    let mut stacks = ExpertStacks::of(graph, planned, scope, weights, constants, form, part)?;
    let dimensions = selection.dimensions(shape, rows);
    let mut routes = graph.local_for(select, "routes", &dimensions)?;
    let mut route_weights = graph.local_for(select, "weights", &dimensions)?;
    if !decode {
        let normalized = selection.select(
            graph,
            select,
            residual,
            [&mut routes, &mut route_weights],
            form,
            shape,
            rows,
            epsilon,
        )?;
        let input =
            expert_input(graph, latent_down, planned, scope, weights, constants, normalized, rows, form, shape)?;
        let pass = super::grouped::class(
            graph,
            group,
            [routes.tensor(), &input],
            stacks.tensors(),
            rows,
            [&shape.experts, &shape.selected],
            constants,
            |graph, grouped| expand(graph, experts, &stacks, grouped, form, shape),
        )?;
        let output = combine(
            graph,
            shared,
            scatter,
            latent_up,
            planned,
            scope,
            weights,
            constants,
            residual,
            sum,
            &pass,
            route_weights.tensor(),
            rows,
            form,
            shape,
            epsilon,
            Some(selection.norm),
        )?;
        return Ok((output, None));
    }
    let base = SumBase::of(
        graph,
        shared,
        planned,
        scope,
        weights,
        constants,
        residual,
        sum,
        rows,
        form,
        shape,
        epsilon,
        Some(selection.norm.clone()),
    )?;
    // A bank-served launch selects, resolves and assigns each choice's
    // source, its landing launch sources the absent choices, and its decode reads
    // where the sources name (`E` = `BANK`); a resident one reads the expert
    // stacks by expert.
    let (normalized, reads, experts_read, sequence, mut sited) = match &mut streamed {
        Some(layer) => {
            let Selection { norm, router_norm, router, bias, expert_scale } = &selection;
            let BankRoute::General(route) = layer.route() else {
                return Err("a general layer's bank holds another form's route".into());
            };
            let bank = layer.numbers();
            let mut ports = layer.decode_ports(graph)?;
            let mut hints = layer.hints(
                graph,
                weights,
                [router_norm, router, bias],
                ports.policy.tensor(),
            )?;
            let mut routing = layer.part(graph, BankPart::Routing)?;
            let tier = layer.route_tier(graph, &ports, constants)?;
            let route = route[hints.ahead as usize].ok_or("a streamed layer's lead was not prepared with its binding")?;
            let select_dims = [
                dimensions.as_slice(),
                &[
                    ("FW", bank.file_width.clone()),
                    ("AHEAD", Dim::from(hints.ahead)),
                    ("FLIGHT", bank.flight.clone()),
                    ("RING", bank.ring_words.clone()),
                    ("GM", bank.map_words.clone()),
                    ("TM", tier.experts.clone()),
                    ("P", bank.choices.clone()),
                    ("EN", bank.next_experts.clone()),
                ],
            ]
            .concat();
            // The step's sources, which the selection writes and its later
            // entries read.
            let mut locals = RouteLocals { sources: graph.local_for(route, "sources", &select_dims)? };
            let selected = graph.enqueue(
                route,
                &select_dims,
                routed_select_streamed::WorkflowArgs {
                    residual: residual.into(),
                    norm: norm.into(),
                    router_norm: router_norm.into(),
                    router: router.into(),
                    bias: bias.into(),
                    expert_scale: (&expert_scale.tensor).into(),
                    routes: routes.tensor_mut().into(),
                    weights: route_weights.tensor_mut().into(),
                    epsilon,
                    score: scalar(&shape.score, "a route's score code")?,
                    normalization: scalar(&shape.normalization, "a route's normalization code")?,
                    normalization_epsilon: form.normalization_epsilon.f32(),
                    scale: form.scale.f32(),
                    map: ports.map.tensor().into(),
                    tier: (&tier.map).into(),
                    policy: ports.policy.tensor().into(),
                    sited: (ports.sited()).into(),
                    step: ports.step.tensor().into(),
                    file: ports.file.tensor_mut().into(),
                    sources: locals.sources.tensor_mut().into(),
                    next_router_norm: (&hints.weights[0][0]).into(),
                    next_router: (&hints.weights[0][1]).into(),
                    next_bias: (&hints.weights[0][2]).into(),
                    after_router_norm: (&hints.weights[1][0]).into(),
                    after_router: (&hints.weights[1][1]).into(),
                    after_bias: (&hints.weights[1][2]).into(),
                    next_policy: (&hints.policy).into(),
                    rings: hints.rings.tensor_mut().into(),
                    routing: routing.tensor_mut().into(),
                },
            )?;
            let decode_shape = DecodeShape { rows };
            let mut sources = locals.sources.tensor().clone();
            layer.land(graph, LandPorts { decode: &mut ports, sources: &mut sources, records: stacks.records() }, &decode_shape, constants)?;
            let reads = layer.reads(graph, &ports, &locals, constants)?;
            let sequence = ports.sequence.take();
            let sited = ports.site.is_some().then_some(ports);
            (selected.value, reads, bank.records.clone(), sequence, sited)
        }
        None => {
            let normalized = selection.select(
                graph,
                select,
                residual,
                [&mut routes, &mut route_weights],
                form,
                shape,
                rows,
                epsilon,
            )?;
            let stacks = RecordTensors {
                gate: stacks.gate.as_ref().map(|(gate, _)| gate.tensor()),
                up: stacks.up.tensor(),
                down: stacks.down.tensor(),
            };
            let reads = DecodeReads::resident(graph, rows, &shape.selected, stacks, constants)?;
            (normalized, reads, shape.experts.clone(), None, None)
        }
    };
    let input = expert_input(graph, latent_down, planned, scope, weights, constants, normalized, rows, form, shape)?;
    // The site request, before the gate/up: the step's CPU choices of the
    // snapshot, over the rows the experts project.
    if let (Some(layer), Some(ports)) = (&mut streamed, &mut sited) {
        layer.site_request(
            graph,
            ports,
            routes.tensor(),
            route_weights.tensor(),
            &reads.sources,
            &input,
            &DecodeShape { rows },
        )?;
    }
    let activation = scalar(&shape.activation, "an expert activation code")?;
    let mut dimensions = shape.expert_dimensions(rows);
    dimensions[2] = ("E", experts_read);
    let map_dimension = reads.dimensions();
    let up_scale = &stacks.up_scale;
    let product = match (experts, &stacks.gate) {
        (ExpertEntries::Gated { decode, .. }, Some((gate, gate_scale))) => {
            graph
                .enqueue(
                    decode,
                    &[
                        dimensions.as_slice(),
                        &[
                            ("GS", gate_scale.extent.clone()),
                            ("US", up_scale.extent.clone()),
                            ("LB", Dim::from(GluLimit::word(form.experts_expansion.limit))),
                        ],
                        &map_dimension,
                    ]
                    .concat(),
                    routed_gate_up::WorkflowArgs {
                        normalized: (&input).into(),
                        routes: routes.tensor().into(),
                        sources: (&reads.sources).into(),
                        expert_gate: gate.tensor().into(),
                        expert_up: stacks.up.tensor().into(),
                        tier_gate: (reads.tier.gate.as_ref().ok_or("a gated layer's reads lack its gate")?).into(),
                        tier_up: (&reads.tier.up).into(),
                        activation,
                        gate_scale: (&gate_scale.tensor).into(),
                        up_scale: (&up_scale.tensor).into(),
                        glu_limit: (&crate::operators::glu_limit(routes.tensor(), form.experts_expansion.limit)).into(),
                    },
                )?
                .value
        }
        (ExpertEntries::Plain { decode, .. }, None) => {
            graph
                .enqueue(
                    decode,
                    &[dimensions.as_slice(), &[("US", up_scale.extent.clone())], &map_dimension].concat(),
                    routed_up::WorkflowArgs {
                        normalized: (&input).into(),
                        routes: routes.tensor().into(),
                        sources: (&reads.sources).into(),
                        expert_up: stacks.up.tensor().into(),
                        tier_up: (&reads.tier.up).into(),
                        up_scale: (&up_scale.tensor).into(),
                        activation,
                    },
                )?
                .value
        }
        _ => return Err("expert entries disagree with its weights".into()),
    };
    let policy_words = match &reads.policy {
        Some(policy) => policy.clone(),
        None => {
            let constant = GraphConstant::i32(graph, &[0; magnitude_kernels::constants::POLICY_WORDS as usize])?;
            let tensor = constant.port().tensor().clone();
            constants.push(constant);
            tensor
        }
    };
    let resident_args = || routed_down::WorkflowArgs {
        base: (&base.onto).into(),
        product: (&product).into(),
        routes: routes.tensor().into(),
        sources: (&reads.sources).into(),
        weights: route_weights.tensor().into(),
        expert_down: stacks.down.tensor().into(),
        tier_down: (&reads.tier.down).into(),
        policy: (&policy_words).into(),
    };
    let down_dimensions = [dimensions.as_slice(), &map_dimension].concat();
    // A layer with a site combines, while its group is not whole
    // (`policy[POLICY_WHOLE]` clear), with the CPU's answer: its projections
    // (`routed_down_project`), beside the CPU, then their sum
    // (`routed_down_sited`), which waits for the answer when a choice of the
    // step is the CPU's; while its group is whole, as a resident layer does
    // (`routed_down`), at no cost and with its numerics. The two are
    // exclusive by their conditions, so one storage holds the combine
    // (`GraphDraft::share`). Any other layer combines resident,
    // unconditioned.
    let combine = match (streamed.as_mut(), sited) {
        (Some(layer), Some(mut ports)) => {
            ports.site.take().zip(layer.site_combine()).map(|(site, combine)| (layer, site, combine, ports))
        }
        _ => None,
    };
    let routed = match combine {
        Some((layer, mut site, (BankCombineEntry::General { project, sited }, site_dimensions), ports)) => {
            let (policy, whole) = (ports.policy, magnitude_kernels::constants::POLICY_WHOLE);
            let resident = graph
                .enqueue_under(down, &down_dimensions, resident_args(), &policy, whole, LaunchEncoding::WhileSet)?
                .value;
            let published = graph
                .enqueue_under(
                    project,
                    &[dimensions.as_slice(), &map_dimension].concat(),
                    routed_down_project::WorkflowArgs {
                        product: (&product).into(),
                        routes: routes.tensor().into(),
                        sources: (&reads.sources).into(),
                        expert_down: stacks.down.tensor().into(),
                        tier_down: (&reads.tier.down).into(),
                    },
                    &policy,
                    whole,
                    LaunchEncoding::WhileClear,
                )?
                .value;
            layer.park_site(graph);
            let summed = graph
                .enqueue_under(
                    sited,
                    &[
                        &[dimensions[0].clone(), dimensions[1].clone(), dimensions[3].clone(), dimensions[4].clone()],
                        site_dimensions.as_slice(),
                    ]
                    .concat(),
                    routed_down_sited::WorkflowArgs {
                        base: (&base.onto).into(),
                        published: (&published).into(),
                        sources: (&reads.sources).into(),
                        weights: route_weights.tensor().into(),
                        step: ports.step.tensor().into(),
                        site: site.table.tensor_mut().into(),
                        site_output: site.output.tensor().into(),
                    },
                    &policy,
                    whole,
                    LaunchEncoding::WhileClear,
                )?
                .value;
            graph.share(&summed, &resident)?;
            resident
        }
        Some((_, _, (BankCombineEntry::Fused { .. }, _), _)) => {
            return Err("a general layer's bank holds another form's combine".into());
        }
        None => graph.enqueue(down, &down_dimensions, resident_args())?.value,
    };
    let output = base.finish(graph, latent_up, planned, scope, weights, constants, routed, rows, shape)?;
    Ok((output, sequence))
}

/// A merge set's pass over a general layer of `scope` (`OnceStage::Pass`): the
/// experts of the crossed rows the merged count names from the merged first
/// row, of the pass's `rows`, read where they live (`grouped::pass`), into
/// the merged ports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn general_pass<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: GeneralRoutedGraphEntries<'a, G>,
    bank: LayerBank<'_, 'a, G>,
    topology: &GeneralRoutedTopology,
    numbers: &GeneralRoutedLayer,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    ports: &mut PassPorts,
    blocks: &Dim,
    rows: &Dim,
) -> Result<WorkflowTensor, GraphError> {
    let (form, shape) = (&topology.form, &numbers.shape);
    let part = bank.stacks();
    let mut layer = StreamedLayer::of(planned, &topology.bank, numbers.bank.as_ref(), scope, bank)?;
    let stacks = ExpertStacks::of(graph, planned, scope, weights, constants, form, part)?;
    super::grouped::pass(
        graph,
        entries.group,
        layer.as_mut(),
        ports,
        blocks,
        stacks.tensors(),
        |tier| Ok(tier.as_ref().map(|_, port| port.tensor().clone())),
        rows,
        [&shape.experts, &shape.selected],
        constants,
        |graph, grouped| expand(graph, entries.experts, &stacks, grouped, form, shape),
    )
}

