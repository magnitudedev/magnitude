//! Recurrent (gated delta net) block graph: normed projection, the in-place
//! state advance publishing the gated rows, and the plain residual output
//! projection (`attention_output`). On a backend of the convolved
//! [`StepForm`], the step row classes' projection also convolves and
//! publishes the successor windows, and their step advances from the
//! convolved channels. State never moves: the
//! layer's window, delta and tape arenas are bound as ports, and per-slot
//! tables select which version (bank and tape rows) each slot reads and which
//! bank it publishes to.

use crate::operators::{Join, RecurrentBank, SublayerInput, scalar, split_bank};
use crate::programs::graph::facts::Numbers;
use seismic::Argument;
use crate::programs::graph::{
    GraphError,
    draft::{ControlSource, GraphDraft},
};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    PackingBound, PlannedWeights, WeightPort, packed_weight, weight,
};
use crate::{
    RecurrentBinding, StateResourcePlan,
    native::{RecurrentKernels, RecurrentStepKernels},
};
use magnitude_family_contracts::{
    GatedDelta, RecurrentHeadMapping, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    attention_output, gated_delta_chunk, gated_delta_project, gated_delta_project_convolved,
    gated_delta_step, gated_delta_step_convolved,
};
use seismic::{
    BackendName, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor,
};

/// How a backend advances the row-sequential (step) row classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StepForm {
    /// `gated_delta_project`, then `gated_delta_step`, which convolves and
    /// publishes the successor windows.
    Step,
    /// `gated_delta_project_convolved`, which also convolves and publishes the
    /// successor windows, then `gated_delta_step_convolved`.
    Convolved,
}

impl StepForm {
    /// The convolved form wherever the checked bundle declares both of its
    /// entries for `backend`, else the step form.
    pub(crate) fn of(backend: BackendName) -> Result<Self, String> {
        let project = seismic::generated::native_implementation_for_backend::<
            gated_delta_project_convolved::Entry,
        >(backend)
        .map_err(|error| error.to_string())?;
        let step = seismic::generated::native_implementation_for_backend::<
            gated_delta_step_convolved::Entry,
        >(backend)
        .map_err(|error| error.to_string())?;
        match (project, step) {
            (Some(_), Some(_)) => Ok(Self::Convolved),
            (None, None) => Ok(Self::Step),
            _ => Err(format!(
                "backend {backend:?} declares only one entry of the convolved recurrent step form"
            )),
        }
    }
}

pub(crate) struct RecurrentGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, gated_delta_project::Entry>,
    pub step: RecurrentStepEntries<'a, G>,
    pub chunk: G::Binding<'a, gated_delta_chunk::Entry>,
    pub output: G::Binding<'a, attention_output::Entry>,
}

/// The step row classes' entries of a [`StepForm`].
pub(crate) enum RecurrentStepEntries<'a, G: GraphDraft + 'a> {
    Step(G::Binding<'a, gated_delta_step::Entry>),
    Convolved {
        project: G::Binding<'a, gated_delta_project_convolved::Entry>,
        step: G::Binding<'a, gated_delta_step_convolved::Entry>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for RecurrentStepEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for RecurrentStepEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a RecurrentKernels> for RecurrentGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a RecurrentKernels) -> Self {
        Self {
            project: &kernels.project,
            step: match &kernels.step {
                RecurrentStepKernels::Step(step) => RecurrentStepEntries::Step(step),
                RecurrentStepKernels::Convolved { project, step } => {
                    RecurrentStepEntries::Convolved { project, step }
                }
            },
            chunk: &kernels.chunk,
            output: &kernels.output,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedRecurrentEntries {
    project: [(&'static str, Element); 6],
    state: [(&'static str, Element); 2],
    form: StepForm,
    output: [(&'static str, Element); 2],
}

impl CheckedRecurrentEntries {
    /// The entries of `binding` on `backend` (its [`StepForm`]).
    pub(crate) fn new(binding: RecurrentBinding, backend: BackendName) -> Result<Self, String> {
        Ok(Self {
            project: [
                ("NW", binding.norm),
                ("QW", binding.qkv),
                ("GW", binding.gate),
                ("AW", binding.alpha),
                ("BW", binding.beta),
                ("A", binding.activation),
            ],
            state: [("RN", binding.recurrent_norm), ("A", binding.activation)],
            form: StepForm::of(backend)?,
            output: [("OW", binding.output), ("A", binding.activation)],
        })
    }

    pub(crate) fn entries(&self) -> RecurrentGraphEntries<'_, NativeGraphMetadata> {
        RecurrentGraphEntries {
            project: &self.project,
            step: match self.form {
                StepForm::Step => RecurrentStepEntries::Step(&self.state[..]),
                StepForm::Convolved => RecurrentStepEntries::Convolved {
                    project: &self.project[..],
                    step: &self.state[..],
                },
            },
            chunk: &self.state,
            output: &self.output,
        }
    }
}

/// Row classes at or above this size advance state with the chunked entry;
/// smaller classes use the row-sequential step. The chunked entry advances
/// slots of at most 16 rows row-sequentially too (with the step's bits), so a
/// class of at most 16 rows gains nothing from it and takes the step.
pub(crate) const CHUNKED_ROWS: u64 = 17;

/// Whether a `rows`-row class advances state with the chunked entry.
pub(crate) fn chunked(rows: &Dim) -> bool {
    rows.at_least(CHUNKED_ROWS)
}

/// The slab banks a recurrent entry takes as its scalar argument: a trace
/// holds for this count alone.
pub(crate) fn slab_banks<const N: usize>(bank: &RecurrentBank<Dim, N>) -> Result<Argument<u32>, GraphError> {
    Ok(bank.slab_banks.word().ok_or("bank slab count exceeds u32")?)
}

/// Ports over a recurrent bank's components, of `elements`, each of the
/// store's banks of the component's extents.
pub(crate) fn bank_ports<G: GraphDraft, const N: usize>(
    graph: &mut G,
    elements: &[Element; N],
    bank: &RecurrentBank<Dim, N>,
) -> Result<[NativePort; N], GraphError> {
    let mut ports = Vec::with_capacity(N);
    for (index, &element) in elements.iter().enumerate() {
        ports.push(graph.port(element, &bank.extents(index))?);
    }
    Ok(ports
        .try_into()
        .unwrap_or_else(|_| unreachable!("one port per component")))
}

/// The layer's recurrent arenas, bound to the state store's tensors per run.
#[derive(Clone)]
pub(crate) struct RecurrentStatePorts {
    pub window: NativePort,
    pub delta: NativePort,
    pub tape: NativePort,
}

impl RecurrentStatePorts {
    /// The ports of a three-component bank (window, delta or state, tape)
    /// and its tape's rows.
    pub(crate) fn of<G: GraphDraft>(
        graph: &mut G,
        elements: &[Element; 3],
        bank: &RecurrentBank<Dim, 3>,
    ) -> Result<(Self, Dim), GraphError> {
        let [window, delta, tape] = bank_ports(graph, elements, bank)?;
        let tape_rows = bank.components[2]
            .first()
            .cloned()
            .ok_or("recurrent tape component has no row axis")?;
        Ok((Self { window, delta, tape }, tape_rows))
    }
}

/// Per-run slot tables: row segments, published row counts, the version each
/// slot reads (bank and tape rows) and the bank it publishes to.
#[derive(Clone)]
pub(crate) struct RecurrentControlPorts {
    pub segments: NativePort,
    pub stop: NativePort,
    pub previous_bank: NativePort,
    pub previous_tape: NativePort,
    pub following_bank: NativePort,
}

/// What decides a recurrent block's graph beside its entries: no model
/// number. Its reals `R`: the model's (`f64`), a key's (`RealIndex`), or
/// its builder's arguments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RecurrentTopology<R = Argument<f64>> {
    /// Whether key heads are grouped (`RecurrentHeadMapping::Grouped`): the
    /// entries' `grouped` argument, a flag no number can carry unguarded.
    pub grouped: bool,
    pub epsilon: R,
    /// The elements of the layer's window, delta and tape components.
    pub bank: [Element; 3],
    /// How the graph takes its launch's slot tables.
    pub controls: ControlSource,
}

impl<R> Numbers<R> for RecurrentTopology<R> {
    type Of<E> = RecurrentTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> RecurrentTopology<E> {
        RecurrentTopology {
            grouped: self.grouped,
            epsilon: f(self.epsilon),
            bank: self.bank,
            controls: self.controls,
        }
    }
}

/// The numbers of one recurrent block at one graph class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RecurrentBlock<D = Dim> {
    pub rows: D,
    pub hidden: D,
    pub slots: D,
    pub key_heads: D,
    pub value_heads: D,
    pub width: D,
    pub convolution_width: D,
    /// The output gate's code (`gated_delta::gate_function`), the entries'
    /// `gate_function` argument.
    pub gate_function: D,
    pub bank: RecurrentBank<D, 3>,
}

impl<D> Numbers<D> for RecurrentBlock<D> {
    type Of<E> = RecurrentBlock<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> RecurrentBlock<E> {
        RecurrentBlock {
            rows: f(self.rows),
            hidden: f(self.hidden),
            slots: f(self.slots),
            key_heads: f(self.key_heads),
            value_heads: f(self.value_heads),
            width: f(self.width),
            convolution_width: f(self.convolution_width),
            gate_function: f(self.gate_function),
            bank: self.bank.map(f),
        }
    }
}

/// The topology and numbers of the recurrent block of `delta` over a
/// `hidden`-wide residual at a class of `rows` rows and `slots` slots, its
/// bank `state`'s components from `component_index` on.
pub(crate) fn split(
    delta: &GatedDelta,
    hidden: u64,
    rows: u64,
    slots: u64,
    epsilon: f32,
    state: &StateResourcePlan,
    component_index: usize,
    controls: ControlSource,
) -> Result<(RecurrentTopology<f64>, RecurrentBlock<u64>), GraphError> {
    let (elements, bank) = split_bank(state, component_index)?;
    Ok((
        RecurrentTopology {
            grouped: matches!(delta.head_mapping, RecurrentHeadMapping::Grouped),
            epsilon: f64::from(epsilon),
            bank: elements,
            controls,
        },
        RecurrentBlock {
            rows,
            hidden,
            slots,
            key_heads: delta.key_heads,
            value_heads: delta.value_heads,
            width: delta.width,
            convolution_width: delta.convolution_width,
            gate_function: crate::operators::code(super::gate_function(delta)),
            bank,
        },
    ))
}

pub(crate) fn recurrent<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: RecurrentGraphEntries<'a, G>,
    topology: &RecurrentTopology,
    block: &RecurrentBlock,
    weights: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    input: SublayerInput<'_>,
    residual: &WorkflowTensor,
    join: &Join,
) -> Result<(WorkflowTensor, RecurrentStatePorts, RecurrentControlPorts), GraphError> {
    let chunked = chunked(&block.rows);
    // The packed projections' bounds. The convolved decode form has no
    // packed form and reads none, so the input projection's are only where
    // `gated_delta_project` runs (an unread port is unbound).
    let projects = chunked || matches!(kernels.step, RecurrentStepEntries::Step(_));
    let mut bound = |graph: &mut G, kind| {
        PackingBound::planned(graph, weights, WeightRole { scope, kind }, ports, constants)
    };
    let projection_bounds = match projects {
        true => Some((
            bound(graph, WeightKind::RecurrentQueryKeyValue)?,
            bound(graph, WeightKind::RecurrentGate)?,
        )),
        false => None,
    };
    let output = packed_weight(
        graph,
        weights,
        scope,
        WeightKind::RecurrentOutput,
        ports,
        constants,
    )?;
    // A mix site's partials have no input norm: the port is empty.
    let input_norm = if input.form.normalized {
        weight(graph, weights, scope, WeightKind::InputNorm, ports)?
    } else {
        GraphConstant::absent(graph, constants)?.tensor().clone()
    };
    let mut weight = |graph: &mut G, kind| weight(graph, weights, scope, kind, ports);
    let qkv_weight = weight(graph, WeightKind::RecurrentQueryKeyValue)?;
    let gate_weight = weight(graph, WeightKind::RecurrentGate)?;
    let alpha_weight = weight(graph, WeightKind::RecurrentAlpha)?;
    let beta_weight = weight(graph, WeightKind::RecurrentBeta)?;
    let convolution = weight(graph, WeightKind::RecurrentConvolution)?;
    let rate = weight(graph, WeightKind::RecurrentDecay)?;
    let time_bias = weight(graph, WeightKind::RecurrentTimeBias)?;
    let recurrent_norm = weight(graph, WeightKind::RecurrentNorm)?;

    let (
        RecurrentStatePorts {
            mut window,
            mut delta,
            mut tape,
        },
        tape_rows,
    ) = RecurrentStatePorts::of(graph, &topology.bank, &block.bank)?;
    let banks = &block.bank.banks;
    let slab_banks = slab_banks(&block.bank)?;
    let gate_function = scalar::<i32>(&block.gate_function, "a recurrent gate code")?;

    let [parts, normalized] = input.form.dimensions(block.hidden.clone());
    let project = |graph: &mut G| {
        let Some((qkv_bound, gate_bound)) = &projection_bounds else {
            return Err("the recurrent projection's packing bounds are absent".into());
        };
        graph
            .enqueue(
                kernels.project,
                &[
                    ("M", block.rows.clone()),
                    ("H", block.hidden.clone()),
                    ("NK", block.key_heads.clone()),
                    ("NV", block.value_heads.clone()),
                    ("W", block.width.clone()),
                    parts.clone(),
                    normalized.clone(),
                    ("QB", qkv_bound.extent.clone()),
                    ("ZB", gate_bound.extent.clone()),
                ],
                gated_delta_project::WorkflowArgs {
                    hidden: input.rows.into(),
                    input_norm: (&input_norm).into(),
                    qkv_weight: (&qkv_weight).into(),
                    gate_weight: (&gate_weight).into(),
                    alpha_weight: (&alpha_weight).into(),
                    beta_weight: (&beta_weight).into(),
                    epsilon: topology.epsilon.f32(),
                    qkv_bound: (&qkv_bound.tensor).into(),
                    gate_bound: (&gate_bound.tensor).into(),
                },
            )
            .map(|projection| projection.value)
    };
    let dimensions = [
        ("M", &block.rows),
        ("B", &block.slots),
        ("S", banks),
        ("NK", &block.key_heads),
        ("NV", &block.value_heads),
        ("W", &block.width),
        ("C", &block.convolution_width),
        ("T", &tape_rows),
    ];
    // Every entry that binds the slot tables gives them the chunk's geometry,
    // whichever advances this class.
    let table = |graph: &mut G, name: &str| {
        graph.control_for(topology.controls, kernels.chunk, name, &dimensions)
    };
    let controls = RecurrentControlPorts {
        segments: table(graph, "segments")?,
        stop: table(graph, "stop")?,
        previous_bank: table(graph, "previous_bank")?,
        previous_tape: table(graph, "previous_tape")?,
        following_bank: table(graph, "following_bank")?,
    };
    // The L2-norm epsilon of the q/k prologue, scaled as the model defines
    // it: a scalar argument, so the trace holds for this width alone.
    let norm_epsilon = (topology.epsilon * block.width.real()).f32();
    let gated = if chunked {
        let projection = project(graph)?;
        graph
            .enqueue(
                kernels.chunk,
                &dimensions,
                gated_delta_chunk::WorkflowArgs {
                    projection: (&projection).into(),
                    convolution: (&convolution).into(),
                    rate: (&rate).into(),
                    time_bias: (&time_bias).into(),
                    recurrent_norm: (&recurrent_norm).into(),
                    segments: controls.segments.tensor().into(),
                    stop: controls.stop.tensor().into(),
                    previous_bank: controls.previous_bank.tensor().into(),
                    previous_tape: controls.previous_tape.tensor().into(),
                    following_bank: controls.following_bank.tensor().into(),
                    window: window.tensor_mut().into(),
                    delta: delta.tensor_mut().into(),
                    tape: tape.tensor_mut().into(),
                    norm_epsilon,
                    epsilon: topology.epsilon.f32(),
                    grouped: topology.grouped.into(),
                    gate_function,
                    slab_banks,
                },
            )?
            .value
    } else {
        match kernels.step {
            RecurrentStepEntries::Step(step) => {
                let projection = project(graph)?;
                graph
                    .enqueue(
                        step,
                        &dimensions,
                        gated_delta_step::WorkflowArgs {
                            projection: (&projection).into(),
                            convolution: (&convolution).into(),
                            rate: (&rate).into(),
                            time_bias: (&time_bias).into(),
                            recurrent_norm: (&recurrent_norm).into(),
                            segments: controls.segments.tensor().into(),
                            stop: controls.stop.tensor().into(),
                            previous_bank: controls.previous_bank.tensor().into(),
                            previous_tape: controls.previous_tape.tensor().into(),
                            following_bank: controls.following_bank.tensor().into(),
                            window: window.tensor_mut().into(),
                            delta: delta.tensor_mut().into(),
                            tape: tape.tensor_mut().into(),
                            norm_epsilon,
                            epsilon: topology.epsilon.f32(),
                            grouped: topology.grouped.into(),
                            gate_function,
                            slab_banks,
                        },
                    )?
                    .value
            }
            RecurrentStepEntries::Convolved { project, step } => {
                // The projection launch convolves and publishes the successor
                // windows; the step advances from its convolved channels.
                let projected = graph.enqueue(
                    project,
                    &[
                        ("M", block.rows.clone()),
                        ("B", block.slots.clone()),
                        ("S", banks.clone()),
                        ("H", block.hidden.clone()),
                        ("NK", block.key_heads.clone()),
                        ("NV", block.value_heads.clone()),
                        ("W", block.width.clone()),
                        ("C", block.convolution_width.clone()),
                        ("T", tape_rows.clone()),
                        parts,
                        normalized,
                    ],
                    gated_delta_project_convolved::WorkflowArgs {
                        hidden: input.rows.into(),
                        input_norm: (&input_norm).into(),
                        qkv_weight: (&qkv_weight).into(),
                        gate_weight: (&gate_weight).into(),
                        alpha_weight: (&alpha_weight).into(),
                        beta_weight: (&beta_weight).into(),
                        convolution: (&convolution).into(),
                        segments: controls.segments.tensor().into(),
                        stop: controls.stop.tensor().into(),
                        previous_bank: controls.previous_bank.tensor().into(),
                        previous_tape: controls.previous_tape.tensor().into(),
                        following_bank: controls.following_bank.tensor().into(),
                        window: window.tensor_mut().into(),
                        epsilon: topology.epsilon.f32(),
                        slab_banks,
                    },
                )?;
                graph
                    .enqueue(
                        step,
                        &[
                            ("M", &block.rows),
                            ("B", &block.slots),
                            ("S", banks),
                            ("NK", &block.key_heads),
                            ("NV", &block.value_heads),
                            ("W", &block.width),
                            ("T", &tape_rows),
                        ],
                        gated_delta_step_convolved::WorkflowArgs {
                            projection: (&projected.r0).into(),
                            convolved: (&projected.r1).into(),
                            rate: (&rate).into(),
                            time_bias: (&time_bias).into(),
                            recurrent_norm: (&recurrent_norm).into(),
                            segments: controls.segments.tensor().into(),
                            stop: controls.stop.tensor().into(),
                            previous_bank: controls.previous_bank.tensor().into(),
                            previous_tape: controls.previous_tape.tensor().into(),
                            following_bank: controls.following_bank.tensor().into(),
                            delta: delta.tensor_mut().into(),
                            tape: tape.tensor_mut().into(),
                            norm_epsilon,
                            epsilon: topology.epsilon.f32(),
                            grouped: topology.grouped.into(),
                            gate_function,
                            slab_banks,
                        },
                    )?
                    .value
            }
        }
    };
    let inject = join.inject(graph, &block.rows, constants)?;
    let [streams, injected] = join.dimensions();
    let output = graph
        .enqueue(
            kernels.output,
            &[
                ("M", block.rows.clone()),
                ("D", block.hidden.clone()),
                streams,
                injected,
                ("Q", block.value_heads.clone()),
                ("W", block.width.clone()),
                ("OS", output.operands.scale.extent.clone()),
                ("OB", output.operands.bound.extent.clone()),
            ],
            attention_output::WorkflowArgs {
                hidden: residual.into(),
                inject: (&inject).into(),
                gated: (&gated).into(),
                output_weight: (&output.weight).into(),
                output_scale: (&output.operands.scale.tensor).into(),
                output_bound: (&output.operands.bound.tensor).into(),
            },
        )?
        .value;
    Ok((
        output,
        RecurrentStatePorts {
            window,
            delta,
            tape,
        },
        controls,
    ))
}
