//! Recurrent (gated delta rule) block graph: normed projection, the in-place
//! state advance publishing the gated rows, and the plain residual output
//! projection (`attention_output`). A Gated DeltaNet layer projects with
//! `gated_delta_project` and advances with the `gated_delta_*` entries; on a
//! backend of the convolved [`StepForm`], the step row classes' projection
//! also convolves and publishes the successor windows, and their step
//! advances from the convolved channels. A Kimi Delta Attention layer
//! projects with `attention_project` (q | k | v, the gate's and decay's down
//! products and beta), expands the down products with `project_rows` and
//! advances with the `kimi_delta_*` entries. State never moves: the
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
    PackedWeight, PackingBound, PlannedWeights, WeightPort, packed_weight, weight,
};
use crate::{
    RecurrentBinding, RecurrentForm, StateResourcePlan,
    native::{RecurrentFormKernels, RecurrentKernels, RecurrentStepKernels},
};
use magnitude_family_contracts::{
    GatedDelta, RecurrentHeadMapping, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    attention_output, attention_project, gated_delta_chunk, gated_delta_project,
    gated_delta_project_convolved, gated_delta_step, gated_delta_step_convolved, kimi_delta_chunk,
    kimi_delta_step, project_rows,
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
    pub form: RecurrentFormEntries<'a, G>,
    pub output: G::Binding<'a, attention_output::Entry>,
}

/// The entries of a layer's [`RecurrentForm`].
pub(crate) enum RecurrentFormEntries<'a, G: GraphDraft + 'a> {
    /// Gated DeltaNet.
    Gated {
        project: G::Binding<'a, gated_delta_project::Entry>,
        step: RecurrentStepEntries<'a, G>,
        chunk: G::Binding<'a, gated_delta_chunk::Entry>,
    },
    /// Kimi Delta Attention: the projection, the gate's and the decay's up
    /// projections, and the state entries.
    Kimi {
        project: G::Binding<'a, attention_project::Entry>,
        gate_up: G::Binding<'a, project_rows::Entry>,
        alpha_up: G::Binding<'a, project_rows::Entry>,
        step: G::Binding<'a, kimi_delta_step::Entry>,
        chunk: G::Binding<'a, kimi_delta_chunk::Entry>,
    },
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
            form: match &kernels.form {
                RecurrentFormKernels::Gated {
                    project,
                    step,
                    chunk,
                } => RecurrentFormEntries::Gated {
                    project,
                    step: match step {
                        RecurrentStepKernels::Step(step) => RecurrentStepEntries::Step(step),
                        RecurrentStepKernels::Convolved { project, step } => {
                            RecurrentStepEntries::Convolved { project, step }
                        }
                    },
                    chunk,
                },
                RecurrentFormKernels::Kimi(kimi) => RecurrentFormEntries::Kimi {
                    project: &kimi.project,
                    gate_up: &kimi.gate_up,
                    alpha_up: &kimi.alpha_up,
                    step: &kimi.step,
                    chunk: &kimi.chunk,
                },
            },
            output: &kernels.output,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedRecurrentEntries {
    form: CheckedForm,
    state: [(&'static str, Element); 2],
    output: [(&'static str, Element); 2],
}

/// The element bindings of a [`RecurrentForm`]'s projection entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedForm {
    Gated {
        project: [(&'static str, Element); 6],
        step: StepForm,
    },
    Kimi {
        project: [(&'static str, Element); 7],
        up: [[(&'static str, Element); 3]; 2],
    },
}

impl CheckedRecurrentEntries {
    /// The entries of `binding` on `backend` (a Gated DeltaNet layer's of its
    /// [`StepForm`]).
    pub(crate) fn new(binding: RecurrentBinding, backend: BackendName) -> Result<Self, String> {
        let form = match binding.form {
            RecurrentForm::PerHead => CheckedForm::Gated {
                project: [
                    ("NW", binding.norm),
                    ("QW", binding.qkv),
                    ("GW", binding.gate),
                    ("AW", binding.alpha),
                    ("BW", binding.beta),
                    ("A", binding.activation),
                ],
                step: StepForm::of(backend)?,
            },
            // The projection's segments q | k | v, the gate's and decay's
            // down products and beta, published in A, the state entries'
            // inputs, as the up projections publish theirs.
            RecurrentForm::PerChannel { gate_up, alpha_up, .. } => CheckedForm::Kimi {
                project: [
                    ("NW", binding.norm),
                    ("QW", binding.qkv),
                    ("GW", binding.gate),
                    ("KW", binding.alpha),
                    ("VW", binding.beta),
                    ("A", binding.activation),
                    ("Y", binding.activation),
                ],
                up: [gate_up, alpha_up].map(|weight| {
                    [("A", binding.activation), ("W", weight), ("Y", binding.activation)]
                }),
            },
        };
        Ok(Self {
            form,
            state: [("RN", binding.recurrent_norm), ("A", binding.activation)],
            output: [("OW", binding.output), ("A", binding.activation)],
        })
    }

    pub(crate) fn entries(&self) -> RecurrentGraphEntries<'_, NativeGraphMetadata> {
        RecurrentGraphEntries {
            form: match &self.form {
                CheckedForm::Gated { project, step } => RecurrentFormEntries::Gated {
                    project,
                    step: match step {
                        StepForm::Step => RecurrentStepEntries::Step(&self.state[..]),
                        StepForm::Convolved => RecurrentStepEntries::Convolved {
                            project: &project[..],
                            step: &self.state[..],
                        },
                    },
                    chunk: &self.state,
                },
                CheckedForm::Kimi {
                    project,
                    up: [gate_up, alpha_up],
                } => RecurrentFormEntries::Kimi {
                    project,
                    gate_up,
                    alpha_up,
                    step: &self.state,
                    chunk: &self.state,
                },
            },
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
    /// The q/k L2-norm epsilon.
    pub key_norm_epsilon: R,
    /// A Kimi Delta Attention layer's decay lower bound
    /// (`gated_delta::decay_lower`), the `kimi_delta_*` entries'
    /// `decay_lower`: none for Gated DeltaNet.
    pub decay_lower: Option<R>,
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
            key_norm_epsilon: f(self.key_norm_epsilon),
            decay_lower: self.decay_lower.map(&mut *f),
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
    /// A Kimi Delta Attention layer's gate and decay ranks
    /// (`gated_delta::ranks`): none for Gated DeltaNet.
    pub ranks: Option<[D; 2]>,
    pub bank: RecurrentBank<D, 3>,
}

impl RecurrentBlock {
    /// The convolved q | k | v channels (`CH`).
    fn channels(&self) -> Dim {
        (&self.key_heads * 2 + &self.value_heads) * &self.width
    }
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
            ranks: self.ranks.map(|ranks| ranks.map(&mut *f)),
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
            key_norm_epsilon: delta.key_norm_epsilon,
            decay_lower: super::decay_lower(delta).map(f64::from),
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
            ranks: super::ranks(delta),
            bank,
        },
    ))
}

/// A layer's projection entries and weights, by its form.
enum Projected<'a, G: GraphDraft + 'a> {
    Gated {
        project: G::Binding<'a, gated_delta_project::Entry>,
        step: RecurrentStepEntries<'a, G>,
        chunk: G::Binding<'a, gated_delta_chunk::Entry>,
        /// The packed projections' bounds, where `gated_delta_project` runs.
        bounds: Option<(PackingBound, PackingBound)>,
        qkv: WorkflowTensor,
        gate: WorkflowTensor,
        alpha: WorkflowTensor,
        beta: WorkflowTensor,
    },
    Kimi {
        project: G::Binding<'a, attention_project::Entry>,
        gate_up: G::Binding<'a, project_rows::Entry>,
        alpha_up: G::Binding<'a, project_rows::Entry>,
        step: G::Binding<'a, kimi_delta_step::Entry>,
        chunk: G::Binding<'a, kimi_delta_chunk::Entry>,
        qkv: PackedWeight,
        gate: PackedWeight,
        alpha: PackedWeight,
        beta: PackedWeight,
    },
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
    // GDN's packed projections' bounds. The convolved decode form has no
    // packed form and reads none, so the input projection's are only where
    // `gated_delta_project` runs (an unread port is unbound).
    let projects = match kernels.form {
        RecurrentFormEntries::Gated { step, .. } => {
            chunked || matches!(step, RecurrentStepEntries::Step(_))
        }
        RecurrentFormEntries::Kimi { .. } => false,
    };
    let mut bound = |graph: &mut G, kind| {
        PackingBound::planned(graph, weights, WeightRole { scope, kind }, ports, constants)
    };
    let mut projection_bounds = match projects {
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
    let projected = match kernels.form {
        RecurrentFormEntries::Gated {
            project,
            step,
            chunk,
        } => {
            let mut weight = |graph: &mut G, kind| weight(graph, weights, scope, kind, ports);
            Projected::Gated {
                project,
                step,
                chunk,
                bounds: projection_bounds.take(),
                qkv: weight(graph, WeightKind::RecurrentQueryKeyValue)?,
                gate: weight(graph, WeightKind::RecurrentGate)?,
                alpha: weight(graph, WeightKind::RecurrentAlpha)?,
                beta: weight(graph, WeightKind::RecurrentBeta)?,
            }
        }
        RecurrentFormEntries::Kimi {
            project,
            gate_up,
            alpha_up,
            step,
            chunk,
        } => {
            let mut weight =
                |graph: &mut G, kind| packed_weight(graph, weights, scope, kind, ports, constants);
            Projected::Kimi {
                project,
                gate_up,
                alpha_up,
                step,
                chunk,
                qkv: weight(graph, WeightKind::RecurrentQueryKeyValue)?,
                gate: weight(graph, WeightKind::RecurrentGate)?,
                alpha: weight(graph, WeightKind::RecurrentAlpha)?,
                beta: weight(graph, WeightKind::RecurrentBeta)?,
            }
        }
    };
    let mut weight = |graph: &mut G, kind| weight(graph, weights, scope, kind, ports);
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
    let table = |graph: &mut G, name: &str| match &projected {
        Projected::Gated { chunk, .. } => {
            graph.control_for(topology.controls, *chunk, name, &dimensions)
        }
        Projected::Kimi { chunk, .. } => {
            graph.control_for(topology.controls, *chunk, name, &dimensions)
        }
    };
    let controls = RecurrentControlPorts {
        segments: table(graph, "segments")?,
        stop: table(graph, "stop")?,
        previous_bank: table(graph, "previous_bank")?,
        previous_tape: table(graph, "previous_tape")?,
        following_bank: table(graph, "following_bank")?,
    };
    // The L2-norm epsilon of the q/k prologue: a scalar argument, so the
    // trace holds for this value alone.
    let norm_epsilon = topology.key_norm_epsilon.f32();
    let gated = match projected {
        Projected::Gated {
            project: project_entry,
            step,
            chunk,
            bounds,
            qkv: qkv_weight,
            gate: gate_weight,
            alpha: alpha_weight,
            beta: beta_weight,
        } => {
            let project = |graph: &mut G| {
                let Some((qkv_bound, gate_bound)) = &bounds else {
                    return Err("the recurrent projection's packing bounds are absent".into());
                };
                graph
                    .enqueue(
                        project_entry,
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
            if chunked {
                let projection = project(graph)?;
                graph
                    .enqueue(
                        chunk,
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
                match step {
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
                        // The projection launch convolves and publishes the
                        // successor windows; the step advances from its
                        // convolved channels.
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
            }
        }
        Projected::Kimi {
            project,
            gate_up,
            alpha_up,
            step,
            chunk,
            qkv,
            gate,
            alpha,
            beta,
        } => {
            let (Some([gate_rank, alpha_rank]), Some(decay_lower)) =
                (&block.ranks, &topology.decay_lower)
            else {
                unreachable!("a per-channel layer's block has its ranks and decay lower bound")
            };
            // q | k | v, the gate's and decay's down products and beta, in A.
            let projection = graph.enqueue(
                project,
                &[
                    ("M", block.rows.clone()),
                    ("D", block.hidden.clone()),
                    ("Q", block.channels()),
                    ("GR", gate_rank.clone()),
                    ("K", alpha_rank.clone()),
                    ("V", block.value_heads.clone()),
                    ("QS", qkv.operands.scale.extent.clone()),
                    ("GS", gate.operands.scale.extent.clone()),
                    ("KS", alpha.operands.scale.extent.clone()),
                    ("VS", beta.operands.scale.extent.clone()),
                    ("QB", qkv.operands.bound.extent.clone()),
                    ("GB", gate.operands.bound.extent.clone()),
                    ("KB", alpha.operands.bound.extent.clone()),
                    ("VB", beta.operands.bound.extent.clone()),
                    parts,
                    normalized,
                ],
                attention_project::WorkflowArgs {
                    hidden: input.rows.into(),
                    input_norm: (&input_norm).into(),
                    query_weight: (&qkv.weight).into(),
                    gate_weight: (&gate.weight).into(),
                    key_weight: (&alpha.weight).into(),
                    value_weight: (&beta.weight).into(),
                    epsilon: topology.epsilon.f32(),
                    project_mode: 0.into(),
                    query_scale: (&qkv.operands.scale.tensor).into(),
                    gate_scale: (&gate.operands.scale.tensor).into(),
                    key_scale: (&alpha.operands.scale.tensor).into(),
                    value_scale: (&beta.operands.scale.tensor).into(),
                    query_bound: (&qkv.operands.bound.tensor).into(),
                    gate_bound: (&gate.operands.bound.tensor).into(),
                    key_bound: (&alpha.operands.bound.tensor).into(),
                    value_bound: (&beta.operands.bound.tensor).into(),
                },
            )?;
            // The gate and decay inputs: their down products through the up
            // projections, in A.
            let gate_up_weight =
                packed_weight(graph, weights, scope, WeightKind::RecurrentGateUp, ports, constants)?;
            let alpha_up_weight =
                packed_weight(graph, weights, scope, WeightKind::RecurrentAlphaUp, ports, constants)?;
            let expand = |graph: &mut G,
                              entry: G::Binding<'a, project_rows::Entry>,
                              source: &WorkflowTensor,
                              rank: &Dim,
                              up: &PackedWeight| {
                graph
                    .enqueue(
                        entry,
                        &[
                            ("M", block.rows.clone()),
                            ("G", Dim::from(1)),
                            ("K", rank.clone()),
                            ("N", &block.value_heads * &block.width),
                            ("WS", up.operands.scale.extent.clone()),
                            ("WB", up.operands.bound.extent.clone()),
                        ],
                        project_rows::WorkflowArgs {
                            source: source.into(),
                            weight: (&up.weight).into(),
                            weight_scale: (&up.operands.scale.tensor).into(),
                            weight_bound: (&up.operands.bound.tensor).into(),
                        },
                    )
                    .map(|expanded| expanded.value)
            };
            let gate_input = expand(graph, gate_up, &projection.r1, gate_rank, &gate_up_weight)?;
            let decay_input = expand(graph, alpha_up, &projection.r2, alpha_rank, &alpha_up_weight)?;
            let decay_lower = decay_lower.f32();
            if chunked {
                graph
                    .enqueue(
                        chunk,
                        &dimensions,
                        kimi_delta_chunk::WorkflowArgs {
                            projection: (&projection.r0).into(),
                            beta: (&projection.r3).into(),
                            gate: (&gate_input).into(),
                            decay: (&decay_input).into(),
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
                            decay_lower,
                            grouped: topology.grouped.into(),
                            gate_function,
                            slab_banks,
                        },
                    )?
                    .value
            } else {
                graph
                    .enqueue(
                        step,
                        &dimensions,
                        kimi_delta_step::WorkflowArgs {
                            projection: (&projection.r0).into(),
                            beta: (&projection.r3).into(),
                            gate: (&gate_input).into(),
                            decay: (&decay_input).into(),
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
                            decay_lower,
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
