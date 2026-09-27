//! Exact demand of every semantic value (Unconstrained only).
//!
//! A value is *demanded* when its exact bits can reach a discrete value, an
//! address, a predicate, a failure, an effect decision or progress. The single
//! seed rule is that every discrete-typed value, and every content class with a
//! discrete element type, is demanded; demand is then closed backward over data
//! edges, content classes and callee summaries. Every control decision is a
//! discrete value, so no control-dependence edge exists. A floating operation
//! may deviate only when its output is not demanded, so deviation (which flows
//! forward along the same edges) never reaches a demanded value.
//!
//! Labels make the analysis context-independent per body: `Internal` marks
//! demand regardless of the call context, and one label per floating result
//! leaf and floating mutable parameter records which outputs a value reaches.
//! Bodies are labelled bottom-up over the acyclic call graph; each call node's
//! context is then assigned top-down. Family instances are per call site, so a
//! `FunctionId` has exactly one context.

use crate::content_classes::ContentClasses;
use seismic_lang::entry::{
    CandidateKind, ParameterAccess, SemanticEventKind, SemanticFunction, SemanticNodeView,
    SemanticProgram, SemanticType,
};
use seismic_lang::ids::{CapabilityId, FamilyId, FunctionId, NodeId, RegionId, SemanticValueId};
use seismic_lang::registry::{representation_info, BackendName};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Unconstrained-only. Built once per candidate domain from the checked
/// program and backend.
pub(crate) struct ProgramDemand {
    /// Every body selectable on the backend of every reached family instance.
    functions: HashMap<FunctionId, FunctionDemand>,
}

pub(crate) struct FunctionDemand {
    demanded_values: Vec<bool>,
    call_demands: HashMap<NodeId, CallDemand>,
    /// The body may reach a source failure, directly or through a callee.
    fallible: bool,
}

/// Which floating outputs of one call (or the entry) are demanded by its
/// context. Discrete outputs are always demanded and are not represented here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CallDemand {
    /// By result ordinal.
    results: Vec<bool>,
    /// By parameter ordinal; only floating mutable parameters can be set.
    mutable_parameters: Vec<bool>,
}

impl CallDemand {
    /// Root: no floating result leaf and no floating mutable parameter is demanded.
    pub fn entry(function: &SemanticFunction) -> Self {
        Self {
            results: vec![false; function.results().len()],
            mutable_parameters: vec![false; function.parameters().len()],
        }
    }

    pub fn any(&self) -> bool {
        self.results
            .iter()
            .chain(&self.mutable_parameters)
            .any(|demanded| *demanded)
    }
}

/// Bottom-up summary of one family instance over its bodies selectable on the
/// backend. Parameter and result ordinals are the family contract's.
struct FamilyDemandSummary {
    /// Parameters demanded regardless of context.
    internal: BTreeSet<u32>,
    /// Floating result ordinal → parameters it depends on.
    through_results: Vec<BTreeSet<u32>>,
    /// Floating mutable parameter → parameters its final contents depend on.
    through_mutable: BTreeMap<u32, BTreeSet<u32>>,
    fallible: bool,
}

impl ProgramDemand {
    pub fn analyze(
        program: &SemanticProgram,
        backend: BackendName,
        supports: impl Fn(CapabilityId) -> bool,
    ) -> Self {
        let mut summarizer = DemandSummarizer {
            program,
            backend,
            supports,
            summaries: HashMap::new(),
            summarizing: HashSet::new(),
            bodies: HashMap::new(),
        };
        summarizer.summarize(program.root());
        let mut functions = HashMap::new();
        for function in summarizer.selectable(program.root()) {
            let demand = CallDemand::entry(program.function(function));
            summarizer.assign(function, demand, &mut functions);
        }
        Self { functions }
    }

    pub fn is_demanded(&self, function: FunctionId, value: SemanticValueId) -> bool {
        self.function(function).demanded_values[value.index()]
    }

    pub fn call_demand(&self, function: FunctionId, call: NodeId) -> &CallDemand {
        self.function(function)
            .call_demands
            .get(&call)
            .unwrap_or_else(|| panic!("{call:?} is not a call node of {function:?}"))
    }

    pub fn is_fallible(&self, function: FunctionId) -> bool {
        self.function(function).fallible
    }

    fn function(&self, function: FunctionId) -> &FunctionDemand {
        self.functions
            .get(&function)
            .unwrap_or_else(|| panic!("{function:?} is not a body selectable on this backend"))
    }
}

const INTERNAL: u32 = 0;

/// A set of demand labels: `INTERNAL`, then one per result ordinal, then one per
/// parameter ordinal (used by floating mutable parameters).
#[derive(Clone, Default)]
struct LabelSet(Vec<u64>);

impl LabelSet {
    fn insert(&mut self, label: u32) {
        let word = label as usize / 64;
        if self.0.len() <= word {
            self.0.resize(word + 1, 0);
        }
        self.0[word] |= 1 << (label % 64);
    }

    fn contains(&self, label: u32) -> bool {
        self.0
            .get(label as usize / 64)
            .is_some_and(|word| word & (1 << (label % 64)) != 0)
    }

    /// Returns whether `self` grew.
    fn union_with(&mut self, other: &Self) -> bool {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        let mut grew = false;
        for (word, other) in self.0.iter_mut().zip(&other.0) {
            grew |= *other & !*word != 0;
            *word |= other;
        }
        grew
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }
}

struct CallSite {
    node: NodeId,
    family: FamilyId,
    inputs: Vec<SemanticValueId>,
    outputs: Vec<SemanticValueId>,
}

/// Context-independent labels of one body.
struct BodyLabels<'a> {
    classes: ContentClasses<'a>,
    /// By point: content classes first, then one point per leaf (used only by
    /// non-tensor leaves).
    labels: Vec<LabelSet>,
    results: u32,
    calls: Vec<CallSite>,
    fallible: bool,
}

impl BodyLabels<'_> {
    fn points(&self, value: SemanticValueId) -> impl Iterator<Item = usize> + '_ {
        point_range(&self.classes, value)
    }

    fn value_labels(&self, value: SemanticValueId) -> LabelSet {
        let mut labels = LabelSet::default();
        for point in self.points(value) {
            labels.union_with(&self.labels[point]);
        }
        labels
    }

    fn demanded(&self, point: usize, demand: &CallDemand) -> bool {
        let labels = &self.labels[point];
        labels.contains(INTERNAL)
            || (0..self.results).any(|k| demand.results[k as usize] && labels.contains(1 + k))
            || demand
                .mutable_parameters
                .iter()
                .enumerate()
                .any(|(p, demanded)| *demanded && labels.contains(1 + self.results + p as u32))
    }

    /// Whether some floating leaf of `value` is demanded under `demand`.
    fn floating_demanded(&self, value: SemanticValueId, demand: &CallDemand) -> bool {
        self.classes
            .leaf_range(value)
            .zip(self.points(value))
            .any(|(leaf, point)| {
                !is_discrete(self.classes.leaf_type(leaf)) && self.demanded(point, demand)
            })
    }
}

/// The point of one leaf: its content class, or the leaf itself when it is not
/// a tensor.
fn leaf_point(classes: &ContentClasses<'_>, leaf: usize) -> usize {
    match classes.leaf_class(leaf) {
        Some(class) => class.index(),
        None => classes.len() + leaf,
    }
}

fn point_range<'c>(
    classes: &'c ContentClasses<'_>,
    value: SemanticValueId,
) -> impl Iterator<Item = usize> + 'c {
    classes
        .leaf_range(value)
        .map(|leaf| leaf_point(classes, leaf))
}

fn is_discrete(ty: &SemanticType) -> bool {
    match ty {
        SemanticType::Scalar(dtype) => !dtype.is_float(),
        SemanticType::Integer | SemanticType::Index { .. } | SemanticType::Range { .. } => true,
        SemanticType::Tensor(tensor) => !representation_info(tensor.representation)
            .decoded
            .is_float(),
        SemanticType::Opaque { .. } => false,
        SemanticType::Tuple(_) | SemanticType::Void => {
            unreachable!("leaf types exclude tuples and void")
        }
    }
}

/// Labels every selectable body bottom-up into family summaries, then assigns
/// each body its call context top-down.
struct DemandSummarizer<'a, S> {
    program: &'a SemanticProgram,
    backend: BackendName,
    supports: S,
    summaries: HashMap<FamilyId, FamilyDemandSummary>,
    summarizing: HashSet<FamilyId>,
    bodies: HashMap<FunctionId, BodyLabels<'a>>,
}

impl<'a, S: Fn(CapabilityId) -> bool> DemandSummarizer<'a, S> {
    /// Bodies of `family` selectable on the backend, reference first.
    fn selectable(&self, family: FamilyId) -> Vec<FunctionId> {
        let family = self.program.family(family);
        std::iter::once(family.reference().candidate())
            .chain(family.alternatives())
            .filter(|candidate| {
                let backend_matches = match candidate.kind {
                    CandidateKind::Portable => true,
                    CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
                        backend == self.backend
                    }
                };
                backend_matches
                    && candidate
                        .requires
                        .iter()
                        .all(|capability| (self.supports)(*capability))
            })
            .map(|candidate| candidate.function)
            .collect()
    }

    fn summarize(&mut self, family: FamilyId) {
        if self.summaries.contains_key(&family) {
            return;
        }
        assert!(
            self.summarizing.insert(family),
            "semantic call graph is cyclic at {family:?}"
        );
        let program = self.program;
        let contract = program.function(program.family(family).reference().function());
        let mut summary = FamilyDemandSummary {
            internal: BTreeSet::new(),
            through_results: vec![BTreeSet::new(); contract.results().len()],
            through_mutable: BTreeMap::new(),
            fallible: false,
        };
        for function in self.selectable(family) {
            let body = self.label_body(function);
            let semantic = program.function(function);
            assert_eq!(
                semantic.parameters().len(),
                contract.parameters().len(),
                "family bodies differ in parameter arity"
            );
            assert_eq!(
                body.results as usize,
                summary.through_results.len(),
                "family bodies differ in result arity"
            );
            for (ordinal, parameter) in semantic.parameters().iter().enumerate() {
                let ordinal = ordinal as u32;
                let labels = body.value_labels(parameter.value);
                if labels.contains(INTERNAL) {
                    summary.internal.insert(ordinal);
                }
                for (k, through) in summary.through_results.iter_mut().enumerate() {
                    if labels.contains(1 + k as u32) {
                        through.insert(ordinal);
                    }
                }
                for p in 0..semantic.parameters().len() as u32 {
                    if labels.contains(1 + body.results + p) {
                        summary
                            .through_mutable
                            .entry(p)
                            .or_default()
                            .insert(ordinal);
                    }
                }
            }
            summary.fallible |= body.fallible;
            self.bodies.insert(function, body);
        }
        self.summarizing.remove(&family);
        self.summaries.insert(family, summary);
    }

    fn label_body(&mut self, function: FunctionId) -> BodyLabels<'a> {
        let program = self.program;
        let semantic = program.function(function);
        let mut calls = Vec::new();
        let mut fallible = false;
        collect_calls(semantic, semantic.root(), &mut calls, &mut fallible);
        for call in &calls {
            self.summarize(call.family);
            fallible |= self.summaries[&call.family].fallible;
        }

        let classes = ContentClasses::analyze(semantic);
        let mut graph = DemandGraph {
            function: semantic,
            classes: &classes,
            labels: vec![LabelSet::default(); classes.len() + classes.leaf_total()],
            edges: vec![Vec::new(); classes.len() + classes.leaf_total()],
        };

        // Seeds: every discrete leaf (a tensor leaf seeds its class).
        for leaf in 0..classes.leaf_total() {
            if is_discrete(classes.leaf_type(leaf)) {
                graph.labels[leaf_point(&classes, leaf)].insert(INTERNAL);
            }
        }
        // Output seeds: floating result leaves and floating mutable parameters.
        let results = semantic.results().len() as u32;
        for (k, result) in semantic.results().iter().enumerate() {
            graph.seed(*result, 1 + k as u32);
        }
        for (p, parameter) in semantic.parameters().iter().enumerate() {
            if parameter.access == ParameterAccess::Mutable {
                graph.seed(parameter.value, 1 + results + p as u32);
            }
        }

        graph.region(semantic.root());
        for call in &calls {
            let summary = &self.summaries[&call.family];
            for parameter in &summary.internal {
                for point in point_range(&classes, call.inputs[*parameter as usize]) {
                    graph.labels[point].insert(INTERNAL);
                }
            }
            for (k, through) in summary.through_results.iter().enumerate() {
                for parameter in through {
                    graph.edge(call.outputs[k], call.inputs[*parameter as usize]);
                }
            }
            for (mutable, through) in &summary.through_mutable {
                for parameter in through {
                    graph.edge(
                        call.inputs[*mutable as usize],
                        call.inputs[*parameter as usize],
                    );
                }
            }
        }
        let labels = graph.close();
        BodyLabels {
            classes,
            labels,
            results,
            calls,
            fallible,
        }
    }

    fn assign(
        &mut self,
        function: FunctionId,
        demand: CallDemand,
        functions: &mut HashMap<FunctionId, FunctionDemand>,
    ) {
        let body = self
            .bodies
            .remove(&function)
            .unwrap_or_else(|| panic!("{function:?} is reached from two body sites"));
        let program = self.program;
        let semantic = program.function(function);
        let demanded_values = semantic
            .values()
            .map(|(value, _)| {
                body.points(value)
                    .any(|point| body.demanded(point, &demand))
            })
            .collect();
        let mut call_demands = HashMap::new();
        for call in &body.calls {
            let contract = program.function(program.family(call.family).reference().function());
            let call_demand = CallDemand {
                results: call
                    .outputs
                    .iter()
                    .map(|output| body.floating_demanded(*output, &demand))
                    .collect(),
                mutable_parameters: contract
                    .parameters()
                    .iter()
                    .zip(&call.inputs)
                    .map(|(parameter, input)| {
                        parameter.access == ParameterAccess::Mutable
                            && body.floating_demanded(*input, &demand)
                    })
                    .collect(),
            };
            for callee in self.selectable(call.family) {
                self.assign(callee, call_demand.clone(), functions);
            }
            call_demands.insert(call.node, call_demand);
        }
        functions.insert(
            function,
            FunctionDemand {
                demanded_values,
                call_demands,
                fallible: body.fallible,
            },
        );
    }
}

/// Collects every call node of `region` and its nested regions, and whether any
/// node may terminate with a source failure.
fn collect_calls(
    function: &SemanticFunction,
    region: RegionId,
    calls: &mut Vec<CallSite>,
    fallible: &mut bool,
) {
    for (id, node) in function.nodes(region) {
        *fallible |= node
            .events()
            .iter()
            .any(|event| matches!(event.kind(), SemanticEventKind::MayFail));
        match node.view() {
            SemanticNodeView::Call {
                family,
                inputs,
                outputs,
            } => calls.push(CallSite {
                node: id,
                family,
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
            }),
            SemanticNodeView::If {
                then, otherwise, ..
            } => {
                collect_calls(function, then, calls, fallible);
                collect_calls(function, otherwise, calls, fallible);
            }
            SemanticNodeView::Loop { body, .. } => collect_calls(function, body, calls, fallible),
            _ => {}
        }
    }
}

/// Backward dependency graph of one body over points (content classes, then
/// non-tensor leaves). `edges[target]` lists the points whose exact value the
/// target depends on.
struct DemandGraph<'g, 'a> {
    function: &'a SemanticFunction,
    classes: &'g ContentClasses<'a>,
    labels: Vec<LabelSet>,
    edges: Vec<Vec<usize>>,
}

impl DemandGraph<'_, '_> {
    fn seed(&mut self, value: SemanticValueId, label: u32) {
        for point in point_range(self.classes, value) {
            self.labels[point].insert(label);
        }
    }

    /// Every point of `target` depends on every point of `source`.
    fn edge(&mut self, target: SemanticValueId, source: SemanticValueId) {
        let sources = point_range(self.classes, source).collect::<Vec<_>>();
        for target in point_range(self.classes, target) {
            self.edges[target].extend(&sources);
        }
    }

    /// Leaf `i` of `target` depends on leaf `i` of `source` (values of one type).
    fn edge_leaves(&mut self, target: SemanticValueId, source: SemanticValueId) {
        let sources = point_range(self.classes, source).collect::<Vec<_>>();
        let targets = point_range(self.classes, target).collect::<Vec<_>>();
        assert_eq!(
            targets.len(),
            sources.len(),
            "leaf-wise dependency over different types"
        );
        for (target, source) in targets.into_iter().zip(sources) {
            self.edges[target].push(source);
        }
    }

    /// Leaves of tuple component `index` and of `value` depend on each other in
    /// the direction given by `component_is_target`.
    fn edge_component(
        &mut self,
        tuple: SemanticValueId,
        index: u32,
        value: SemanticValueId,
        component_is_target: bool,
    ) {
        let component = self
            .classes
            .component_range(tuple, index)
            .map(|leaf| leaf_point(self.classes, leaf))
            .collect::<Vec<_>>();
        let value = point_range(self.classes, value).collect::<Vec<_>>();
        assert_eq!(
            component.len(),
            value.len(),
            "tuple component type differs from its value"
        );
        for (component, value) in component.into_iter().zip(value) {
            let (target, source) = if component_is_target {
                (component, value)
            } else {
                (value, component)
            };
            self.edges[target].push(source);
        }
    }

    fn region(&mut self, region: RegionId) {
        let function = self.function;
        for (_, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::Primitive { inputs, output, .. }
                | SemanticNodeView::Intrinsic { inputs, output, .. }
                | SemanticNodeView::Elementwise { inputs, output, .. } => {
                    for input in inputs {
                        self.edge(output, *input);
                    }
                }
                SemanticNodeView::TuplePack { inputs, output } => {
                    for (index, input) in inputs.iter().enumerate() {
                        self.edge_component(output, index as u32, *input, true);
                    }
                }
                SemanticNodeView::TupleGet {
                    tuple,
                    index,
                    output,
                } => self.edge_component(tuple, index, output, false),
                SemanticNodeView::Reduce { input, output, .. }
                | SemanticNodeView::Copy { input, output }
                | SemanticNodeView::RepresentationConvert { input, output, .. } => {
                    self.edge(output, input)
                }
                SemanticNodeView::ElementRead { place, output, .. } => self.edge(output, place),
                SemanticNodeView::ElementWrite { place, value, .. } => self.edge(place, value),
                SemanticNodeView::Store {
                    destination, value, ..
                } => self.edge(destination, value),
                SemanticNodeView::Atomic {
                    place, arguments, ..
                } => {
                    for argument in arguments {
                        self.edge(place, *argument);
                    }
                }
                SemanticNodeView::If {
                    captures,
                    outputs,
                    then,
                    otherwise,
                    ..
                } => {
                    for arm in [then, otherwise] {
                        let arm_region = function.region(arm);
                        for (output, result) in outputs.iter().zip(arm_region.results()) {
                            self.edge_leaves(*output, *result);
                        }
                        for (parameter, capture) in arm_region.parameters().iter().zip(captures) {
                            self.edge_leaves(*parameter, *capture);
                        }
                        self.region(arm);
                    }
                }
                SemanticNodeView::Loop {
                    captures,
                    body,
                    carries,
                    ..
                } => {
                    let parameters = function.region(body).parameters();
                    for (parameter, capture) in parameters[1..].iter().zip(captures) {
                        self.edge_leaves(*parameter, *capture);
                    }
                    for carry in carries {
                        for target in [carry.result, carry.parameter] {
                            self.edge_leaves(target, carry.yielded);
                            self.edge_leaves(target, carry.initial);
                        }
                    }
                    self.region(body);
                }
                // Call edges come from the callee summary.
                SemanticNodeView::Call { .. } => {}
                // Geometry only: discrete operands are seeded, contents untouched.
                SemanticNodeView::Alloc { .. }
                | SemanticNodeView::Fill { .. }
                | SemanticNodeView::Extent { .. }
                | SemanticNodeView::View { .. }
                | SemanticNodeView::Check { .. } => {}
            }
        }
    }

    /// Closes labels backward over every edge. Labels only grow, so the
    /// worklist terminates.
    fn close(mut self) -> Vec<LabelSet> {
        let mut work = (0..self.labels.len())
            .filter(|point| !self.labels[*point].is_empty())
            .collect::<Vec<_>>();
        while let Some(target) = work.pop() {
            let labels = self.labels[target].clone();
            for &source in &self.edges[target] {
                if self.labels[source].union_with(&labels) {
                    work.push(source);
                }
            }
        }
        self.labels
    }
}
