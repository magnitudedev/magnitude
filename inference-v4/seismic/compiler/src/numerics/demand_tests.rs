use super::demand::ProgramDemand;
use crate::content_classes::ContentClasses;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::{
    ElementBindings, SemanticFunction, SemanticNodeView, SemanticProgram, SemanticType,
};
use seismic_lang::ids::{FunctionId, NodeId, RegionId, SemanticValueId};
use seismic_lang::registry::{self, BackendName};
use seismic_lang::types::DType;

const SAMPLING: &str = include_str!("../../../../seismic-std/lib/kernels/sampling.seismic");
const GELU: &str = include_str!("../../../../seismic-std/lib/kernels/gelu.seismic");
const RMS_NORM: &str = include_str!("../../../../seismic-std/lib/kernels/rms_norm.seismic");
const LINEAR: &str = include_str!("../../../../seismic-std/lib/kernels/linear.seismic");
const ELEMENTWISE: &str = include_str!("../../../../seismic-std/lib/kernels/elementwise.seismic");
const MATMUL: &str = include_str!("../../../../seismic-std/lib/constructs/matmul.seismic");
const ROUTED_SUFFIX: &str = include_str!("../../../../engine/lib/routed_suffix.seismic");

struct Analyzed {
    program: SemanticProgram,
    demand: ProgramDemand,
}

fn analyze(files: &[(&str, &str)], entry: &str, bindings: &ElementBindings) -> Analyzed {
    let module = check_source(SourceSet::new(
        files
            .iter()
            .map(|(path, text)| SourceFile {
                path: (*path).into(),
                text: (*text).into(),
            })
            .collect(),
    ))
    .unwrap_or_else(|error| panic!("demand fixture does not check: {error:?}"));
    let entry = module
        .entry(module.entry_named(entry).expect("fixture entry"), bindings)
        .unwrap_or_else(|error| panic!("demand fixture entry does not build: {error:?}"));
    let program = entry.into_parts().program;
    let demand = ProgramDemand::analyze(&program, BackendName::Cpu, |_| false);
    Analyzed { program, demand }
}

fn f32_bindings(parameters: &[&str]) -> ElementBindings {
    parameters
        .iter()
        .fold(ElementBindings::new(), |bindings, parameter| {
            bindings.bind(parameter, registry::dense(DType::F32))
        })
}

impl Analyzed {
    fn root(&self) -> FunctionId {
        self.program
            .family(self.program.root())
            .reference()
            .function()
    }

    fn function(&self, id: FunctionId) -> &SemanticFunction {
        self.program.function(id)
    }

    /// Call nodes of `function` in execution order, with their callee's
    /// reference body.
    fn calls(&self, function: FunctionId) -> Vec<(NodeId, FunctionId, Vec<SemanticValueId>)> {
        fn visit(
            analyzed: &Analyzed,
            function: &SemanticFunction,
            region: RegionId,
            calls: &mut Vec<(NodeId, FunctionId, Vec<SemanticValueId>)>,
        ) {
            for (id, node) in function.nodes(region) {
                match node.view() {
                    SemanticNodeView::Call { family, inputs, .. } => calls.push((
                        id,
                        analyzed.program.family(family).reference().function(),
                        inputs.to_vec(),
                    )),
                    SemanticNodeView::If {
                        then, otherwise, ..
                    } => {
                        visit(analyzed, function, then, calls);
                        visit(analyzed, function, otherwise, calls);
                    }
                    SemanticNodeView::Loop { body, .. } => visit(analyzed, function, body, calls),
                    _ => {}
                }
            }
        }
        let mut calls = Vec::new();
        let semantic = self.function(function);
        visit(self, semantic, semantic.root(), &mut calls);
        calls
    }

    /// The call of `function` whose callee is named `name`, by occurrence.
    fn call(
        &self,
        function: FunctionId,
        name: &str,
        occurrence: usize,
    ) -> (NodeId, FunctionId, Vec<SemanticValueId>) {
        self.calls(function)
            .into_iter()
            .filter(|(_, callee, _)| self.function(*callee).name() == name)
            .nth(occurrence)
            .unwrap_or_else(|| panic!("no call #{occurrence} of {name}"))
    }

    /// The caller argument bound to the callee parameter named `parameter`.
    fn argument(
        &self,
        call: &(NodeId, FunctionId, Vec<SemanticValueId>),
        parameter: &str,
    ) -> SemanticValueId {
        let ordinal = self
            .function(call.1)
            .parameters()
            .iter()
            .position(|candidate| candidate.name == parameter)
            .unwrap_or_else(|| panic!("callee has no parameter {parameter}"));
        call.2[ordinal]
    }

    fn parameter(&self, function: FunctionId, name: &str) -> SemanticValueId {
        self.function(function)
            .parameters()
            .iter()
            .find(|parameter| parameter.name == name)
            .unwrap_or_else(|| panic!("no parameter {name}"))
            .value
    }

    /// Values of a single-file program whose span covers exactly `snippet`.
    fn spanned(&self, function: FunctionId, text: &str, snippet: &str) -> Vec<SemanticValueId> {
        let values = self
            .function(function)
            .values()
            .filter(|(_, info)| &text[info.span.start as usize..info.span.end as usize] == snippet)
            .map(|(value, _)| value)
            .collect::<Vec<_>>();
        assert!(!values.is_empty(), "no value spans `{snippet}`");
        values
    }

    fn demanded(&self, function: FunctionId, text: &str, snippet: &str) -> bool {
        let answers = self
            .spanned(function, text, snippet)
            .into_iter()
            .map(|value| self.demand.is_demanded(function, value))
            .collect::<Vec<_>>();
        assert!(
            answers.iter().all(|answer| *answer == answers[0]),
            "values spanning `{snippet}` disagree: {answers:?}"
        );
        answers[0]
    }
}

fn is_floating(ty: &SemanticType) -> bool {
    match ty {
        SemanticType::Scalar(dtype) => dtype.is_float(),
        SemanticType::Tensor(tensor) => registry::representation_info(tensor.representation)
            .decoded
            .is_float(),
        _ => false,
    }
}

#[test]
#[ignore = "blocked on A1-L6: the checker panics with 'checked branch capture has no then parameter' (lang/src/check/entry_build.rs:4521) while building sample_rows; sampling_reduction_demands_every_float_value covers the same demand paths today"]
fn sampling_demands_every_float_value() {
    let analyzed = analyze(
        &[("sampling.seismic", SAMPLING)],
        "sample_rows",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let floating = analyzed
        .function(root)
        .values()
        .filter(|(_, info)| is_floating(&info.ty))
        .map(|(value, _)| value)
        .collect::<Vec<_>>();
    assert!(!floating.is_empty());
    for value in floating {
        assert!(
            analyzed.demand.is_demanded(root, value),
            "sampling float value {value:?} is not demanded"
        );
    }
    assert!(analyzed.demanded(root, SAMPLING, "log(-log(uniform))"));
}

#[test]
fn gelu_demands_erf_conditions_but_not_arm_polynomials() {
    let analyzed = analyze(
        &[("gelu.seismic", GELU)],
        "gelu",
        &f32_bindings(&["T", "U"]),
    );
    let root = analyzed.root();
    let erf = analyzed.call(root, "erf_values", 0);
    assert!(
        !analyzed.demand.call_demand(root, erf.0).any(),
        "the erf result feeds only the floating output"
    );
    let body = erf.1;
    // Conditions and everything they compare are exact.
    assert!(analyzed.demanded(body, GELU, "source[i]"));
    assert!(analyzed.demanded(body, GELU, "abs(x)"));
    assert!(analyzed.demanded(body, GELU, "magnitude < 0.84375"));
    assert!(analyzed.demanded(body, GELU, "x != x"));
    // The float→integer cast operand is discrete-consumed.
    assert!(analyzed.demanded(body, GELU, "magnitude * 256.0"));
    // Arm polynomials only produce floating results.
    assert!(!analyzed.demanded(body, GELU, "x * x"));
    assert!(!analyzed.demanded(body, GELU, "magnitude - 1.0"));
    assert!(!analyzed.demanded(body, GELU, "exp(-z * z - 0.5625)"));
    assert!(!analyzed.demanded(body, GELU, "1.0 - complement"));
    // The erf argument is demanded in the caller through the callee summary;
    // the published GELU value is not.
    assert!(analyzed.demanded(root, GELU, "source * 0.7071067811865476"));
    assert!(analyzed.demanded(root, GELU, "f32(x[rows])"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
    assert!(!analyzed.demanded(root, GELU, "erf_values(source * 0.7071067811865476)"));
    assert!(!analyzed.demanded(root, GELU, "0.5 * source"));
    assert!(!analyzed.demanded(root, GELU, "1.0 + probability"));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "result")));
}

#[test]
#[ignore = "blocked on A1-L5/A1-L6: the checker rejects the fixture (rms_norm 'mathematical quantity conversion to float is not yet supported' x2; routed_suffix 'cannot assign index to an element of dtype i32' and 'borrowed tensor leaf cannot cross an owned boundary'); router_demand_crosses_two_mutable_call_hops covers the same demand paths today"]
fn routed_suffix_demands_the_router_path_but_not_the_experts() {
    let analyzed = analyze(
        &[
            ("rms_norm.seismic", RMS_NORM),
            ("linear.seismic", LINEAR),
            ("elementwise.seismic", ELEMENTWISE),
            ("matmul.seismic", MATMUL),
            ("routed_suffix.seismic", ROUTED_SUFFIX),
        ],
        "qwen_routed_suffix",
        &f32_bindings(&[
            "A", "NW", "RW", "SRW", "EGW", "EUW", "EDW", "SGW", "SUW", "SDW",
        ]),
    );
    let root = analyzed.root();
    let norm = analyzed.call(root, "rms_norm", 0);
    let router = analyzed.call(root, "linear", 0);
    let topk = analyzed.call(root, "route_topk", 0);
    assert!(analyzed.demand.call_demand(root, norm.0).any());
    assert!(analyzed.demand.call_demand(root, router.0).any());
    assert!(!analyzed.demand.call_demand(root, topk.0).any());
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.argument(&norm, "y")));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.argument(&router, "y")));
    for parameter in ["residual", "norm", "router", "routes"] {
        assert!(
            analyzed
                .demand
                .is_demanded(root, analyzed.parameter(root, parameter)),
            "{parameter} is on the routing path"
        );
    }
    for parameter in [
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_router",
        "scores",
    ] {
        assert!(
            !analyzed
                .demand
                .is_demanded(root, analyzed.parameter(root, parameter)),
            "{parameter} only produces floating outputs"
        );
    }
    for (name, occurrence) in [
        ("routed_input", 0),
        ("routed_input", 1),
        ("routed_activation", 0),
        ("routed_output", 0),
        ("linear", 1),
        ("linear", 2),
        ("linear", 3),
        ("silu", 0),
        ("multiply", 0),
        ("shared_coefficient", 0),
        ("routed_merge", 0),
    ] {
        let call = analyzed.call(root, name, occurrence);
        assert!(
            !analyzed.demand.call_demand(root, call.0).any(),
            "{name} #{occurrence} publishes only floats nobody demands"
        );
    }
    // Inside the router body the softmax decides routes; the normalized scores do not.
    let topk_body = topk.1;
    let semantic = analyzed.function(topk_body);
    let exp = semantic
        .values()
        .find(|(_, info)| {
            &ROUTED_SUFFIX[info.span.start as usize..info.span.end as usize]
                == "exp(values - maximum)"
        })
        .expect("route_topk softmax")
        .0;
    assert!(analyzed.demand.is_demanded(topk_body, exp));
    let normalized_score = semantic
        .values()
        .find(|(_, info)| {
            &ROUTED_SUFFIX[info.span.start as usize..info.span.end as usize]
                == "f32(selected[i]) / denominator"
        })
        .expect("route_topk score normalization")
        .0;
    assert!(!analyzed.demand.is_demanded(topk_body, normalized_score));
}

#[test]
fn float_predicate_demands_its_operands() {
    // P33: the float predicate selects which element is written.
    const P33: &str = "fn probe(x: &tensor[4] f32, out: &mut tensor[2] f32):
    let s = x[0] * x[1] - x[2]
    if s > 0.0:
        out[0] = 1.0
    else:
        out[1] = 1.0
";
    let analyzed = analyze(&[("p33.seismic", P33)], "probe", &ElementBindings::new());
    let root = analyzed.root();
    assert!(analyzed.demanded(root, P33, "x[0] * x[1] - x[2]"));
    assert!(analyzed.demanded(root, P33, "x[0] * x[1]"));
    assert!(analyzed.demanded(root, P33, "s > 0.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "out")));
}

#[test]
fn one_helper_at_two_demand_contexts() {
    // P36: per-call-site instantiation gives each call its own context.
    const P36: &str = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:
    return x * 3.0
fn probe[N](x: &tensor[N] f32, out: &mut tensor[N] f32) -> i32:
    let a = scale(x)
    let b = scale(x)
    out[0:N] = b
    return reduce(a, 0, argmax)
";
    let analyzed = analyze(&[("p36.seismic", P36)], "probe", &ElementBindings::new());
    let root = analyzed.root();
    let a = analyzed.call(root, "scale", 0);
    let b = analyzed.call(root, "scale", 1);
    assert_ne!(a.1, b.1, "each call site owns its family instance");
    assert!(analyzed.demand.call_demand(root, a.0).any());
    assert!(!analyzed.demand.call_demand(root, b.0).any());
    assert!(analyzed.demanded(a.1, P36, "x * 3.0"));
    assert!(!analyzed.demanded(b.1, P36, "x * 3.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "out")));
}

#[test]
fn loop_carry_demand_reaches_the_initial_value() {
    // A zero-trip loop returns the initial value, so a demanded result demands it.
    const DEMANDED: &str = "fn probe[N](x: &tensor[N] f32) -> i32:
    let mut acc = x[0] + 1.0
    for i in 0..N:
        acc = acc * 2.0
    return i32(acc)
";
    let analyzed = analyze(
        &[("carry.seismic", DEMANDED)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    assert!(analyzed.demanded(root, DEMANDED, "x[0] + 1.0"));
    assert!(analyzed.demanded(root, DEMANDED, "acc * 2.0"));

    const FLOATING: &str = "fn probe[N](x: &tensor[N] f32) -> f32:
    let mut acc = x[0] + 1.0
    for i in 0..N:
        acc = acc * 2.0
    return acc
";
    let analyzed = analyze(
        &[("carry.seismic", FLOATING)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    assert!(!analyzed.demanded(root, FLOATING, "x[0] + 1.0"));
    assert!(!analyzed.demanded(root, FLOATING, "acc * 2.0"));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
}

#[test]
fn calls_with_mutable_float_and_integer_parameters() {
    const UPDATE: &str = "fn update(v: &mut tensor[4] f32, c: &mut tensor[4] i32, x: &tensor[4] f32, y: &tensor[4] f32):
    for i in 0..4:
        v[i] = x[i] * 2.0
        c[i] = i32(y[i] + 1.0)
fn probe(x: &tensor[4] f32, y: &tensor[4] f32, v: &mut tensor[4] f32, c: &mut tensor[4] i32):
    update(v, c, x, y)
fn observe(x: &tensor[4] f32, y: &tensor[4] f32, v: &mut tensor[4] f32, c: &mut tensor[4] i32) -> i32:
    update(v, c, x, y)
    return i32(v[0])
";
    // The integer state is demanded in every context; the float state only
    // where the caller reads it into a discrete value.
    let analyzed = analyze(
        &[("update.seismic", UPDATE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let update = analyzed.call(root, "update", 0);
    assert!(!analyzed.demand.call_demand(root, update.0).any());
    assert!(analyzed.demanded(update.1, UPDATE, "y[i] + 1.0"));
    assert!(!analyzed.demanded(update.1, UPDATE, "x[i] * 2.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "y")));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "c")));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "v")));

    let analyzed = analyze(
        &[("update.seismic", UPDATE)],
        "observe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let update = analyzed.call(root, "update", 0);
    assert!(analyzed.demand.call_demand(root, update.0).any());
    assert!(analyzed.demanded(update.1, UPDATE, "x[i] * 2.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "v")));
}

#[test]
fn fallibility_is_summarized_through_calls() {
    const SOURCE: &str = "fn pick[N](x: &tensor[N] f32, i: i32) -> f32:
    return x[i]
fn probe[N](x: &tensor[N] f32, i: i32) -> f32:
    return pick(x, i) * 2.0
";
    let analyzed = analyze(
        &[("pick.seismic", SOURCE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let pick = analyzed.call(root, "pick", 0);
    assert!(analyzed.demand.is_fallible(pick.1));
    assert!(analyzed.demand.is_fallible(root));
    // The index is discrete, so the element it selects is read exactly, but the
    // element's value itself is not demanded.
    assert!(!analyzed.demand.call_demand(root, pick.0).any());
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
}

#[test]
fn sampling_reduction_demands_every_float_value() {
    // The sampling shape: an int→float cast upstream, a float rebound inside an
    // `If`, and an argmax reduction into an i32 result.
    const SOURCE: &str =
        "fn probe[V](logits: &tensor[V] f32, draws: &tensor[2] u32, result: &mut tensor[1] i32):
    let mut scores = tensor[V] f32
    for token in 0..V:
        let mut value = logits[token]
        if draws[0] == 1:
            let uniform = (f32(draws[1] >> 9) + 0.5) * 0.00000011920928955078125
            value = value - log(-log(uniform))
        scores[token] = value
    result[0] = reduce(scores, 0, argmax)
";
    let analyzed = analyze(
        &[("sampling.seismic", SOURCE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let floating = analyzed
        .function(root)
        .values()
        .filter(|(_, info)| is_floating(&info.ty))
        .map(|(value, _)| value)
        .collect::<Vec<_>>();
    assert!(!floating.is_empty());
    for value in floating {
        assert!(
            analyzed.demand.is_demanded(root, value),
            "sampling float value {value:?} is not demanded"
        );
    }
    assert!(analyzed.demanded(root, SOURCE, "f32(draws[1] >> 9)"));
    assert!(analyzed.demanded(root, SOURCE, "log(-log(uniform))"));
    assert!(analyzed.demanded(root, SOURCE, "value - log(-log(uniform))"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "logits")));
}

#[test]
fn router_demand_crosses_two_mutable_call_hops() {
    // The routed-suffix shape: norm → project → pick through `&mut` float
    // outputs, where only `pick` compares floats. The sibling projection
    // publishes only floats nobody demands.
    const SOURCE: &str = "fn norm[N](x: &tensor[N] f32, y: &mut tensor[N] f32):
    for i in 0..N:
        y[i] = x[i] * 2.0
fn project[N](x: &tensor[N] f32, y: &mut tensor[N] f32):
    for i in 0..N:
        y[i] = x[i] + 1.0
fn pick(scores: &tensor[4] f32, routes: &mut tensor[1] i32):
    if scores[0] > scores[1]:
        routes[0] = 0
    else:
        routes[0] = 1
fn probe(x: &tensor[4] f32, w: &tensor[4] f32, normalized: &mut tensor[4] f32, logits: &mut tensor[4] f32, routes: &mut tensor[1] i32, expert: &mut tensor[4] f32):
    norm(x, normalized)
    project(normalized, logits)
    pick(logits, routes)
    project(w, expert)
";
    let analyzed = analyze(
        &[("router.seismic", SOURCE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let norm = analyzed.call(root, "norm", 0);
    let router = analyzed.call(root, "project", 0);
    let pick = analyzed.call(root, "pick", 0);
    let expert = analyzed.call(root, "project", 1);
    assert!(analyzed.demand.call_demand(root, norm.0).any());
    assert!(analyzed.demand.call_demand(root, router.0).any());
    assert!(!analyzed.demand.call_demand(root, pick.0).any());
    assert!(!analyzed.demand.call_demand(root, expert.0).any());
    for parameter in ["x", "normalized", "logits", "routes"] {
        assert!(
            analyzed
                .demand
                .is_demanded(root, analyzed.parameter(root, parameter)),
            "{parameter} is on the routing path"
        );
    }
    for parameter in ["w", "expert"] {
        assert!(
            !analyzed
                .demand
                .is_demanded(root, analyzed.parameter(root, parameter)),
            "{parameter} only produces floating outputs"
        );
    }
    assert!(analyzed.demanded(norm.1, SOURCE, "x[i] * 2.0"));
    assert!(analyzed.demanded(router.1, SOURCE, "x[i] + 1.0"));
    assert!(!analyzed.demanded(expert.1, SOURCE, "x[i] + 1.0"));
    assert!(analyzed.demanded(pick.1, SOURCE, "scores[0] > scores[1]"));
}

#[test]
fn carried_tuple_demands_only_the_discrete_consumed_component() {
    // Component 0 is a float nobody demands, component 1 an integer, component
    // 2 a float cast to the i32 result. A wrong component offset or a reversed
    // tuple edge moves demand between the two float components.
    const SOURCE: &str =
        "fn probe[N](x: &tensor[N] f32, y: &tensor[N] f32, out: &mut tensor[1] f32) -> i32:
    let mut state = (y[0] * 2.0, i32(0), x[0] + 1.0)
    for i in 0..N:
        let (free, count, exact) = state
        state = (free * 5.0, count + 1, exact * 3.0)
    let (published, total, decided) = state
    out[0] = published
    return i32(decided) + total
";
    let analyzed = analyze(
        &[("tuple.seismic", SOURCE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    assert!(analyzed.demanded(root, SOURCE, "x[0] + 1.0"));
    assert!(analyzed.demanded(root, SOURCE, "exact * 3.0"));
    assert!(!analyzed.demanded(root, SOURCE, "y[0] * 2.0"));
    assert!(!analyzed.demanded(root, SOURCE, "free * 5.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "y")));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "out")));
}

#[test]
fn tuple_components_keep_their_tensor_classes() {
    // A tuple of a demanded tensor and an undemanded float: the tensor
    // component shares its base's class through pack and projection, and only
    // it reaches the discrete result.
    const SOURCE: &str =
        "fn probe(x: &tensor[4] f32, y: &tensor[4] f32, out: &mut tensor[1] f32) -> i32:
    let pair = (y[0] * 2.0, x[1:3])
    let (scale, window) = pair
    out[0] = scale
    return i32(window[0])
";
    let analyzed = analyze(
        &[("pair.seismic", SOURCE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    let function = analyzed.function(root);
    let classes = ContentClasses::analyze(function);
    let x = analyzed.parameter(root, "x");
    let (pair, window) = function
        .nodes(function.root())
        .find_map(|(_, node)| match node.view() {
            SemanticNodeView::TupleGet {
                tuple,
                index: 1,
                output,
            } => Some((tuple, output)),
            _ => None,
        })
        .expect("the window projection");
    assert_eq!(classes.class(window), classes.class(x));
    // The tensor is the second leaf of the pair: the float leaf has no class.
    let pair_leaves = classes.leaf_range(pair);
    assert_eq!(pair_leaves.len(), 2);
    assert_eq!(classes.leaf_class(pair_leaves.start), None);
    assert_eq!(
        classes.component_range(pair, 1),
        pair_leaves.start + 1..pair_leaves.end
    );
    assert_eq!(
        classes.leaf_class(pair_leaves.start + 1),
        Some(classes.class(x))
    );
    assert_ne!(
        classes.class(x),
        classes.class(analyzed.parameter(root, "y"))
    );
    assert!(analyzed.demand.is_demanded(root, x));
    assert!(!analyzed.demanded(root, SOURCE, "y[0] * 2.0"));
    assert!(!analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "y")));
}

#[test]
fn scalar_captured_by_an_if_arm_is_demanded() {
    // `s` is compared nowhere outside the arm; only the capture edge carries
    // the arm's demand back to it.
    const SOURCE: &str = "fn probe(x: &tensor[4] f32, flag: i32) -> i32:
    let s = x[0] * 2.0
    let mut r = i32(0)
    if flag > 0:
        r = i32(s)
    return r
";
    let analyzed = analyze(&[("if.seismic", SOURCE)], "probe", &ElementBindings::new());
    let root = analyzed.root();
    assert!(analyzed.demanded(root, SOURCE, "x[0] * 2.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
}

#[test]
fn scalar_captured_by_a_loop_body_is_demanded() {
    const SOURCE: &str = "fn probe[N](x: &tensor[N] f32) -> i32:
    let s = x[0] * 2.0
    let mut r = i32(0)
    for i in 0..N:
        r = r + i32(s)
    return r
";
    let analyzed = analyze(
        &[("loop.seismic", SOURCE)],
        "probe",
        &ElementBindings::new(),
    );
    let root = analyzed.root();
    assert!(analyzed.demanded(root, SOURCE, "x[0] * 2.0"));
    assert!(analyzed
        .demand
        .is_demanded(root, analyzed.parameter(root, "x")));
}
