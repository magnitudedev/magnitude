//! Emits a compiled `seismic::EntryContract` implementation for every exported
//! entry in a checked Seismic module.
//!
//! All admission checks (`where`), default specialization resolution
//! (`default`, `least where`, `greatest where`, defaults-first walk), result
//! tensor sizing, scratch buffer sizing (`bytes`, `when`), active launch
//! counting (`launch ... when`, `repeat`), and operand form declarations
//! (`ScratchOperand`) are lowered directly to compiled Rust with zero
//! `.seismicbundle` decoding and zero `HashMap` allocation.

use crate::BuildError;
use seismic_lang::checked::{
    CheckedModule, ElementDomain, ElementQuery, ElementSummary, ElementTarget, ElementTerm,
    ElementUses, Encoded, EntryInfo, NativeComparison, NativeCondition, NativeDefault,
    NativeImplementation, NativeNatExpr, NativeOperand, NativeParameter, NativeScratch,
    ParameterSummaryKind, ResultSummaryKind,
};
use seismic_lang::entry::{AliasRule, ElementBindings, ParameterKind, ResultKind, TensorAccess};
use seismic_lang::expr::{AnyExpr, BinaryOp, BoolExpr, CmpOp, ExprArena, NaryOp, NatExpr, NodeView, SymbolId, UnaryOp};
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{self, BackendName, RowPart};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

fn backend_variant(backend: BackendName) -> &'static str {
    match backend {
        BackendName::Cpu => "Cpu",
        BackendName::Metal => "Metal",
        BackendName::Cuda => "Cuda",
        BackendName::Vulkan => "Vulkan",
    }
}

fn backend_slug(backend: BackendName) -> &'static str {
    match backend {
        BackendName::Cpu => "cpu",
        BackendName::Metal => "metal",
        BackendName::Cuda => "cuda",
        BackendName::Vulkan => "vulkan",
    }
}

fn row_part_variant(part: RowPart) -> &'static str {
    match part {
        RowPart::Codes => "Codes",
        RowPart::Coefficients => "Coefficients",
    }
}

/// Emit `impl seismic::EntryContract for Entry` inside the current entry's
/// generated module.
pub(crate) fn render_contract(
    out: &mut String,
    module: &CheckedModule,
    entry: &EntryInfo,
) -> Result<(), BuildError> {
    let natives: Vec<(BackendName, &NativeImplementation)> = BackendName::ALL
        .into_iter()
        .filter_map(|backend| {
            module
                .native_implementation(entry.id, backend)
                .map(|native| (backend, native))
        })
        .collect();

    let schema_info = if natives.is_empty() {
        ExtractedSchemaInfo {
            tensor_results: Vec::new(),
            tensor_params: Vec::new(),
            result_dims: BTreeSet::new(),
            non_monotonic_result_dims: BTreeSet::new(),
            schema_word_count: 0,
            tensor_param_count: 0,
            bound_tensor_param_count: 0,
            tensor_result_count: 0,
            result_ranks: Vec::new(),
            disjoint: Vec::new(),
            parks: false,
        }
    } else {
        extract_schema_info(module, entry, &natives)?
    };

    if !natives.is_empty() {
        out.push_str("  #[allow(unused_variables, unused_parens, unreachable_code, clippy::all)]\n");
        out.push_str("  mod __contract {\n");
        render_contract_admit_helpers(out, entry, &natives);
        render_contract_default_params_helpers(out, entry, &natives);
        render_contract_sizing_helpers(out, entry, &natives, &schema_info.tensor_results);
        render_contract_sizing_bound_helpers(out, entry, &natives, &schema_info.tensor_results);
        render_contract_scratch_configs_helpers(out, entry, &natives);
        render_contract_sizing_at_point_helpers(out, entry, &natives, &schema_info.tensor_results);
        render_contract_active_launches_helpers(out, entry, &natives);
        render_contract_launch_configs_helpers(out, entry, &natives);
        render_contract_launch_mask_helpers(out, entry, &natives);
        out.push_str("  }\n");
    }

    out.push_str("  #[allow(unused_variables, unused_parens, unreachable_code, clippy::all)]\n");
    out.push_str("  impl seismic::EntryContract for Entry {\n");

    render_contract_admit(out, &natives);
    render_contract_default_params(out, &natives);
    render_contract_sizing(out, &natives);
    render_contract_node_shape(out, &natives, &schema_info);
    render_contract_scratch_meta(out, &natives);
    render_contract_param_spec(out, entry, &natives, &schema_info.tensor_params);
    render_contract_sizing_bound(out, &natives);
    render_contract_scratch_configs(out, &natives);
    render_contract_sizing_at_point(out, &natives);
    render_contract_active_launches(out, &natives);
    render_contract_launch_configs(out, &natives);
    render_contract_launch_mask(out, &natives);

    out.push_str("  }\n");
    Ok(())
}

/// One tensor result leaf extracted from the entry's `CallSchema`: its element
/// summary and lowered Rust expressions for each axis extent.
struct LoweredTensorResult {
    element: ElementSummary,
    axis_exprs: Vec<String>,
}

/// One tensor parameter leaf extracted from the entry's `CallSchema`: its name,
/// element summary, and lowered Rust expressions for each axis extent.
struct LoweredTensorParam {
    name: String,
    element: ElementSummary,
    axis_exprs: Vec<String>,
}

/// All static schema information extracted once per entry from `CallSchema`.
struct ExtractedSchemaInfo {
    tensor_results: Vec<LoweredTensorResult>,
    tensor_params: Vec<LoweredTensorParam>,
    /// Dimension names referenced in any tensor result axis expression.
    result_dims: BTreeSet<String>,
    /// Subset of `result_dims` not proven monotonically non-decreasing across
    /// all tensor result axis expressions.
    non_monotonic_result_dims: BTreeSet<String>,
    schema_word_count: usize,
    tensor_param_count: usize,
    bound_tensor_param_count: usize,
    tensor_result_count: usize,
    result_ranks: Vec<Option<usize>>,
    disjoint: Vec<[usize; 2]>,
    parks: bool,
}

/// Build one valid `ElementBindings` for `domain` so we can monomorphize
/// `entry` once at build time to inspect its result tensor axis `NatExpr`s.
fn sample_element_bindings(domain: &ElementDomain) -> Option<ElementBindings> {
    let params = domain.parameters();
    if params.is_empty() {
        return Some(ElementBindings::new());
    }

    let all_reps = registry::representations();
    let conversions = registry::representation_conversions();

    let is_conversion_param = |name: &str| {
        domain.conversions().iter().any(|c| {
            c.source == name
                || matches!(&c.target, ElementTarget::Parameter(target) if target == name)
        })
    };

    // Build compact candidate lists for each parameter, preferring dense f32/bf16/f16 first.
    let mut candidates: Vec<Vec<RepresentationId>> = Vec::with_capacity(params.len());
    for param in params {
        let mut list: Vec<RepresentationId> = Vec::new();
        for preferred in ["f32", "bf16", "f16"] {
            if let Some(id) = registry::representation(preferred) {
                if param.uses.admits(id) {
                    list.push(id);
                }
            }
        }
        if is_conversion_param(&param.name) {
            for conv in conversions {
                for id in [conv.source, conv.destination] {
                    if param.uses.admits(id) && !list.contains(&id) {
                        list.push(id);
                    }
                }
            }
            for info in all_reps {
                if param.uses.admits(info.id) && !list.contains(&info.id) {
                    list.push(info.id);
                }
            }
        } else {
            let mut seen_operands: BTreeSet<u64> = list
                .iter()
                .map(|&id| registry::representation_info(id).operands.code())
                .collect();
            for info in all_reps {
                if param.uses.admits(info.id) {
                    let code = info.operands.code();
                    if seen_operands.insert(code) {
                        list.push(info.id);
                    }
                }
            }
        }
        candidates.push(list);
    }

    let mut chosen: Vec<Option<RepresentationId>> = vec![None; params.len()];
    fn search(
        idx: usize,
        domain: &ElementDomain,
        candidates: &[Vec<RepresentationId>],
        chosen: &mut [Option<RepresentationId>],
    ) -> Option<ElementBindings> {
        let params = domain.parameters();
        if idx == params.len() {
            let mut bindings = ElementBindings::new();
            for (param, rep) in params.iter().zip(chosen.iter()) {
                bindings = bindings.bind(&param.name, rep.unwrap());
            }
            return domain.admit(&bindings).ok().map(|()| bindings);
        }

        for &rep in &candidates[idx] {
            chosen[idx] = Some(rep);
            // Prune conversions whose endpoints are already assigned.
            let conv_ok = domain.conversions().iter().all(|conv| {
                let src = params
                    .iter()
                    .position(|p| p.name == conv.source)
                    .and_then(|i| chosen[i]);
                let dst = match &conv.target {
                    ElementTarget::Parameter(name) => params
                        .iter()
                        .position(|p| p.name == *name)
                        .and_then(|i| chosen[i]),
                    ElementTarget::Concrete(id) => Some(*id),
                };
                match (src, dst) {
                    (Some(s), Some(t)) => {
                        registry::representation_conversion(s, t).is_some()
                            || registry::dense_conversion(s, t).is_some()
                    }
                    _ => true,
                }
            });
            if !conv_ok {
                continue;
            }
            // Prune predicates whose terms are already assigned.
            let pred_ok = domain.predicates().iter().all(|pred| {
                let eval = |term: &ElementTerm| match term {
                    ElementTerm::Constant(v) => Some(*v),
                    ElementTerm::Operands(name) => params
                        .iter()
                        .position(|p| p.name == *name)
                        .and_then(|i| chosen[i])
                        .map(|id| registry::representation_info(id).operands.code()),
                };
                match (eval(&pred.left), eval(&pred.right)) {
                    (Some(l), Some(r)) => if pred.equal { l == r } else { l != r },
                    _ => true,
                }
            });
            if !pred_ok {
                continue;
            }
            if let Some(found) = search(idx + 1, domain, candidates, chosen) {
                return Some(found);
            }
        }
        chosen[idx] = None;
        None
    }

    search(0, domain, &candidates, &mut chosen)
}

fn extract_schema_info(
    module: &CheckedModule,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) -> Result<ExtractedSchemaInfo, BuildError> {
    let bindings = sample_element_bindings(&entry.element_domain).unwrap_or_else(|| {
        panic!(
            "entry `{}` has an uninhabited element domain",
            entry.name
        )
    });
    let logical = module.entry(entry.id, &bindings).map_err(BuildError::Source)?;
    let schema = logical.schema();
    let arena = logical.arena();

    let mut sym_to_dim: HashMap<SymbolId, usize> = HashMap::new();
    for dim in schema.dimensions() {
        let idx = entry
            .dimensions
            .iter()
            .position(|name| name == &dim.name)
            .unwrap_or_else(|| {
                panic!(
                    "entry `{}` schema dimension `{}` missing from EntryInfo::dimensions",
                    entry.name, dim.name
                )
            });
        sym_to_dim.insert(dim.symbol, idx);
    }

    let mut tensor_results = Vec::new();
    let mut result_dim_indices = BTreeSet::new();
    let mut non_monotonic_result_dim_indices = BTreeSet::new();
    let mut result_ranks = Vec::with_capacity(schema.results().len());
    for (summary, leaf) in entry.results.iter().zip(schema.results().iter()) {
        if let (
            ResultSummaryKind::Tensor { element, .. },
            ResultKind::Tensor { axes, .. },
        ) = (&summary.kind, &leaf.kind)
        {
            let mut axis_exprs = Vec::with_capacity(axes.len());
            for axis in axes {
                let mut axis_dims = BTreeSet::new();
                axis_exprs.push(lower_schema_nat(arena, *axis, &sym_to_dim, Some(&mut axis_dims)));
                for &dim_idx in &axis_dims {
                    if !schema_expr_is_non_decreasing_in(
                        arena,
                        (*axis).into(),
                        &sym_to_dim,
                        dim_idx,
                    ) {
                        non_monotonic_result_dim_indices.insert(dim_idx);
                    }
                    result_dim_indices.insert(dim_idx);
                }
            }
            result_ranks.push(Some(axes.len()));
            tensor_results.push(LoweredTensorResult {
                element: element.clone(),
                axis_exprs,
            });
        } else {
            result_ranks.push(None);
        }
    }
    let result_dims: BTreeSet<String> = result_dim_indices
        .into_iter()
        .map(|idx| entry.dimensions[idx].clone())
        .collect();
    let non_monotonic_result_dims: BTreeSet<String> = non_monotonic_result_dim_indices
        .into_iter()
        .map(|idx| entry.dimensions[idx].clone())
        .collect();

    // CPU natives intentionally compile one unspecialized kernel (empty `statics`),
    // so intersect `statics` across the static-specializing (non-CPU) backends.
    let mut specializing = natives
        .iter()
        .filter(|(backend, _)| *backend != BackendName::Cpu);
    let mut static_dims: BTreeSet<&str> = specializing
        .next()
        .map(|(_, native)| native.statics.iter().map(String::as_str).collect())
        .unwrap_or_default();
    for (_, native) in specializing {
        let backend_statics: BTreeSet<&str> =
            native.statics.iter().map(String::as_str).collect();
        static_dims.retain(|d| backend_statics.contains(d));
    }

    let mut tensor_params = Vec::new();
    let mut tensor_param_count = 0usize;
    let mut bound_tensor_param_count = 0usize;
    for (summary, leaf) in entry.parameters.iter().zip(schema.parameters().iter()) {
        if let (
            ParameterSummaryKind::Tensor { element, .. },
            ParameterKind::Tensor { access, axes, .. },
        ) = (&summary.kind, &leaf.kind)
        {
            tensor_param_count += 1;
            if !matches!(access, TensorAccess::Host { .. }) {
                bound_tensor_param_count += 1;
            }
            let mut axis_exprs = Vec::with_capacity(axes.len());
            for (axis_idx, axis) in axes.iter().enumerate() {
                assert_axis_expr_non_decreasing(
                    arena,
                    (*axis).into(),
                    &sym_to_dim,
                    &entry.dimensions,
                    &static_dims,
                )
                .map_err(|reason| {
                    BuildError::Environment(format!(
                        "entry `{}` tensor parameter `{}` axis {axis_idx} is not monotonically non-decreasing: {reason}",
                        entry.name, summary.name
                    ))
                })?;
                axis_exprs.push(lower_schema_nat(arena, *axis, &sym_to_dim, None));
            }
            tensor_params.push(LoweredTensorParam {
                name: summary.name.clone(),
                element: element.clone(),
                axis_exprs,
            });
        }
    }

    let schema_word_count = schema.dimensions().len()
        + schema
            .parameters()
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { axes, .. } => axes.len() * 2,
                ParameterKind::Range { .. } => 2,
                ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => 1,
            })
            .sum::<usize>()
        + schema
            .results()
            .iter()
            .map(|result| match &result.kind {
                ResultKind::Tensor { axes, .. } => axes.len() * 2,
                ResultKind::Range { .. } | ResultKind::Scalar(_) | ResultKind::Index { .. } => 0,
            })
            .sum::<usize>();

    let param_ordinal = |id| {
        schema
            .parameters()
            .iter()
            .position(|parameter| parameter.id == id)
            .expect("alias rule names a schema parameter")
    };
    let disjoint = schema
        .aliases()
        .iter()
        .filter_map(|rule| match *rule {
            AliasRule::Disjoint(first, second) => Some([param_ordinal(first), param_ordinal(second)]),
            _ => None,
        })
        .collect();

    Ok(ExtractedSchemaInfo {
        tensor_result_count: tensor_results.len(),
        tensor_results,
        tensor_params,
        result_dims,
        non_monotonic_result_dims,
        schema_word_count,
        tensor_param_count,
        bound_tensor_param_count,
        result_ranks,
        disjoint,
        parks: schema.parks(),
    })
}

/// Verify at build time that a tensor parameter axis expression is
/// monotonically non-decreasing in every dynamic (non-static) `u64` dimension.
///
/// `magnitude-executor`'s graph regime sizing (`programs/graph/sizing.rs`)
/// relies on this invariant to evaluate maximum parameter byte capacities at
/// the upper corner `(max(axis_0), ..., max(axis_k))` of each regime slice box
/// rather than enumerating interior points.
///
/// Non-decreasing operations (`NatConst`, non-negative `IntConst`, `Symbol`,
/// `Add`, `Mul`, `NaryOp::Product`, `Min`, `Max`, `Div`/`CeilDiv`/`AlignUp` by
/// a positive constant, and `Sub` whose RHS depends only on constants or static
/// dimensions such as 0/1 structural flags `VK`/`PR`) pass. Anything that can
/// decrease as a dynamic dimension increases (`Sub` with a dynamic dimension on
/// the RHS, `Div`/`CeilDiv`/`AlignUp` by a non-constant or zero, `Rem`,
/// `Select`, or unknown nodes) returns `Err`.
fn assert_axis_expr_non_decreasing(
    arena: &ExprArena,
    node: AnyExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    dim_names: &[String],
    static_dims: &BTreeSet<&str>,
) -> Result<(), String> {
    match arena.view(node) {
        NodeView::NatConst(_) => Ok(()),
        NodeView::IntConst(v) => {
            if v >= 0 {
                Ok(())
            } else {
                Err(format!("negative integer constant {v}"))
            }
        }
        NodeView::Symbol(sym) => {
            if sym_to_dim.contains_key(&sym) {
                Ok(())
            } else {
                Err(format!("unknown symbol {sym:?}"))
            }
        }
        NodeView::Unary {
            op: UnaryOp::NatFromInt | UnaryOp::IntFromNat,
            operand,
        } => assert_axis_expr_non_decreasing(arena, operand, sym_to_dim, dim_names, static_dims),
        NodeView::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Min | BinaryOp::Max => {
                assert_axis_expr_non_decreasing(arena, lhs, sym_to_dim, dim_names, static_dims)?;
                assert_axis_expr_non_decreasing(arena, rhs, sym_to_dim, dim_names, static_dims)
            }
            BinaryOp::Sub => {
                assert_axis_expr_non_decreasing(arena, lhs, sym_to_dim, dim_names, static_dims)?;
                if let Some(dim) =
                    first_dynamic_dim(arena, rhs, sym_to_dim, dim_names, static_dims)?
                {
                    Err(format!(
                        "subtraction has dynamic dimension `{dim}` on RHS, which can decrease when `{dim}` increases"
                    ))
                } else {
                    Ok(())
                }
            }
            BinaryOp::Div | BinaryOp::CeilDiv | BinaryOp::AlignUp => {
                assert_axis_expr_non_decreasing(arena, lhs, sym_to_dim, dim_names, static_dims)?;
                match eval_const_u64(arena, rhs) {
                    Some(v) if v > 0 => Ok(()),
                    Some(_) => Err(format!("{op:?} has zero divisor/alignment")),
                    None => Err(format!(
                        "{op:?} has non-constant RHS, which can decrease when a dimension increases"
                    )),
                }
            }
            other => Err(format!("non-monotone binary op {other:?}")),
        },
        NodeView::Nary {
            op: NaryOp::Product,
            operands,
        } => {
            for &op in operands {
                assert_axis_expr_non_decreasing(arena, op, sym_to_dim, dim_names, static_dims)?;
            }
            Ok(())
        }
        NodeView::Select { .. } => {
            Err("conditional `Select` is not guaranteed non-decreasing".to_owned())
        }
        other => Err(format!("unsupported axis expression node {other:?}")),
    }
}

fn eval_const_u64(arena: &ExprArena, node: AnyExpr) -> Option<u64> {
    match arena.view(node) {
        NodeView::NatConst(v) => Some(v),
        NodeView::IntConst(v) => u64::try_from(v).ok(),
        NodeView::Unary {
            op: UnaryOp::NatFromInt | UnaryOp::IntFromNat,
            operand,
        } => eval_const_u64(arena, operand),
        NodeView::Binary { op, lhs, rhs } => {
            let l = eval_const_u64(arena, lhs)?;
            let r = eval_const_u64(arena, rhs)?;
            match op {
                BinaryOp::Add => l.checked_add(r),
                BinaryOp::Sub => l.checked_sub(r),
                BinaryOp::Mul => l.checked_mul(r),
                BinaryOp::Div => l.checked_div(r),
                BinaryOp::CeilDiv => (r > 0).then(|| l.div_ceil(r)),
                BinaryOp::AlignUp => (r > 0).then(|| l.div_ceil(r).checked_mul(r)).flatten(),
                BinaryOp::Min => Some(l.min(r)),
                BinaryOp::Max => Some(l.max(r)),
                _ => None,
            }
        }
        NodeView::Nary {
            op: NaryOp::Product,
            operands,
        } => {
            let mut acc = 1u64;
            for &op in operands {
                acc = acc.checked_mul(eval_const_u64(arena, op)?)?;
            }
            Some(acc)
        }
        _ => None,
    }
}

fn first_dynamic_dim<'a>(
    arena: &ExprArena,
    node: AnyExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    dim_names: &'a [String],
    static_dims: &BTreeSet<&str>,
) -> Result<Option<&'a str>, String> {
    match arena.view(node) {
        NodeView::NatConst(_) => Ok(None),
        NodeView::IntConst(v) => {
            if v >= 0 {
                Ok(None)
            } else {
                Err(format!("negative integer constant {v}"))
            }
        }
        NodeView::Symbol(sym) => {
            let &idx = sym_to_dim
                .get(&sym)
                .ok_or_else(|| format!("unknown symbol {sym:?}"))?;
            let name = dim_names[idx].as_str();
            if static_dims.contains(name) {
                Ok(None)
            } else {
                Ok(Some(name))
            }
        }
        NodeView::Unary {
            op: UnaryOp::NatFromInt | UnaryOp::IntFromNat,
            operand,
        } => first_dynamic_dim(arena, operand, sym_to_dim, dim_names, static_dims),
        NodeView::Binary { lhs, rhs, .. } => {
            if let Some(dim) = first_dynamic_dim(arena, lhs, sym_to_dim, dim_names, static_dims)? {
                return Ok(Some(dim));
            }
            first_dynamic_dim(arena, rhs, sym_to_dim, dim_names, static_dims)
        }
        NodeView::Nary { operands, .. } => {
            for &op in operands {
                if let Some(dim) =
                    first_dynamic_dim(arena, op, sym_to_dim, dim_names, static_dims)?
                {
                    return Ok(Some(dim));
                }
            }
            Ok(None)
        }
        NodeView::Select { .. } => {
            Err("conditional `Select` on RHS of subtraction".to_owned())
        }
        other => Err(format!("unsupported axis expression node {other:?}")),
    }
}

/// Conservative check for whether a schema expression mentions `target_dim`
/// (returning `true` for unknown nodes/symbols so callers remain sound).
fn schema_expr_mentions_dim(
    arena: &ExprArena,
    node: AnyExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    target_dim: usize,
) -> bool {
    match arena.view(node) {
        NodeView::NatConst(_) | NodeView::IntConst(_) | NodeView::BoolConst(_) => false,
        NodeView::Symbol(sym) => sym_to_dim.get(&sym).is_none_or(|&idx| idx == target_dim),
        NodeView::Unary { operand, .. } => {
            schema_expr_mentions_dim(arena, operand, sym_to_dim, target_dim)
        }
        NodeView::Binary { lhs, rhs, .. } | NodeView::Cmp { lhs, rhs, .. } => {
            schema_expr_mentions_dim(arena, lhs, sym_to_dim, target_dim)
                || schema_expr_mentions_dim(arena, rhs, sym_to_dim, target_dim)
        }
        NodeView::Nary { operands, .. } => operands
            .iter()
            .any(|&op| schema_expr_mentions_dim(arena, op, sym_to_dim, target_dim)),
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => {
            schema_expr_mentions_dim(arena, cond.into(), sym_to_dim, target_dim)
                || schema_expr_mentions_dim(arena, then, sym_to_dim, target_dim)
                || schema_expr_mentions_dim(arena, otherwise, sym_to_dim, target_dim)
        }
        _ => true,
    }
}

/// Conservative proof that a schema `NatExpr` is monotonically non-decreasing
/// in `target_dim` (holding all other dimensions fixed).
fn schema_expr_is_non_decreasing_in(
    arena: &ExprArena,
    node: AnyExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    target_dim: usize,
) -> bool {
    if !schema_expr_mentions_dim(arena, node, sym_to_dim, target_dim) {
        return true;
    }
    match arena.view(node) {
        NodeView::NatConst(_) => true,
        NodeView::IntConst(v) => v >= 0,
        NodeView::Symbol(sym) => sym_to_dim.contains_key(&sym),
        NodeView::Unary {
            op: UnaryOp::NatFromInt | UnaryOp::IntFromNat,
            operand,
        } => schema_expr_is_non_decreasing_in(arena, operand, sym_to_dim, target_dim),
        NodeView::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Min | BinaryOp::Max => {
                schema_expr_is_non_decreasing_in(arena, lhs, sym_to_dim, target_dim)
                    && schema_expr_is_non_decreasing_in(arena, rhs, sym_to_dim, target_dim)
            }
            BinaryOp::Sub | BinaryOp::Div | BinaryOp::CeilDiv | BinaryOp::AlignUp => {
                schema_expr_is_non_decreasing_in(arena, lhs, sym_to_dim, target_dim)
                    && !schema_expr_mentions_dim(arena, rhs, sym_to_dim, target_dim)
            }
            _ => false,
        },
        NodeView::Nary {
            op: NaryOp::Product,
            operands,
        } => operands
            .iter()
            .all(|&op| schema_expr_is_non_decreasing_in(arena, op, sym_to_dim, target_dim)),
        _ => false,
    }
}

fn lower_schema_nat(
    arena: &ExprArena,
    node: NatExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    used_dims: Option<&mut BTreeSet<usize>>,
) -> String {
    lower_schema_any_nat(arena, node.into(), sym_to_dim, used_dims)
}

fn lower_schema_any_nat(
    arena: &ExprArena,
    node: AnyExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    mut used_dims: Option<&mut BTreeSet<usize>>,
) -> String {
    match arena.view(node) {
        NodeView::NatConst(v) => format!("{v}u64"),
        NodeView::IntConst(v) => {
            assert!(v >= 0, "negative IntConst {v} in tensor axis expression");
            format!("{}u64", v as u64)
        }
        NodeView::Symbol(sym) => {
            let idx = *sym_to_dim
                .get(&sym)
                .unwrap_or_else(|| panic!("unknown symbol {sym:?} in tensor axis expression"));
            if let Some(dims) = used_dims {
                dims.insert(idx);
            }
            format!("__d_{idx}")
        }
        NodeView::Unary {
            op: UnaryOp::NatFromInt | UnaryOp::IntFromNat,
            operand,
        } => lower_schema_any_nat(arena, operand, sym_to_dim, used_dims),
        NodeView::Binary { op, lhs, rhs } => {
            let l = lower_schema_any_nat(arena, lhs, sym_to_dim, used_dims.as_deref_mut());
            let r = lower_schema_any_nat(arena, rhs, sym_to_dim, used_dims);
            match op {
                BinaryOp::Add => format!("(({l}).checked_add({r}).ok_or(seismic::ContractError::ArithmeticOverflow)?)"),
                BinaryOp::Sub => format!("(({l}).checked_sub({r}).ok_or(seismic::ContractError::ArithmeticOverflow)?)"),
                BinaryOp::Mul => format!("(({l}).checked_mul({r}).ok_or(seismic::ContractError::ArithmeticOverflow)?)"),
                BinaryOp::Div => format!("(({l}).checked_div({r}).ok_or(seismic::ContractError::ArithmeticOverflow)?)"),
                BinaryOp::Rem => format!("(({l}).checked_rem({r}).ok_or(seismic::ContractError::ArithmeticOverflow)?)"),
                BinaryOp::CeilDiv => format!("seismic::contract::rt::ceil_div({l}, {r})?"),
                BinaryOp::AlignUp => format!("seismic::contract::rt::align_up({l}, {r})?"),
                BinaryOp::Min => format!("(({l}).min({r}))"),
                BinaryOp::Max => format!("(({l}).max({r}))"),
                other => panic!("unsupported binary op {other:?} in tensor axis expression"),
            }
        }
        NodeView::Nary {
            op: NaryOp::Product,
            operands,
        } => {
            if operands.is_empty() {
                "1u64".to_owned()
            } else {
                let mut acc = lower_schema_any_nat(arena, operands[0], sym_to_dim, used_dims.as_deref_mut());
                for &op in &operands[1..] {
                    let rhs = lower_schema_any_nat(arena, op, sym_to_dim, used_dims.as_deref_mut());
                    acc = format!("(({acc}).checked_mul({rhs}).ok_or(seismic::ContractError::ArithmeticOverflow)?)");
                }
                acc
            }
        }
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => {
            let c = lower_schema_bool(arena, cond, sym_to_dim, used_dims.as_deref_mut());
            let t = lower_schema_any_nat(arena, then, sym_to_dim, used_dims.as_deref_mut());
            let f = lower_schema_any_nat(arena, otherwise, sym_to_dim, used_dims);
            format!("(if {c} {{ {t} }} else {{ {f} }})")
        }
        other => panic!("unsupported NodeView {other:?} in tensor axis expression"),
    }
}

fn lower_schema_bool(
    arena: &ExprArena,
    node: BoolExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    used_dims: Option<&mut BTreeSet<usize>>,
) -> String {
    lower_schema_any_bool(arena, node.into(), sym_to_dim, used_dims)
}

fn lower_schema_any_bool(
    arena: &ExprArena,
    node: AnyExpr,
    sym_to_dim: &HashMap<SymbolId, usize>,
    mut used_dims: Option<&mut BTreeSet<usize>>,
) -> String {
    match arena.view(node) {
        NodeView::BoolConst(b) => if b { "true" } else { "false" }.to_owned(),
        NodeView::Unary {
            op: UnaryOp::Not,
            operand,
        } => format!("(!({}))", lower_schema_any_bool(arena, operand, sym_to_dim, used_dims)),
        NodeView::Binary { op, lhs, rhs } => {
            let l = lower_schema_any_bool(arena, lhs, sym_to_dim, used_dims.as_deref_mut());
            let r = lower_schema_any_bool(arena, rhs, sym_to_dim, used_dims);
            match op {
                BinaryOp::And => format!("(({l}) && ({r}))"),
                BinaryOp::Or => format!("(({l}) || ({r}))"),
                BinaryOp::Implies => format!("((!({l})) || ({r}))"),
                BinaryOp::Iff => format!("(({l}) == ({r}))"),
                other => panic!("unsupported boolean binary op {other:?} in tensor axis expression"),
            }
        }
        NodeView::Nary { op, operands } => match op {
            NaryOp::All => {
                if operands.is_empty() {
                    "true".to_owned()
                } else {
                    operands
                        .iter()
                        .map(|&o| format!("({})", lower_schema_any_bool(arena, o, sym_to_dim, used_dims.as_deref_mut())))
                        .collect::<Vec<_>>()
                        .join(" && ")
                }
            }
            NaryOp::Any => {
                if operands.is_empty() {
                    "false".to_owned()
                } else {
                    operands
                        .iter()
                        .map(|&o| format!("({})", lower_schema_any_bool(arena, o, sym_to_dim, used_dims.as_deref_mut())))
                        .collect::<Vec<_>>()
                        .join(" || ")
                }
            }
            other => panic!("unsupported boolean nary op {other:?} in tensor axis expression"),
        },
        NodeView::Cmp { op, lhs, rhs } => {
            let l = lower_schema_any_nat(arena, lhs, sym_to_dim, used_dims.as_deref_mut());
            let r = lower_schema_any_nat(arena, rhs, sym_to_dim, used_dims);
            let cmp = match op {
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
            };
            format!("(({l}) {cmp} ({r}))")
        }
        other => panic!("unsupported boolean NodeView {other:?} in tensor axis expression"),
    }
}

/// Emit element binding and `ElementDomain` validation into `out`, binding
/// `__e_0`, `__e_1`, ... in `entry.element_parameters` order.
fn emit_element_binding(out: &mut String, entry: &EntryInfo, indent: &str) {
    let n = entry.element_parameters.len();
    if n == 0 {
        let _ = writeln!(
            out,
            "{indent}if !elements.is_empty() {{ return Err(seismic::ContractError::UnsupportedElements); }}"
        );
        return;
    }

    let _ = writeln!(
        out,
        "{indent}if elements.len() != {n} {{ return Err(seismic::ContractError::UnsupportedElements); }}"
    );
    let _ = writeln!(
        out,
        "{indent}let mut __e: [Option<seismic::Element>; {n}] = [None; {n}];"
    );
    let _ = writeln!(out, "{indent}for &(name, elem) in elements {{");
    let _ = writeln!(out, "{indent}  let slot = match name {{");
    for (i, param) in entry.element_parameters.iter().enumerate() {
        let _ = writeln!(out, "{indent}    {param:?} => &mut __e[{i}],");
    }
    let _ = writeln!(
        out,
        "{indent}    _ => return Err(seismic::ContractError::UnsupportedElements),"
    );
    let _ = writeln!(out, "{indent}  }};");
    let _ = writeln!(
        out,
        "{indent}  if slot.is_some() {{ return Err(seismic::ContractError::UnsupportedElements); }}"
    );
    let _ = writeln!(out, "{indent}  *slot = Some(elem);");
    let _ = writeln!(out, "{indent}}}");
    for i in 0..n {
        let _ = writeln!(
            out,
            "{indent}let __e_{i} = __e[{i}].ok_or(seismic::ContractError::UnsupportedElements)?;"
        );
    }

    let elem_idx = |name: &str| -> usize {
        entry
            .element_parameters
            .iter()
            .position(|p| p == name)
            .expect("element domain names an entry element parameter")
    };

    for param in entry.element_domain.parameters() {
        let idx = elem_idx(&param.name);
        let ElementUses {
            decoded_read,
            component_read,
            integer_code_read,
            stored,
            record_stored,
            conversion_source,
            partial_copy,
        } = param.uses;
        let _ = writeln!(
            out,
            "{indent}seismic::contract::rt::admits_uses(seismic::contract::rt::ElementUses {{ decoded_read: {decoded_read}, component_read: {component_read}, integer_code_read: {integer_code_read}, stored: {stored}, record_stored: {record_stored}, conversion_source: {conversion_source}, partial_copy: {partial_copy} }}, __e_{idx})?;"
        );
    }

    for conv in entry.element_domain.conversions() {
        let src = format!("__e_{}", elem_idx(&conv.source));
        let dst = match &conv.target {
            ElementTarget::Parameter(name) => format!("__e_{}", elem_idx(name)),
            ElementTarget::Concrete(id) => {
                let name = registry::representation_info(*id).name;
                format!("seismic::contract::rt::element_named({name:?})")
            }
        };
        let _ = writeln!(
            out,
            "{indent}seismic::contract::rt::admits_conversion({src}, {dst})?;"
        );
    }

    for pred in entry.element_domain.predicates() {
        let term = |t: &ElementTerm| match t {
            ElementTerm::Constant(v) => format!("{v}u64"),
            ElementTerm::Operands(name) => {
                format!("seismic::contract::rt::query_operands(__e_{})", elem_idx(name))
            }
        };
        let l = term(&pred.left);
        let r = term(&pred.right);
        let op = if pred.equal { "==" } else { "!=" };
        let _ = writeln!(
            out,
            "{indent}if !(({l}) {op} ({r})) {{ return Err(seismic::ContractError::UnsupportedElements); }}"
        );
    }
}

/// Collect every element parameter name that `native` queries with
/// `ElementQuery::UnitColumns` anywhere in its declaration (since
/// `native_implementation_at` returns `None` if any `unit_columns` read has no
/// value).
fn unit_column_reads(native: &NativeImplementation) -> Vec<String> {
    let mut reads = BTreeSet::new();
    fn visit_expr(expr: &NativeNatExpr, reads: &mut BTreeSet<String>) {
        match expr {
            NativeNatExpr::Element(name, ElementQuery::UnitColumns) => {
                reads.insert(name.clone());
            }
            NativeNatExpr::Constant(_)
            | NativeNatExpr::Dimension(_)
            | NativeNatExpr::Parameter(_)
            | NativeNatExpr::Element(_, _)
            | NativeNatExpr::Device(_) => {}
            NativeNatExpr::Add(l, r)
            | NativeNatExpr::Sub(l, r)
            | NativeNatExpr::Mul(l, r)
            | NativeNatExpr::Div(l, r)
            | NativeNatExpr::Rem(l, r)
            | NativeNatExpr::CeilDiv(l, r)
            | NativeNatExpr::Min(l, r)
            | NativeNatExpr::Max(l, r) => {
                visit_expr(l, reads);
                visit_expr(r, reads);
            }
        }
    }
    fn visit_cond(cond: &NativeCondition, reads: &mut BTreeSet<String>) {
        match cond {
            NativeCondition::Compare { left, right, .. } => {
                visit_expr(left, reads);
                visit_expr(right, reads);
            }
            NativeCondition::And(l, r) | NativeCondition::Or(l, r) => {
                visit_cond(l, reads);
                visit_cond(r, reads);
            }
        }
    }
    fn visit_param(param: &NativeParameter, reads: &mut BTreeSet<String>) {
        match &param.default {
            NativeDefault::Value(e) => visit_expr(e, reads),
            NativeDefault::Least(c) | NativeDefault::Greatest(c) => visit_cond(c, reads),
        }
    }
    for p in &native.params {
        visit_param(p, &mut reads);
    }
    if let Some(c) = &native.constraint {
        visit_cond(c, &mut reads);
    }
    for s in &native.scratch {
        visit_expr(&s.bytes, &mut reads);
        if let Some(w) = &s.when {
            visit_cond(w, &mut reads);
        }
        if let Some(NativeOperand::Formed { pairs, formed, nvfp4, .. }) = &s.operand {
            visit_cond(pairs, &mut reads);
            visit_cond(formed, &mut reads);
            visit_cond(nvfp4, &mut reads);
        }
    }
    for l in &native.launches {
        for p in &l.params {
            visit_param(p, &mut reads);
        }
        if let Some(w) = &l.when {
            visit_cond(w, &mut reads);
        }
        for e in l.groups.iter().chain(&l.group_extent) {
            visit_expr(e, &mut reads);
        }
        visit_expr(&l.shared_bytes, &mut reads);
    }
    if let Some(r) = &native.repeat {
        visit_expr(&r.count, &mut reads);
    }
    if let Some(f) = &native.formation {
        for op in &f.operands {
            visit_expr(&op.rows, &mut reads);
            visit_expr(&op.width, &mut reads);
            visit_expr(&op.groups, &mut reads);
            visit_cond(&op.formed, &mut reads);
            visit_cond(&op.tiles, &mut reads);
        }
    }
    reads.into_iter().collect()
}

fn emit_unit_column_checks(
    out: &mut String,
    entry: &EntryInfo,
    native: &NativeImplementation,
    indent: &str,
) {
    for name in unit_column_reads(native) {
        let idx = entry
            .element_parameters
            .iter()
            .position(|p| p == &name)
            .expect("unit_columns names an entry element parameter");
        let _ = writeln!(
            out,
            "{indent}let _ = seismic::contract::rt::query_unit_columns(__e_{idx})?;"
        );
    }
}

/// Flattened list of all declared tuning parameters of one native
/// implementation (entry-wide first, then each launch's local parameters).
struct ParamSlot<'a> {
    launch: Option<usize>,
    param: &'a NativeParameter,
}

fn collect_params(native: &NativeImplementation) -> Vec<ParamSlot<'_>> {
    let mut out = Vec::new();
    for p in &native.params {
        out.push(ParamSlot {
            launch: None,
            param: p,
        });
    }
    for (launch_idx, launch) in native.launches.iter().enumerate() {
        for p in &launch.params {
            out.push(ParamSlot {
                launch: Some(launch_idx),
                param: p,
            });
        }
    }
    out
}

/// Bind static dimensions (from `dim_names`) out of `dims` into `__d_0`, `__d_1`, ...
fn emit_dim_binding(out: &mut String, dim_names: &[String], indent: &str) {
    let n = dim_names.len();
    if n == 0 {
        return;
    }
    let _ = writeln!(out, "{indent}let mut __d: [Option<u64>; {n}] = [None; {n}];");
    let _ = writeln!(out, "{indent}for &(name, val) in dims {{");
    let _ = writeln!(out, "{indent}  match name {{");
    for (i, name) in dim_names.iter().enumerate() {
        let _ = writeln!(out, "{indent}    {name:?} => __d[{i}] = Some(val),");
    }
    let _ = writeln!(out, "{indent}    _ => {{}}");
    let _ = writeln!(out, "{indent}  }}");
    let _ = writeln!(out, "{indent}}}");
    for (i, name) in dim_names.iter().enumerate() {
        let _ = writeln!(
            out,
            "{indent}let __d_{i} = __d[{i}].ok_or(seismic::ContractError::MissingDimension({name:?}))?;"
        );
    }
}

/// Bind and domain-check all declared tuning parameters from `params` into
/// `__p_0`, `__p_1`, ...
fn emit_param_binding(out: &mut String, slots: &[ParamSlot<'_>], indent: &str) {
    let n = slots.len();
    if n == 0 {
        let _ = writeln!(
            out,
            "{indent}if let Some(&(name, _)) = params.first() {{ return Err(seismic::ContractError::InvalidParameter(\"unexpected\")); }}"
        );
        return;
    }
    let _ = writeln!(out, "{indent}let mut __p: [Option<u64>; {n}] = [None; {n}];");
    let _ = writeln!(out, "{indent}for &(name, val) in params {{");
    let _ = writeln!(out, "{indent}  let mut matched = false;");
    let _ = writeln!(out, "{indent}  let mut assigned = false;");
    for (i, slot) in slots.iter().enumerate() {
        let pname = &slot.param.name;
        let _ = writeln!(
            out,
            "{indent}  if !assigned && name == {pname:?} {{ matched = true; if __p[{i}].is_none() {{ __p[{i}] = Some(val); assigned = true; }} }}"
        );
    }
    let _ = writeln!(
        out,
        "{indent}  if !matched {{ return Err(seismic::ContractError::InvalidParameter(\"unknown\")); }}"
    );
    let _ = writeln!(
        out,
        "{indent}  if !assigned {{ return Err(seismic::ContractError::InvalidParameter(\"duplicate\")); }}"
    );
    let _ = writeln!(out, "{indent}}}");
    for (i, slot) in slots.iter().enumerate() {
        let pname = &slot.param.name;
        let _ = writeln!(
            out,
            "{indent}let __p_{i} = __p[{i}].ok_or(seismic::ContractError::InvalidParameter({pname:?}))?;"
        );
        let vals = format_u64_slice(&slot.param.values);
        let _ = writeln!(
            out,
            "{indent}if !{vals}.contains(&__p_{i}) {{ return Err(seismic::ContractError::ParameterOutsideDomain {{ parameter: {pname:?}, value: __p_{i} }}); }}"
        );
    }
}

fn format_u64_slice(values: &[u64]) -> String {
    if values.is_empty() {
        return "[0u64; 0]".to_owned();
    }
    let mut s = String::from("[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{v}u64");
    }
    s.push(']');
    s
}

struct ExprCtx<'a> {
    backend: BackendName,
    elements: &'a [String],
    dims: &'a [String],
    params: &'a [ParamSlot<'a>],
    /// When lowering inside a launch's `when` condition, resolve its local
    /// parameters first.
    launch: Option<usize>,
    /// When lowering a parameter's `least where` / `greatest where` default,
    /// only that parameter is in scope under `__cand`.
    default_param: Option<&'a str>,
}

impl<'a> ExprCtx<'a> {
    fn resolve_param_index(&self, name: &str) -> Option<usize> {
        if let Some(launch) = self.launch {
            if let Some(idx) = self
                .params
                .iter()
                .position(|s| s.launch == Some(launch) && s.param.name == name)
            {
                return Some(idx);
            }
        }
        if let Some(idx) = self
            .params
            .iter()
            .position(|s| s.launch.is_none() && s.param.name == name)
        {
            return Some(idx);
        }
        let mut launch_matches = self
            .params
            .iter()
            .enumerate()
            .filter(|(_, s)| s.launch.is_some() && s.param.name == name)
            .map(|(i, _)| i);
        let first = launch_matches.next()?;
        launch_matches.next().is_none().then_some(first)
    }

    fn expect_param_index(&self, name: &str) -> usize {
        self.resolve_param_index(name)
            .unwrap_or_else(|| panic!("unknown or ambiguous parameter `{name}` in native expression"))
    }

    fn lower_nat(&self, expr: &NativeNatExpr) -> String {
        match expr {
            NativeNatExpr::Constant(v) => format!("{v}u64"),
            NativeNatExpr::Dimension(name) => {
                let idx = self
                    .dims
                    .iter()
                    .position(|d| d == name)
                    .unwrap_or_else(|| panic!("unknown dimension `{name}` in native expression"));
                format!("__d_{idx}")
            }
            NativeNatExpr::Parameter(name) => {
                if self.default_param == Some(name.as_str()) {
                    return "__cand".to_owned();
                }
                let idx = self.expect_param_index(name);
                format!("__p_{idx}")
            }
            NativeNatExpr::Element(elem_name, query) => {
                let idx = self
                    .elements
                    .iter()
                    .position(|e| e == elem_name)
                    .unwrap_or_else(|| panic!("unknown element `{elem_name}` in native expression"));
                match query {
                    ElementQuery::Operands => {
                        format!("seismic::contract::rt::query_operands(__e_{idx})")
                    }
                    ElementQuery::Form(form) => format!(
                        "seismic::contract::rt::query_form(seismic::BackendName::{}, {:?}, __e_{idx})",
                        backend_variant(self.backend),
                        form.name()
                    ),
                    ElementQuery::UnitColumns => {
                        format!("seismic::contract::rt::query_unit_columns(__e_{idx})?")
                    }
                    ElementQuery::Packed => {
                        format!("seismic::contract::rt::query_packed(__e_{idx})")
                    }
                    ElementQuery::DenseBytes => {
                        format!("seismic::contract::rt::query_dense_bytes(__e_{idx})")
                    }
                    ElementQuery::Axis(axis) => format!(
                        "seismic::contract::rt::query_axis({:?}, __e_{idx})",
                        axis.name()
                    ),
                    ElementQuery::RowBytes(part, values) => format!(
                        "seismic::contract::rt::query_row_bytes(seismic::contract::rt::RowPart::{}, {values}u64, __e_{idx})",
                        row_part_variant(*part)
                    ),
                }
            }
            NativeNatExpr::Device(name) => format!(
                "{{ let __dev: u64 = Err(seismic::ContractError::DeviceDependent({name:?}))?; __dev }}"
            ),
            NativeNatExpr::Add(l, r) => format!(
                "(({}).checked_add({}).ok_or(seismic::ContractError::ArithmeticOverflow)?)",
                self.lower_nat(l),
                self.lower_nat(r)
            ),
            NativeNatExpr::Sub(l, r) => format!(
                "(({}).checked_sub({}).ok_or(seismic::ContractError::ArithmeticOverflow)?)",
                self.lower_nat(l),
                self.lower_nat(r)
            ),
            NativeNatExpr::Mul(l, r) => format!(
                "(({}).checked_mul({}).ok_or(seismic::ContractError::ArithmeticOverflow)?)",
                self.lower_nat(l),
                self.lower_nat(r)
            ),
            NativeNatExpr::Div(l, r) => format!(
                "(({}).checked_div({}).ok_or(seismic::ContractError::ArithmeticOverflow)?)",
                self.lower_nat(l),
                self.lower_nat(r)
            ),
            NativeNatExpr::Rem(l, r) => format!(
                "(({}).checked_rem({}).ok_or(seismic::ContractError::ArithmeticOverflow)?)",
                self.lower_nat(l),
                self.lower_nat(r)
            ),
            NativeNatExpr::CeilDiv(l, r) => format!(
                "seismic::contract::rt::ceil_div({}, {})?",
                self.lower_nat(l),
                self.lower_nat(r)
            ),
            NativeNatExpr::Min(l, r) => {
                format!("(({}).min({}))", self.lower_nat(l), self.lower_nat(r))
            }
            NativeNatExpr::Max(l, r) => {
                format!("(({}).max({}))", self.lower_nat(l), self.lower_nat(r))
            }
        }
    }

    fn lower_cond(&self, cond: &NativeCondition) -> String {
        match cond {
            NativeCondition::Compare {
                comparison,
                left,
                right,
            } => {
                let l = self.lower_nat(left);
                let r = self.lower_nat(right);
                let op = match comparison {
                    NativeComparison::Lt => "<",
                    NativeComparison::Le => "<=",
                    NativeComparison::Gt => ">",
                    NativeComparison::Ge => ">=",
                    NativeComparison::Eq => "==",
                    NativeComparison::Ne => "!=",
                };
                format!("(({l}) {op} ({r}))")
            }
            NativeCondition::And(l, r) => {
                format!("(({}) && ({}))", self.lower_cond(l), self.lower_cond(r))
            }
            NativeCondition::Or(l, r) => {
                format!("(({}) || ({}))", self.lower_cond(l), self.lower_cond(r))
            }
        }
    }
}

fn flatten_conjuncts<'a>(cond: &'a NativeCondition, out: &mut Vec<&'a NativeCondition>) {
    match cond {
        NativeCondition::And(l, r) => {
            flatten_conjuncts(l, out);
            flatten_conjuncts(r, out);
        }
        other => out.push(other),
    }
}

fn render_backend_dispatch(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
    helper_prefix: &str,
    call_args: &str,
) {
    if natives.is_empty() {
        out.push_str("      Err(seismic::ContractError::NoImplementation(backend))\n");
        out.push_str("    }\n");
        return;
    }
    out.push_str("      match backend {\n");
    for &(backend, _) in natives {
        let variant = backend_variant(backend);
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "        seismic::BackendName::{variant} => __contract::{helper_prefix}_{slug}({call_args}),"
        );
    }
    if natives.len() < BackendName::ALL.len() {
        out.push_str("        _ => Err(seismic::ContractError::NoImplementation(backend)),\n");
    }
    out.push_str("      }\n");
    out.push_str("    }\n");
}

fn emit_conjunct_guards(
    out: &mut String,
    ctx: &ExprCtx<'_>,
    conjuncts: &[&NativeCondition],
    on_false: &str,
    indent: &str,
) {
    for c in conjuncts {
        let cond_expr = ctx.lower_cond(c);
        let _ = writeln!(
            out,
            "{indent}if !((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})()?) {{ {on_false} }}"
        );
    }
}

fn emit_constraint_guard(
    out: &mut String,
    ctx: &ExprCtx<'_>,
    constraint: &NativeCondition,
    indent: &str,
) {
    let mut conjuncts = Vec::new();
    flatten_conjuncts(constraint, &mut conjuncts);
    emit_conjunct_guards(
        out,
        ctx,
        &conjuncts,
        "return Err(seismic::ContractError::DomainViolation);",
        indent,
    );
}

fn render_contract_admit(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_admit(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[(&str, u64)]) -> Result<(), seismic::ContractError> {\n",
    );
    render_backend_dispatch(out, natives, "admit", "elements, dims, params");
}

fn render_contract_admit_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn admit_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[(&str, u64)]) -> Result<(), seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &native.statics, "      ");
        let slots = collect_params(native);
        emit_param_binding(out, &slots, "      ");
        if let Some(constraint) = &native.constraint {
            let ctx = ExprCtx {
                backend,
                elements: &entry.element_parameters,
                dims: &native.statics,
                params: &slots,
                launch: None,
                default_param: None,
            };
            emit_constraint_guard(out, &ctx, constraint, "      ");
        }
        out.push_str("      Ok(())\n");
        out.push_str("    }\n");
    }
}

fn render_contract_default_params(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_default_params(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<Vec<(&'static str, u64)>, seismic::ContractError> {\n",
    );
    render_backend_dispatch(out, natives, "default_params", "elements, dims");
}

fn render_contract_default_params_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn default_params_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<Vec<(&'static str, u64)>, seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &native.statics, "      ");
        let slots = collect_params(native);
        let ctx = ExprCtx {
            backend,
            elements: &entry.element_parameters,
            dims: &native.statics,
            params: &slots,
            launch: None,
            default_param: None,
        };

        // Compute each parameter's default and its defaults-first candidate order.
        for (k, slot) in slots.iter().enumerate() {
            let pname = &slot.param.name;
            let vals = format_u64_slice(&slot.param.values);
            match &slot.param.default {
                NativeDefault::Value(expr) => {
                    let default_ctx = ExprCtx {
                        backend,
                        elements: &entry.element_parameters,
                        dims: &native.statics,
                        params: &slots,
                        launch: slot.launch,
                        default_param: None,
                    };
                    let e = default_ctx.lower_nat(expr);
                    let _ = writeln!(
                        out,
                        "      let __def_{k}: u64 = (|| -> Result<u64, seismic::ContractError> {{ Ok({e}) }})()?;"
                    );
                    let _ = writeln!(
                        out,
                        "      if !{vals}.contains(&__def_{k}) {{ return Err(seismic::ContractError::NoDefault({pname:?})); }}"
                    );
                }
                NativeDefault::Least(cond) | NativeDefault::Greatest(cond) => {
                    let mut sorted = slot.param.values.clone();
                    sorted.sort_unstable();
                    if matches!(slot.param.default, NativeDefault::Greatest(_)) {
                        sorted.reverse();
                    }
                    let sorted_lit = format_u64_slice(&sorted);
                    let default_ctx = ExprCtx {
                        backend,
                        elements: &entry.element_parameters,
                        dims: &native.statics,
                        params: &slots,
                        launch: slot.launch,
                        default_param: Some(pname.as_str()),
                    };
                    let cond_expr = default_ctx.lower_cond(cond);
                    let _ = writeln!(
                        out,
                        "      let __def_{k}: u64 = (|| -> Result<u64, seismic::ContractError> {{ for __cand in {sorted_lit} {{ if {cond_expr} {{ return Ok(__cand); }} }} Err(seismic::ContractError::NoDefault({pname:?})) }})()?;"
                    );
                }
            }
            let len = slot.param.values.len();
            if len == 1 {
                let _ = writeln!(out, "      let __ord_{k} = [__def_{k}];");
            } else {
                let _ = writeln!(out, "      let mut __ord_{k} = [0u64; {len}];");
                let _ = writeln!(out, "      __ord_{k}[0] = __def_{k};");
                let _ = writeln!(out, "      let mut __pos_{k} = 1usize;");
                let _ = writeln!(out, "      for __v in {vals} {{");
                let _ = writeln!(
                    out,
                    "        if __v != __def_{k} {{ __ord_{k}[__pos_{k}] = __v; __pos_{k} += 1; }}"
                );
                let _ = writeln!(out, "      }}");
            }
        }

        // Partition top-level conjuncts by the deepest parameter index they read,
        // pruning invalid prefixes at the earliest loop depth.
        let mut conjuncts = Vec::new();
        if let Some(constraint) = &native.constraint {
            flatten_conjuncts(constraint, &mut conjuncts);
        }
        let mut static_conjuncts: Vec<&NativeCondition> = Vec::new();
        let mut depth_conjuncts: Vec<Vec<&NativeCondition>> = vec![Vec::new(); slots.len()];
        for c in &conjuncts {
            let mut names = Vec::new();
            c.parameters(&mut names);
            let max_depth = names
                .iter()
                .map(|n| ctx.expect_param_index(n))
                .max();
            match max_depth {
                None => static_conjuncts.push(*c),
                Some(d) => depth_conjuncts[d].push(*c),
            }
        }

        emit_conjunct_guards(
            out,
            &ctx,
            &static_conjuncts,
            "return Err(seismic::ContractError::DomainViolation);",
            "      ",
        );

        if slots.is_empty() {
            if let Some(constraint) = &native.constraint {
                emit_constraint_guard(out, &ctx, constraint, "      ");
            }
            out.push_str("      Ok(Vec::new())\n");
        } else {
            let mut indent = String::from("      ");
            for k in 0..slots.len() {
                let _ = writeln!(out, "{indent}for &__p_{k} in &__ord_{k} {{");
                indent.push_str("  ");
                for c in &depth_conjuncts[k] {
                    let cond_expr = ctx.lower_cond(c);
                    let _ = writeln!(
                        out,
                        "{indent}if matches!((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})(), Ok(false)) {{ continue; }}"
                    );
                }
            }
            // Innermost leaf: verify full constraint in exact order to preserve
            // arithmetic error semantics, then return the first admissible tuple.
            let mut result_vec = String::from("vec![");
            for (k, slot) in slots.iter().enumerate() {
                if k > 0 {
                    result_vec.push_str(", ");
                }
                let _ = write!(result_vec, "({:?}, __p_{k})", slot.param.name);
            }
            result_vec.push(']');

            if !conjuncts.is_empty() {
                let cond_expr = lower_conjunction(&ctx, &conjuncts);
                let _ = writeln!(out, "{indent}if {cond_expr} {{");
                let _ = writeln!(out, "{indent}  return Ok({result_vec});");
                let _ = writeln!(out, "{indent}}}");
            } else {
                let _ = writeln!(out, "{indent}return Ok({result_vec});");
            }

            for _ in 0..slots.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            out.push_str("      Err(seismic::ContractError::DomainViolation)\n");
        }

        out.push_str("    }\n");
    }
}

fn render_contract_sizing(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_sizing(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[(&str, u64)]) -> Result<seismic::EntrySizing, seismic::ContractError> {\n",
    );
    render_backend_dispatch(out, natives, "sizing", "elements, dims, params");
}

fn render_contract_sizing_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
    tensor_results: &[LoweredTensorResult],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn sizing_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[(&str, u64)]) -> Result<seismic::EntrySizing, seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &entry.dimensions, "      ");
        let slots = collect_params(native);
        emit_param_binding(out, &slots, "      ");

        let ctx = ExprCtx {
            backend,
            elements: &entry.element_parameters,
            dims: &entry.dimensions,
            params: &slots,
            launch: None,
            default_param: None,
        };

        if let Some(constraint) = &native.constraint {
            emit_constraint_guard(out, &ctx, constraint, "      ");
        }

        // Results
        if tensor_results.is_empty() {
            out.push_str("      let __results = Vec::new();\n");
        } else {
            out.push_str(
                "      let __results = (|| -> Result<Vec<seismic::TensorResultSpec>, seismic::ContractError> {\n",
            );
            out.push_str("        Ok(vec![\n");
            for result in tensor_results {
                let elem_expr = match &result.element {
                    ElementSummary::Fixed(name) => {
                        format!("seismic::contract::rt::element_named({name:?})")
                    }
                    ElementSummary::Parameter(name) => {
                        let idx = entry
                            .element_parameters
                            .iter()
                            .position(|p| p == name)
                            .expect("result element names an entry element parameter");
                        format!("__e_{idx}")
                    }
                };
                let extents = result.axis_exprs.join(", ");
                let _ = writeln!(
                    out,
                    "          {{ let __el = {elem_expr}; let __bytes = seismic::contract::rt::canonical_bytes(__el, &[{extents}])?; seismic::TensorResultSpec {{ element: __el, bytes: __bytes }} }},"
                );
            }
            out.push_str("        ])\n");
            out.push_str("      })()?;\n");
        }

        // Scratch buffers + operand forms
        let has_operands = native.scratch.iter().any(|s| s.operand.is_some());
        if native.scratch.is_empty() {
            out.push_str("      let __scratch = Vec::new();\n");
        } else {
            out.push_str("      let __scratch = vec![\n");
            for scratch in &native.scratch {
                let bytes_expr = scratch_bytes_expr(&ctx, scratch);
                let emission = match scratch.emission {
                    Some(r) => format!("Some({r}u32)"),
                    None => "None".to_owned(),
                };
                let _ = writeln!(
                    out,
                    "        seismic::ScratchSpec {{ bytes: {bytes_expr}, sync: {}, emission: {emission} }},",
                    scratch.sync
                );
            }
            out.push_str("      ];\n");
        }

        if !has_operands {
            out.push_str("      let __operand_forms = Vec::new();\n");
        } else {
            out.push_str(
                "      let __operand_forms = (|| -> Result<Vec<seismic::OperandFormSpec>, seismic::ContractError> {\n",
            );
            out.push_str("        Ok(vec![\n");
            for (scratch_index, scratch) in native.scratch.iter().enumerate() {
                if let Some(operand) = &scratch.operand {
                    let (role_expr, rows, norm) = match operand {
                        NativeOperand::Formed {
                            parameter,
                            formation,
                            pairs,
                            rows,
                            norm,
                            formed,
                            nvfp4,
                        } => {
                            let pairs_expr = ctx.lower_cond(pairs);
                            let formed_expr = ctx.lower_cond(formed);
                            let nvfp4_expr = ctx.lower_cond(nvfp4);
                            (
                                format!(
                                    "seismic::OperandRole::Formed {{ parameter: {parameter}u32, formation: {formation}u32, pairs: {pairs_expr}, formed: {formed_expr}, nvfp4: {nvfp4_expr} }}"
                                ),
                                *rows,
                                *norm,
                            )
                        }
                        NativeOperand::Emitted { result, rows, norm } => (
                            format!("seismic::OperandRole::Emitted {{ result: {result}u32 }}"),
                            *rows,
                            *norm,
                        ),
                    };
                    let norm_expr = match norm {
                        Some(n) => format!(
                            "Some(seismic::NativeOperandNorm {{ weight: {}u32, epsilon: {}u32 }})",
                            n.weight, n.epsilon
                        ),
                        None => "None".to_owned(),
                    };
                    let _ = writeln!(
                        out,
                        "          seismic::OperandFormSpec {{ scratch_index: {scratch_index}usize, role: {role_expr}, rows: {rows}, norm: {norm_expr} }},"
                    );
                }
            }
            out.push_str("        ])\n");
            out.push_str("      })()?;\n");
        }

        // Active launches (including repeat rounds if present)
        out.push_str(
            "      let __active_launches: u64 = (|| -> Result<u64, seismic::ContractError> {\n",
        );
        for (launch_idx, launch) in native.launches.iter().enumerate() {
            match &launch.when {
                None => {
                    let _ = writeln!(out, "        let __act_{launch_idx}: u64 = 1u64;");
                }
                Some(when) => {
                    let lctx = ExprCtx {
                        backend,
                        elements: &entry.element_parameters,
                        dims: &entry.dimensions,
                        params: &slots,
                        launch: Some(launch_idx),
                        default_param: None,
                    };
                    let w = lctx.lower_cond(when);
                    let _ = writeln!(
                        out,
                        "        let __act_{launch_idx}: u64 = if (|| -> Result<bool, seismic::ContractError> {{ Ok({w}) }})()? {{ 1u64 }} else {{ 0u64 }};"
                    );
                }
            }
        }

        let sum_range = |range: std::ops::Range<usize>| -> String {
            if range.is_empty() {
                "0u64".to_owned()
            } else {
                range
                    .map(|i| format!("__act_{i}"))
                    .collect::<Vec<_>>()
                    .join(" + ")
            }
        };

        match &native.repeat {
            None => {
                let total = sum_range(0..native.launches.len());
                let _ = writeln!(out, "        Ok({total})");
            }
            Some(repeat) => {
                let rounds = ctx.lower_nat(&repeat.count);
                let before = sum_range(0..repeat.first);
                let body = sum_range(repeat.first..repeat.first + repeat.launches);
                let after = sum_range(repeat.first + repeat.launches..native.launches.len());
                let _ = writeln!(out, "        let __rounds: u64 = {rounds};");
                let _ = writeln!(
                    out,
                    "        ({body}).checked_mul(__rounds).and_then(|v| v.checked_add({before})).and_then(|v| v.checked_add({after})).ok_or(seismic::ContractError::ArithmeticOverflow)"
                );
            }
        }
        out.push_str("      })()?;\n");

        out.push_str(
            "      Ok(seismic::EntrySizing { results: __results, scratch: __scratch, active_launches: __active_launches, operand_forms: __operand_forms })\n",
        );
        out.push_str("    }\n");
    }
}

fn collect_native_nat_dims<'a>(expr: &'a NativeNatExpr, dims: &mut BTreeSet<&'a str>) {
    match expr {
        NativeNatExpr::Dimension(name) => {
            dims.insert(name.as_str());
        }
        NativeNatExpr::Constant(_)
        | NativeNatExpr::Parameter(_)
        | NativeNatExpr::Element(_, _)
        | NativeNatExpr::Device(_) => {}
        NativeNatExpr::Add(l, r)
        | NativeNatExpr::Sub(l, r)
        | NativeNatExpr::Mul(l, r)
        | NativeNatExpr::Div(l, r)
        | NativeNatExpr::Rem(l, r)
        | NativeNatExpr::CeilDiv(l, r)
        | NativeNatExpr::Min(l, r)
        | NativeNatExpr::Max(l, r) => {
            collect_native_nat_dims(l, dims);
            collect_native_nat_dims(r, dims);
        }
    }
}

fn collect_native_cond_dims<'a>(cond: &'a NativeCondition, dims: &mut BTreeSet<&'a str>) {
    match cond {
        NativeCondition::Compare { left, right, .. } => {
            collect_native_nat_dims(left, dims);
            collect_native_nat_dims(right, dims);
        }
        NativeCondition::And(l, r) | NativeCondition::Or(l, r) => {
            collect_native_cond_dims(l, dims);
            collect_native_cond_dims(r, dims);
        }
    }
}

fn native_nat_mentions_dim(expr: &NativeNatExpr, dim: &str) -> bool {
    match expr {
        NativeNatExpr::Dimension(name) => name == dim,
        NativeNatExpr::Constant(_)
        | NativeNatExpr::Parameter(_)
        | NativeNatExpr::Element(_, _)
        | NativeNatExpr::Device(_) => false,
        NativeNatExpr::Add(l, r)
        | NativeNatExpr::Sub(l, r)
        | NativeNatExpr::Mul(l, r)
        | NativeNatExpr::Div(l, r)
        | NativeNatExpr::Rem(l, r)
        | NativeNatExpr::CeilDiv(l, r)
        | NativeNatExpr::Min(l, r)
        | NativeNatExpr::Max(l, r) => {
            native_nat_mentions_dim(l, dim) || native_nat_mentions_dim(r, dim)
        }
    }
}

/// Conservative proof that a `NativeNatExpr` (which evaluates to `u64 >= 0`) is
/// monotonically non-decreasing in `dim` (holding all other dimensions,
/// elements, and tuning parameters fixed).
fn native_nat_is_non_decreasing_in(expr: &NativeNatExpr, dim: &str) -> bool {
    match expr {
        NativeNatExpr::Dimension(_)
        | NativeNatExpr::Constant(_)
        | NativeNatExpr::Parameter(_)
        | NativeNatExpr::Element(_, _)
        | NativeNatExpr::Device(_) => true,
        NativeNatExpr::Add(l, r)
        | NativeNatExpr::Mul(l, r)
        | NativeNatExpr::Min(l, r)
        | NativeNatExpr::Max(l, r) => {
            native_nat_is_non_decreasing_in(l, dim) && native_nat_is_non_decreasing_in(r, dim)
        }
        NativeNatExpr::Sub(l, r) | NativeNatExpr::Div(l, r) | NativeNatExpr::CeilDiv(l, r) => {
            native_nat_is_non_decreasing_in(l, dim) && !native_nat_mentions_dim(r, dim)
        }
        NativeNatExpr::Rem(l, r) => {
            !native_nat_mentions_dim(l, dim) && !native_nat_mentions_dim(r, dim)
        }
    }
}

/// Conservative proof that a `NativeCondition` is monotonically non-decreasing
/// in `dim` under `false < true` (increasing `dim` can only flip `false -> true`,
/// never `true -> false`). Because inactive scratch buffers charge `1` and
/// active buffers charge `bytes.max(1) >= 1`, a non-decreasing `when` predicate
/// paired with non-decreasing `bytes` is monotonically non-decreasing.
fn native_cond_is_non_decreasing_in(cond: &NativeCondition, dim: &str) -> bool {
    match cond {
        NativeCondition::And(l, r) | NativeCondition::Or(l, r) => {
            native_cond_is_non_decreasing_in(l, dim) && native_cond_is_non_decreasing_in(r, dim)
        }
        NativeCondition::Compare {
            comparison,
            left,
            right,
        } => {
            if !native_nat_mentions_dim(left, dim) && !native_nat_mentions_dim(right, dim) {
                return true;
            }
            match comparison {
                NativeComparison::Gt | NativeComparison::Ge => {
                    native_nat_is_non_decreasing_in(left, dim)
                        && !native_nat_mentions_dim(right, dim)
                }
                NativeComparison::Lt | NativeComparison::Le => {
                    !native_nat_mentions_dim(left, dim)
                        && native_nat_is_non_decreasing_in(right, dim)
                }
                NativeComparison::Eq | NativeComparison::Ne => false,
            }
        }
    }
}

fn render_contract_node_shape(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
    schema_info: &ExtractedSchemaInfo,
) {
    out.push_str(
        "    fn contract_node_shape(backend: seismic::BackendName) -> Result<seismic::ContractNodeShape, seismic::ContractError> {\n",
    );
    if natives.is_empty() {
        out.push_str("      Err(seismic::ContractError::NoImplementation(backend))\n");
        out.push_str("    }\n");
        return;
    }

    let mut disjoint_lit = String::from("&[");
    for (i, [a, b]) in schema_info.disjoint.iter().enumerate() {
        if i > 0 {
            disjoint_lit.push_str(", ");
        }
        let _ = write!(disjoint_lit, "[{a}usize, {b}usize]");
    }
    disjoint_lit.push(']');

    let mut ranks_lit = String::from("&[");
    for (i, rank) in schema_info.result_ranks.iter().enumerate() {
        if i > 0 {
            ranks_lit.push_str(", ");
        }
        match rank {
            Some(r) => {
                let _ = write!(ranks_lit, "Some({r}usize)");
            }
            None => ranks_lit.push_str("None"),
        }
    }
    ranks_lit.push(']');

    out.push_str("      match backend {\n");
    for &(backend, native) in natives {
        let variant = backend_variant(backend);
        let buffers = schema_info.tensor_param_count + schema_info.tensor_result_count + native.scratch.len();
        let arguments = schema_info.bound_tensor_param_count + schema_info.tensor_result_count + native.scratch.len();
        let emitted = native
            .scratch
            .iter()
            .filter(|s| matches!(s.operand, Some(NativeOperand::Emitted { .. })))
            .count();
        let runtime_params = if native.launch_scoped() {
            native.params.iter().filter(|p| !p.code).count()
                + native
                    .launches
                    .iter()
                    .map(|l| l.params.iter().filter(|p| !p.code).count())
                    .sum::<usize>()
        } else {
            0
        };
        let words = schema_info.schema_word_count + emitted + runtime_params;
        let launches = native.launches.len();
        let mut statics_lit = String::from("&[");
        for (i, name) in native.statics.iter().enumerate() {
            if i > 0 {
                statics_lit.push_str(", ");
            }
            let _ = write!(statics_lit, "{name:?}");
        }
        statics_lit.push(']');

        // Exact dynamic dimensions read anywhere inside `contract_sizing_at_point`:
        // tensor result axis extents, scratch byte expressions, and scratch `when` predicates.
        let mut sizing_dims: BTreeSet<&str> = schema_info
            .result_dims
            .iter()
            .map(String::as_str)
            .collect();
        for scratch in &native.scratch {
            collect_native_nat_dims(&scratch.bytes, &mut sizing_dims);
            if let Some(when) = &scratch.when {
                collect_native_cond_dims(when, &mut sizing_dims);
            }
        }
        sizing_dims.retain(|name| !native.statics.iter().any(|s| s == name));
        let mut sizing_dims_lit = String::from("&[");
        for (i, name) in sizing_dims.iter().enumerate() {
            if i > 0 {
                sizing_dims_lit.push_str(", ");
            }
            let _ = write!(sizing_dims_lit, "{name:?}");
        }
        sizing_dims_lit.push(']');

        let mut monotonic_sizing_dims_lit = String::from("&[");
        let mut mono_idx = 0usize;
        for &dim in &sizing_dims {
            let is_monotonic = !schema_info.non_monotonic_result_dims.contains(dim)
                && native.scratch.iter().all(|scratch| {
                    native_nat_is_non_decreasing_in(&scratch.bytes, dim)
                        && scratch
                            .when
                            .as_ref()
                            .is_none_or(|when| native_cond_is_non_decreasing_in(when, dim))
                });
            if is_monotonic {
                if mono_idx > 0 {
                    monotonic_sizing_dims_lit.push_str(", ");
                }
                let _ = write!(monotonic_sizing_dims_lit, "{dim:?}");
                mono_idx += 1;
            }
        }
        monotonic_sizing_dims_lit.push(']');

        let mut launch_when_dims: BTreeSet<&str> = BTreeSet::new();
        if backend == BackendName::Cuda {
            for launch in &native.launches {
                if let Some(when) = &launch.when {
                    collect_native_cond_dims(when, &mut launch_when_dims);
                }
            }
            launch_when_dims.retain(|name| !native.statics.iter().any(|s| s == name));
        }
        let mut launch_when_dims_lit = String::from("&[");
        for (i, name) in launch_when_dims.iter().enumerate() {
            if i > 0 {
                launch_when_dims_lit.push_str(", ");
            }
            let _ = write!(launch_when_dims_lit, "{name:?}");
        }
        launch_when_dims_lit.push(']');

        let conditions: usize = native.launches.iter().map(|l| l.conditions.len()).sum();
        let _ = writeln!(
            out,
            "        seismic::BackendName::{variant} => Ok(seismic::ContractNodeShape {{ buffers: {buffers}usize, arguments: {arguments}usize, words: {words}usize, launches: {launches}usize, conditions: {conditions}usize, disjoint: {disjoint_lit}, parks: {}, result_ranks: {ranks_lit}, statics: {statics_lit}, sizing_dims: {sizing_dims_lit}, monotonic_sizing_dims: {monotonic_sizing_dims_lit}, launch_when_dims: {launch_when_dims_lit} }}),",
            schema_info.parks
        );
    }
    if natives.len() < BackendName::ALL.len() {
        out.push_str("        _ => Err(seismic::ContractError::NoImplementation(backend)),\n");
    }
    out.push_str("      }\n");
    out.push_str("    }\n");
}

fn render_contract_scratch_meta(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_scratch_meta(backend: seismic::BackendName) -> Result<seismic::ContractScratchMeta, seismic::ContractError> {\n",
    );
    if natives.is_empty() {
        out.push_str("      Err(seismic::ContractError::NoImplementation(backend))\n");
        out.push_str("    }\n");
        return;
    }

    out.push_str("      match backend {\n");
    for &(backend, native) in natives {
        let variant = backend_variant(backend);
        if native.scratch.is_empty() {
            let _ = writeln!(
                out,
                "        seismic::BackendName::{variant} => Ok(seismic::ContractScratchMeta {{ scratch: &[] }}),"
            );
            continue;
        }
        let mut scratch_lit = String::from("&[");
        for (i, scratch) in native.scratch.iter().enumerate() {
            if i > 0 {
                scratch_lit.push_str(", ");
            }
            let emission = match scratch.emission {
                Some(r) => format!("Some({r}u32)"),
                None => "None".to_owned(),
            };
            let operand = match &scratch.operand {
                None => "None".to_owned(),
                Some(NativeOperand::Formed {
                    parameter,
                    formation,
                    rows,
                    norm,
                    ..
                }) => {
                    let norm_expr = match norm {
                        Some(n) => format!(
                            "Some(seismic::NativeOperandNorm {{ weight: {}u32, epsilon: {}u32 }})",
                            n.weight, n.epsilon
                        ),
                        None => "None".to_owned(),
                    };
                    format!(
                        "Some(seismic::sizing::ScratchOperand::Formed {{ parameter: {parameter}u32, formation: {formation}u32, rows: {rows}, norm: {norm_expr} }})"
                    )
                }
                Some(NativeOperand::Emitted { result, rows, norm }) => {
                    let norm_expr = match norm {
                        Some(n) => format!(
                            "Some(seismic::NativeOperandNorm {{ weight: {}u32, epsilon: {}u32 }})",
                            n.weight, n.epsilon
                        ),
                        None => "None".to_owned(),
                    };
                    format!(
                        "Some(seismic::sizing::ScratchOperand::Emitted {{ result: {result}u32, rows: {rows}, norm: {norm_expr} }})"
                    )
                }
            };
            let _ = write!(
                scratch_lit,
                "seismic::ScratchMeta {{ sync: {}, emission: {emission}, operand: {operand} }}",
                scratch.sync
            );
        }
        scratch_lit.push(']');
        let _ = writeln!(
            out,
            "        seismic::BackendName::{variant} => Ok(seismic::ContractScratchMeta {{ scratch: {scratch_lit} }}),"
        );
    }
    if natives.len() < BackendName::ALL.len() {
        out.push_str("        _ => Err(seismic::ContractError::NoImplementation(backend)),\n");
    }
    out.push_str("      }\n");
    out.push_str("    }\n");
}

fn render_contract_param_spec(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
    tensor_params: &[LoweredTensorParam],
) {
    out.push_str(
        "    fn contract_param_spec(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], parameter: &str, dims: &[(&str, u64)]) -> Result<seismic::TensorParamSpec, seismic::ContractError> {\n",
    );
    if natives.is_empty() {
        out.push_str("      Err(seismic::ContractError::NoImplementation(backend))\n");
        out.push_str("    }\n");
        return;
    }
    // Verify backend has a native implementation.
    out.push_str("      match backend {\n");
    out.push_str("        ");
    for (i, &(backend, _)) in natives.iter().enumerate() {
        if i > 0 {
            out.push_str(" | ");
        }
        let _ = write!(out, "seismic::BackendName::{}", backend_variant(backend));
    }
    out.push_str(" => {}\n");
    if natives.len() < BackendName::ALL.len() {
        out.push_str("        _ => return Err(seismic::ContractError::NoImplementation(backend)),\n");
    }
    out.push_str("      }\n");
    emit_element_binding(out, entry, "      ");
    let non_empty_unit_cols = natives
        .iter()
        .filter(|(_, native)| !unit_column_reads(native).is_empty())
        .count();
    if non_empty_unit_cols > 0 {
        out.push_str("      match backend {\n");
        for &(backend, native) in natives {
            if !unit_column_reads(native).is_empty() {
                let variant = backend_variant(backend);
                let _ = writeln!(out, "        seismic::BackendName::{variant} => {{");
                emit_unit_column_checks(out, entry, native, "          ");
                out.push_str("        }\n");
            }
        }
        if non_empty_unit_cols < BackendName::ALL.len() {
            out.push_str("        _ => {}\n");
        }
        out.push_str("      }\n");
    }
    out.push_str("      match parameter {\n");
    for param in tensor_params {
        let _ = writeln!(
            out,
            "        {:?} => (|| -> Result<seismic::TensorParamSpec, seismic::ContractError> {{",
            param.name
        );
        // Bind only the entry dimensions referenced by this parameter's axis expressions.
        for (idx, dim_name) in entry.dimensions.iter().enumerate() {
            let needle = format!("__d_{idx}");
            let used = param.axis_exprs.iter().any(|expr| {
                // Match exact identifier token __d_{idx} (not __d_10 when searching for __d_1).
                expr.match_indices(&needle).any(|(pos, _)| {
                    let after = expr.as_bytes().get(pos + needle.len());
                    !matches!(after, Some(b'0'..=b'9'))
                })
            });
            if used {
                let _ = writeln!(
                    out,
                    "          let __d_{idx}: u64 = dims.iter().find(|(n, _)| *n == {dim_name:?}).map(|(_, v)| *v).ok_or(seismic::ContractError::MissingDimension({dim_name:?}))?;"
                );
            }
        }
        let elem_expr = match &param.element {
            ElementSummary::Fixed(name) => {
                format!("seismic::contract::rt::element_named({name:?})")
            }
            ElementSummary::Parameter(name) => {
                let idx = entry
                    .element_parameters
                    .iter()
                    .position(|p| p == name)
                    .expect("param element names an entry element parameter");
                format!("__e_{idx}")
            }
        };
        let extents = param.axis_exprs.join(", ");
        let _ = writeln!(out, "          let __el = {elem_expr};");
        let _ = writeln!(out, "          let __extents = vec![{extents}];");
        let _ = writeln!(
            out,
            "          let __bytes = seismic::contract::rt::canonical_bytes(__el, &__extents)?;"
        );
        out.push_str(
            "          Ok(seismic::TensorParamSpec { element: __el, extents: __extents, bytes: __bytes })\n",
        );
        out.push_str("        })(),\n");
    }
    out.push_str("        _ => Err(seismic::ContractError::InvalidParameter(\"unknown\")),\n");
    out.push_str("      }\n");
    out.push_str("    }\n");
}

struct ParamComponent<'a> {
    params: Vec<usize>,
    conjuncts: Vec<&'a NativeCondition>,
}

struct ScratchComponent<'a> {
    outer: Vec<usize>,
    outer_conjuncts: Vec<&'a NativeCondition>,
    residuals: Vec<ParamComponent<'a>>,
}

struct FactoredConstraints<'a> {
    outer_indices: Vec<usize>,
    static_conjuncts: Vec<&'a NativeCondition>,
    unreferenced: Vec<usize>,
    independent: Vec<ParamComponent<'a>>,
    scratch: Vec<ScratchComponent<'a>>,
}

fn dsu_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn dsu_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = dsu_find(parent, a);
    let rb = dsu_find(parent, b);
    if ra != rb {
        if ra < rb {
            parent[rb] = ra;
        } else {
            parent[ra] = rb;
        }
    }
}

fn launch_when_outer_indices(
    native: &NativeImplementation,
    slots: &[ParamSlot<'_>],
) -> Vec<usize> {
    let mut indices = BTreeSet::new();
    for (launch_idx, launch) in native.launches.iter().enumerate() {
        if let Some(when) = &launch.when {
            let mut names = Vec::new();
            when.parameters(&mut names);
            for name in &names {
                if let Some(k) = slots
                    .iter()
                    .position(|s| s.launch == Some(launch_idx) && s.param.name == *name)
                    .or_else(|| {
                        slots
                            .iter()
                            .position(|s| s.launch.is_none() && s.param.name == *name)
                    })
                {
                    indices.insert(k);
                }
            }
        }
    }
    indices.into_iter().collect()
}

fn factor_constraints<'a>(
    native: &'a NativeImplementation,
    slots: &[ParamSlot<'_>],
    ctx: &ExprCtx<'_>,
) -> FactoredConstraints<'a> {
    let mut scratch_param_names = Vec::new();
    for s in &native.scratch {
        s.parameters(&mut scratch_param_names);
    }
    let outer_indices: Vec<usize> = (0..slots.len())
        .filter(|&k| slots[k].launch.is_none() && scratch_param_names.contains(&slots[k].param.name))
        .collect();
    factor_constraints_for(native, slots, ctx, outer_indices)
}

fn factor_constraints_for<'a>(
    native: &'a NativeImplementation,
    slots: &[ParamSlot<'_>],
    ctx: &ExprCtx<'_>,
    outer_indices: Vec<usize>,
) -> FactoredConstraints<'a> {
    let mut is_outer = vec![false; slots.len()];
    for &k in &outer_indices {
        is_outer[k] = true;
    }

    let mut conjuncts = Vec::new();
    if let Some(constraint) = &native.constraint {
        flatten_conjuncts(constraint, &mut conjuncts);
    }

    let mut parent: Vec<usize> = (0..slots.len()).collect();
    let mut static_conjuncts: Vec<&'a NativeCondition> = Vec::new();
    let mut param_conjuncts: Vec<(usize, &'a NativeCondition)> = Vec::new();

    for c in conjuncts {
        let mut names = Vec::new();
        c.parameters(&mut names);
        let mut indices: Vec<usize> = names
            .iter()
            .map(|n| ctx.expect_param_index(n))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            static_conjuncts.push(c);
        } else {
            for &other in &indices[1..] {
                dsu_union(&mut parent, indices[0], other);
            }
            param_conjuncts.push((indices[0], c));
        }
    }

    let mut comp_members: Vec<Vec<usize>> = vec![Vec::new(); slots.len()];
    for k in 0..slots.len() {
        let r = dsu_find(&mut parent, k);
        comp_members[r].push(k);
    }
    let mut comp_conjuncts: Vec<Vec<&'a NativeCondition>> = vec![Vec::new(); slots.len()];
    for (rep, c) in param_conjuncts {
        let r = dsu_find(&mut parent, rep);
        comp_conjuncts[r].push(c);
    }

    let mut unreferenced = Vec::new();
    let mut independent = Vec::new();
    let mut scratch = Vec::new();

    for r in 0..slots.len() {
        if comp_members[r].is_empty() {
            continue;
        }
        let mut outer = Vec::new();
        let mut inner = Vec::new();
        for &k in &comp_members[r] {
            if is_outer[k] {
                outer.push(k);
            } else {
                inner.push(k);
            }
        }
        let conjs = std::mem::take(&mut comp_conjuncts[r]);
        if outer.is_empty() && conjs.is_empty() {
            debug_assert_eq!(inner.len(), 1);
            unreferenced.push(inner[0]);
        } else if outer.is_empty() {
            independent.push(ParamComponent {
                params: inner,
                conjuncts: conjs,
            });
        } else {
            let mut outer_conjuncts: Vec<&'a NativeCondition> = Vec::new();
            let mut residuals: Vec<ParamComponent<'a>> = Vec::new();
            if inner.is_empty() {
                outer_conjuncts = conjs;
            } else {
                let mut inner_pos: Vec<Option<usize>> = vec![None; slots.len()];
                for (li, &k) in inner.iter().enumerate() {
                    inner_pos[k] = Some(li);
                }
                let mut rc_parent: Vec<usize> = (0..inner.len()).collect();
                let mut rc_conj_list: Vec<(usize, &'a NativeCondition)> = Vec::new();
                for c in conjs {
                    let mut names = Vec::new();
                    c.parameters(&mut names);
                    let mut local_inner: Vec<usize> = names
                        .iter()
                        .filter_map(|n| inner_pos[ctx.expect_param_index(n)])
                        .collect();
                    local_inner.sort_unstable();
                    local_inner.dedup();
                    if local_inner.is_empty() {
                        outer_conjuncts.push(c);
                    } else {
                        for &other in &local_inner[1..] {
                            dsu_union(&mut rc_parent, local_inner[0], other);
                        }
                        rc_conj_list.push((local_inner[0], c));
                    }
                }
                let mut rc_members: Vec<Vec<usize>> = vec![Vec::new(); inner.len()];
                for (li, &k) in inner.iter().enumerate() {
                    let root = dsu_find(&mut rc_parent, li);
                    rc_members[root].push(k);
                }
                let mut rc_conjs: Vec<Vec<&'a NativeCondition>> = vec![Vec::new(); inner.len()];
                for (rep, c) in rc_conj_list {
                    let root = dsu_find(&mut rc_parent, rep);
                    rc_conjs[root].push(c);
                }
                for root in 0..inner.len() {
                    if !rc_members[root].is_empty() {
                        residuals.push(ParamComponent {
                            params: std::mem::take(&mut rc_members[root]),
                            conjuncts: std::mem::take(&mut rc_conjs[root]),
                        });
                    }
                }
            }
            scratch.push(ScratchComponent {
                outer,
                outer_conjuncts,
                residuals,
            });
        }
    }

    FactoredConstraints {
        outer_indices,
        static_conjuncts,
        unreferenced,
        independent,
        scratch,
    }
}

fn partition_component_conjuncts<'a>(
    loop_order: &[usize],
    conjuncts: &[&'a NativeCondition],
    ctx: &ExprCtx<'_>,
) -> Vec<Vec<&'a NativeCondition>> {
    let mut local_depth: HashMap<usize, usize> = HashMap::with_capacity(loop_order.len());
    for (depth, &k) in loop_order.iter().enumerate() {
        local_depth.insert(k, depth);
    }
    let mut depth_conjuncts: Vec<Vec<&'a NativeCondition>> = vec![Vec::new(); loop_order.len()];
    for &c in conjuncts {
        let mut names = Vec::new();
        c.parameters(&mut names);
        if let Some(max_depth) = names
            .iter()
            .filter_map(|n| local_depth.get(&ctx.expect_param_index(n)).copied())
            .max()
        {
            depth_conjuncts[max_depth].push(c);
        }
    }
    depth_conjuncts
}

fn lower_conjunction(ctx: &ExprCtx<'_>, conjuncts: &[&NativeCondition]) -> String {
    conjuncts
        .iter()
        .map(|c| {
            format!(
                "(|| -> Result<bool, seismic::ContractError> {{ Ok({}) }})()?",
                ctx.lower_cond(c)
            )
        })
        .collect::<Vec<_>>()
        .join(" && ")
}

fn emit_default_first_order(
    out: &mut String,
    backend: BackendName,
    entry: &EntryInfo,
    slots: &[ParamSlot<'_>],
    ctx: &ExprCtx<'_>,
    k: usize,
    indent: &str,
) {
    let slot = &slots[k];
    if slot.param.values.len() <= 1 {
        return;
    }
    let pname = &slot.param.name;
    let vals = format_u64_slice(&slot.param.values);
    let _ = writeln!(out, "{indent}let __ord_{k} = {{");
    let _ = writeln!(out, "{indent}  let mut __arr = {vals};");
    match &slot.param.default {
        NativeDefault::Value(expr) => {
            let default_ctx = ExprCtx {
                backend,
                elements: &entry.element_parameters,
                dims: ctx.dims,
                params: slots,
                launch: slot.launch,
                default_param: None,
            };
            let e = default_ctx.lower_nat(expr);
            let _ = writeln!(
                out,
                "{indent}  if let Ok(__def) = (|| -> Result<u64, seismic::ContractError> {{ Ok({e}) }})() {{"
            );
        }
        NativeDefault::Least(cond) | NativeDefault::Greatest(cond) => {
            let mut sorted = slot.param.values.clone();
            sorted.sort_unstable();
            if matches!(slot.param.default, NativeDefault::Greatest(_)) {
                sorted.reverse();
            }
            let sorted_lit = format_u64_slice(&sorted);
            let default_ctx = ExprCtx {
                backend,
                elements: &entry.element_parameters,
                dims: ctx.dims,
                params: slots,
                launch: slot.launch,
                default_param: Some(pname.as_str()),
            };
            let cond_expr = default_ctx.lower_cond(cond);
            let _ = writeln!(
                out,
                "{indent}  if let Ok(__def) = (|| -> Result<u64, seismic::ContractError> {{ for __cand in {sorted_lit} {{ if {cond_expr} {{ return Ok(__cand); }} }} Err(seismic::ContractError::NoDefault({pname:?})) }})() {{"
            );
        }
    }
    let _ = writeln!(
        out,
        "{indent}    if let Some(__idx) = __arr.iter().position(|&__v| __v == __def) {{"
    );
    let _ = writeln!(out, "{indent}      __arr[..=__idx].rotate_right(1);");
    let _ = writeln!(out, "{indent}    }}");
    let _ = writeln!(out, "{indent}  }}");
    let _ = writeln!(out, "{indent}  __arr");
    let _ = writeln!(out, "{indent}}};");
}

fn scratch_bytes_expr(ctx: &ExprCtx<'_>, scratch: &NativeScratch) -> String {
    let inner = match &scratch.when {
        Some(when) => {
            let w = ctx.lower_cond(when);
            let b = ctx.lower_nat(&scratch.bytes);
            format!("if {w} {{ ({b}).max(1u64) }} else {{ 1u64 }}")
        }
        None => {
            let b = ctx.lower_nat(&scratch.bytes);
            format!("({b}).max(1u64)")
        }
    };
    format!("(|| -> Result<u64, seismic::ContractError> {{ Ok({inner}) }})()?")
}

/// Emit the `comp.outer` loops, `outer_conjuncts` checks, and independent
/// residual inner-component checks for one `ScratchComponent`.
///
/// Returns the indentation inside the innermost `comp.outer` loop after all
/// residual checks pass; the caller emits the admitted-tuple body and closes
/// `comp.outer.len()` braces.
///
/// When `used_by_meta` is `Some(flags)`, each residual component captures
/// `__rc_w_{k}: u64` for every `k` in `rc.params` with `flags[k]`.
fn emit_scratch_outer_and_residuals(
    out: &mut String,
    ci: usize,
    comp: &ScratchComponent<'_>,
    slots: &[ParamSlot<'_>],
    ctx: &ExprCtx<'_>,
    use_default_order: bool,
    used_by_meta: Option<&[bool]>,
    base_indent: &str,
) -> String {
    let outer_depth_conjuncts =
        partition_component_conjuncts(&comp.outer, &comp.outer_conjuncts, ctx);
    let mut indent = base_indent.to_owned();
    for (depth, &k) in comp.outer.iter().enumerate() {
        let iter_expr = if !use_default_order || slots[k].param.values.len() <= 1 {
            format_u64_slice(&slots[k].param.values)
        } else {
            format!("__ord_{k}")
        };
        let _ = writeln!(out, "{indent}for __p_{k} in {iter_expr} {{");
        indent.push_str("  ");
        if depth + 1 < comp.outer.len() {
            for c in &outer_depth_conjuncts[depth] {
                let cond_expr = ctx.lower_cond(c);
                let _ = writeln!(
                    out,
                    "{indent}if matches!((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})(), Ok(false)) {{ continue; }}"
                );
            }
        }
    }
    if !comp.outer_conjuncts.is_empty() {
        emit_conjunct_guards(out, ctx, &comp.outer_conjuncts, "continue;", &indent);
    }

    for (rc_idx, rc) in comp.residuals.iter().enumerate() {
        let meta_rc: Vec<usize> = match used_by_meta {
            Some(flags) => rc.params.iter().copied().filter(|&k| flags[k]).collect(),
            None => Vec::new(),
        };
        if rc.conjuncts.is_empty() {
            if rc.params.iter().any(|&k| slots[k].param.values.is_empty()) {
                let _ = writeln!(out, "{indent}continue;");
            } else {
                for &k in &meta_rc {
                    if !use_default_order || slots[k].param.values.len() <= 1 {
                        let v = slots[k].param.values[0];
                        let _ = writeln!(out, "{indent}let __rc_w_{k}: u64 = {v}u64;");
                    } else {
                        let _ = writeln!(out, "{indent}let __rc_w_{k}: u64 = __ord_{k}[0];");
                    }
                }
            }
            continue;
        }

        let rc_depth_conjuncts = partition_component_conjuncts(&rc.params, &rc.conjuncts, ctx);
        let _ = writeln!(out, "{indent}let mut __rc_ok_{rc_idx} = false;");
        for &k in &meta_rc {
            let _ = writeln!(out, "{indent}let mut __rc_w_{k}: u64 = 0u64;");
        }
        let mut rc_indent = indent.clone();
        for (depth, &k) in rc.params.iter().enumerate() {
            let iter_expr = if !use_default_order || slots[k].param.values.len() <= 1 {
                format_u64_slice(&slots[k].param.values)
            } else {
                format!("__ord_{k}")
            };
            if depth == 0 {
                let _ = writeln!(
                    out,
                    "{rc_indent}'__rc_{ci}_{rc_idx}: for __p_{k} in {iter_expr} {{"
                );
            } else {
                let _ = writeln!(out, "{rc_indent}for __p_{k} in {iter_expr} {{");
            }
            rc_indent.push_str("  ");
            if depth + 1 < rc.params.len() {
                for c in &rc_depth_conjuncts[depth] {
                    let cond_expr = ctx.lower_cond(c);
                    let _ = writeln!(
                        out,
                        "{rc_indent}if matches!((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})(), Ok(false)) {{ continue; }}"
                    );
                }
            }
        }
        let rc_all = lower_conjunction(ctx, &rc.conjuncts);
        let _ = writeln!(out, "{rc_indent}if {rc_all} {{");
        for &k in &meta_rc {
            let _ = writeln!(out, "{rc_indent}  __rc_w_{k} = __p_{k};");
        }
        let _ = writeln!(out, "{rc_indent}  __rc_ok_{rc_idx} = true;");
        let _ = writeln!(out, "{rc_indent}  break '__rc_{ci}_{rc_idx};");
        let _ = writeln!(out, "{rc_indent}}}");
        for _ in 0..rc.params.len() {
            rc_indent.truncate(rc_indent.len() - 2);
            let _ = writeln!(out, "{rc_indent}}}");
        }
        let _ = writeln!(out, "{indent}if !__rc_ok_{rc_idx} {{ continue; }}");
    }

    indent
}

fn render_contract_sizing_bound(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_sizing_bound(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<seismic::EntrySizing, seismic::ContractError> {\n",
    );
    render_backend_dispatch(out, natives, "sizing_bound", "elements, dims");
}

fn render_contract_sizing_bound_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
    tensor_results: &[LoweredTensorResult],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn sizing_bound_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<seismic::EntrySizing, seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &entry.dimensions, "      ");
        let slots = collect_params(native);
        let ctx = ExprCtx {
            backend,
            elements: &entry.element_parameters,
            dims: &entry.dimensions,
            params: &slots,
            launch: None,
            default_param: None,
        };

        let factored = factor_constraints(native, &slots, &ctx);

        // Identify parameters referenced by operand_forms or active_launches so
        // independent/unreferenced components bind a defaults-first witness for them.
        let mut used_by_meta = vec![false; slots.len()];
        for scratch in &native.scratch {
            if let Some(NativeOperand::Formed { pairs, formed, nvfp4, .. }) = &scratch.operand {
                let mut names = Vec::new();
                pairs.parameters(&mut names);
                formed.parameters(&mut names);
                nvfp4.parameters(&mut names);
                for n in &names {
                    used_by_meta[ctx.expect_param_index(n)] = true;
                }
            }
        }
        for (launch_idx, launch) in native.launches.iter().enumerate() {
            if let Some(when) = &launch.when {
                let lctx = ExprCtx {
                    backend,
                    elements: &entry.element_parameters,
                    dims: &entry.dimensions,
                    params: &slots,
                    launch: Some(launch_idx),
                    default_param: None,
                };
                let mut names = Vec::new();
                when.parameters(&mut names);
                for n in &names {
                    used_by_meta[lctx.expect_param_index(n)] = true;
                }
            }
        }
        if let Some(repeat) = &native.repeat {
            let mut names = Vec::new();
            repeat.count.parameters(&mut names);
            for n in &names {
                used_by_meta[ctx.expect_param_index(n)] = true;
            }
        }

        // Check static conjuncts up front before any parameter component.
        emit_conjunct_guards(
            out,
            &ctx,
            &factored.static_conjuncts,
            "return Err(seismic::ContractError::DomainViolation);",
            "      ",
        );

        // Unreferenced singletons: 0 loop iterations.
        for &k in &factored.unreferenced {
            if slots[k].param.values.is_empty() {
                out.push_str("          return Err(seismic::ContractError::DomainViolation);\n");
            } else if used_by_meta[k] {
                if slots[k].param.values.len() == 1 {
                    let v = slots[k].param.values[0];
                    let _ = writeln!(out, "          let __p_{k}: u64 = {v}u64;");
                } else {
                    emit_default_first_order(out, backend, entry, &slots, &ctx, k, "          ");
                    let _ = writeln!(out, "          let __p_{k}: u64 = __ord_{k}[0];");
                }
            }
        }

        // Independent non-scratch components: check each once, break on first admitted tuple.
        for (ci, comp) in factored.independent.iter().enumerate() {
            for &k in &comp.params {
                emit_default_first_order(out, backend, entry, &slots, &ctx, k, "          ");
            }
            let depth_conjuncts = partition_component_conjuncts(&comp.params, &comp.conjuncts, &ctx);
            let meta_inner: Vec<usize> = comp
                .params
                .iter()
                .copied()
                .filter(|&k| used_by_meta[k])
                .collect();
            let _ = writeln!(out, "          let mut __comp_ok_{ci} = false;");
            for &k in &meta_inner {
                let _ = writeln!(out, "          let mut __w_{k}: u64 = 0u64;");
            }
            let mut indent = String::from("          ");
            for (depth, &k) in comp.params.iter().enumerate() {
                let iter_expr = if slots[k].param.values.len() <= 1 {
                    format_u64_slice(&slots[k].param.values)
                } else {
                    format!("__ord_{k}")
                };
                if depth == 0 {
                    let _ = writeln!(out, "{indent}'__comp_{ci}: for __p_{k} in {iter_expr} {{");
                } else {
                    let _ = writeln!(out, "{indent}for __p_{k} in {iter_expr} {{");
                }
                indent.push_str("  ");
                if depth + 1 < comp.params.len() {
                    for c in &depth_conjuncts[depth] {
                        let cond_expr = ctx.lower_cond(c);
                        let _ = writeln!(
                            out,
                            "{indent}if matches!((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})(), Ok(false)) {{ continue; }}"
                        );
                    }
                }
            }
            let all_cond = lower_conjunction(&ctx, &comp.conjuncts);
            let _ = writeln!(out, "{indent}if {all_cond} {{");
            for &k in &meta_inner {
                let _ = writeln!(out, "{indent}  __w_{k} = __p_{k};");
            }
            let _ = writeln!(out, "{indent}  __comp_ok_{ci} = true;");
            let _ = writeln!(out, "{indent}  break '__comp_{ci};");
            let _ = writeln!(out, "{indent}}}");
            for _ in 0..comp.params.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            let _ = writeln!(
                out,
                "          if !__comp_ok_{ci} {{ return Err(seismic::ContractError::DomainViolation); }}"
            );
            for &k in &meta_inner {
                let _ = writeln!(out, "          let __p_{k}: u64 = __w_{k};");
            }
        }

        // Helper closure to emit active launches computation into `__active_launches`.
        let emit_active_launches = |out: &mut String, indent: &str| {
            let _ = writeln!(
                out,
                "{indent}__active_launches = (|| -> Result<u64, seismic::ContractError> {{"
            );
            for (launch_idx, launch) in native.launches.iter().enumerate() {
                match &launch.when {
                    None => {
                        let _ = writeln!(out, "{indent}  let __act_{launch_idx}: u64 = 1u64;");
                    }
                    Some(when) => {
                        let lctx = ExprCtx {
                            backend,
                            elements: &entry.element_parameters,
                            dims: &entry.dimensions,
                            params: &slots,
                            launch: Some(launch_idx),
                            default_param: None,
                        };
                        let w = lctx.lower_cond(when);
                        let _ = writeln!(
                            out,
                            "{indent}  let __act_{launch_idx}: u64 = if (|| -> Result<bool, seismic::ContractError> {{ Ok({w}) }})()? {{ 1u64 }} else {{ 0u64 }};"
                        );
                    }
                }
            }
            let sum_range = |range: std::ops::Range<usize>| -> String {
                if range.is_empty() {
                    "0u64".to_owned()
                } else {
                    range.map(|i| format!("__act_{i}")).collect::<Vec<_>>().join(" + ")
                }
            };
            match &native.repeat {
                None => {
                    let total = sum_range(0..native.launches.len());
                    let _ = writeln!(out, "{indent}  Ok({total})");
                }
                Some(repeat) => {
                    let rounds = ctx.lower_nat(&repeat.count);
                    let before = sum_range(0..repeat.first);
                    let body = sum_range(repeat.first..repeat.first + repeat.launches);
                    let after = sum_range(repeat.first + repeat.launches..native.launches.len());
                    let _ = writeln!(out, "{indent}  let __rounds: u64 = {rounds};");
                    let _ = writeln!(
                        out,
                        "{indent}  ({body}).checked_mul(__rounds).and_then(|v| v.checked_add({before})).and_then(|v| v.checked_add({after})).ok_or(seismic::ContractError::ArithmeticOverflow)"
                    );
                }
            }
            let _ = writeln!(out, "{indent}}})()?;");
        };

        // Helper closure to emit operand forms into `__operand_forms`.
        let has_operands = native.scratch.iter().any(|s| s.operand.is_some());
        let emit_operand_forms = |out: &mut String, indent: &str| {
            if !has_operands {
                return;
            }
            let _ = writeln!(
                out,
                "{indent}__operand_forms = (|| -> Result<Vec<seismic::OperandFormSpec>, seismic::ContractError> {{"
            );
            let _ = writeln!(out, "{indent}  Ok(vec![");
            for (scratch_index, scratch) in native.scratch.iter().enumerate() {
                if let Some(operand) = &scratch.operand {
                    let (role_expr, rows, norm) = match operand {
                        NativeOperand::Formed {
                            parameter,
                            formation,
                            pairs,
                            rows,
                            norm,
                            formed,
                            nvfp4,
                        } => {
                            let pairs_expr = ctx.lower_cond(pairs);
                            let formed_expr = ctx.lower_cond(formed);
                            let nvfp4_expr = ctx.lower_cond(nvfp4);
                            (
                                format!(
                                    "seismic::OperandRole::Formed {{ parameter: {parameter}u32, formation: {formation}u32, pairs: {pairs_expr}, formed: {formed_expr}, nvfp4: {nvfp4_expr} }}"
                                ),
                                *rows,
                                *norm,
                            )
                        }
                        NativeOperand::Emitted { result, rows, norm } => (
                            format!("seismic::OperandRole::Emitted {{ result: {result}u32 }}"),
                            *rows,
                            *norm,
                        ),
                    };
                    let norm_expr = match norm {
                        Some(n) => format!(
                            "Some(seismic::NativeOperandNorm {{ weight: {}u32, epsilon: {}u32 }})",
                            n.weight, n.epsilon
                        ),
                        None => "None".to_owned(),
                    };
                    let _ = writeln!(
                        out,
                        "{indent}    seismic::OperandFormSpec {{ scratch_index: {scratch_index}usize, role: {role_expr}, rows: {rows}, norm: {norm_expr} }},"
                    );
                }
            }
            let _ = writeln!(out, "{indent}  ])");
            let _ = writeln!(out, "{indent}}})()?;");
        };

        let scratch_len = native.scratch.len();
        if scratch_len > 0 {
            let _ = writeln!(out, "          let mut __scratch_max = [0u64; {scratch_len}];");
        }
        out.push_str("          let mut __active_launches: u64 = 0u64;\n");
        if has_operands {
            out.push_str("          let mut __operand_forms = Vec::new();\n");
        } else {
            out.push_str("          let __operand_forms = Vec::new();\n");
        }

        if factored.scratch.is_empty() {
            for (i, scratch) in native.scratch.iter().enumerate() {
                let bytes_expr = scratch_bytes_expr(&ctx, scratch);
                let _ = writeln!(out, "          __scratch_max[{i}] = {bytes_expr};");
            }
            emit_operand_forms(out, "          ");
            emit_active_launches(out, "          ");
        } else if factored.scratch.len() == 1 {
            let comp = &factored.scratch[0];
            for &k in comp
                .outer
                .iter()
                .chain(comp.residuals.iter().flat_map(|rc| rc.params.iter()))
            {
                emit_default_first_order(out, backend, entry, &slots, &ctx, k, "          ");
            }
            let meta_inner: Vec<usize> = comp
                .residuals
                .iter()
                .flat_map(|rc| rc.params.iter().copied())
                .filter(|&k| used_by_meta[k])
                .collect();
            out.push_str("          let mut __any_admitted = false;\n");
            let mut indent = emit_scratch_outer_and_residuals(
                out,
                0,
                comp,
                &slots,
                &ctx,
                true,
                Some(&used_by_meta),
                "          ",
            );
            for (i, scratch) in native.scratch.iter().enumerate() {
                let bytes_expr = scratch_bytes_expr(&ctx, scratch);
                let _ = writeln!(out, "{indent}__scratch_max[{i}] = __scratch_max[{i}].max({bytes_expr});");
            }
            let _ = writeln!(out, "{indent}if !__any_admitted {{");
            let inner_indent = format!("{indent}  ");
            for &k in &meta_inner {
                let _ = writeln!(out, "{inner_indent}let __p_{k}: u64 = __rc_w_{k};");
            }
            emit_operand_forms(out, &inner_indent);
            emit_active_launches(out, &inner_indent);
            let _ = writeln!(out, "{inner_indent}__any_admitted = true;");
            let _ = writeln!(out, "{indent}}}");
            for _ in 0..comp.outer.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            out.push_str("          if !__any_admitted { return Err(seismic::ContractError::DomainViolation); }\n");
        } else {
            // Multiple disjoint scratch-coupled components: enumerate each independently,
            // then take the product of their admitted outer subsets.
            for (ci, comp) in factored.scratch.iter().enumerate() {
                for &k in comp
                    .outer
                    .iter()
                    .chain(comp.residuals.iter().flat_map(|rc| rc.params.iter()))
                {
                    emit_default_first_order(out, backend, entry, &slots, &ctx, k, "          ");
                }
                let meta_inner: Vec<usize> = comp
                    .residuals
                    .iter()
                    .flat_map(|rc| rc.params.iter().copied())
                    .filter(|&k| used_by_meta[k])
                    .collect();
                let _ = writeln!(out, "          let mut __sub_{ci}: Vec<u64> = Vec::new();");
                for &k in &meta_inner {
                    let _ = writeln!(out, "          let mut __w_{k}: u64 = 0u64;");
                }
                let mut indent = emit_scratch_outer_and_residuals(
                    out,
                    ci,
                    comp,
                    &slots,
                    &ctx,
                    true,
                    Some(&used_by_meta),
                    "          ",
                );
                if !meta_inner.is_empty() {
                    let _ = writeln!(out, "{indent}if __sub_{ci}.is_empty() {{");
                    for &k in &meta_inner {
                        let _ = writeln!(out, "{indent}  __w_{k} = __rc_w_{k};");
                    }
                    let _ = writeln!(out, "{indent}}}");
                }
                for &ok in &comp.outer {
                    let _ = writeln!(out, "{indent}__sub_{ci}.push(__p_{ok});");
                }
                for _ in 0..comp.outer.len() {
                    indent.truncate(indent.len() - 2);
                    let _ = writeln!(out, "{indent}}}");
                }
                let _ = writeln!(
                    out,
                    "          if __sub_{ci}.is_empty() {{ return Err(seismic::ContractError::DomainViolation); }}"
                );
                for &k in &meta_inner {
                    let _ = writeln!(out, "          let __p_{k}: u64 = __w_{k};");
                }
            }

            out.push_str("          let mut __any_admitted = false;\n");
            let mut indent = String::from("          ");
            for (ci, comp) in factored.scratch.iter().enumerate() {
                let stride = comp.outer.len();
                let _ = writeln!(out, "{indent}for __chunk_{ci} in __sub_{ci}.chunks_exact({stride}usize) {{");
                indent.push_str("  ");
                for (j, &ok) in comp.outer.iter().enumerate() {
                    let _ = writeln!(out, "{indent}let __p_{ok}: u64 = __chunk_{ci}[{j}usize];");
                }
            }
            for (i, scratch) in native.scratch.iter().enumerate() {
                let bytes_expr = scratch_bytes_expr(&ctx, scratch);
                let _ = writeln!(out, "{indent}__scratch_max[{i}] = __scratch_max[{i}].max({bytes_expr});");
            }
            let _ = writeln!(out, "{indent}if !__any_admitted {{");
            let inner_indent = format!("{indent}  ");
            emit_operand_forms(out, &inner_indent);
            emit_active_launches(out, &inner_indent);
            let _ = writeln!(out, "{indent}  __any_admitted = true;");
            let _ = writeln!(out, "{indent}}}");
            for _ in 0..factored.scratch.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
        }

        // Results
        if tensor_results.is_empty() {
            out.push_str("      let __results = Vec::new();\n");
        } else {
            out.push_str(
                "      let __results = (|| -> Result<Vec<seismic::TensorResultSpec>, seismic::ContractError> {\n",
            );
            out.push_str("        Ok(vec![\n");
            for result in tensor_results {
                let elem_expr = match &result.element {
                    ElementSummary::Fixed(name) => {
                        format!("seismic::contract::rt::element_named({name:?})")
                    }
                    ElementSummary::Parameter(name) => {
                        let idx = entry
                            .element_parameters
                            .iter()
                            .position(|p| p == name)
                            .expect("result element names an entry element parameter");
                        format!("__e_{idx}")
                    }
                };
                let extents = result.axis_exprs.join(", ");
                let _ = writeln!(
                    out,
                    "          {{ let __el = {elem_expr}; let __bytes = seismic::contract::rt::canonical_bytes(__el, &[{extents}])?; seismic::TensorResultSpec {{ element: __el, bytes: __bytes }} }},"
                );
            }
            out.push_str("        ])\n");
            out.push_str("      })()?;\n");
        }

        // Scratch specs from __scratch_max
        if scratch_len == 0 {
            out.push_str("      let __scratch = Vec::new();\n");
        } else {
            out.push_str("      let __scratch = vec![\n");
            for (i, scratch) in native.scratch.iter().enumerate() {
                let emission = match scratch.emission {
                    Some(r) => format!("Some({r}u32)"),
                    None => "None".to_owned(),
                };
                let _ = writeln!(
                    out,
                    "        seismic::ScratchSpec {{ bytes: __scratch_max[{i}], sync: {}, emission: {emission} }},",
                    scratch.sync
                );
            }
            out.push_str("      ];\n");
        }

        out.push_str(
            "      Ok(seismic::EntrySizing { results: __results, scratch: __scratch, active_launches: __active_launches, operand_forms: __operand_forms })\n",
        );
        out.push_str("    }\n");
    }
}

fn render_contract_scratch_configs(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_scratch_configs(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<seismic::contract::ScratchConfigs, seismic::ContractError> {\n",
    );
    render_backend_dispatch(out, natives, "scratch_configs", "elements, dims");
}

fn render_contract_scratch_configs_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn scratch_configs_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<seismic::contract::ScratchConfigs, seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &native.statics, "      ");
        let slots = collect_params(native);
        let ctx = ExprCtx {
            backend,
            elements: &entry.element_parameters,
            dims: &native.statics,
            params: &slots,
            launch: None,
            default_param: None,
        };

        let factored = factor_constraints(native, &slots, &ctx);

        emit_conjunct_guards(
            out,
            &ctx,
            &factored.static_conjuncts,
            "return Err(seismic::ContractError::DomainViolation);",
            "      ",
        );

        for &k in &factored.unreferenced {
            if slots[k].param.values.is_empty() {
                out.push_str("          return Err(seismic::ContractError::DomainViolation);\n");
            }
        }

        for (ci, comp) in factored.independent.iter().enumerate() {
            let depth_conjuncts = partition_component_conjuncts(&comp.params, &comp.conjuncts, &ctx);
            let _ = writeln!(out, "          let mut __comp_ok_{ci} = false;");
            let mut indent = String::from("          ");
            for (depth, &k) in comp.params.iter().enumerate() {
                let iter_expr = format_u64_slice(&slots[k].param.values);
                if depth == 0 {
                    let _ = writeln!(out, "{indent}'__comp_{ci}: for __p_{k} in {iter_expr} {{");
                } else {
                    let _ = writeln!(out, "{indent}for __p_{k} in {iter_expr} {{");
                }
                indent.push_str("  ");
                if depth + 1 < comp.params.len() {
                    for c in &depth_conjuncts[depth] {
                        let cond_expr = ctx.lower_cond(c);
                        let _ = writeln!(
                            out,
                            "{indent}if matches!((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})(), Ok(false)) {{ continue; }}"
                        );
                    }
                }
            }
            let all_cond = lower_conjunction(&ctx, &comp.conjuncts);
            let _ = writeln!(out, "{indent}if {all_cond} {{");
            let _ = writeln!(out, "{indent}  __comp_ok_{ci} = true;");
            let _ = writeln!(out, "{indent}  break '__comp_{ci};");
            let _ = writeln!(out, "{indent}}}");
            for _ in 0..comp.params.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            let _ = writeln!(
                out,
                "          if !__comp_ok_{ci} {{ return Err(seismic::ContractError::DomainViolation); }}"
            );
        }

        if factored.scratch.is_empty() {
            out.push_str(
                "          Ok(seismic::contract::ScratchConfigs { arity: 0usize, tuples: Vec::new() })\n",
            );
        } else if factored.scratch.len() == 1 {
            let comp = &factored.scratch[0];
            let arity = comp.outer.len();
            out.push_str("          let mut __tuples: Vec<u64> = Vec::new();\n");
            let mut indent = emit_scratch_outer_and_residuals(
                out,
                0,
                comp,
                &slots,
                &ctx,
                false,
                None,
                "          ",
            );
            for &ok in &comp.outer {
                let _ = writeln!(out, "{indent}__tuples.push(__p_{ok});");
            }
            for _ in 0..comp.outer.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            out.push_str(
                "          if __tuples.is_empty() { return Err(seismic::ContractError::DomainViolation); }\n",
            );
            let _ = writeln!(
                out,
                "          Ok(seismic::contract::ScratchConfigs {{ arity: {arity}usize, tuples: __tuples }})"
            );
        } else {
            for (ci, comp) in factored.scratch.iter().enumerate() {
                let _ = writeln!(out, "          let mut __sub_{ci}: Vec<u64> = Vec::new();");
                let mut indent = emit_scratch_outer_and_residuals(
                    out,
                    ci,
                    comp,
                    &slots,
                    &ctx,
                    false,
                    None,
                    "          ",
                );
                for &ok in &comp.outer {
                    let _ = writeln!(out, "{indent}__sub_{ci}.push(__p_{ok});");
                }
                for _ in 0..comp.outer.len() {
                    indent.truncate(indent.len() - 2);
                    let _ = writeln!(out, "{indent}}}");
                }
                let _ = writeln!(
                    out,
                    "          if __sub_{ci}.is_empty() {{ return Err(seismic::ContractError::DomainViolation); }}"
                );
            }

            let arity = factored.outer_indices.len();
            out.push_str("          let mut __tuples: Vec<u64> = Vec::new();\n");
            let mut indent = String::from("          ");
            for (ci, comp) in factored.scratch.iter().enumerate() {
                let stride = comp.outer.len();
                let _ = writeln!(out, "{indent}for __chunk_{ci} in __sub_{ci}.chunks_exact({stride}usize) {{");
                indent.push_str("  ");
                for (j, &ok) in comp.outer.iter().enumerate() {
                    let _ = writeln!(out, "{indent}let __p_{ok}: u64 = __chunk_{ci}[{j}usize];");
                }
            }
            for &ok in &factored.outer_indices {
                let _ = writeln!(out, "{indent}__tuples.push(__p_{ok});");
            }
            for _ in 0..factored.scratch.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            let _ = writeln!(
                out,
                "          Ok(seismic::contract::ScratchConfigs {{ arity: {arity}usize, tuples: __tuples }})"
            );
        }

        out.push_str("    }\n");
    }
}

fn render_contract_sizing_at_point(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_sizing_at_point(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], configs: &seismic::contract::ScratchConfigs, max_results: &mut [u64], max_scratch: &mut [u64]) -> Result<(), seismic::ContractError> {\n",
    );
    render_backend_dispatch(
        out,
        natives,
        "sizing_at_point",
        "elements, dims, configs, max_results, max_scratch",
    );
}

fn render_contract_sizing_at_point_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
    tensor_results: &[LoweredTensorResult],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn sizing_at_point_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], configs: &seismic::contract::ScratchConfigs, max_results: &mut [u64], max_scratch: &mut [u64]) -> Result<(), seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &entry.dimensions, "      ");

        // Tensor results (zero heap allocations).
        for (i, result) in tensor_results.iter().enumerate() {
            let elem_expr = match &result.element {
                ElementSummary::Fixed(name) => {
                    format!("seismic::contract::rt::element_named({name:?})")
                }
                ElementSummary::Parameter(name) => {
                    let idx = entry
                        .element_parameters
                        .iter()
                        .position(|p| p == name)
                        .expect("result element names an entry element parameter");
                    format!("__e_{idx}")
                }
            };
            let extents = result.axis_exprs.join(", ");
            let _ = writeln!(
                out,
                "      {{ let __bytes = (|| -> Result<u64, seismic::ContractError> {{ let __el = {elem_expr}; seismic::contract::rt::canonical_bytes(__el, &[{extents}]) }})()?; max_results[{i}usize] = max_results[{i}usize].max(__bytes); }}"
            );
        }

        // Scratch buffers (zero heap allocations, no parameter admission loops).
        if !native.scratch.is_empty() {
            let slots = collect_params(native);
            let ctx = ExprCtx {
                backend,
                elements: &entry.element_parameters,
                dims: &entry.dimensions,
                params: &slots,
                launch: None,
                default_param: None,
            };
            let mut scratch_param_names = Vec::new();
            for s in &native.scratch {
                s.parameters(&mut scratch_param_names);
            }
            let outer_indices: Vec<usize> = (0..slots.len())
                .filter(|&k| slots[k].launch.is_none() && scratch_param_names.contains(&slots[k].param.name))
                .collect();

            let mut param_scratch: Vec<(usize, &NativeScratch)> = Vec::new();
            for (i, scratch) in native.scratch.iter().enumerate() {
                let mut sp = Vec::new();
                scratch.parameters(&mut sp);
                if sp.is_empty() {
                    let bytes_expr = scratch_bytes_expr(&ctx, scratch);
                    let _ = writeln!(
                        out,
                        "      max_scratch[{i}usize] = max_scratch[{i}usize].max({bytes_expr});"
                    );
                } else {
                    param_scratch.push((i, scratch));
                }
            }

            if !param_scratch.is_empty() {
                let arity = outer_indices.len();
                let _ = writeln!(out, "      debug_assert_eq!(configs.arity, {arity}usize);");
                let _ = writeln!(
                    out,
                    "      for __cfg in configs.tuples.chunks_exact({arity}usize) {{"
                );
                for (j, &ok) in outer_indices.iter().enumerate() {
                    let _ = writeln!(out, "        let __p_{ok}: u64 = __cfg[{j}usize];");
                }
                for (i, scratch) in param_scratch {
                    let bytes_expr = scratch_bytes_expr(&ctx, scratch);
                    let _ = writeln!(
                        out,
                        "        max_scratch[{i}usize] = max_scratch[{i}usize].max({bytes_expr});"
                    );
                }
                out.push_str("      }\n");
            }
        }

        out.push_str("      Ok(())\n");
        out.push_str("    }\n");
    }
}

fn render_contract_active_launches(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_active_launches(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[(&str, u64)], words: &[(&str, u64, bool)]) -> Result<u64, seismic::ContractError> {\n",
    );
    render_backend_dispatch(
        out,
        natives,
        "active_launches",
        "elements, dims, params, words",
    );
}

fn render_contract_active_launches_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) {
    for &(backend, native) in natives {
        let slug = backend_slug(backend);
        let _ = writeln!(
            out,
            "    pub(super) fn active_launches_{slug}(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[(&str, u64)], words: &[(&str, u64, bool)]) -> Result<u64, seismic::ContractError> {{"
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &entry.dimensions, "      ");
        let slots = collect_params(native);
        emit_param_binding(out, &slots, "      ");

        let ctx = ExprCtx {
            backend,
            elements: &entry.element_parameters,
            dims: &entry.dimensions,
            params: &slots,
            launch: None,
            default_param: None,
        };

        if let Some(constraint) = &native.constraint {
            emit_constraint_guard(out, &ctx, constraint, "      ");
        }

        let emit_launch_active = |out: &mut String, launch_idx: usize, indent: &str| {
            let launch = &native.launches[launch_idx];
            if launch.conditions.is_empty() {
                match &launch.when {
                    None => {
                        let _ = writeln!(out, "{indent}let __act_{launch_idx}: u64 = 1u64;");
                    }
                    Some(when) => {
                        let lctx = ExprCtx {
                            backend,
                            elements: &entry.element_parameters,
                            dims: &entry.dimensions,
                            params: &slots,
                            launch: Some(launch_idx),
                            default_param: None,
                        };
                        let w = lctx.lower_cond(when);
                        let _ = writeln!(
                            out,
                            "{indent}let __act_{launch_idx}: u64 = if (|| -> Result<bool, seismic::ContractError> {{ Ok({w}) }})()? {{ 1u64 }} else {{ 0u64 }};"
                        );
                    }
                }
            } else {
                let mut block = String::from("(|| -> Result<u64, seismic::ContractError> { ");
                let mut terms = Vec::with_capacity(launch.conditions.len());
                for (ci, cond) in launch.conditions.iter().enumerate() {
                    let word = &cond.word;
                    let index = cond.index;
                    let _ = write!(
                        block,
                        "let __w_{ci} = words.iter().find(|&&(w, idx, _)| w == {word:?} && idx == {index}u64).map(|&(_, _, s)| s).ok_or(seismic::ContractError::MissingConditionWord {{ word: {word:?}, index: {index}u64 }})?; "
                    );
                    match cond.encoded {
                        Encoded::WhileSet => terms.push(format!("__w_{ci}")),
                        Encoded::WhileClear => terms.push(format!("!__w_{ci}")),
                    }
                }
                let _ = write!(block, "Ok(u64::from({})) }})()?", terms.join(" && "));
                match &launch.when {
                    None => {
                        let _ = writeln!(out, "{indent}let __act_{launch_idx}: u64 = {block};");
                    }
                    Some(when) => {
                        let lctx = ExprCtx {
                            backend,
                            elements: &entry.element_parameters,
                            dims: &entry.dimensions,
                            params: &slots,
                            launch: Some(launch_idx),
                            default_param: None,
                        };
                        let w = lctx.lower_cond(when);
                        let _ = writeln!(
                            out,
                            "{indent}let __act_{launch_idx}: u64 = if (|| -> Result<bool, seismic::ContractError> {{ Ok({w}) }})()? {{ {block} }} else {{ 0u64 }};"
                        );
                    }
                }
            }
        };

        let sum_range = |range: std::ops::Range<usize>| -> String {
            if range.is_empty() {
                "0u64".to_owned()
            } else {
                range
                    .map(|i| format!("__act_{i}"))
                    .collect::<Vec<_>>()
                    .join(" + ")
            }
        };

        match &native.repeat {
            None => {
                for launch_idx in 0..native.launches.len() {
                    emit_launch_active(out, launch_idx, "      ");
                }
                let total = sum_range(0..native.launches.len());
                let _ = writeln!(out, "      Ok({total})");
            }
            Some(repeat) => {
                let rounds = ctx.lower_nat(&repeat.count);
                let _ = writeln!(
                    out,
                    "      let __rounds: u64 = (|| -> Result<u64, seismic::ContractError> {{ Ok({rounds}) }})()?;"
                );
                for launch_idx in 0..repeat.first {
                    emit_launch_active(out, launch_idx, "      ");
                }
                let body_range = repeat.first..repeat.first + repeat.launches;
                if body_range.is_empty() {
                    out.push_str("      let __body: u64 = 0u64;\n");
                } else {
                    out.push_str("      let __body: u64 = if __rounds == 0u64 {\n");
                    out.push_str("        0u64\n");
                    out.push_str("      } else {\n");
                    for launch_idx in body_range.clone() {
                        emit_launch_active(out, launch_idx, "        ");
                    }
                    let body_sum = sum_range(body_range);
                    let _ = writeln!(
                        out,
                        "        ({body_sum}).checked_mul(__rounds).ok_or(seismic::ContractError::ArithmeticOverflow)?"
                    );
                    out.push_str("      };\n");
                }
                for launch_idx in (repeat.first + repeat.launches)..native.launches.len() {
                    emit_launch_active(out, launch_idx, "      ");
                }
                let before = sum_range(0..repeat.first);
                let after = sum_range(repeat.first + repeat.launches..native.launches.len());
                let _ = writeln!(
                    out,
                    "      (__body).checked_add({before}).and_then(|v| v.checked_add({after})).ok_or(seismic::ContractError::ArithmeticOverflow)"
                );
            }
        }

        out.push_str("    }\n");
    }
}

fn render_contract_launch_configs(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_launch_configs(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<Vec<Vec<u64>>, seismic::ContractError> {\n",
    );
    if natives.is_empty() {
        out.push_str("      Err(seismic::ContractError::NoImplementation(backend))\n");
        out.push_str("    }\n");
        return;
    }
    out.push_str("      match backend {\n");
    for &(backend, _) in natives {
        let variant = backend_variant(backend);
        if backend == BackendName::Cuda {
            out.push_str(
                "        seismic::BackendName::Cuda => __contract::launch_configs_cuda(elements, dims),\n",
            );
        } else {
            let _ = writeln!(
                out,
                "        seismic::BackendName::{variant} => Ok(vec![Vec::new()]),"
            );
        }
    }
    if natives.len() < BackendName::ALL.len() {
        out.push_str("        _ => Err(seismic::ContractError::NoImplementation(backend)),\n");
    }
    out.push_str("      }\n");
    out.push_str("    }\n");
}

fn render_contract_launch_configs_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) {
    for &(backend, native) in natives {
        if backend != BackendName::Cuda {
            continue;
        }
        out.push_str(
            "    pub(super) fn launch_configs_cuda(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)]) -> Result<Vec<Vec<u64>>, seismic::ContractError> {\n",
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &native.statics, "      ");
        let slots = collect_params(native);
        let ctx = ExprCtx {
            backend,
            elements: &entry.element_parameters,
            dims: &native.statics,
            params: &slots,
            launch: None,
            default_param: None,
        };
        let outer_indices = launch_when_outer_indices(native, &slots);
        let factored = factor_constraints_for(native, &slots, &ctx, outer_indices);

        emit_conjunct_guards(
            out,
            &ctx,
            &factored.static_conjuncts,
            "return Err(seismic::ContractError::DomainViolation);",
            "      ",
        );

        for &k in &factored.unreferenced {
            if slots[k].param.values.is_empty() {
                out.push_str("      return Err(seismic::ContractError::DomainViolation);\n");
            }
        }

        for (ci, comp) in factored.independent.iter().enumerate() {
            let depth_conjuncts = partition_component_conjuncts(&comp.params, &comp.conjuncts, &ctx);
            let _ = writeln!(out, "      let mut __comp_ok_{ci} = false;");
            let mut indent = String::from("      ");
            for (depth, &k) in comp.params.iter().enumerate() {
                let iter_expr = format_u64_slice(&slots[k].param.values);
                if depth == 0 {
                    let _ = writeln!(out, "{indent}'__comp_{ci}: for __p_{k} in {iter_expr} {{");
                } else {
                    let _ = writeln!(out, "{indent}for __p_{k} in {iter_expr} {{");
                }
                indent.push_str("  ");
                if depth + 1 < comp.params.len() {
                    for c in &depth_conjuncts[depth] {
                        let cond_expr = ctx.lower_cond(c);
                        let _ = writeln!(
                            out,
                            "{indent}if matches!((|| -> Result<bool, seismic::ContractError> {{ Ok({cond_expr}) }})(), Ok(false)) {{ continue; }}"
                        );
                    }
                }
            }
            let all_cond = lower_conjunction(&ctx, &comp.conjuncts);
            let _ = writeln!(out, "{indent}if {all_cond} {{");
            let _ = writeln!(out, "{indent}  __comp_ok_{ci} = true;");
            let _ = writeln!(out, "{indent}  break '__comp_{ci};");
            let _ = writeln!(out, "{indent}}}");
            for _ in 0..comp.params.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            let _ = writeln!(
                out,
                "      if !__comp_ok_{ci} {{ return Err(seismic::ContractError::DomainViolation); }}"
            );
        }

        if factored.scratch.is_empty() {
            out.push_str("      Ok(vec![Vec::new()])\n");
        } else if factored.scratch.len() == 1 {
            let comp = &factored.scratch[0];
            out.push_str("      let mut __configs: Vec<Vec<u64>> = Vec::new();\n");
            let mut indent = emit_scratch_outer_and_residuals(
                out,
                0,
                comp,
                &slots,
                &ctx,
                false,
                None,
                "      ",
            );
            let elems = comp
                .outer
                .iter()
                .map(|&ok| format!("__p_{ok}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "{indent}__configs.push(vec![{elems}]);");
            for _ in 0..comp.outer.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            out.push_str(
                "      if __configs.is_empty() { return Err(seismic::ContractError::DomainViolation); }\n",
            );
            out.push_str("      Ok(__configs)\n");
        } else {
            for (ci, comp) in factored.scratch.iter().enumerate() {
                let _ = writeln!(out, "      let mut __sub_{ci}: Vec<u64> = Vec::new();");
                let mut indent = emit_scratch_outer_and_residuals(
                    out,
                    ci,
                    comp,
                    &slots,
                    &ctx,
                    false,
                    None,
                    "      ",
                );
                for &ok in &comp.outer {
                    let _ = writeln!(out, "{indent}__sub_{ci}.push(__p_{ok});");
                }
                for _ in 0..comp.outer.len() {
                    indent.truncate(indent.len() - 2);
                    let _ = writeln!(out, "{indent}}}");
                }
                let _ = writeln!(
                    out,
                    "      if __sub_{ci}.is_empty() {{ return Err(seismic::ContractError::DomainViolation); }}"
                );
            }

            out.push_str("      let mut __configs: Vec<Vec<u64>> = Vec::new();\n");
            let mut indent = String::from("      ");
            for (ci, comp) in factored.scratch.iter().enumerate() {
                let stride = comp.outer.len();
                let _ = writeln!(
                    out,
                    "{indent}for __chunk_{ci} in __sub_{ci}.chunks_exact({stride}usize) {{"
                );
                indent.push_str("  ");
                for (j, &ok) in comp.outer.iter().enumerate() {
                    let _ = writeln!(out, "{indent}let __p_{ok}: u64 = __chunk_{ci}[{j}usize];");
                }
            }
            let elems = factored
                .outer_indices
                .iter()
                .map(|&ok| format!("__p_{ok}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "{indent}__configs.push(vec![{elems}]);");
            for _ in 0..factored.scratch.len() {
                indent.truncate(indent.len() - 2);
                let _ = writeln!(out, "{indent}}}");
            }
            out.push_str("      Ok(__configs)\n");
        }

        out.push_str("    }\n");
    }
}

fn render_contract_launch_mask(
    out: &mut String,
    natives: &[(BackendName, &NativeImplementation)],
) {
    out.push_str(
        "    fn contract_launch_mask(backend: seismic::BackendName, elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[u64]) -> Result<u64, seismic::ContractError> {\n",
    );
    if natives.is_empty() {
        out.push_str("      Err(seismic::ContractError::NoImplementation(backend))\n");
        out.push_str("    }\n");
        return;
    }
    out.push_str("      match backend {\n");
    for &(backend, _) in natives {
        let variant = backend_variant(backend);
        if backend == BackendName::Cuda {
            out.push_str(
                "        seismic::BackendName::Cuda => __contract::launch_mask_cuda(elements, dims, params),\n",
            );
        } else {
            let _ = writeln!(out, "        seismic::BackendName::{variant} => Ok(0u64),");
        }
    }
    if natives.len() < BackendName::ALL.len() {
        out.push_str("        _ => Err(seismic::ContractError::NoImplementation(backend)),\n");
    }
    out.push_str("      }\n");
    out.push_str("    }\n");
}

fn render_contract_launch_mask_helpers(
    out: &mut String,
    entry: &EntryInfo,
    natives: &[(BackendName, &NativeImplementation)],
) {
    for &(backend, native) in natives {
        if backend != BackendName::Cuda {
            continue;
        }
        assert!(
            native.launches.len() <= 64,
            "entry `{}` CUDA implementation declares {} launches (at most 64 supported)",
            entry.name,
            native.launches.len()
        );
        out.push_str(
            "    pub(super) fn launch_mask_cuda(elements: &[(&str, seismic::Element)], dims: &[(&str, u64)], params: &[u64]) -> Result<u64, seismic::ContractError> {\n",
        );
        emit_element_binding(out, entry, "      ");
        emit_unit_column_checks(out, entry, native, "      ");
        emit_dim_binding(out, &entry.dimensions, "      ");
        let slots = collect_params(native);
        let outer_indices = launch_when_outer_indices(native, &slots);
        let arity = outer_indices.len();
        let _ = writeln!(
            out,
            "      if params.len() != {arity}usize {{ return Err(seismic::ContractError::InvalidParameter(\"launch_params\")); }}"
        );
        for (i, &k) in outer_indices.iter().enumerate() {
            let _ = writeln!(out, "      let __p_{k}: u64 = params[{i}usize];");
        }
        let mut static_mask = 0u64;
        for (launch_idx, launch) in native.launches.iter().enumerate() {
            if launch.when.is_none() {
                static_mask |= 1u64 << launch_idx;
            }
        }
        let has_conditional = native.launches.iter().any(|l| l.when.is_some());
        if !has_conditional {
            let _ = writeln!(out, "      Ok({static_mask}u64)");
        } else {
            let _ = writeln!(out, "      let mut __mask: u64 = {static_mask}u64;");
            for (launch_idx, launch) in native.launches.iter().enumerate() {
                if let Some(when) = &launch.when {
                    let lctx = ExprCtx {
                        backend,
                        elements: &entry.element_parameters,
                        dims: &entry.dimensions,
                        params: &slots,
                        launch: Some(launch_idx),
                        default_param: None,
                    };
                    let w = lctx.lower_cond(when);
                    let _ = writeln!(
                        out,
                        "      if (|| -> Result<bool, seismic::ContractError> {{ Ok({w}) }})()? {{ __mask |= 1u64 << {launch_idx}u32; }}"
                    );
                }
            }
            out.push_str("      Ok(__mask)\n");
        }
        out.push_str("    }\n");
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{
        check_source, NativeComparison, NativeCondition, NativeDefault, NativeImplementation,
        NativeNatExpr, NativeParameter, NativeParameterRole, NativeScratch, NativeSpecialization,
        SourceFile, SourceSet,
    };

    fn param(name: &str, values: Vec<u64>) -> NativeParameter {
        let default = NativeDefault::Value(NativeNatExpr::Constant(values[0]));
        NativeParameter {
            name: name.to_owned(),
            code: false,
            arithmetic: false,
            form: false,
            values,
            default,
            role: NativeParameterRole::Declared,
        }
    }

    fn cmp(comparison: NativeComparison, left: NativeNatExpr, right: NativeNatExpr) -> NativeCondition {
        NativeCondition::Compare {
            comparison,
            left,
            right,
        }
    }

    fn and(a: NativeCondition, b: NativeCondition) -> NativeCondition {
        NativeCondition::And(Box::new(a), Box::new(b))
    }

    fn or(a: NativeCondition, b: NativeCondition) -> NativeCondition {
        NativeCondition::Or(Box::new(a), Box::new(b))
    }

    fn p(name: &str) -> NativeNatExpr {
        NativeNatExpr::Parameter(name.to_owned())
    }

    fn c(v: u64) -> NativeNatExpr {
        NativeNatExpr::Constant(v)
    }

    fn add(a: NativeNatExpr, b: NativeNatExpr) -> NativeNatExpr {
        NativeNatExpr::Add(Box::new(a), Box::new(b))
    }

    #[test]
    fn residual_inner_factoring_splits_disjoint_and_coupled_components() {
        // Outer param `OUT` in [0, 1] is read by scratch.
        // Inner params:
        //   A in [1, 2]: disjoint residual component #1 (`OUT == 0 or A == 2`)
        //   B in [3, 4]: disjoint residual component #2 (`OUT == 0 or B == 4`)
        //   C in [1, 2], D in [2, 4]: coupled pair residual component #3
        //     (`(OUT == 1 and C + D == 6) or (OUT == 0 and C + D == 99)`)
        // Satisfiable for OUT = 1 (A=2, B=4, C=2, D=4); unsatisfiable for OUT = 0 (C+D==99).
        let constraint = and(
            and(
                or(cmp(NativeComparison::Eq, p("OUT"), c(0)), cmp(NativeComparison::Eq, p("A"), c(2))),
                or(cmp(NativeComparison::Eq, p("OUT"), c(0)), cmp(NativeComparison::Eq, p("B"), c(4))),
            ),
            or(
                and(
                    cmp(NativeComparison::Eq, p("OUT"), c(1)),
                    cmp(NativeComparison::Eq, add(p("C"), p("D")), c(6)),
                ),
                and(
                    cmp(NativeComparison::Eq, p("OUT"), c(0)),
                    cmp(NativeComparison::Eq, add(p("C"), p("D")), c(99)),
                ),
            ),
        );

        let dummy_module = check_source(SourceSet::new(vec![SourceFile {
            path: "probe.seismic".to_owned(),
            text: "fn probe(dst: &mut tensor[2] f32):\n    dst = zeros_like(dst)\n".to_owned(),
        }]))
        .expect("dummy module checks cleanly");
        let entry = dummy_module
            .entry_named("probe")
            .expect("probe entry exists");

        let native = NativeImplementation {
            entry,
            backend: BackendName::Metal,
            declared_in: "test.seismic".to_owned(),
            source_path: "metal/test.metal".to_owned(),
            statics: Vec::new(),
            params: vec![
                param("OUT", vec![0, 1]),
                param("A", vec![1, 2]),
                param("B", vec![3, 4]),
                param("C", vec![1, 2]),
                param("D", vec![2, 4]),
            ],
            elements: Vec::new(),
            records: Vec::new(),
            launch_records: Vec::new(),
            constraint: Some(constraint),
            scratch: vec![NativeScratch {
                name: "buf".to_owned(),
                bytes: add(p("OUT"), c(1)),
                sync: false,
                when: None,
                operand: None,
                emission: None,
            }],
            launches: Vec::new(),
            repeat: None,
            formation: None,
            segments: None,
            ordered_segments: Vec::new(),
        };

        let slots = collect_params(&native);
        let ctx = ExprCtx {
            backend: BackendName::Metal,
            elements: &[],
            dims: &[],
            params: &slots,
            launch: None,
            default_param: None,
        };
        let factored = factor_constraints(&native, &slots, &ctx);

        assert_eq!(factored.outer_indices, vec![0]);
        assert!(factored.unreferenced.is_empty());
        assert!(factored.independent.is_empty());
        assert_eq!(factored.scratch.len(), 1);

        let comp = &factored.scratch[0];
        assert_eq!(comp.outer, vec![0]);
        let mut residual_param_sets: Vec<Vec<usize>> = comp
            .residuals
            .iter()
            .map(|r| r.params.clone())
            .collect();
        residual_param_sets.sort();
        // Two disjoint singleton components ([A]=1, [B]=2) plus one coupled pair ([C, D]=[3, 4]).
        assert_eq!(residual_param_sets, vec![vec![1], vec![2], vec![3, 4]]);

        // Cross-check satisfiable (OUT=1) and unsatisfiable (OUT=0) against NativeImplementation::scratch_configurations.
        let admitted = native
            .scratch_configurations(&NativeSpecialization::new())
            .expect("OUT=1 is satisfiable");
        let admitted_outs: Vec<u64> = admitted
            .iter()
            .map(|spec| spec.param("OUT").expect("OUT bound"))
            .collect();
        assert_eq!(admitted_outs, vec![1]);
    }
}
