//! Witness -> concrete execution IR. Deterministic: consumes the selected
//! implementations, covers and numbers and returns one execution or a diagnostic.
//! Never chooses, never repairs.
//!
//! Mapping rules (fixed, not selected here):
//! - a root-level `parallel` region of the entry is one launch (`StmtKind::Parallel`) whose
//!   work items are the pieces; every other region is an ordered loop over pieces inside
//!   its owner; root stages and invocation-scope statements are consecutive root statements;
//! - a slice is the range `[lo + p*w, lo + (p+1)*w)`; only dividing widths are instantiated;
//! - calls are inlined with the witness-selected candidate; views stay references;
//! - tile-valued computation is one element loop per execution unit, one loop per selected
//!   elementwise interval; region results are local tiles with leading piece axes.
use super::family::{CandidateRef, Family, Witness};
use super::sir::Program;
use super::sir::{Mode, RegionMode};
use super::types as st;
use crate::exec::lowered_ir::{AliasRequirement, LoweredIr, ResultBinding};
use crate::exec::ir::{Expr, ExprKind, Var, VarKind};
use crate::exec::types::Ty;
use crate::sym::Atom;
use crate::types::DType;

mod context;
mod expr;
mod region;
mod stmt;
mod walk;

use context::{Instantiation, Value};

/// Targets whose structural mapping gives the inner owner regions of a launch participants of
/// their own (Metal: one SIMD group per inner owner inside the threadgroup of the launch
/// piece). Their execution IR keeps such a region as a `Parallel` statement nested in the
/// launch; for every other target it is ordered loops over its pieces, like any nested region.
pub fn owner_regions(target: &str) -> bool {
    target == "metal"
}

/// The structural half of the inner-owner rule, shared with the backends' accounting: `region`
/// is a statement-position `parallel` region without merge or result over domains, and every
/// binder refines a distinct binder of the launch (`launch_binders`, slices of the same body).
/// Position is the other half: the region sits in the owner body of that launch, outside
/// element loops, branches and other inner owner regions.
pub fn inner_owner_region(
    body: &crate::sir::Body,
    launch_binders: &[st::SliceId],
    region: &crate::sir::Region,
) -> bool {
    use crate::sir::{RegionSource, SliceParent, VarKind};
    if region.mode != RegionMode::Parallel
        || region.merge.is_some()
        || region.result.is_some()
        || region.source != RegionSource::Domains
        || region.binders.is_empty()
    {
        return false;
    }
    let mut refined = Vec::new();
    for binder in &region.binders {
        let Some(VarKind::Slice(slice)) = body.vars.get(*binder).map(|declared| &declared.kind)
        else {
            return false;
        };
        let Some(SliceParent::Refine(parent)) = body
            .slices
            .get(slice.0 as usize)
            .map(|declared| &declared.parent)
        else {
            return false;
        };
        if !launch_binders.contains(parent) || refined.contains(parent) {
            return false;
        }
        refined.push(*parent);
    }
    true
}

pub fn instantiate(
    program: &Program,
    family: &Family,
    witness: &Witness,
) -> Result<LoweredIr, String> {
    let mut inst = Instantiation::new(program, family, witness);
    let entry = family
        .occurrences
        .first()
        .ok_or_else(|| format!("family of `{}` has no entry occurrence", family.entry))?;
    let choice = *witness.choices.get(&entry.id).ok_or_else(|| {
        format!(
            "witness selects no implementation of entry `{}`",
            family.entry
        )
    })?;
    let mut frame = inst.frame(CandidateRef {
        occurrence: entry.id,
        candidate: choice,
    })?;
    let definition = frame.definition;

    // Entry parameters are execution variables 0..n in declaration order.
    let mut params = Vec::with_capacity(definition.params.len());
    let mut index_params = Vec::new();
    let mut range_params = Vec::new();
    let mut source_ordinals = Vec::with_capacity(definition.params.len());
    for param in &definition.params {
        source_ordinals.push(params.len());
        if let st::Ty::Range(bound) = &param.ty {
            let bound = inst.resolve(&frame, bound)?;
            let start = format!("{}_start", param.name);
            let end = format!("{}_end", param.name);
            let mut endpoints = Vec::new();
            for name in [&start, &end] {
                let ordinal = params.len();
                let atom = Atom::Param(name.clone());
                let ty = Ty::Scalar(DType::I32);
                inst.vars.push(crate::exec::ir::Var { name: name.clone(), ty: ty.clone(), span: definition.span, kind: crate::exec::ir::VarKind::Param(ordinal) });
                endpoints.push(context::symbol(crate::sym::Sym::atom(atom), definition.span));
                params.push((name.clone(), ty));
            }
            *frame.vars.get_mut(param.var).ok_or_else(|| format!("entry parameter `{}` has no variable", param.name))? = Some(Value::Range(endpoints.remove(0), endpoints.remove(0)));
            range_params.push(crate::exec::lowered_ir::RangeParameter { name: param.name.clone(), start, end, bound });
            continue;
        }
        let ordinal = params.len();
        let (ty, kind) = match &param.ty {
            st::Ty::Tensor(shaped) | st::Ty::View(shaped) => (Ty::Tensor(inst.shaped(&frame, shaped)?), crate::exec::ir::VarKind::Param(ordinal)),
            st::Ty::Scalar(dtype) => (Ty::Scalar(*dtype), crate::exec::ir::VarKind::Param(ordinal)),
            st::Ty::Index(bound) => {
                index_params.push((param.name.clone(), inst.resolve(&frame, bound)?));
                (Ty::Scalar(DType::I32), crate::exec::ir::VarKind::Index(Atom::Param(param.name.clone())))
            }
            other => return Err(format!("entry `{}` parameter `{}` has type `{other}`, which is not part of the invocation ABI", definition.name, param.name)),
        };
        let id = inst.vars.len();
        // A bounded index parameter is read as its symbol: phase formation and emission
        // resolve it from `index_params`, not from a variable binding of some phase.
        let reference = match &kind {
            crate::exec::ir::VarKind::Index(atom) => {
                context::symbol(crate::sym::Sym::atom(atom.clone()), definition.span)
            }
            _ => crate::exec::ir::Expr {
                kind: crate::exec::ir::ExprKind::Var(id),
                ty: ty.clone(),
                sym: None,
                span: definition.span,
            },
        };
        inst.vars.push(crate::exec::ir::Var {
            name: param.name.clone(),
            ty: ty.clone(),
            span: definition.span,
            kind,
        });
        let slot = frame
            .vars
            .get_mut(param.var)
            .ok_or_else(|| format!("entry parameter `{}` has no variable", param.name))?;
        *slot = Some(if matches!(ty, Ty::Scalar(_)) {
            Value::Scalar(reference)
        } else {
            Value::Shaped(reference)
        });
        params.push((param.name.clone(), ty));
    }
    // This is the flattened invocation prefix; logical ranges contribute two ABI scalars.
    let source_param_count = params.len();

    // Every written parameter is disjoint from every other tensor parameter; a declared
    // `alias` pair may coincide exactly.
    let mut alias_requirements = Vec::new();
    for (source_left, a) in definition.params.iter().enumerate() {
        for (source_right, b) in definition.params.iter().enumerate().skip(source_left + 1) {
            let left = source_ordinals[source_left];
            let right = source_ordinals[source_right];
            let tensors =
                matches!(params[left].1, Ty::Tensor(_)) && matches!(params[right].1, Ty::Tensor(_));
            if tensors && (a.mode != Mode::In || b.mode != Mode::In) {
                let exact_allowed = definition
                    .aliases
                    .iter()
                    .any(|&(x, y)| (x, y) == (source_left, source_right) || (y, x) == (source_left, source_right));
                alias_requirements.push(AliasRequirement {
                    left,
                    right,
                    exact_allowed,
                });
            }
        }
    }

    let body = frame.body;
    let mut stmts = inst.block(&mut frame, &body.block, true)?;
    let mut prologue = std::mem::take(&mut frame.returns.prologue);
    prologue.append(&mut stmts);

    fn specialized_result(
        inst: &Instantiation<'_>,
        frame: &context::Frame<'_>,
        ty: &st::Ty,
    ) -> Result<Ty, String> {
        match ty {
            st::Ty::Tensor(shaped) => Ok(Ty::Tensor(inst.shaped(frame, shaped)?)),
            st::Ty::Tuple(items) => items
                .iter()
                .map(|item| specialized_result(inst, frame, item))
                .collect::<Result<Vec<_>, _>>()
                .map(Ty::Tuple),
            st::Ty::Void => Ok(Ty::Void),
            other => Err(format!(
                "entry `{}` result `{other}` is not an owned tensor or tuple of owned tensors",
                frame.definition.name
            )),
        }
    }
    fn bind_results<'a>(
        inst: &mut Instantiation<'a>,
        ty: &Ty,
        value: Value<'a>,
        path: &mut Vec<u32>,
        params: &mut Vec<(String, Ty)>,
        bindings: &mut Vec<ResultBinding>,
        names: &mut std::collections::BTreeSet<String>,
        out: &mut Vec<crate::exec::ir::Stmt>,
    ) -> Result<(), String> {
        match (ty, value) {
            (Ty::Tensor(shape), Value::Shaped(source)) => {
                let suffix = if path.is_empty() {
                    String::new()
                } else {
                    path.iter().map(|item| format!(".{item}")).collect()
                };
                let name = format!("$return{suffix}");
                let parameter = params.len();
                let ty = Ty::Tensor(shape.clone());
                let variable = inst.vars.len();
                inst.vars.push(Var {
                    name: name.clone(),
                    ty: ty.clone(),
                    span: source.span,
                    kind: VarKind::Param(parameter),
                });
                let destination = Expr {
                    kind: ExprKind::Var(variable),
                    ty: ty.clone(),
                    sym: None,
                    span: source.span,
                };
                params.push((name.clone(), ty));
                bindings.push(ResultBinding {
                    path: path.clone(),
                    parameter,
                });
                names.insert(name);
                inst.copy_into(destination, crate::syntax::ast::AssignOp::Assign, source, out)
            }
            (Ty::Tuple(types), Value::Tuple(values)) if types.len() == values.len() => {
                for (ordinal, (ty, value)) in types.iter().zip(values).enumerate() {
                    path.push(ordinal as u32);
                    bind_results(inst, ty, value, path, params, bindings, names, out)?;
                    path.pop();
                }
                Ok(())
            }
            (expected, actual) => Err(format!(
                "entry owned result `{expected}` has no matching execution value `{actual:?}`"
            )),
        }
    }

    let result = specialized_result(&inst, &frame, &definition.result)?;
    let returned = frame.returns.locals.or(frame.returns.direct).unwrap_or_default();
    let root = match (&result, returned.len()) {
        (Ty::Void, 0) => Value::Void,
        (Ty::Tuple(_), _) => Value::Tuple(returned),
        (_, 1) => returned.into_iter().next().unwrap(),
        (Ty::Void, _) => return Err(format!("entry `{}` returns a value despite its void contract", definition.name)),
        _ => return Err(format!("entry `{}` does not return its complete owned result", definition.name)),
    };
    let mut result_bindings = Vec::new();
    let mut result_names = std::collections::BTreeSet::new();
    if !matches!(result, Ty::Void) {
        bind_results(
            &mut inst,
            &result,
            root,
            &mut Vec::new(),
            &mut params,
            &mut result_bindings,
            &mut result_names,
            &mut prologue,
        )?;
    }

    let ir = LoweredIr {
        name: family.entry.clone(),
        backend: family.target.clone(),
        ownership: crate::exec::lowered_ir::Ownership {
            results: result_names,
            ..Default::default()
        },
        alias_requirements,
        params,
        source_param_count,
        result,
        result_bindings,
        index_params,
        range_params,
        vars: inst.vars,
        body: prologue,
        shapes: family
            .workload
            .shapes
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
    };
    crate::exec::verify::lowered(&ir)?;
    Ok(ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::ir::{ExprKind, Stmt, StmtKind};
    use crate::family::{self, Workload};
    use crate::program::{compile, SourceFile};

    /// The witness written out by hand as a rule: the first candidate of every active
    /// occurrence, singleton covers, and the given numbers for the sites in family order.
    fn build(
        sources: &[&str],
        entry: &str,
        workload: &Workload,
        numbers: &[i64],
    ) -> (Family, LoweredIr) {
        build_for_target(sources, entry, "cpu", workload, numbers)
    }

    fn build_for_target(
        sources: &[&str],
        entry: &str,
        target: &str,
        workload: &Workload,
        numbers: &[i64],
    ) -> (Family, LoweredIr) {
        let files: Vec<SourceFile> = sources
            .iter()
            .enumerate()
            .map(|(i, text)| SourceFile {
                path: format!("test{i}.seismic"),
                text: text.to_string(),
            })
            .collect();
        let program = compile(&files).unwrap_or_else(|d| {
            panic!(
                "{}",
                d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n")
            )
        });
        let supports = |_: &crate::sir::IntrinsicUse| Ok(());
        let environment = family::TargetEnvironment {
            target,
            capability_fingerprint: "instantiate-test-capabilities-v1",
            supports_intrinsic: &supports,
        };
        let family = family::construct(&program, entry, &environment, workload).expect("family");
        let mut witness = Witness::default();
        let mut pending = vec![family.occurrences[0].id];
        let mut numbers = numbers.iter();
        while let Some(occurrence) = pending.pop() {
            witness.choices.insert(occurrence, 0);
            let candidate = &family.occurrence(occurrence).candidates[0];
            pending.extend(candidate.children.iter().rev());
            for site in &candidate.sites {
                witness
                    .sites
                    .insert(*site, *numbers.next().expect("a number per active site"));
            }
            for sequence in &candidate.sequences {
                let units = family.sequences[sequence.0 as usize].units.len() as u32;
                witness
                    .covers
                    .insert(*sequence, (0..units).map(|u| (u, u + 1)).collect());
            }
        }
        let ir = instantiate(&program, &family, &witness).expect("instantiation");
        crate::exec::verify::lowered(&ir).expect("executable IR");
        (family, ir)
    }

    #[test]
    fn logical_intrinsic_is_preserved_with_owned_destination() {
        let source = "\
fn product(a: &tensor[2, 3] f32, b: &tensor[3, 2] f32, y: &mut tensor[2, 2] f32) for metal requires metal.matrix:
    let value = metal.matrix.matmul(a, b, accumulation=f32)
    for i in 0..2:
        for j in 0..2:
            y[i, j] = value[i, j]
";
        let (_, ir) = build_for_target(&[source], "product", "metal", &Workload::default(), &[]);
        assert!(any(&ir.body, &|statement| {
            matches!(
                &statement.kind,
                StmtKind::Assign {
                    target: crate::exec::ir::Expr {
                        kind: ExprKind::Var(_),
                        ty: Ty::Tile(_),
                        ..
                    },
                    op: crate::syntax::ast::AssignOp::Assign,
                    value: crate::exec::ir::Expr {
                        kind: ExprKind::Intrinsic {
                            op: crate::intrinsics::Operation::MatrixMatmul,
                            ..
                        },
                        ty: Ty::Tile(_),
                        ..
                    },
                }
            )
        }));
    }

    #[test]
    fn owned_tuple_result_flattens_to_hidden_destinations() {
        let source = "\
fn pair[N](x: tensor[N] f32, y: tensor[N] f32) -> (tensor[N] f32, tensor[N] f32):
    return x, y
";
        let workload = Workload { shapes: [("N".into(), 4)].into(), ..Default::default() };
        let (_, ir) = build(&[source], "pair", &workload, &[]);
        assert_eq!(ir.source_param_count, 2);
        assert_eq!(ir.result_bindings.len(), 2);
        assert_eq!(ir.result_bindings[0].path, [0]);
        assert_eq!(ir.result_bindings[1].path, [1]);
        assert_eq!(ir.params[ir.result_bindings[0].parameter].0, "$return.0");
        assert_eq!(ir.params[ir.result_bindings[1].parameter].0, "$return.1");
        assert!(ir.ownership.results.contains("$return.0"));
        assert!(ir.ownership.results.contains("$return.1"));
        crate::exec::verify::lowered(&ir).unwrap();
    }

    #[test]
    fn owned_tensor_snapshot_materializes_before_result_binding() {
        let source = "\
fn snapshot[N](x: &tensor[N] f32) -> tensor[N] f32:
    return to_owned(x)
";
        let workload = Workload { shapes: [("N".into(), 4)].into(), ..Default::default() };
        let (_, ir) = build(&[source], "snapshot", &workload, &[]);
        assert_eq!(ir.result_bindings.len(), 1);
        assert!(any(&ir.body, &|statement| matches!(
            &statement.kind,
            StmtKind::Assign { value: crate::exec::ir::Expr { kind: ExprKind::Load { .. }, ty: Ty::Tile(_), .. }, .. }
        )));
        assert!(ir.ownership.results.contains("$return"));
        crate::exec::verify::lowered(&ir).unwrap();
    }

    #[test]
    fn entry_range_flattens_to_an_ordered_scalar_pair() {
        let source = "\
fn fill[N](selected: range[N]):
    for i in selected:
        let value = i
";
        let workload = Workload { shapes: [("N".into(), 5)].into(), ..Default::default() };
        let (_, ir) = build(&[source], "fill", &workload, &[]);
        assert_eq!(ir.params.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), ["selected_start", "selected_end"]);
        assert_eq!(ir.source_param_count, 2);
        assert_eq!(ir.range_params[0].name, "selected");
        assert_eq!(ir.range_params[0].bound.as_constant(), Some(5));
        let first = crate::abi::ScalarParameter::from_lowered(&ir, "selected_start", DType::I32).unwrap();
        assert!(matches!(first.range.as_ref().map(|r| r.endpoint), Some(crate::abi::RangeEndpoint::Start)));
    }

    fn any(body: &[Stmt], test: &dyn Fn(&Stmt) -> bool) -> bool {
        body.iter().any(|s| {
            test(s)
                || match &s.kind {
                    StmtKind::Parallel { body, .. }
                    | StmtKind::Owned { body, .. }
                    | StmtKind::Range { body, .. } => any(body, test),
                    StmtKind::If { then, els, .. } => any(then, test) || any(els, test),
                    _ => false,
                }
        })
    }




}
