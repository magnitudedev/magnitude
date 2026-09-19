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
use super::syntax::ast::Mode;
use super::types as st;
use crate::exec::lowered_ir::{AliasRequirement, LoweredIr};
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
    if region.mode != crate::syntax::ast::RegionMode::Parallel
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
    for (ordinal, param) in definition.params.iter().enumerate() {
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

    // Every written parameter is disjoint from every other tensor parameter; a declared
    // `alias` pair may coincide exactly.
    let mut alias_requirements = Vec::new();
    for (left, a) in definition.params.iter().enumerate() {
        for (right, b) in definition.params.iter().enumerate().skip(left + 1) {
            let tensors =
                matches!(params[left].1, Ty::Tensor(_)) && matches!(params[right].1, Ty::Tensor(_));
            if tensors && (a.mode != Mode::In || b.mode != Mode::In) {
                let exact_allowed = definition
                    .aliases
                    .iter()
                    .any(|&(x, y)| (x, y) == (left, right) || (y, x) == (left, right));
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

    let ir = LoweredIr {
        name: family.entry.clone(),
        backend: family.target.clone(),
        ownership: Default::default(),
        alias_requirements,
        params,
        index_params,
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
    use crate::exec::ir::{Builtin, ExprKind, Stmt, StmtKind};
    use crate::family::{self, SiteKind, Workload};
    use crate::program::{compile, SourceFile};

    const MATMUL: &str = "\
fn matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout acc: tile[M, N] f32):
    for i, j in owned(acc):
        let mut s = acc[i, j]
        for k in axis(a, 1):
            s = fma(f32(a[i, k]), f32(b[j, k]), s)
        acc[i, j] = s
";

    const LINEAR: &str = "\
fn linear[M, N, K](x: tensor[M, K] T, weight: tensor[N, K] U, out y: tensor[M, N] V):
    parallel [rows, cols] in (0..M, 0..N):
        let mut acc = zeros_like(y[rows, cols], dtype=f32)
        ordered [k] in 0..K:
            matmul(load(x[rows, k]), load(weight[cols, k]), into=acc)
        publish acc to y[rows, cols]
";

    const RMS_NORM: &str = "\
fn rms_norm[R, W](x: tensor[R, W] T, weight: tensor[W] U, out y: tensor[R, W] V, eps: f32):
    parallel [rows] in 0..R:
        let w = f32(weight)
        for row in rows:
            let t = f32(x[row])
            let ss = reduce(t * t, 0, sum)
            publish t * rsqrt(ss / f32(W) + eps) * w to y[row]
";

    const SUM_SQUARES: &str = "\
admit fn sum_squares[R, W](x: tensor[R, W] T, out y: tensor[R, 1] f32):
    parallel [rows] in 0..R:
        for row in rows:
            let total = parallel [part] in 0..W:
                let v = f32(x[row, part])
                yield reduce(v * v, 0, sum)
            merge (left, right) identity f32(0.0):
                yield left + right
            let mut t = zeros_like(y[row], dtype=f32)
            for i in owned(t):
                t[i] = total
            publish t to y[row]
";

    fn workload(shapes: &[(&str, i64)], elems: &[&str]) -> Workload {
        Workload {
            shapes: shapes.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            elems: elems
                .iter()
                .map(|k| (k.to_string(), crate::types::Elem::Dtype(DType::F32)))
                .collect(),
            ..Workload::default()
        }
    }

    /// The witness written out by hand as a rule: the first candidate of every active
    /// occurrence, singleton covers, and the given numbers for the sites in family order.
    fn build(
        sources: &[&str],
        entry: &str,
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
        let family = family::construct(&program, entry, "cpu", workload).expect("family");
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

    #[test]
    fn linear_is_one_launch_over_the_selected_pieces() {
        let (family, ir) = build(
            &[MATMUL, LINEAR],
            "linear",
            &workload(&[("M", 2), ("N", 8), ("K", 128)], &["T", "U", "V"]),
            &[1, 4, 64],
        );
        assert!(family
            .sites
            .iter()
            .all(|s| matches!(s.kind, SiteKind::Width { .. })));
        let [Stmt {
            kind: StmtKind::Parallel { extents, body, .. },
            ..
        }] = ir.body.as_slice()
        else {
            panic!("{:#?}", ir.body)
        };
        assert_eq!(
            extents.iter().map(|e| e.as_constant()).collect::<Vec<_>>(),
            vec![Some(2), Some(2)]
        );
        // The ordered K traversal is a loop over 128 / 64 windows inside each owner.
        assert!(any(
            body,
            &|s| matches!(&s.kind, StmtKind::Range { hi, .. } if hi.as_constant() == Some(2))
        ));
        assert!(any(
            body,
            &|s| matches!(&s.kind, StmtKind::Expr(e) if matches!(e.kind, ExprKind::Builtin { name: Builtin::Store, .. }))
        ));
        assert_eq!(ir.alias_requirements.len(), 2);
    }

    #[test]
    fn rms_norm_reduces_and_publishes_each_row_inside_its_owner() {
        let (_, ir) = build(
            &[RMS_NORM],
            "rms_norm",
            &workload(&[("R", 6), ("W", 16)], &["T", "U", "V"]),
            &[3],
        );
        let [Stmt {
            kind: StmtKind::Parallel { extents, body, .. },
            ..
        }] = ir.body.as_slice()
        else {
            panic!("{:#?}", ir.body)
        };
        assert_eq!(extents[0].as_constant(), Some(2));
        assert!(any(
            body,
            &|s| matches!(&s.kind, StmtKind::Assign { value, .. } if matches!(value.kind, ExprKind::Builtin { name: Builtin::Reduce, .. }))
        ));
        assert!(any(
            body,
            &|s| matches!(&s.kind, StmtKind::Assign { value, .. } if matches!(value.kind, ExprKind::Load { .. }))
        ));
        assert_eq!(ir.params.len(), 4);
    }

    #[test]
    fn nested_merge_is_the_adjacent_pair_recurrence_over_the_selected_parts() {
        // Sites in family order: rows width 1, merge partition count 4.
        let (_, ir) = build(
            &[SUM_SQUARES],
            "sum_squares",
            &workload(&[("R", 2), ("W", 12)], &["T"]),
            &[1, 4],
        );
        let [Stmt {
            kind: StmtKind::Parallel { body, .. },
            ..
        }] = ir.body.as_slice()
        else {
            panic!("{:#?}", ir.body)
        };
        // Four partials: a level of two pairs, then a level of one pair.
        for pairs in [2, 1] {
            assert!(any(
                body,
                &|s| matches!(&s.kind, StmtKind::Range { hi, body, .. } if hi.as_constant() == Some(pairs) && !body.is_empty())
            ));
        }
        assert!(any(
            body,
            &|s| matches!(&s.kind, StmtKind::Assign { value, .. } if matches!(&value.kind, ExprKind::TileAlloc { shape, .. } if shape[0].as_constant() == Some(4)))
        ));
    }
}
