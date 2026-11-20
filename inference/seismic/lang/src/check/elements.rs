//! Element-parameter uses accumulated while checking one body (L11, L22).
//! A definition's uses become its `ElementDomain`, the only owner of which
//! representations each element parameter may be bound to.

use crate::checked::{
    ElementConversion, ElementDomain, ElementParameter, ElementPredicate, ElementTarget,
    ElementTerm, ElementUses,
};
use crate::types::Elem;
use std::collections::BTreeMap;

/// The uses of every element parameter of the body being checked.
#[derive(Clone, Debug, Default)]
pub(crate) struct ElementUseSet {
    uses: BTreeMap<String, ElementUses>,
    conversions: Vec<ElementConversion>,
    predicates: Vec<ElementPredicate>,
}

impl ElementUseSet {
    fn entry(&mut self, parameter: &str) -> &mut ElementUses {
        self.uses.entry(parameter.to_owned()).or_default()
    }

    /// A read of an element of a `T` tensor: its decoded value, as `f32`.
    pub(crate) fn decoded_read(&mut self, element: &Elem) {
        if let Elem::Param(parameter) = element {
            self.entry(parameter).decoded_read = true;
        }
    }

    pub(crate) fn component_read(
        &mut self,
        element: &Elem,
        component: crate::registry::QuantizedComponent,
    ) {
        if let Elem::Param(parameter) = element {
            let uses = self.entry(parameter);
            uses.component_read = true;
            uses.integer_code_read |= component.requires_integer_code();
        }
    }

    /// A scalar/elementwise write, `tensor[...]` allocation, or writable
    /// intrinsic operand of `T` storage (requires `ReadWrite` dense float).
    pub(crate) fn stored(&mut self, element: &Elem) {
        if let Elem::Param(parameter) = element {
            self.entry(parameter).stored = true;
        }
    }

    /// A `&mut` parameter, a result tensor leaf, or a store of whole trailing
    /// matrices of `T` storage (`registry::record_writable`).
    pub(crate) fn record_stored(&mut self, element: &Elem) {
        if let Elem::Param(parameter) = element {
            self.entry(parameter).record_stored = true;
        }
    }

    /// `to_owned` of a `T` view selecting part of its packing axis (L22).
    pub(crate) fn partial_copy(&mut self, element: &Elem) {
        if let Elem::Param(parameter) = element {
            self.entry(parameter).partial_copy = true;
        }
    }

    /// `repack[U = target](t)` with `t: tensor[..] source`.
    pub(crate) fn conversion(&mut self, source: &Elem, target: ElementTarget) {
        if let Elem::Param(parameter) = source {
            self.entry(parameter).conversion_source = true;
            let conversion = ElementConversion {
                source: parameter.clone(),
                target,
            };
            if !self.conversions.contains(&conversion) {
                self.conversions.push(conversion);
            }
        }
    }

    /// The uses a call imposes on the caller: every caller element parameter
    /// passed as a callee element argument takes the callee contract's uses of
    /// that argument, and the callee's conversions and element predicates are
    /// re-expressed over the caller's elements.
    pub(crate) fn call(&mut self, callee: &ElementDomain, arguments: &[(String, Elem)]) {
        let argument = |name: &str| {
            arguments
                .iter()
                .find(|(parameter, _)| parameter == name)
                .map(|(_, element)| element)
        };
        for parameter in callee.parameters() {
            let Some(Elem::Param(own)) = argument(&parameter.name) else {
                continue;
            };
            self.entry(own).merge(parameter.uses);
        }
        for conversion in callee.conversions() {
            let Some(source) = argument(&conversion.source) else {
                continue;
            };
            let target = match &conversion.target {
                ElementTarget::Concrete(representation) => ElementTarget::Concrete(*representation),
                ElementTarget::Parameter(name) => match argument(name) {
                    Some(Elem::Param(own)) => ElementTarget::Parameter(own.clone()),
                    Some(Elem::Repr(representation)) => ElementTarget::Concrete(*representation),
                    Some(Elem::Dtype(dtype)) => {
                        ElementTarget::Concrete(crate::registry::dense(*dtype))
                    }
                    None => continue,
                },
            };
            self.conversion(source, target);
        }
        // A callee element bound to a representation contributes its constant
        // recipe; one with no argument here constrains nothing of the caller's.
        let term = |term: &ElementTerm| match term {
            ElementTerm::Constant(value) => Some(ElementTerm::Constant(*value)),
            ElementTerm::Operands(name) => match argument(name)? {
                Elem::Param(own) => Some(ElementTerm::Operands(own.clone())),
                Elem::Repr(representation) => Some(ElementTerm::Constant(
                    crate::registry::representation_info(*representation)
                        .operands
                        .code(),
                )),
                Elem::Dtype(dtype) => Some(ElementTerm::Constant(
                    crate::registry::representation_info(crate::registry::dense(*dtype))
                        .operands
                        .code(),
                )),
            },
        };
        for predicate in callee.predicates() {
            let (Some(left), Some(right)) = (term(&predicate.left), term(&predicate.right)) else {
                continue;
            };
            self.predicate(ElementPredicate {
                equal: predicate.equal,
                left,
                right,
            });
        }
    }

    /// The element domain of a definition with these element parameters and
    /// its own `where` element predicates.
    pub(crate) fn domain(
        mut self,
        parameters: &[String],
        predicates: &[ElementPredicate],
    ) -> ElementDomain {
        for predicate in predicates {
            self.predicate(predicate.clone());
        }
        ElementDomain::new(
            parameters
                .iter()
                .map(|name| ElementParameter {
                    name: name.clone(),
                    uses: self.uses.get(name).copied().unwrap_or_default(),
                })
                .collect(),
            self.conversions,
            self.predicates,
        )
    }

    fn predicate(&mut self, predicate: ElementPredicate) {
        if !self.predicates.contains(&predicate) {
            self.predicates.push(predicate);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::checked::{check_source, SourceFile, SourceSet};

    /// A caller's element passed to a callee element takes every one of the
    /// callee's uses of it.
    #[test]
    fn a_call_imposes_every_callee_use() {
        use super::ElementUseSet;
        use crate::types::Elem;
        let mut callee = ElementUseSet::default();
        let own = Elem::Param("U".into());
        callee.decoded_read(&own);
        callee.component_read(&own, crate::registry::QuantizedComponent::Code);
        callee.stored(&own);
        callee.record_stored(&own);
        callee.partial_copy(&own);
        callee.conversion(&own, crate::checked::ElementTarget::Parameter("U".into()));
        let callee = callee.domain(&["U".into()], &[]);
        let mut caller = ElementUseSet::default();
        caller.call(&callee, &[("U".into(), Elem::Param("T".into()))]);
        let caller = caller.domain(&["T".into()], &[]);
        assert_eq!(caller.parameters()[0].uses, callee.parameters()[0].uses);
    }

    #[test]
    fn an_entry_admits_only_bindings_its_element_predicates_hold() {
        use crate::entry::ElementBindings;
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "shared.seismic".into(),
            text: "fn probe[N](q: &tensor[16 * N] Q, k: &tensor[16 * N] K) -> tensor[16 * N] f32 where operands(Q) == operands(K):\n    let mut out = tensor[16 * N] f32\n    parallel for i in 0..16 * N:\n        out[i] = q[i] + k[i]\n    return out\n".into(),
        }]))
        .unwrap();
        let id = module.entry_named("probe").unwrap();
        let w4a4 = crate::registry::representation("nvfp4g16a4").unwrap();
        let w4a16 = crate::registry::representation("nvfp4g16").unwrap();
        let bind = |q, k| ElementBindings::new().bind("Q", q).bind("K", k);
        assert!(module.entry(id, &bind(w4a4, w4a4)).is_ok());
        assert!(module.entry(id, &bind(w4a16, w4a16)).is_ok());
        let refused = module.entry(id, &bind(w4a4, w4a16)).err().unwrap().to_string();
        assert!(refused.contains("where operands(Q) == operands(K)"), "{refused}");
    }

    #[test]
    fn repack_result_entry_admits_packed_resident_targets_and_rejects_external_or_unregistered() {
        use crate::checked::ElementBindingError;
        use crate::entry::ElementBindings;
        use crate::registry;
        use crate::types::DType;

        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "repack.seismic".into(),
            text: "fn repack_weight[B, N, K](source: &tensor[B, N, K] E) -> tensor[B, N, K] U:\n    return repack[U = U](source)\n".into(),
        }]))
        .unwrap();
        let id = module.entry_named("repack_weight").unwrap();
        let domain = &module.entries()[id.index()].element_domain;

        let u_uses = domain
            .parameters()
            .iter()
            .find(|p| p.name == "U")
            .unwrap()
            .uses;
        assert!(u_uses.record_stored);
        assert!(!u_uses.stored);

        let gguf_q4_k = registry::representation("gguf_q4_k").unwrap();
        let q4k_rows16 = registry::representation("q4k@rows16").unwrap();
        let gguf_q2_k = registry::representation("gguf_q2_k").unwrap();
        let q2k_rows16 = registry::representation("q2k@rows16").unwrap();
        let f32_repr = registry::dense(DType::F32);
        let bind = |e, u| ElementBindings::new().bind("E", e).bind("U", u);

        assert_eq!(domain.admit(&bind(gguf_q4_k, q4k_rows16)), Ok(()));
        assert_eq!(domain.admit(&bind(gguf_q2_k, q2k_rows16)), Ok(()));
        assert_eq!(domain.admit(&bind(f32_repr, f32_repr)), Ok(()));

        assert!(matches!(
            domain.admit(&bind(gguf_q4_k, gguf_q4_k)),
            Err(ElementBindingError::Inadmissible {
                ref parameter,
                representation,
                ..
            }) if parameter == "U" && representation == gguf_q4_k
        ));
        assert_eq!(
            domain.admit(&bind(gguf_q4_k, q2k_rows16)),
            Err(ElementBindingError::NoConversion {
                source: gguf_q4_k,
                target: q2k_rows16,
            })
        );
        assert_eq!(
            domain.admit(&bind(gguf_q4_k, f32_repr)),
            Err(ElementBindingError::NoConversion {
                source: gguf_q4_k,
                target: f32_repr,
            })
        );
    }

    #[test]
    fn allocating_or_element_writing_result_still_requires_stored_dense_float() {
        use crate::checked::ElementBindingError;
        use crate::entry::ElementBindings;
        use crate::registry;
        use crate::types::DType;

        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "stored_results.seismic".into(),
            text: "\
fn alloc_and_write[N](x: &tensor[N] f32) -> tensor[N] U:
    let mut out = tensor[N] U
    parallel for i in 0..N:
        out[i] = x[i]
    return out

fn alloc_records[B, N, K](src: &tensor[N, K] E) -> tensor[B, N, K] U:
    let mut out = tensor[B, N, K] U
    parallel for b in 0..B:
        out[b, :, :] = repack[U = U](src)
    return out

fn element_write[N](x: &tensor[N] f32, dst: &mut tensor[N] U):
    parallel for i in 0..N:
        dst[i] = x[i]
"
            .into(),
        }]))
        .unwrap();

        let gguf_q4_k = registry::representation("gguf_q4_k").unwrap();
        let q4k_rows16 = registry::representation("q4k@rows16").unwrap();
        let f32_repr = registry::dense(DType::F32);
        let bf16_repr = registry::dense(DType::BF16);

        for name in ["alloc_and_write", "element_write"] {
            let id = module.entry_named(name).unwrap();
            let domain = &module.entries()[id.index()].element_domain;
            let u_uses = domain
                .parameters()
                .iter()
                .find(|p| p.name == "U")
                .unwrap()
                .uses;
            assert!(u_uses.stored, "{name} must require `stored`");
            assert_eq!(domain.admit(&ElementBindings::new().bind("U", f32_repr)), Ok(()));
            assert_eq!(domain.admit(&ElementBindings::new().bind("U", bf16_repr)), Ok(()));
            assert!(matches!(
                domain.admit(&ElementBindings::new().bind("U", q4k_rows16)),
                Err(ElementBindingError::Inadmissible {
                    ref parameter,
                    representation,
                    ..
                }) if parameter == "U" && representation == q4k_rows16
            ));
        }

        let alloc_records_id = module.entry_named("alloc_records").unwrap();
        let alloc_records_domain = &module.entries()[alloc_records_id.index()].element_domain;
        let u_uses = alloc_records_domain
            .parameters()
            .iter()
            .find(|p| p.name == "U")
            .unwrap()
            .uses;
        assert!(u_uses.stored, "tensor[...] allocation must require `stored`");
        assert_eq!(
            alloc_records_domain.admit(&ElementBindings::new().bind("E", f32_repr).bind("U", f32_repr)),
            Ok(())
        );
        assert!(matches!(
            alloc_records_domain.admit(&ElementBindings::new().bind("E", gguf_q4_k).bind("U", q4k_rows16)),
            Err(ElementBindingError::Inadmissible {
                ref parameter,
                representation,
                ..
            }) if parameter == "U" && representation == q4k_rows16
        ));
    }

    #[test]
    fn caller_delegating_to_result_only_element_helper_inherits_uses_and_conversions() {
        use crate::checked::{CpuBinding, ElementBindingError};
        use crate::entry::ElementBindings;
        use crate::registry;
        use crate::types::DType;

        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "delegate.seismic".into(),
            text: "\
fn make_dense[N](x: &tensor[N] f32) -> tensor[N] A:
    let mut out = tensor[N] A
    parallel for i in 0..N:
        out[i] = x[i]
    return out

fn call_dense[N](x: &tensor[N] f32) -> tensor[N] A:
    return make_dense(x)

fn repack_inner[B, N, K](source: &tensor[B, N, K] E) -> tensor[B, N, K] U:
    return repack[U = U](source)

fn call_repack[B, N, K](source: &tensor[B, N, K] E) -> tensor[B, N, K] U:
    return repack_inner(source)
"
            .into(),
        }]))
        .unwrap();

        let gguf_q4_k = registry::representation("gguf_q4_k").unwrap();
        let q4k_rows16 = registry::representation("q4k@rows16").unwrap();
        let q2k_rows16 = registry::representation("q2k@rows16").unwrap();
        let f32_repr = registry::dense(DType::F32);

        let dense_id = module.entry_named("call_dense").unwrap();
        let dense_domain = &module.entries()[dense_id.index()].element_domain;
        let a_uses = dense_domain
            .parameters()
            .iter()
            .find(|p| p.name == "A")
            .unwrap()
            .uses;
        assert!(a_uses.stored);
        assert_eq!(dense_domain.cpu_binding("A"), Some(CpuBinding::Dense));
        assert_eq!(
            dense_domain.admit(&ElementBindings::new().bind("A", f32_repr)),
            Ok(())
        );
        assert!(matches!(
            dense_domain.admit(&ElementBindings::new().bind("A", q4k_rows16)),
            Err(ElementBindingError::Inadmissible {
                ref parameter,
                representation,
                ..
            }) if parameter == "A" && representation == q4k_rows16
        ));

        let repack_id = module.entry_named("call_repack").unwrap();
        let repack_domain = &module.entries()[repack_id.index()].element_domain;
        let u_uses = repack_domain
            .parameters()
            .iter()
            .find(|p| p.name == "U")
            .unwrap()
            .uses;
        assert!(u_uses.record_stored);
        assert!(!u_uses.stored);
        assert_eq!(repack_domain.cpu_binding("U"), Some(CpuBinding::Packed));
        assert_eq!(
            repack_domain.admit(&ElementBindings::new().bind("E", gguf_q4_k).bind("U", q4k_rows16)),
            Ok(())
        );
        assert_eq!(
            repack_domain.admit(&ElementBindings::new().bind("E", gguf_q4_k).bind("U", q2k_rows16)),
            Err(ElementBindingError::NoConversion {
                source: gguf_q4_k,
                target: q2k_rows16,
            })
        );
    }
}
