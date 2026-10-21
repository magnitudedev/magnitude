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

    /// A write, allocation, installation, result leaf or `&mut` parameter of
    /// `T` storage.
    pub(crate) fn stored(&mut self, element: &Elem) {
        if let Elem::Param(parameter) = element {
            self.entry(parameter).stored = true;
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
            let uses = self.entry(own);
            uses.decoded_read |= parameter.uses.decoded_read;
            uses.component_read |= parameter.uses.component_read;
            uses.integer_code_read |= parameter.uses.integer_code_read;
            uses.stored |= parameter.uses.stored;
            uses.conversion_source |= parameter.uses.conversion_source;
            uses.partial_copy |= parameter.uses.partial_copy;
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
}
