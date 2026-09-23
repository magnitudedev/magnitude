//! Element-parameter uses accumulated while checking one body (L11, L22).
//! A definition's uses become its `ElementDomain`, the only owner of which
//! representations each element parameter may be bound to.

use crate::checked::{ElementConversion, ElementDomain, ElementParameter, ElementTarget, ElementUses};
use crate::types::Elem;
use std::collections::BTreeMap;

/// The uses of every element parameter of the body being checked.
#[derive(Clone, Debug, Default)]
pub(crate) struct ElementUseSet {
    uses: BTreeMap<String, ElementUses>,
    conversions: Vec<ElementConversion>,
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
    /// that argument, and the callee's conversions are re-expressed over the
    /// caller's elements.
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
    }

    /// The element domain of a definition with these element parameters.
    pub(crate) fn domain(self, parameters: &[String]) -> ElementDomain {
        ElementDomain::new(
            parameters
                .iter()
                .map(|name| ElementParameter {
                    name: name.clone(),
                    uses: self.uses.get(name).copied().unwrap_or_default(),
                })
                .collect(),
            self.conversions,
        )
    }
}
