//! Reusable finite model definitions with explicit interfaces.
//!
//! Definitions contain fully inspected immutable models, not recursive callbacks.
//! Instantiation is eager; factor guards preserve the semantics of alternatives.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentInterface {
    pub variables: Vec<VarId>,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fragment {
    model: Model,
    interface: FragmentInterface,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentInstance {
    pub variables: Vec<VarId>,
    pub factors: Vec<FactorId>,
}
impl FragmentInstance {
    pub fn variable(&self, local: VarId) -> Option<VarId> {
        self.variables.get(local.0).copied()
    }
}
impl Fragment {
    pub fn new(model: Model, interface: Vec<VarId>) -> Result<Self> {
        model.validate()?;
        if unique_scope(interface.iter().copied()).len() != interface.len() {
            return Err(Error::InvalidModel(
                "fragment interface variables must be distinct".into(),
            ));
        }
        for variable in &interface {
            if variable.0 >= model.variables.len() {
                return Err(Error::InvalidModel(
                    "fragment interface references an absent variable".into(),
                ));
            }
        }
        Ok(Self {
            model,
            interface: FragmentInterface {
                variables: interface,
            },
        })
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn interface(&self) -> &FragmentInterface {
        &self.interface
    }
    pub fn validate(&self) -> Result<()> {
        Self::new(self.model.clone(), self.interface.variables.clone()).map(|_| ())
    }
}
impl ModelBuilder {
    /// Instantiates the same definition without confusing analysis reuse with
    /// execution sharing: every private variable/factor gets its own identity.
    /// Bindings must be distinct and lie within the corresponding interface
    /// domains. Sharing an event requires explicitly binding that interface.
    pub fn instantiate(
        &mut self,
        name: impl AsRef<str>,
        fragment: &Fragment,
        bindings: &[VarId],
        guards: Vec<Literal>,
    ) -> Result<FragmentInstance> {
        self.instantiate_mode(name.as_ref(), fragment, bindings, guards, false)
    }
    /// Retains the complete mapped body as one inspectable construction region.
    /// Variable allocation is eager; search materializes factor children in
    /// budgeted steps. This never invokes an opaque model-expansion callback.
    pub fn instantiate_lazy(
        &mut self,
        name: impl AsRef<str>,
        fragment: &Fragment,
        bindings: &[VarId],
        guards: Vec<Literal>,
    ) -> Result<FragmentInstance> {
        self.instantiate_mode(name.as_ref(), fragment, bindings, guards, true)
    }
    fn instantiate_mode(
        &mut self,
        name: &str,
        fragment: &Fragment,
        bindings: &[VarId],
        guards: Vec<Literal>,
        lazy: bool,
    ) -> Result<FragmentInstance> {
        fragment.validate()?;
        if fragment.model.units != self.units {
            return Err(Error::InvalidModel(
                "fragment objective units differ from parent".into(),
            ));
        }
        if bindings.len() != fragment.interface.variables.len() {
            return Err(Error::InvalidModel(
                "fragment binding arity differs from its interface".into(),
            ));
        }
        if unique_scope(bindings.iter().copied()).len() != bindings.len() {
            return Err(Error::InvalidModel("fragment interface bindings must be distinct; model sharing explicitly in the definition".into()));
        }
        let mut mapping = vec![None; fragment.model.variables.len()];
        for (local, global) in fragment.interface.variables.iter().zip(bindings) {
            let domain = self
                .variables
                .get(global.0)
                .ok_or_else(|| {
                    Error::InvalidModel("fragment binding references an absent variable".into())
                })?
                .domain
                .clone();
            let admitted = &fragment.model.variables[local.0].domain;
            if domain.intersect(admitted)?.cardinality() != domain.cardinality() {
                return Err(Error::InvalidModel(
                    "fragment binding domain exceeds its declared interface domain".into(),
                ));
            }
            mapping[local.0] = Some(*global);
        }
        for guard in &guards {
            if guard.variable.0 >= self.variables.len() {
                return Err(Error::InvalidModel(
                    "fragment guard references an absent variable".into(),
                ));
            }
        }
        if fragment
            .model
            .variables
            .iter()
            .enumerate()
            .any(|(i, variable)| mapping[i].is_none() && variable.domain.is_empty())
        {
            return Err(Error::InvalidModel(
                "fragment private variable cannot have an empty declared domain".into(),
            ));
        }
        let old_guard_len = self.guards.len();
        self.guards.extend(guards);
        let first_factor = self.factors.len();
        for (index, variable) in fragment.model.variables.iter().enumerate() {
            if mapping[index].is_none() {
                let global = self.local_variable(
                    format!("{}.{}", name, variable.name),
                    variable.domain.clone(),
                )?;
                mapping[index] = Some(global);
            }
        }
        let mapping: Vec<_> = mapping
            .into_iter()
            .map(|v| v.expect("all fragment variables mapped"))
            .collect();
        if lazy {
            let factors = fragment
                .model
                .factors
                .iter()
                .map(|factor| factor.remap(&mapping))
                .collect();
            self.factor(Vec::new(), FactorKind::Fragment { factors });
        } else {
            for factor in &fragment.model.factors {
                let factor = factor.remap(&mapping);
                self.factor(factor.guards, factor.kind);
            }
        }
        self.guards.truncate(old_guard_len);
        Ok(FragmentInstance {
            variables: mapping,
            factors: (first_factor..self.factors.len()).map(FactorId).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn preparation() -> Fragment {
        let mut b = ModelBuilder::new();
        let representation = b.variable("representation", Domain::boolean());
        let local = b.variable("local", Domain::interval(1, 3).unwrap());
        b.constraint(Constraint::LinearLe {
            terms: vec![LinearTerm::new(local, 1)],
            rhs: 2,
        });
        b.cost(Cost::Constant(6));
        Fragment::new(b.build().unwrap(), vec![representation]).unwrap()
    }
    #[test]
    fn occurrences_are_not_shared_execution() {
        let mut b = ModelBuilder::new();
        let representation = b.variable("representation", Domain::boolean());
        let fragment = preparation();
        let a = b
            .instantiate("a", &fragment, &[representation], vec![])
            .unwrap();
        let z = b
            .instantiate("b", &fragment, &[representation], vec![])
            .unwrap();
        assert_ne!(a.variables[1], z.variables[1]);
        b.cost(Cost::Constant(2));
        let m = b.build().unwrap();
        assert_eq!(
            m.validate_assignment(&[0, 1, 1]).unwrap().exact_cost,
            Some(14)
        );
    }
    #[test]
    fn inactive_locals_have_one_value_and_no_body_cost() {
        let mut b = ModelBuilder::new();
        let active = b.variable("active", Domain::boolean());
        let representation = b.variable("representation", Domain::boolean());
        let f = b
            .instantiate(
                "optional",
                &preparation(),
                &[representation],
                vec![Literal::new(active, 1)],
            )
            .unwrap();
        let m = b.build().unwrap();
        let mut values = vec![0, 0, 1];
        assert_eq!(m.validate_assignment(&values).unwrap().exact_cost, Some(0));
        values[f.variables[1].0] = 2;
        assert!(m.validate_assignment(&values).unwrap().infeasible);
        values[active.0] = 1;
        assert_eq!(m.validate_assignment(&values).unwrap().exact_cost, Some(6));
    }
    #[test]
    fn scopes_include_outer_guards() {
        let mut b = ModelBuilder::new();
        let active = b.variable("active", Domain::boolean());
        let representation = b.variable("representation", Domain::boolean());
        let instance = b
            .instantiate(
                "optional",
                &preparation(),
                &[representation],
                vec![Literal::new(active, 1)],
            )
            .unwrap();
        let m = b.build().unwrap();
        for id in instance.factors {
            assert!(m.factors[id.0].scope().contains(&active));
        }
    }
    #[test]
    fn eager_and_lazy_definitions_have_identical_assignment_semantics() {
        fn model(lazy: bool) -> Model {
            let mut b = ModelBuilder::new();
            let active = b.variable("active", Domain::boolean());
            let representation = b.variable("representation", Domain::boolean());
            let guards = vec![Literal::new(active, 1)];
            if lazy {
                b.instantiate_lazy("optional", &preparation(), &[representation], guards)
                    .unwrap();
            } else {
                b.instantiate("optional", &preparation(), &[representation], guards)
                    .unwrap();
            }
            b.build().unwrap()
        }
        let eager = model(false);
        let lazy = model(true);
        for active in 0..=1 {
            for representation in 0..=1 {
                for local in 1..=3 {
                    assert_eq!(
                        eager
                            .validate_assignment(&[active, representation, local])
                            .unwrap(),
                        lazy.validate_assignment(&[active, representation, local])
                            .unwrap()
                    );
                }
            }
        }
        let bundle = lazy
            .factors()
            .iter()
            .find(|f| matches!(f.kind, FactorKind::Fragment { .. }))
            .unwrap();
        assert!(bundle.scope().contains(&VarId(0)));
        assert!(bundle.scope().contains(&VarId(2)));
    }
}
