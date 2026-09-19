//! Original lowering domains represented as decisions and arithmetic values.
//! This does not finish lowering or supply an execution cost. Family construction
//! must attach each decision's guarded IR consequences before solving it.
use magnitude_solver::model::{Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::{Error, Symbolic, Value};
use seismic_lang::lowered_ir::{Alternative, Decision};

pub struct Choice {
    pub ordinal: VarId,
    /// Width/capacity/cut itself, rather than its arbitrary diagnostic ordinal.
    pub numeric: Option<Value>,
    presence: Option<VarId>,
    source: Decision,
}
impl Choice {
    /// A guarded decision belongs only to its defining topology alternative.
    /// Inactive choices have canonical private values and reconstruct as absent.
    pub fn append(
        builder: &mut ModelBuilder,
        name: &str,
        decision: &Decision,
        presence: Option<VarId>,
    ) -> Result<Self, Error> {
        match presence {
            Some(presence) => builder.when(Literal::new(presence, 1), |builder| {
                Self::build(builder, name, decision, Some(presence))
            }),
            None => Self::build(builder, name, decision, None),
        }
    }
    fn build(
        builder: &mut ModelBuilder,
        name: &str,
        decision: &Decision,
        presence: Option<VarId>,
    ) -> Result<Self, Error> {
        let count = decision.alternatives.len();
        let maximum = count
            .checked_sub(1)
            .ok_or_else(|| Error::Invalid("empty source choice domain".into()))?;
        let maximum = i64::try_from(maximum)
            .map_err(|_| Error::Unsupported("source choice ordinal exceeds i64".into()))?;
        let ordinal = builder
            .local_variable(
                format!("{name}.ordinal"),
                Domain::interval(0, maximum).map_err(invalid)?,
            )
            .map_err(invalid)?;
        let numeric = if let Some(domain) = decision.alternatives.numeric() {
            let runs = domain.runs().collect::<Vec<_>>();
            let mut bounds = Vec::with_capacity(runs.len());
            let mut covered = 0usize;
            for run in &runs {
                if run.ordinal != covered || run.count == 0 {
                    return Err(Error::Invalid(
                        "numeric runs do not partition source ordinals".into(),
                    ));
                }
                covered = covered
                    .checked_add(run.count)
                    .ok_or_else(|| Error::Unsupported("source ordinal count overflow".into()))?;
                let last = run
                    .get(covered - 1)
                    .ok_or_else(|| Error::Unsupported("numeric source value exceeds i64".into()))?;
                bounds.push((run.first.min(last), run.first.max(last)));
            }
            if covered != count {
                return Err(Error::Invalid(
                    "numeric runs omit source alternatives".into(),
                ));
            }
            let lo = bounds
                .iter()
                .map(|b| b.0)
                .min()
                .ok_or_else(|| Error::Invalid("empty numeric source domain".into()))?;
            let hi = bounds.iter().map(|b| b.1).max().unwrap();
            let value = Symbolic::new(builder, name)
                .positive("value", Domain::interval(lo, hi).map_err(invalid)?)?;
            let selectors = if runs.len() > 1 {
                let selectors = (0..runs.len())
                    .map(|index| {
                        builder
                            .local_variable(format!("{name}.run{index}"), Domain::boolean())
                            .map_err(invalid)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                builder.constraint(Constraint::ExactlyOne {
                    variables: selectors.clone(),
                });
                selectors
            } else {
                Vec::new()
            };
            for (index, run) in runs.iter().enumerate() {
                let guards = selectors
                    .get(index)
                    .map(|&v| vec![Literal::new(v, 1)])
                    .unwrap_or_default();
                let end = run.ordinal + run.count - 1;
                builder.guarded_constraint(
                    guards.clone(),
                    Constraint::InDomain {
                        variable: ordinal,
                        domain: Domain::interval(run.ordinal as i64, end as i64)
                            .map_err(invalid)?,
                    },
                );
                let offset = i128::from(run.first) - i128::from(run.stride) * run.ordinal as i128;
                for sign in [1i64, -1] {
                    builder.guarded_constraint(
                        guards.clone(),
                        Constraint::LinearLe {
                            terms: vec![
                                LinearTerm::new(value.id(), sign),
                                LinearTerm::new(ordinal, -sign * run.stride),
                            ],
                            rhs: i128::from(sign) * offset,
                        },
                    );
                }
            }
            Some(value)
        } else {
            None
        };
        Ok(Self {
            ordinal,
            numeric,
            presence,
            source: decision.clone(),
        })
    }
    pub fn source(&self) -> &Decision {
        &self.source
    }
    /// The enclosing family still owns model identity and full assignment
    /// validation. This check establishes original-domain replay correspondence.
    pub fn reconstruct(&self, values: &[i64]) -> Result<Option<(usize, Alternative)>, Error> {
        let read = |id: VarId| {
            values
                .get(id.0)
                .copied()
                .ok_or_else(|| Error::Reconstruction("missing source choice value".into()))
        };
        if let Some(presence) = self.presence {
            match read(presence)? {
                0 => return Ok(None),
                1 => {}
                _ => {
                    return Err(Error::Reconstruction(
                        "source presence is not boolean".into(),
                    ));
                }
            }
        }
        let index = usize::try_from(read(self.ordinal)?)
            .map_err(|_| Error::Reconstruction("negative source choice ordinal".into()))?;
        let selected = self.source.alternatives.get(index).ok_or_else(|| {
            Error::Reconstruction("source choice lies outside its original domain".into())
        })?;
        if let Some(numeric) = self.numeric {
            let expected = self
                .source
                .alternatives
                .numeric()
                .and_then(|d| d.value(index));
            if expected != Some(read(numeric.id())?) {
                return Err(Error::Reconstruction(
                    "source numeric value differs from selected alternative".into(),
                ));
            }
        }
        Ok(Some((index, selected)))
    }
}
fn invalid(error: magnitude_solver::Error) -> Error {
    Error::Invalid(error.to_string())
}
