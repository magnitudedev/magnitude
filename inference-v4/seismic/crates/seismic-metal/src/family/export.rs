//! One retained Metal family, common mathematical model, and checked assignment
//! reconstruction. This module contains no search and no candidate score table.
use super::{
    decomposition,
    implementation::Registry,
};
use crate::{execution::Execution, tuning::Conditions};
use magnitude_solver::{
    FeasibleSolution,
    model::{Cost, LinearTerm, ModelBuilder, ObligationKind},
};
use seismic_accounting::{
    schedule::symbolic::Encoding,
    workload::{DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner::{
    self as compiler,
    family::{Export, Reconstructed, Reconstruction},
};
use std::sync::Arc;

struct Family {
    source: compiler::source::Binding,
    conditions: Conditions,
    decomposition: Option<decomposition::Binding>,
    implementation: Registry,
    schedule: Option<super::grouping::Binding>,
}
pub fn export(
    input: compiler::Input<'_>,
    conditions: Conditions,
    workload: ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Export<Execution>, String> {
    let mut builder = ModelBuilder::new();
    builder.units(format!(
        "{}/{} seconds",
        conditions.hardware.timebase.seconds_numerator,
        conditions.hardware.timebase.seconds_denominator
    ));
    let source = Arc::new(compiler::source::Binding::construct(input, "metal")?.expanded()?);
    let name = format!("metal.{}", source.template().name);
    let mut source = compiler::source::Binding::append(&mut builder, source)?;
    let source_template = super::source::Template::construct(&mut builder, &mut source)?;
    let mut family = Family {
        source,
        conditions,
        decomposition: None,
        implementation: Registry::new(name.clone()),
        schedule: None,
    };
    let prepared_schedule = prepare(&mut builder, &name, &source_template, &mut family, &workload, limits)?;
    if let Some(prepared) = prepared_schedule {
        let mut encoding = Encoding::new(&mut builder, &name, prepared.resources(), prepared.horizon()?).map_err(|e| e.to_string())?;
        encoding.bind_timebase(&family.conditions.hardware.timebase).map_err(|e| e.to_string())?;
        family.schedule = Some(prepared.append(&mut builder, &mut encoding)?);
        let completion = encoding.finish(&mut builder).map_err(|e| e.to_string())?;
        builder.cost(Cost::Linear { constant: 0, terms: vec![LinearTerm::new(completion, 1)] });
    } else {
        builder.cost(Cost::Constant(0));
    }
    Export::new(
        Arc::new(builder.build().map_err(|e| e.to_string())?),
        family,
    )
}
/// Append each compiler-owned definition once. Unsupported construction keeps
/// the original family as a coverage obligation, without retaining a partial
/// narrowing or invoking a selected backend preparation path.
fn prepare(
    builder: &mut ModelBuilder,
    name: &str,
    source: &super::source::Template,
    family: &mut Family,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Option<super::grouping::Prepared>, String> {
    use seismic_accounting::{algebra::Error, workload::DerivationError};
    let mut next = builder.clone();
    let decomposition = match decomposition::Binding::append_retained(&mut next, name, source,
        &family.conditions.form, &family.conditions.capacities) {
        Ok(binding) => binding,
        Err(Error::Unsupported(reason)) => {
            builder.obligation(vec![], ObligationKind::Construction, reason);
            return Ok(None);
        },
        Err(error) => return Err(error.to_string()),
    };
    *builder = next;
    let mut next = builder.clone();
    let work = match decomposition.work_template(&mut next, source, &family.conditions.capacities, limits) {
        Ok(work) => work,
        Err(Error::Unsupported(reason)) => {
            builder.obligation(vec![], ObligationKind::Construction, reason);
            family.decomposition = Some(decomposition);
            return Ok(None);
        },
        Err(error) => return Err(error.to_string()),
    };
    *builder = next;
    let template = family.implementation.complete_work(builder, &work)?;
    let terminal = Arc::new(template.operations.append(builder, &format!("{name}.terminal"))?);
    let mut next = builder.clone();
    let prepared = super::grouping::Prepared::derive(&mut next, name, &template.execution, template.emission, Some(terminal),
        &decomposition.launches(), &family.conditions.hardware, workload, limits, None);
    family.decomposition = Some(decomposition);
    match prepared {
        Ok(prepared) => { *builder = next; Ok(Some(prepared)) },
        Err(DerivationError::Analysis(reason)) => Err(reason),
        Err(error) => {
            builder.obligation(vec![], ObligationKind::Analysis, error.to_string());
            Ok(None)
        },
    }
}
impl Reconstruction<Execution> for Family {
    fn reconstruct(
        &self,
        witness: &FeasibleSolution,
        lower_bound: u64,
    ) -> Result<Reconstructed<Execution>, String> {
        let values = witness.values();
        let (source, mut decisions) = self.source.reconstruct(values)?;
        let decomposition = self
            .decomposition
            .as_ref()
            .ok_or("Metal decomposition is unresolved")?
            .reconstruct(values)
            .map_err(|e| e.to_string())?;
        let (mut execution, implementation) = self.implementation.instantiate(values)?;
        execution.install_mappings(&decomposition.mappings, values)?;
        execution.install_dispatches(&decomposition.dispatches)?;
        execution.install_source(&source, &decomposition.decomposition, values)?;
        let (emitted, objective, terminal) = self.schedule.as_ref().ok_or("Metal accounting is unresolved")?.reconstruct(values, lower_bound, &decomposition.dispatches)?;
        execution.install_terminal(emitted)?;
        decisions.extend(decomposition.decisions);
        decisions.extend(implementation);
        decisions.extend(terminal);
        Ok(Reconstructed {
            execution,
            source,
            objective,
            decisions,
        })
    }
}
