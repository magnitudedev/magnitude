//! Source-owned classification of replay dependencies. Content-dependent work
//! and addressing require a checked concrete reference execution before trials.
//! Allocation geometry remains metadata-bound for ordinary resource admission.
use super::ObservationError;
use seismic_lang::entry::{
    LogicalEntryView, NumericalRole, RegionKind, SemanticFunction, SemanticNodeView, SemanticType,
    ValueOrigin,
};
use seismic_lang::expr::SymbolKind;
use seismic_lang::ids::{FamilyId, RegionId, SemanticValueId};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObservationSupport {
    pub content_control: bool,
    pub content_addressing: bool,
    pub content_checks: bool,
}
impl ObservationSupport {
    pub fn depends_on_contents(self) -> bool {
        self.content_control || self.content_addressing || self.content_checks
    }
}
#[derive(Clone, Copy)]
enum UseSite {
    Control,
    Address,
    Check,
    Geometry,
}

type Known = HashMap<SemanticValueId, bool>;

pub fn assess_observation_support(
    entry: LogicalEntryView<'_>,
) -> Result<ObservationSupport, ObservationError> {
    let parameters = entry
        .schema()
        .parameters()
        .iter()
        .map(|parameter| {
            !matches!(
                parameter.kind,
                seismic_lang::entry::ParameterKind::Tensor { .. }
            )
        })
        .collect::<Vec<_>>();
    let mut support = ObservationSupport::default();
    function(entry, entry.program().root(), &parameters, &mut support)?;
    Ok(support)
}

fn known(function: &SemanticFunction, value: SemanticValueId, values: &Known) -> bool {
    if let Some(known) = values.get(&value) {
        return *known;
    }
    match function.value(value).origin {
        ValueOrigin::Node(node) => {
            let node = function.node(node);
            match node.view() {
                SemanticNodeView::Extent { .. } => true,
                SemanticNodeView::Primitive { .. }
                | SemanticNodeView::TuplePack { .. }
                | SemanticNodeView::TupleGet { .. } => node
                    .dependencies()
                    .iter()
                    .all(|value| known(function, *value, values)),
                _ => false,
            }
        }
        _ => false,
    }
}
fn require(
    function: &SemanticFunction,
    value: SemanticValueId,
    values: &Known,
    site: UseSite,
    support: &mut ObservationSupport,
) -> Result<(), ObservationError> {
    if known(function, value, values) {
        return Ok(());
    }
    match site {
        UseSite::Control => support.content_control = true,
        UseSite::Address => support.content_addressing = true,
        UseSite::Check => support.content_checks = true,
        UseSite::Geometry => {
            return Err(ObservationError::Unsupported(format!(
                "generated contents affect allocation or view geometry in {}",
                function.name()
            )))
        }
    }
    Ok(())
}
fn function(
    entry: LogicalEntryView<'_>,
    family: FamilyId,
    parameters: &[bool],
    support: &mut ObservationSupport,
) -> Result<Vec<bool>, ObservationError> {
    let reference = entry
        .program()
        .family(family)
        .candidates()
        .iter()
        .find(|candidate| candidate.numerical == NumericalRole::Reference)
        .expect("checked family has a reference");
    let function = entry.program().function(reference.function);
    let mut values = function
        .parameters()
        .iter()
        .zip(parameters)
        .map(|(parameter, known)| (parameter.value, *known))
        .collect::<Known>();
    region(entry, function, function.root(), &mut values, support)?;
    for (_, value) in function.values() {
        if let SemanticType::Tensor(tensor) = &value.ty {
            for axis in &tensor.axes {
                for symbol in entry.arena().free_symbols((*axis).into()) {
                    if let SymbolKind::RuntimeValue(value) = entry.arena().symbol_kind(symbol) {
                        require(function, value, &values, UseSite::Geometry, support)?;
                    }
                }
            }
        }
    }
    Ok(function
        .results()
        .iter()
        .map(|value| known(function, *value, &values))
        .collect())
}
fn captures(
    function: &SemanticFunction,
    child: RegionId,
    inputs: &[SemanticValueId],
    values: &mut Known,
) {
    for (parameter, input) in function.region(child).parameters().iter().zip(inputs) {
        values.insert(*parameter, known(function, *input, values));
    }
}
fn region(
    entry: LogicalEntryView<'_>,
    function: &SemanticFunction,
    region_id: RegionId,
    values: &mut Known,
    support: &mut ObservationSupport,
) -> Result<(), ObservationError> {
    for (_, node) in function.nodes(region_id) {
        match node.view() {
            SemanticNodeView::If {
                condition,
                captures: inputs,
                outputs,
                then,
                otherwise,
            } => {
                require(function, condition, values, UseSite::Control, support)?;
                captures(function, then, inputs, values);
                captures(function, otherwise, inputs, values);
                region(entry, function, then, values, support)?;
                region(entry, function, otherwise, values, support)?;
                for ((output, a), b) in outputs
                    .iter()
                    .zip(function.region(then).results())
                    .zip(function.region(otherwise).results())
                {
                    values.insert(
                        *output,
                        known(function, *a, values) && known(function, *b, values),
                    );
                }
            }
            SemanticNodeView::Loop {
                start,
                end,
                captures: inputs,
                outputs,
                body,
                carries,
                ..
            } => {
                require(function, start, values, UseSite::Control, support)?;
                require(function, end, values, UseSite::Control, support)?;
                // Carried data may change across iterations. It is never assumed
                // metadata merely because the initial value was a constant.
                for (parameter, input) in function
                    .region(body)
                    .parameters()
                    .iter()
                    .skip(1)
                    .zip(inputs)
                {
                    values.insert(*parameter, known(function, *input, values));
                }
                for carry in carries {
                    values.insert(carry.parameter, false);
                }
                if let RegionKind::LoopBody { binder_value, .. } = function.region(body).kind() {
                    values.insert(*binder_value, true);
                }
                region(entry, function, body, values, support)?;
                for output in outputs {
                    values.insert(*output, false);
                }
            }
            SemanticNodeView::Check { condition, .. } => {
                require(function, condition, values, UseSite::Check, support)?
            }
            SemanticNodeView::ElementRead { indices, .. }
            | SemanticNodeView::ElementWrite { indices, .. } => {
                for index in indices {
                    require(function, *index, values, UseSite::Address, support)?;
                }
            }
            SemanticNodeView::View { .. } => {
                for value in node.dependencies().into_iter().skip(1) {
                    require(function, value, values, UseSite::Geometry, support)?;
                }
            }
            SemanticNodeView::Atomic { .. } => {
                return Err(ObservationError::Unsupported(
                    "content-controlled atomic observation needs an explicit input contract".into(),
                ))
            }
            SemanticNodeView::Call {
                family,
                inputs,
                outputs,
            } => {
                let parameters = inputs
                    .iter()
                    .map(|value| known(function, *value, values))
                    .collect::<Vec<_>>();
                let returned = self::function(entry, family, &parameters, support)?;
                for (output, known) in outputs.iter().zip(returned) {
                    values.insert(*output, known);
                }
            }
            SemanticNodeView::Primitive { .. }
            | SemanticNodeView::Intrinsic { .. }
            | SemanticNodeView::Elementwise { .. }
            | SemanticNodeView::Reduce { .. }
            | SemanticNodeView::Alloc { .. }
            | SemanticNodeView::Fill { .. }
            | SemanticNodeView::Copy { .. }
            | SemanticNodeView::RepresentationConvert { .. }
            | SemanticNodeView::Store { .. }
            | SemanticNodeView::TuplePack { .. }
            | SemanticNodeView::TupleGet { .. }
            | SemanticNodeView::Extent { .. } => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ElementBindings;

    fn assess(source: &str) -> Result<ObservationSupport, ObservationError> {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "observer-support.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        assess_observation_support(entry.as_view())
    }
    #[test]
    fn tensor_arithmetic_is_supported_but_content_control_is_reported() {
        assert!(assess("fn probe[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut y = zeros_like(x)\n    parallel for i in 0..N:\n        y[i] = x[i] * x[i]\n    return y\n").is_ok());
        let result = assess("fn probe(x: &tensor[1] f32) -> f32:\n    let mut y = 0.0\n    if x[0] > 0.0:\n        y = x[0]\n    return y\n");
        assert!(result.unwrap().content_control);
    }
}
