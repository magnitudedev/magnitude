//! Decide which region-local pure tensor values need no physical storage.
//! This consumes the checked dependency/effect graph. It does not rewrite the
//! invocation domain or infer that unrelated places cannot alias.
use super::*;
pub(super) fn values(function: &SemanticFunction) -> BTreeSet<SemanticValueId> {
    let mut computed = BTreeSet::new();
    let mut regions = vec![function.root()];
    while let Some(region) = regions.pop() {
        for (_, node) in function.nodes(region) {
            let output = match node.view() {
                SemanticNodeView::Elementwise {
                    primitive,
                    inputs,
                    output,
                } if total_elementwise(function, primitive, inputs) => Some(output),
                SemanticNodeView::View {
                    transform, output, ..
                } if !matches!(transform, ViewTransform::Plane { .. }) => Some(output),
                SemanticNodeView::If {
                    then, otherwise, ..
                } => {
                    regions.extend([then, otherwise]);
                    None
                }
                SemanticNodeView::Loop { body, .. } => {
                    regions.push(body);
                    None
                }
                _ => None,
            };
            if let Some(output) = output {
                if matches!(&function.value(output).ty, SemanticType::Tensor(tensor)
                    if matches!(registry::representation_info(tensor.representation).kind, RepresentationKind::Dense(_)))
                {
                    computed.insert(output);
                }
            }
        }
    }
    computed
}

/// Failure is part of executing the definition, even if its result is unused.
/// Only a recipe with no failure continuation may move into a later consumer.
fn total_elementwise(
    function: &SemanticFunction,
    primitive: &PrimitiveId,
    inputs: &[SemanticValueId],
) -> bool {
    match primitive {
        PrimitiveId::Constant(_) | PrimitiveId::Select | PrimitiveId::Decode => true,
        _=>source_scalar::recipe(function,primitive,inputs)
            .is_some_and(|recipe|recipe.failures().is_empty()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::{ElementBindings, NumericalRole};
    fn analyze(source: &str) -> usize {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "streaming.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let program = entry.as_view().program();
        let function = program
            .family(program.root())
            .candidates()
            .iter()
            .find(|c| c.numerical == NumericalRole::Reference)
            .unwrap()
            .function;
        values(program.function(function)).len()
    }
    #[test]
    fn closed_views_and_elementwise_values_stream_into_their_reduction() {
        assert!(analyze("fn probe[N](x: &tensor[N] bf16) -> tensor[1] f32:\n    let a = f32(reshape(x, (N, 1)))\n    return reduce(a * a, 0, sum)\n")>=3);
    }
    #[test]
    fn escaping_total_tensor_remains_a_complete_computed_product() {
        assert_eq!(analyze("fn probe[N](x: &tensor[N] f32) -> (tensor[N] f32, f32):\n    let a = x + x\n    return (a, reduce(a, 0, sum))\n"),1);
    }
    #[test]
    fn potentially_failing_tensor_definition_does_not_move_into_a_consumer() {
        assert_eq!(analyze("fn probe[N](x: &tensor[N] i32) -> i32:\n    let a = x / x\n    return reduce(a, 0, sum)\n"), 0);
    }

    #[test]
    fn snapshot_materialization_is_owned_by_construction() {
        assert_eq!(analyze("fn probe[N](x: &mut tensor[N] f32) -> f32:\n    let a = x + x\n    parallel for i in 0..N:\n        x[i] = 0.0\n    return reduce(a, 0, sum)\n"),1);
    }
}
