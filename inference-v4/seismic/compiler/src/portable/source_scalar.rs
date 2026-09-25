//! Source operation results and their failure effects share one construction.
use super::*;
use seismic_lang::reference_math::{ReferenceRecipe, ScalarOp};

use seismic_lang::failure::{SourceFailure, SourceFailureCause};

pub(super) type SourceChecks = Vec<(SourceFailure, seismic_lang::span::Span)>;
pub(super) type SourceStatuses = BTreeMap<SourceFailure, (PortableTensor, PortableValue)>;

pub(super) fn operation(primitive: &PrimitiveId) -> Option<ScalarOp> {
    seismic_lang::reference_math::scalar_operation(primitive)
}

/// The checked operation already contains explicit quantity-to-word casts.
/// Select its scalar recipe without introducing another conversion boundary.
pub(super) fn recipe(
    function: &SemanticFunction,
    primitive: &PrimitiveId,
    inputs: &[SemanticValueId],
) -> Option<Arc<ReferenceRecipe>> {
    function.scalar_recipe(primitive, inputs)
}

pub(super) fn checks(
    function: &SemanticFunction,
    node: NodeId,
    primitive: &PrimitiveId,
    inputs: &[SemanticValueId],
) -> SourceChecks {
    let Some(recipe) = recipe(function, primitive, inputs) else {
        return Vec::new();
    };
    let mut checks = Vec::new();
    for &(_, cause) in recipe.failures() {
        let site = SourceFailure::at(function, node, SourceFailureCause::Scalar(cause));
        if !checks.iter().any(|(existing, _)| *existing == site) {
            checks.push((site, function.node(node).span()));
        }
    }
    checks
}

pub(super) fn lower<B: seismic_native_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    function: &SemanticFunction,
    node: NodeId,
    statuses: &SourceStatuses,
    alive: PortableValue,
    primitive: &PrimitiveId,
    args: &[PortableValue],
    output: &SemanticType,
) -> (PortableValue, PortableValue) {
    if let Some(operation) = operation(primitive) {
        if args
            .iter()
            .all(|arg| matches!(arg.ty(), ValueType::Scalar(_) | ValueType::Bool))
            && !matches!(output, SemanticType::Index { .. })
        {
            return kernel.source_scalar(operation, args, alive, |cause| {
                statuses[&SourceFailure::at(function, node, SourceFailureCause::Scalar(cause))]
                    .clone()
            });
        }
    }
    (
        lower_scalar_primitive(kernel, primitive, args, output),
        alive,
    )
}

pub(super) fn finish<B: seismic_native_target::TargetFamily>(
    builder: &mut ImplementationBuilder<'_, B>,
    path: &str,
    checks: SourceChecks,
    statuses: Vec<crate::implementation::PortableSourceStatus>,
) {
    assert_eq!(checks.len(), statuses.len());
    for ((failure, span), status) in checks.into_iter().zip(statuses) {
        builder.portable_finish_source_check(
            status,
            CheckSite {
                failure,
                path: path.to_owned(),
                line: span.start,
            },
        );
    }
}

/// Ordered realization retains the participant's terminal state across every
/// element. A stopped element does not read or evaluate the next element.
pub(super) fn nested<B: seismic_native_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    axes: &[PortableValue],
    axis: usize,
    index: &mut Vec<PortableValue>,
    alive: PortableValue,
    body: &mut dyn FnMut(
        &mut PortableBuilder<'_, B>,
        &[PortableValue],
        PortableValue,
    ) -> PortableValue,
) -> PortableValue {
    if axis == axes.len() {
        return body(kernel, index, alive);
    }
    let zero = kernel.index_constant(0);
    kernel.repeat(
        zero,
        axes[axis],
        vec![alive],
        &[registry::IntrinsicUniformity::Varying],
        |kernel, binder, carry| {
            index.push(binder);
            let alive = nested(kernel, axes, axis + 1, index, carry[0], body);
            index.pop();
            vec![alive]
        },
    )[0]
}
