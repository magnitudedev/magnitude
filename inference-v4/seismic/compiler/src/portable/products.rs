//! Structured transport of complete source products through the existing SSA
//! branch. The tensor producer and storage identities remain the same owners.
use super::*;

impl SegmentBound {
    pub(super) fn scalar_fields(&self) -> Vec<PortableValue> {
        match self {
            Self::Scalar(v) => vec![*v],
            Self::Opaque(v) => vec![v.value()],
            Self::Tensor(t) => t.scalar_fields(),
            Self::Range { start, end } => [start.scalar_fields(), end.scalar_fields()].concat(),
            Self::Tuple(values) => values.iter().flat_map(Self::scalar_fields).collect(),
            Self::Unit => vec![],
        }
    }
    pub(super) fn with_scalar_fields<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        fields: &mut impl Iterator<Item = PortableValue>,
    ) -> Self {
        match self {
            Self::Scalar(_) => Self::Scalar(fields.next().expect("scalar product field")),
            Self::Opaque(v) => Self::Opaque(
                kernel.opaque_with_value(*v, fields.next().expect("opaque product field")),
            ),
            Self::Tensor(t) => Self::Tensor(t.with_scalar_fields(kernel, fields)),
            Self::Range { start, end } => Self::Range {
                start: Box::new(start.with_scalar_fields(kernel, fields)),
                end: Box::new(end.with_scalar_fields(kernel, fields)),
            },
            Self::Tuple(values) => Self::Tuple(
                values
                    .iter()
                    .map(|v| v.with_scalar_fields(kernel, fields))
                    .collect(),
            ),
            Self::Unit => Self::Unit,
        }
    }
}

impl SegmentTensor {
    fn scalar_fields(&self) -> Vec<PortableValue> {
        let mut fields = self.axes.clone();
        match &self.value {
            TensorDefinitionValue::Physical(t) => fields.extend(t.scalar_fields()),
            TensorDefinitionValue::Elementwise { inputs, result, .. } => {
                fields.extend(inputs.iter().flat_map(SegmentBound::scalar_fields));
                fields.extend(result.scalar_fields());
            }
            TensorDefinitionValue::Reduce { input, result, .. } => {
                fields.extend(input.scalar_fields());
                fields.extend(result.scalar_fields());
            }
            TensorDefinitionValue::View { base, transform } => {
                fields.extend(base.scalar_fields());
                if let SegmentViewTransform::Slice(axes) = transform {
                    for axis in axes {
                        match axis {
                            SegmentSliceAxis::Point(v) | SegmentSliceAxis::Range { start: v } => {
                                fields.push(*v)
                            }
                            SegmentSliceAxis::Full => {}
                        }
                    }
                }
            }
            TensorDefinitionValue::Selected {
                condition,
                then,
                otherwise,
            } => {
                fields.push(*condition);
                fields.extend(then.scalar_fields());
                fields.extend(otherwise.scalar_fields());
            }
        }
        fields
    }
    fn with_scalar_fields<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        fields: &mut impl Iterator<Item = PortableValue>,
    ) -> Self {
        let axes = self
            .axes
            .iter()
            .map(|_| fields.next().expect("tensor axis field"))
            .collect();
        let value = match &self.value {
            TensorDefinitionValue::Physical(t) => {
                TensorDefinitionValue::Physical(t.with_scalar_fields(kernel, fields))
            }
            TensorDefinitionValue::Elementwise {
                primitive,
                inputs,
                result,
            } => {
                let inputs = inputs
                    .iter()
                    .map(|v| v.with_scalar_fields(kernel, fields))
                    .collect();
                let result = result.with_scalar_fields(fields);
                TensorDefinitionValue::Elementwise {
                    primitive: primitive.clone(),
                    inputs,
                    result,
                }
            }
            TensorDefinitionValue::Reduce {
                op,
                axis,
                input_dtype,
                input,
                result,
            } => TensorDefinitionValue::Reduce {
                op: *op,
                axis: *axis,
                input_dtype: *input_dtype,
                input: Arc::new(input.with_scalar_fields(kernel, fields)),
                result: result.with_scalar_fields(fields),
            },
            TensorDefinitionValue::View { base, transform } => {
                let base = Arc::new(base.with_scalar_fields(kernel, fields));
                let transform = match transform {
                    SegmentViewTransform::Slice(axes) => SegmentViewTransform::Slice(
                        axes.iter()
                            .map(|axis| match axis {
                                SegmentSliceAxis::Full => SegmentSliceAxis::Full,
                                SegmentSliceAxis::Point(_) => SegmentSliceAxis::Point(
                                    fields.next().expect("slice point field"),
                                ),
                                SegmentSliceAxis::Range { .. } => SegmentSliceAxis::Range {
                                    start: fields.next().expect("slice start field"),
                                },
                            })
                            .collect(),
                    ),
                    _ => transform.clone(),
                };
                TensorDefinitionValue::View { base, transform }
            }
            TensorDefinitionValue::Selected {
                then, otherwise, ..
            } => TensorDefinitionValue::Selected {
                condition: fields.next().expect("selected condition field"),
                then: Arc::new(then.with_scalar_fields(kernel, fields)),
                otherwise: Arc::new(otherwise.with_scalar_fields(kernel, fields)),
            },
        };
        Self { axes, value }
    }
}

pub(super) fn join_fields(
    then: &SegmentBound,
    otherwise: &SegmentBound,
    out: &mut Vec<(Option<PortableValue>, Option<PortableValue>)>,
) {
    match (then, otherwise) {
        (SegmentBound::Scalar(a), SegmentBound::Scalar(b)) => out.push((Some(*a), Some(*b))),
        (SegmentBound::Opaque(a), SegmentBound::Opaque(b)) => {
            assert_eq!(
                (a.capability(), a.name()),
                (b.capability(), b.name()),
                "opaque branch products differ"
            );
            out.push((Some(a.value()), Some(b.value())));
        }
        (SegmentBound::Tensor(a), SegmentBound::Tensor(b)) => {
            assert_eq!(
                a.axes.len(),
                b.axes.len(),
                "tensor branch product ranks differ"
            );
            out.extend(
                a.axes
                    .iter()
                    .zip(&b.axes)
                    .map(|(a, b)| (Some(*a), Some(*b))),
            );
            out.extend(a.scalar_fields().into_iter().map(|v| (Some(v), None)));
            out.extend(b.scalar_fields().into_iter().map(|v| (None, Some(v))));
        }
        (SegmentBound::Range { start: a, end: b }, SegmentBound::Range { start: c, end: d }) => {
            join_fields(a, c, out);
            join_fields(b, d, out);
        }
        (SegmentBound::Tuple(a), SegmentBound::Tuple(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                join_fields(a, b, out);
            }
        }
        (SegmentBound::Unit, SegmentBound::Unit) => {}
        _ => panic!("checked branch product kinds differ"),
    }
}

pub(super) fn joined<B: seismic_native_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    condition: PortableValue,
    then: &SegmentBound,
    otherwise: &SegmentBound,
    fields: &mut impl Iterator<Item = PortableValue>,
) -> SegmentBound {
    match (then, otherwise) {
        (SegmentBound::Scalar(_), SegmentBound::Scalar(_)) => {
            SegmentBound::Scalar(fields.next().expect("joined scalar"))
        }
        (SegmentBound::Opaque(a), SegmentBound::Opaque(_)) => SegmentBound::Opaque(
            kernel.opaque_with_value(*a, fields.next().expect("joined opaque")),
        ),
        (SegmentBound::Tensor(a), SegmentBound::Tensor(b)) => {
            let axes = a
                .axes
                .iter()
                .map(|_| fields.next().expect("joined axis"))
                .collect();
            let then = Arc::new(a.with_scalar_fields(kernel, fields));
            let otherwise = Arc::new(b.with_scalar_fields(kernel, fields));
            SegmentBound::Tensor(SegmentTensor {
                axes,
                value: TensorDefinitionValue::Selected {
                    condition,
                    then,
                    otherwise,
                },
            })
        }
        (SegmentBound::Range { start: a, end: b }, SegmentBound::Range { start: c, end: d }) => {
            SegmentBound::Range {
                start: Box::new(joined(kernel, condition, a, c, fields)),
                end: Box::new(joined(kernel, condition, b, d, fields)),
            }
        }
        (SegmentBound::Tuple(a), SegmentBound::Tuple(b)) => SegmentBound::Tuple(
            a.iter()
                .zip(b)
                .map(|(a, b)| joined(kernel, condition, a, b, fields))
                .collect(),
        ),
        (SegmentBound::Unit, SegmentBound::Unit) => SegmentBound::Unit,
        _ => unreachable!("branch schema already checked"),
    }
}
