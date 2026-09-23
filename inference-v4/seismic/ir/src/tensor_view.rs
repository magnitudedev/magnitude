//! One logical tensor view over an actual backing. The index carrier changes
//! when a view is instantiated into a kernel; its ordered coordinate map does
//! not. Bounds and source failure order remain with the constructing operation.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorView<P, I> {
    pub(crate) place: P,
    pub(crate) extents: Vec<I>,
    pub(crate) steps: Vec<ViewStep<I>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewStep<I> {
    Plane { plane: u32, axis: usize },
    Slice(Vec<SliceAxis<I>>),
    Transpose(Vec<u32>),
    Reshape { from: Vec<I>, to: Vec<I> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceAxis<I> {
    Point(I),
    Range { start: I, end: I },
    Full,
}

/// Exact natural coordinate arithmetic; the instantiating owner selects its
/// expression or physical index carrier.
#[derive(Clone, Copy, Debug)]
pub enum CoordinateOp {
    Add,
    Mul,
    Div,
    Rem,
}

impl<I: Copy> ViewStep<I> {
    /// Invert one dense logical step. Plane projection belongs to its registered
    /// representation resolver and is deliberately not a dense coordinate map.
    pub fn dense_coordinates(
        &self,
        index: &[I],
        zero: I,
        mut binary: impl FnMut(CoordinateOp, I, I) -> I,
    ) -> Option<Vec<I>> {
        Some(match self {
            Self::Plane { .. } => return None,
            Self::Slice(axes) => {
                let mut values = index.iter().copied();
                let output = axes
                    .iter()
                    .map(|axis| match *axis {
                        SliceAxis::Point(value) => value,
                        SliceAxis::Range { start, .. } => binary(
                            CoordinateOp::Add,
                            start,
                            values.next().expect("slice mapping rank is closed"),
                        ),
                        SliceAxis::Full => values.next().expect("slice mapping rank is closed"),
                    })
                    .collect();
                assert!(values.next().is_none(), "slice mapping left an output axis");
                output
            }
            Self::Transpose(permutation) => {
                assert_eq!(permutation.len(), index.len());
                let mut output = vec![zero; permutation.len()];
                for (axis, source) in permutation.iter().zip(index) {
                    output[*axis as usize] = *source;
                }
                output
            }
            Self::Reshape { from, to } => {
                assert_eq!(to.len(), index.len());
                let mut linear = zero;
                for (coordinate, extent) in index.iter().zip(to) {
                    linear = binary(CoordinateOp::Mul, linear, *extent);
                    linear = binary(CoordinateOp::Add, linear, *coordinate);
                }
                let mut output = vec![zero; from.len()];
                for axis in (0..from.len()).rev() {
                    output[axis] = binary(CoordinateOp::Rem, linear, from[axis]);
                    linear = binary(CoordinateOp::Div, linear, from[axis]);
                }
                output
            }
        })
    }
}

impl<P, I: Copy> TensorView<P, I> {
    pub fn new(place: P, extents: Vec<I>) -> Self {
        Self {
            place,
            extents,
            steps: Vec::new(),
        }
    }

    pub fn direct_backing(&self) -> Option<&P> {
        self.steps.is_empty().then_some(&self.place)
    }

    pub fn backing(&self) -> &P {
        &self.place
    }
    pub fn extents(&self) -> &[I] {
        &self.extents
    }
    pub fn steps(&self) -> &[ViewStep<I>] {
        &self.steps
    }

    pub(crate) fn has_plane(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, ViewStep::Plane { .. }))
    }

    pub fn slice(mut self, axes: Vec<SliceAxis<I>>, mut subtract: impl FnMut(I, I) -> I) -> Self {
        assert_eq!(axes.len(), self.extents.len(), "slice rank mismatch");
        let extents = axes
            .iter()
            .zip(&self.extents)
            .filter_map(|(axis, extent)| match *axis {
                SliceAxis::Point(_) => None,
                SliceAxis::Range { start, end } => Some(subtract(end, start)),
                SliceAxis::Full => Some(*extent),
            })
            .collect();
        self.steps.push(ViewStep::Slice(axes));
        self.extents = extents;
        self
    }

    pub fn transpose(mut self, permutation: Vec<u32>) -> Self {
        assert_eq!(
            permutation.len(),
            self.extents.len(),
            "transpose rank mismatch"
        );
        let mut seen = vec![false; permutation.len()];
        let extents = permutation
            .iter()
            .map(|axis| {
                let axis = *axis as usize;
                assert!(
                    axis < seen.len() && !seen[axis],
                    "invalid transpose permutation"
                );
                seen[axis] = true;
                self.extents[axis]
            })
            .collect();
        self.steps.push(ViewStep::Transpose(permutation));
        self.extents = extents;
        self
    }

    pub fn reshape(mut self, extents: Vec<I>) -> Self {
        self.steps.push(ViewStep::Reshape {
            from: self.extents.clone(),
            to: extents.clone(),
        });
        self.extents = extents;
        self
    }

    pub fn dense_coordinates(
        &self,
        index: &[I],
        zero: I,
        mut binary: impl FnMut(CoordinateOp, I, I) -> I,
    ) -> Option<Vec<I>> {
        assert_eq!(
            index.len(),
            self.extents.len(),
            "logical tensor index rank mismatch"
        );
        let mut index = index.to_vec();
        for step in self.steps.iter().rev() {
            index = step.dense_coordinates(&index, zero, &mut binary)?;
        }
        Some(index)
    }

    /// Visit actual operand fields, preserving the order used by structured
    /// result transport. The backing is a separate typed operand, not an index.
    pub fn scalar_fields(&self) -> Vec<I> {
        let mut fields = Vec::new();
        self.map(
            |_| (),
            |value| {
                fields.push(*value);
                *value
            },
        );
        fields
    }

    /// Substitute the actual backing and every coordinate operand together.
    /// This preserves the map through calls, kernel instantiation and joins.
    pub fn map<Q, J>(
        &self,
        mut backing: impl FnMut(&P) -> Q,
        mut index: impl FnMut(&I) -> J,
    ) -> TensorView<Q, J> {
        let place = backing(&self.place);
        let extents = self.extents.iter().map(&mut index).collect();
        let steps = self
            .steps
            .iter()
            .map(|step| match step {
                ViewStep::Plane { plane, axis } => ViewStep::Plane {
                    plane: *plane,
                    axis: *axis,
                },
                ViewStep::Transpose(permutation) => ViewStep::Transpose(permutation.clone()),
                ViewStep::Slice(axes) => ViewStep::Slice(
                    axes.iter()
                        .map(|axis| match axis {
                            SliceAxis::Point(value) => SliceAxis::Point(index(value)),
                            SliceAxis::Range { start, end } => SliceAxis::Range {
                                start: index(start),
                                end: index(end),
                            },
                            SliceAxis::Full => SliceAxis::Full,
                        })
                        .collect(),
                ),
                ViewStep::Reshape { from, to } => ViewStep::Reshape {
                    from: from.iter().map(&mut index).collect(),
                    to: to.iter().map(&mut index).collect(),
                },
            })
            .collect();
        TensorView {
            place,
            extents,
            steps,
        }
    }
}
