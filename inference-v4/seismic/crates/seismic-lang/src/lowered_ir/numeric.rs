//! Exact ordinal-to-value definitions of numeric lowering choices. Concrete
//! replay and mathematical export consume these same arithmetic runs.
use super::{Alternative, Alternatives, FoldWindows};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericKind {
    OutputWidth,
    PacketWidth,
    ReductionCut,
    ReductionFrontier,
    ReductionSegment,
    UnrollWidth,
    MatrixPanelWidth,
    StreamCapacity,
    PreparationWindow,
}
impl NumericKind {
    fn of(alternative: &Alternative) -> Option<(Self, i64)> {
        Some(match *alternative {
            Alternative::OutputWidth(value) => (Self::OutputWidth, value),
            Alternative::PacketWidth(value) => (Self::PacketWidth, value),
            Alternative::ReductionCut(value) => (Self::ReductionCut, value),
            Alternative::ReductionFrontier(value) => (Self::ReductionFrontier, value),
            Alternative::ReductionSegment(value) => (Self::ReductionSegment, value),
            Alternative::UnrollWidth(value) => (Self::UnrollWidth, value),
            Alternative::MatrixPanelWidth(value) => (Self::MatrixPanelWidth, value),
            Alternative::StreamCapacity(value) => (Self::StreamCapacity, value),
            Alternative::PreparationWindow(value) => (Self::PreparationWindow, value),
            _ => return None,
        })
    }
    fn alternative(self, value: i64) -> Alternative {
        match self {
            Self::OutputWidth => Alternative::OutputWidth(value),
            Self::PacketWidth => Alternative::PacketWidth(value),
            Self::ReductionCut => Alternative::ReductionCut(value),
            Self::ReductionFrontier => Alternative::ReductionFrontier(value),
            Self::ReductionSegment => Alternative::ReductionSegment(value),
            Self::UnrollWidth => Alternative::UnrollWidth(value),
            Self::MatrixPanelWidth => Alternative::MatrixPanelWidth(value),
            Self::StreamCapacity => Alternative::StreamCapacity(value),
            Self::PreparationWindow => Alternative::PreparationWindow(value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericRun {
    pub ordinal: usize,
    pub count: usize,
    pub first: i64,
    pub stride: i64,
}
impl NumericRun {
    pub fn get(self, ordinal: usize) -> Option<i64> {
        let index = ordinal.checked_sub(self.ordinal)?;
        if index >= self.count {
            return None;
        }
        i64::try_from(i128::from(self.first) + index as i128 * i128::from(self.stride)).ok()
    }
    pub fn contains(self, value: i64) -> bool {
        if self.count == 0 {
            return false;
        }
        if self.stride == 0 {
            return value == self.first;
        }
        let delta = i128::from(value) - i128::from(self.first);
        let stride = i128::from(self.stride);
        delta % stride == 0 && (0..self.count as i128).contains(&(delta / stride))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NumericChoices<'a> {
    Run { kind: NumericKind, run: NumericRun },
    Windows(&'a FoldWindows),
    Explicit { kind: NumericKind, alternatives: &'a [Alternative] },
}
impl NumericChoices<'_> {
    pub fn kind(self) -> NumericKind {
        match self {
            Self::Run { kind, .. } => kind,
            Self::Windows(_) => NumericKind::PreparationWindow,
            Self::Explicit { kind, .. } => kind,
        }
    }
    /// Runs partition ORIGINAL ordinals. Small aligned windows remain explicit;
    /// the arbitrarily long aligned suffix occupies one arithmetic run.
    pub fn runs(&self) -> impl Iterator<Item = NumericRun> + '_ {
        let (single, small, tail): (Option<NumericRun>, &[i64], Option<NumericRun>) = match *self {
            Self::Run { run, .. } => (Some(run), &[], None),
            Self::Explicit { .. } => (None, &[], None),
            Self::Windows(windows) => (
                None,
                windows.small(),
                (windows.multiples > 0).then_some(NumericRun {
                    ordinal: windows.small.len(),
                    count: windows.multiples,
                    first: windows.first,
                    stride: windows.stride,
                }),
            ),
        };
        single
            .into_iter()
            .chain(
                small
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &first)| NumericRun {
                        ordinal,
                        count: 1,
                        first,
                        stride: 0,
                    }),
            )
            .chain(tail)
            .chain(match *self { Self::Explicit { alternatives, .. } => alternatives, _ => &[] }
                .iter().enumerate().map(|(ordinal, alternative)| NumericRun {
                    ordinal, count: 1, first: NumericKind::of(alternative).expect("validated numeric domain").1, stride: 0,
                }))
    }
    pub fn value(&self, ordinal: usize) -> Option<i64> {
        self.runs().find_map(|run| run.get(ordinal))
    }
    pub fn get(self, ordinal: usize) -> Option<Alternative> {
        self.value(ordinal)
            .map(|value| self.kind().alternative(value))
    }
    pub fn contains_value(&self, value: i64) -> bool {
        self.runs().any(|run| run.contains(value))
    }
}
impl Alternatives {
    pub fn numeric(&self) -> Option<NumericChoices<'_>> {
        let (kind, first, stride, count) = match self {
            Self::Explicit(alternatives) => {
                let (kind, _) = NumericKind::of(alternatives.first()?)?;
                if alternatives.iter().all(|alternative| NumericKind::of(alternative).is_some_and(|(candidate, _)| candidate == kind)) {
                    return Some(NumericChoices::Explicit { kind, alternatives });
                }
                return None;
            },
            Self::FoldWindows(windows) => return Some(NumericChoices::Windows(windows)),
            Self::OutputWidths { maximum } => (NumericKind::OutputWidth, 1, 1, *maximum as usize),
            Self::UnrollWidths { maximum } => (NumericKind::UnrollWidth, 1, 1, *maximum as usize),
            Self::MatrixPanelWidths { maximum } => {
                (NumericKind::MatrixPanelWidth, 1, 1, *maximum as usize)
            }
            Self::PacketWidths { maximum } => {
                (NumericKind::PacketWidth, *maximum, -1, *maximum as usize)
            }
            Self::ReductionSegments { maximum } => (
                NumericKind::ReductionSegment,
                *maximum,
                -1,
                *maximum as usize,
            ),
            Self::ReductionCuts { first, last } => (
                NumericKind::ReductionCut,
                *first,
                1,
                (last - first + 1) as usize,
            ),
            Self::ReductionFrontiers { first, last } => (
                NumericKind::ReductionFrontier,
                *first,
                1,
                (last - first + 1) as usize,
            ),
            Self::StreamCapacities(range) => {
                (NumericKind::StreamCapacity, range.maximum, -1, range.count)
            }
        };
        Some(NumericChoices::Run {
            kind,
            run: NumericRun {
                ordinal: 0,
                first,
                stride,
                count,
            },
        })
    }
}
