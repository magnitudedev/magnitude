//! Lowered IR for one backend, sharing the common typed IR nodes.
//! Expansion decisions are retained with the function. Further execution choices
//! are not yet closed here; this representation is not Tuned IR.
pub mod numeric;
use crate::ir::{Stmt, Var, VarId};
use crate::sym::Sym;
use crate::types::{Elem, Ty};
use std::collections::HashMap;

/// A function after inlining for one backend and one shape binding.
#[derive(Clone, Debug, PartialEq)]
pub struct LoweredIr {
    pub name: String,
    pub backend: String,
    pub ownership: crate::composition::Ownership,
    /// Source parallel binding requirements, retained before output regrouping.
    pub alias_requirements: Vec<AliasRequirement>,
    pub params: Vec<(String, Ty)>,
    pub index_params: Vec<(String, Sym)>,
    pub vars: Vec<Var>,
    pub body: Vec<Stmt>,
    /// Shape parameters and their concrete values.
    pub shapes: HashMap<String, i64>,
    /// Which lowering block was chosen for each call site, for inspection and the tuning cache.
    pub selections: Vec<Selection>,
    /// Validated decisions that produced this expanded program.
    pub decisions: Vec<DecisionRecord>,
}

impl LoweredIr {
    /// Substitute already selected compiler operands without transforming the
    /// computation or changing retained variable and operation identities.
    pub fn specialize_parameters(&mut self, parameters: &HashMap<String, Sym>) {
        specialize_statements(&mut self.body, parameters);
        for variable in &mut self.vars { variable.ty = crate::lower::subst_ty(&variable.ty, parameters); }
        for (_, ty) in &mut self.params { *ty = crate::lower::subst_ty(ty, parameters); }
        for (_, extent) in &mut self.index_params { *extent = crate::lower::subst_sym(extent, parameters, &HashMap::new()); }
    }
}

pub fn specialize_statements(body: &mut [Stmt], parameters: &HashMap<String, Sym>) {
    let atoms = parameters.iter().map(|(name, value)| (crate::sym::Atom::Param(name.clone()), value.clone())).collect::<Vec<_>>();
    for statement in body { crate::composition::remap(statement, &HashMap::new(), &atoms); }
}

/// Parameter ordinals whose storage must be disjoint unless their exact typed
/// per-item regions were proved equal in the source computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasRequirement {
    pub left: usize,
    pub right: usize,
    pub exact_allowed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub construct: String,
    pub shape_args: Vec<i64>,
    pub choice: Choice,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Choice {
    /// Index into the construct's blocks for this backend, in file order.
    Block(usize),
    Portable,
}

/// A legal optimization decision, with no performance ordering implied.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub kind: DecisionKind,
    pub alternatives: Alternatives,
}

/// Numeric execution domains are indexed symbolically. An axis with billions of
/// elements does not require billions of allocations to expose its legal pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Alternatives {
    Explicit(Vec<Alternative>),
    StreamCapacities(StreamCapacities),
    OutputWidths { maximum: i64 },
    PacketWidths { maximum: i64 },
    ReductionCuts { first: i64, last: i64 },
    ReductionFrontiers { first: i64, last: i64 },
    ReductionSegments { maximum: i64 },
    UnrollWidths { maximum: i64 },
    MatrixPanelWidths { maximum: i64 },
    FoldWindows(FoldWindows),
}

/// Every legal preparation window, without enumerating potentially long segments.
/// Packet decoders impose their shared alignment; decoded snapshots alone admit
/// all widths and use the fold's ordinary guarded final window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldWindows {
    small: Vec<i64>,
    first: i64,
    stride: i64,
    multiples: usize,
}
impl FoldWindows {
    pub fn new(segment: i64, groups: &[u32]) -> Result<Self, String> {
        if groups.is_empty() && segment > 0 {
            return Ok(Self { small: Vec::new(), first: 1, stride: 1,
                multiples: usize::try_from(segment).map_err(|_| "snapshot window domain exceeds usize")? });
        }
        if segment <= 0
            || groups
                .iter()
                .any(|&g| g == 0 || (segment % i64::from(g) != 0 && i64::from(g) % segment != 0))
        {
            return Err("packet window domain needs complete aligned segments".into());
        }
        fn gcd(mut a: i64, mut b: i64) -> i64 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        let mut stride = 1i64;
        for &group in groups {
            let group = i64::from(group);
            stride = (stride / gcd(stride, group))
                .checked_mul(group)
                .ok_or("packet window alignment overflow")?;
        }
        let maximum_group = i64::from(*groups.iter().max().unwrap());
        let small: Vec<i64> = (1..=maximum_group.min(segment))
            .filter(|&width| {
                let tail = segment % width;
                groups.iter().all(|&group| {
                    let group = i64::from(group);
                    ((width % group == 0 && segment % group == 0)
                        || (group % width == 0 && segment % width == 0))
                        && (tail == 0
                            || (tail % group == 0 && (segment - tail) % group == 0)
                            || (group % tail == 0
                                && segment % tail == 0
                                && (segment - tail) % tail == 0))
                })
            })
            .collect();
        let first = (maximum_group / stride + 1)
            .checked_mul(stride)
            .ok_or("packet window stride overflow")?;
        let multiples = if first > segment {
            0
        } else {
            usize::try_from((segment - first) / stride + 1)
                .map_err(|_| "packet window domain exceeds usize")?
        };
        small
            .len()
            .checked_add(multiples)
            .ok_or("packet window domain length overflow")?;
        Ok(Self {
            small,
            first,
            stride,
            multiples,
        })
    }
    pub fn small(&self) -> &[i64] {
        &self.small
    }
    pub fn progression(&self) -> (i64, i64, usize) {
        (self.first, self.stride, self.multiples)
    }
    pub fn len(&self) -> usize {
        self.small.len() + self.multiples
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, index: usize) -> Option<i64> {
        numeric::NumericChoices::Windows(self).value(index)
    }
    pub fn contains(&self, width: i64) -> bool {
        self.small.binary_search(&width).is_ok()
            || (width >= self.first
                && (width - self.first) % self.stride == 0
                && (width - self.first) / self.stride < (self.multiples as i64))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCapacities {
    maximum: i64,
    count: usize,
}
impl From<Vec<Alternative>> for Alternatives {
    fn from(values: Vec<Alternative>) -> Self { Self::Explicit(values) }
}
impl Alternatives {
    pub fn output_widths(maximum: i64) -> Result<Self, String> {
        if maximum <= 0 || usize::try_from(maximum).is_err() {
            return Err("invalid independent output extent".into());
        }
        Ok(Self::OutputWidths { maximum })
    }
    pub fn reduction_segments(maximum:i64)->Result<Self,String> {if maximum<=0 || usize::try_from(maximum).is_err(){return Err("invalid fold segment extent".into());} Ok(Self::ReductionSegments{maximum})}
    pub fn reduction_cuts(first: i64, last: i64) -> Result<Self,String> {
        if first < 1 || first > last || usize::try_from(last-first+1).is_err() {return Err("invalid reduction cut domain".into());}
        Ok(Self::ReductionCuts{first,last})
    }
    pub fn reduction_frontiers(first: i64, last: i64) -> Result<Self,String> {
        if first < 2 || first > last || usize::try_from(last-first+1).is_err() {return Err("invalid reduction frontier domain".into());}
        Ok(Self::ReductionFrontiers{first,last})
    }
    pub fn stream_capacities(maximum: i64) -> Result<Self, String> {
        if maximum <= 0 { return Err("stream domain must have a positive capacity".into()); }
        let count = usize::try_from(maximum).map_err(|_| "stream domain cardinality overflow")?;
        Ok(Self::StreamCapacities(StreamCapacities { maximum, count }))
    }
    pub fn len(&self) -> usize {
        match self { Self::FoldWindows(w) => w.len(), Self::OutputWidths { maximum } | Self::PacketWidths { maximum } | Self::UnrollWidths { maximum } | Self::MatrixPanelWidths { maximum } => *maximum as usize, Self::ReductionSegments{maximum}=>*maximum as usize, Self::Explicit(v) => v.len(), Self::StreamCapacities(r) => r.count, Self::ReductionCuts{first,last} | Self::ReductionFrontiers{first,last} => (last-first+1) as usize }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn get(&self, index: usize) -> Option<Alternative> {
        match self {
            Self::Explicit(values) => values.get(index).cloned(),
            _ => self.numeric()?.get(index),
        }
    }
    pub fn contains(&self, alternative: &Alternative) -> bool {
        match (self, alternative) {
            (Self::FoldWindows(w), Alternative::PreparationWindow(n)) => w.contains(*n),
            (Self::Explicit(v), a) => v.contains(a),
            (Self::OutputWidths { maximum }, Alternative::OutputWidth(n)) => (1..=*maximum).contains(n),
            (Self::MatrixPanelWidths { maximum }, Alternative::MatrixPanelWidth(n)) => (1..=*maximum).contains(n),
            (Self::UnrollWidths { maximum }, Alternative::UnrollWidth(n)) => (1..=*maximum).contains(n),
            (Self::PacketWidths { maximum }, Alternative::PacketWidth(n)) => (1..=*maximum).contains(n),
            (Self::ReductionSegments{maximum},Alternative::ReductionSegment(n))=>(1..=*maximum).contains(n),
            (Self::StreamCapacities(r), Alternative::StreamCapacity(n)) => (1..=r.maximum).contains(n),
            (Self::ReductionCuts{first,last}, Alternative::ReductionCut(n)) => (*first..=*last).contains(n),
            (Self::ReductionFrontiers{first,last}, Alternative::ReductionFrontier(n)) => (*first..=*last).contains(n),
            _ => false,
        }
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = Alternative> + ExactSizeIterator + '_ {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
    pub fn capacity_interval(&self) -> Option<(i64, i64)> {
        match self { Self::StreamCapacities(r) => Some((r.maximum, 1)), _ => None }
    }
}

/// One retained call whose independent output dimension participates in a
/// common source-coordinate group. Position distinguishes repeated calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputGroupCall {
    pub position: usize,
    pub construct: String,
    pub parameter: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecisionKind {
    OutputGroup { coordinate: VarId, extent: i64, calls: Vec<OutputGroupCall> },
    OutputRemainders { domains: Vec<Vec<Sym>> },
    GroupEpilogue { outputs: Vec<VarId> },
    MatrixPanel { iteration: VarId, iterations: i64, operands: Vec<VarId> },
    Representation { variable: VarId },
    PacketDecode { variable: VarId, group: i64 },
    ReductionSegments {extent:i64},
    Intermediate { variable: VarId, publication: usize },
    ReductionBranch { start: i64, end: i64, fields: Vec<Ty> },
    ReductionFrontier { merge: usize, leaves: Sym, maximum: i64, fields: Vec<Ty> },
    Reduction { merge: String, extent: Sym, fields: Vec<Ty> },
    ParallelFusion { boundary: usize, other: usize, left_domain: Vec<Sym>, right_domain: Vec<Sym> },
    StreamFusion { first: usize, second: usize, extent: Sym },
    FoldOperand { input: usize, ty: Ty },
    FoldState { fields: Vec<Ty> },
    FoldTraversal { segment: i64, window: i64 },
    FoldPreparation { segment: i64 },
    FoldCoefficients { input: usize, segment: i64, window: i64 },
    FoldWords { input: usize, segment: i64, window: i64 },
    PacketDecoder { variable: VarId, width: i64 },
    ReductionInput { input: usize, variable: VarId, segment: i64 },
    ReductionFusion { first: usize, second: usize, extent: Sym },
    RangeFusion { first: usize, second: usize, lo: Sym, hi: Sym },
    Stream { piece: crate::sym::Atom, extent: Sym, maximum: i64 },
    Construct {
        name: String,
        shape_args: Vec<Sym>,
        element_args: Vec<Elem>,
    },
    Producer {
        variable: VarId,
        name: String,
        ty: Ty,
    },
}

/// Construction of independent output epilogues after retained grouped calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupEpilogue { Unrolled, Serial }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Alternative {
    OutputWidth(i64),
    GroupEpilogue(GroupEpilogue),
    UnrollWidth(i64),
    MatrixPanelWidth(i64),
    PreparationWindow(i64),
    StepOperand(crate::reduction::structured::StepOperand),
    StepState(crate::reduction::structured::StepState),
    CoefficientScope(crate::reduction::structured::PreparationScope),
    WordScope(crate::reduction::structured::PreparationScope),
    PacketDecoder(crate::repr::PacketDecoder),
    /// Preserve the representation-owned compact packet storage.
    Encoded,
    /// Materialize exact decoded F32 values alongside the packet snapshot.
    Decoded,
    /// Decode complete coefficient groups, retaining their words and coefficients
    /// across all codes. Ownership is over groups of the same F32 cache.
    DecodedPackets,
    /// Capture one fold segment in its original packed representation. Ordinary
    /// load and storage choices decide whether that snapshot is materialized.
    SegmentSnapshot,
    Direct,
    InputSnapshot(crate::reduction::structured::PreparationScope),
    /// Consecutive codes assigned to one owner within a coefficient group.
    PacketWidth(i64),
    ReductionSegment(i64),
    ReductionCut(i64),
    ReductionFrontier(i64),
    ReductionTree(crate::reduction::structured::Tree),
    ParallelFusion { shared_axes: usize, refine_consumer: bool },
    RetainLocal,
    Separate,
    Concatenate,
    Fuse,
    StreamCapacity(i64),
    Body(Choice),
    Materialize,
    Recompute,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    pub domain: Decision,
    pub selected: Alternative,
}

#[cfg(test)]
mod window_tests {
    use super::*;
    #[test]
    fn packet_windows_cover_alignments_and_keep_large_progressions_compact() {
        for (segment, groups) in [
            (256, vec![64]),
            (512, vec![32, 256]),
            (192, vec![64]),
            (6, vec![12]),
            (60, vec![6, 10]),
        ] {
            let domain = FoldWindows::new(segment, &groups).unwrap();
            let values = (0..domain.len())
                .map(|i| domain.get(i).unwrap())
                .collect::<Vec<_>>();
            assert!(values.windows(2).all(|p| p[0] < p[1]));
            for width in 1..=segment {
                let expected = groups.iter().all(|&g| {
                    let g = i64::from(g);
                    (0..segment / width).all(|i| {
                        let alignment = if width < g { width } else { g };
                        (width % g == 0 || g % width == 0)
                            && segment % alignment == 0
                            && (i * width) % alignment == 0
                    }) && {
                        let tail = segment % width;
                        tail == 0 || {
                            let alignment = if tail < g { tail } else { g };
                            (tail % g == 0 || g % tail == 0)
                                && segment % alignment == 0
                                && (segment - tail) % alignment == 0
                        }
                    }
                });
                assert_eq!(
                    domain.contains(width),
                    expected,
                    "{segment}/{groups:?}/{width}"
                );
                assert_eq!(values.contains(&width), expected);
            }
        }
        let large = FoldWindows::new(1i64 << 40, &[64, 256]).unwrap();
        assert_eq!(large.small().len(), 9);
        assert_eq!(large.get(large.len() - 1), Some(1i64 << 40));
        assert_eq!(large.get(large.len()), None);
        assert!(large.contains((1i64 << 40) - 256));
        assert!(!large.contains((1i64 << 40) - 1));
    }
}
