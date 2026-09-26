//! Per-invocation member selection: the selection axes of an entry, invocation
//! buckets, the published selection table and the invocation log that ranking
//! measures from. Selection is plain data; it never admits a member (the domain
//! does) and never measures (ranking does).

use crate::prepared::InvocationContract;
use seismic_lang::entry::ParameterKind;
use seismic_lang::expr::compiled::{CompiledPredicate, InvocationValues};
use seismic_lang::expr::{BigUint, SymbolId, SymbolValue};
use seismic_lang::types::DType;
use std::collections::BTreeMap;

pub const INVOCATION_LOG_BUCKETS: usize = 1024;
pub const MAX_REGIONS: usize = 64;
/// One octave in total, in L1 bucket units.
pub const REGION_RADIUS: u64 = 4;
pub const MAX_DESIGNATED_DEFAULTS: usize = 4;

/// Ordered selection axes of one entry: every Nat dimension symbol, Index value,
/// Range start/end, and I32/U32/Bool scalar symbol of the schema, in schema order
/// (dimensions, then parameters). Floating scalars and strides are not axes.
#[derive(Clone, Debug)]
pub struct SelectionAxes {
    axes: Box<[(SymbolId, AxisSort)]>,
    symbols: Box<[SymbolId]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisSort {
    Natural,
    Signed,
    Boolean,
}

impl SelectionAxes {
    pub fn of(contract: &InvocationContract) -> Self {
        let schema = contract.schema();
        let mut axes = schema
            .dimensions()
            .iter()
            .map(|dimension| (dimension.symbol, AxisSort::Natural))
            .collect::<Vec<_>>();
        let mut symbols = Vec::new();
        for parameter in schema.parameters() {
            match &parameter.kind {
                ParameterKind::Tensor { .. } => {}
                ParameterKind::Scalar { dtype, symbol } => {
                    symbols.push(*symbol);
                    match dtype {
                        DType::I32 => axes.push((*symbol, AxisSort::Signed)),
                        DType::U32 => axes.push((*symbol, AxisSort::Natural)),
                        DType::Bool => axes.push((*symbol, AxisSort::Boolean)),
                        DType::F32 | DType::BF16 | DType::F16 => {}
                    }
                }
                ParameterKind::Index { symbol, .. } => {
                    symbols.push(*symbol);
                    axes.push((*symbol, AxisSort::Natural));
                }
                ParameterKind::Range { start, end, .. } => {
                    symbols.extend([*start, *end]);
                    axes.extend([(*start, AxisSort::Natural), (*end, AxisSort::Natural)]);
                }
            }
        }
        Self {
            axes: axes.into(),
            symbols: symbols.into(),
        }
    }

    /// One coordinate per axis. Panics if an axis is unbound or bound to a value
    /// of another sort: `values` come from a validated invocation (P3).
    pub fn bucket(&self, values: &InvocationValues) -> InvocationBucket {
        InvocationBucket(
            self.axes
                .iter()
                .map(|(symbol, sort)| {
                    let value = values
                        .get(*symbol)
                        .unwrap_or_else(|| panic!("selection axis {symbol:?} is unbound"));
                    match (sort, value) {
                        (AxisSort::Natural, SymbolValue::Nat(value)) => natural_coordinate(&value),
                        (AxisSort::Natural, SymbolValue::U32(value)) => {
                            natural_coordinate(&BigUint::from(value))
                        }
                        (AxisSort::Signed, SymbolValue::I32(value)) => {
                            signed_coordinate(value < 0, &BigUint::from(value.unsigned_abs()))
                        }
                        (AxisSort::Boolean, SymbolValue::Bool(value)) => i64::from(value),
                        (sort, value) => {
                            panic!(
                                "selection axis {symbol:?} of sort {sort:?} is bound to {value:?}"
                            )
                        }
                    }
                })
                .collect(),
        )
    }

    /// Every symbol selection may read that is not known before issue: the
    /// `CallScalar` symbol of every scalar parameter of the schema (axes and
    /// floating scalars alike).
    pub fn reads(&self) -> &[SymbolId] {
        &self.symbols
    }
}

/// Quarter-octave coordinate: v < 16 → v; otherwise with e = bit_length(v) − 1
/// and m = the two bits below the leading bit, 16 + 4·(e − 4) + m.
fn natural_coordinate(value: &BigUint) -> i64 {
    let bits = value.bits();
    if bits <= 4 {
        return i64::try_from(value).expect("a value below 16 is an i64");
    }
    let exponent = bits - 1;
    let mantissa =
        u64::try_from((value >> (exponent - 2)) & BigUint::from(3u8)).expect("two bits are a u64");
    i64::try_from(16 + 4 * (exponent - 4) + mantissa)
        .expect("a bucket coordinate of an addressable value is an i64")
}

/// Negative v → −natural(|v|) − 1, so negative values never share a bucket with
/// non-negative ones.
fn signed_coordinate(negative: bool, magnitude: &BigUint) -> i64 {
    let natural = natural_coordinate(magnitude);
    if negative {
        -natural - 1
    } else {
        natural
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InvocationBucket(Box<[i64]>);

impl InvocationBucket {
    /// L1 distance. Panics for buckets of different axis counts, which belong
    /// to different entries (P3).
    pub fn distance(&self, other: &Self) -> u64 {
        assert_eq!(
            self.0.len(),
            other.0.len(),
            "buckets of different selection axes"
        );
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .sum()
    }
}

/// Position in `PreparedKernel::members`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemberIndex(u32);

impl MemberIndex {
    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("a publication has at most u32::MAX members"))
    }
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// A conjunction of compiled invocation predicates.
#[derive(Clone, Debug)]
pub struct CompiledConjunction {
    terms: Box<[CompiledPredicate]>,
    reads: Box<[SymbolId]>,
}

impl CompiledConjunction {
    /// The empty conjunction (true). Used by L1a's placeholder and by the
    /// authored-native route.
    pub fn always() -> Self {
        Self {
            terms: Box::new([]),
            reads: Box::new([]),
        }
    }

    /// Terms in order, stop at the first false; panics on an evaluation error
    /// (validated values, P3).
    pub fn holds(&self, values: &InvocationValues) -> bool {
        self.terms.iter().all(|term| {
            term.evaluate(values).unwrap_or_else(|error| {
                panic!("validated invocation could not evaluate a conjunct: {error:?}")
            })
        })
    }

    pub fn reads(&self) -> &[SymbolId] {
        &self.reads
    }
}

#[derive(Clone, Debug)]
pub enum MemberApplicability {
    Total,
    Guarded(CompiledConjunction),
}

impl MemberApplicability {
    /// Panics on a guard evaluation error: `values` come from a validated
    /// invocation (P3).
    pub fn holds(&self, values: &InvocationValues) -> bool {
        match self {
            Self::Total => true,
            Self::Guarded(guard) => guard.holds(values),
        }
    }
}

/// The invocation preference of one designated default member.
#[derive(Clone, Debug)]
pub enum DefaultPreference {
    Always,
    When(CompiledConjunction),
}

impl DefaultPreference {
    pub fn holds(&self, values: &InvocationValues) -> bool {
        match self {
            Self::Always => true,
            Self::When(preference) => preference.holds(values),
        }
    }
}

/// The one selection object of a publication. Member 0 is the general member.
#[derive(Clone, Debug)]
pub struct SelectionTable {
    /// Designated defaults in A4 order; empty → member 0 is the default.
    defaults: Box<[(DefaultPreference, MemberIndex)]>,
    /// Measured regions, strictly sorted by bucket, at most `MAX_REGIONS`.
    regions: Box<[(InvocationBucket, MemberIndex)]>,
}

impl SelectionTable {
    /// Panics if `member_count` is zero (member 0 is always the general member),
    /// an index is ≥ `member_count`, the regions are not strictly
    /// sorted by bucket, or a list exceeds its limit (constructor-owned facts).
    pub(crate) fn new(
        member_count: usize,
        defaults: Vec<(DefaultPreference, MemberIndex)>,
        regions: Vec<(InvocationBucket, MemberIndex)>,
    ) -> Self {
        assert!(
            member_count > 0,
            "a publication always has its general member"
        );
        assert!(
            defaults.len() <= MAX_DESIGNATED_DEFAULTS,
            "more designated defaults than a domain designates"
        );
        assert!(regions.len() <= MAX_REGIONS, "more regions than retained");
        assert!(
            defaults
                .iter()
                .map(|(_, member)| member)
                .chain(regions.iter().map(|(_, member)| member))
                .all(|member| member.as_usize() < member_count),
            "selection names a member outside its publication"
        );
        assert!(
            regions.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "selection regions are not strictly sorted by bucket"
        );
        Self {
            defaults: defaults.into(),
            regions: regions.into(),
        }
    }

    /// 1. The nearest region within `REGION_RADIUS` of `bucket` (L1; ties go to
    ///    the lexicographically smaller bucket), if its member applies.
    /// 2. Otherwise the first designated default whose preference and
    ///    applicability hold.
    /// 3. Otherwise member 0, the general member, which is total.
    pub(crate) fn select(
        &self,
        values: &InvocationValues,
        bucket: &InvocationBucket,
        applicable: impl Fn(MemberIndex) -> bool,
    ) -> MemberIndex {
        let mut nearest: Option<(u64, MemberIndex)> = None;
        for (region, member) in self.regions.iter() {
            let distance = region.distance(bucket);
            if distance <= REGION_RADIUS && nearest.is_none_or(|(closest, _)| distance < closest) {
                nearest = Some((distance, *member));
            }
        }
        if let Some((_, member)) = nearest {
            if applicable(member) {
                return member;
            }
        }
        self.defaults
            .iter()
            .find(|(preference, member)| preference.holds(values) && applicable(*member))
            .map_or(MemberIndex(0), |(_, member)| *member)
    }
}

/// The distinct invocation buckets one prepared kernel was bound with.
#[derive(Debug)]
pub struct InvocationLog<S> {
    buckets: BTreeMap<InvocationBucket, LoggedInvocation<S>>,
    overflow: u64,
}

#[derive(Debug)]
pub struct LoggedInvocation<S> {
    pub values: InvocationValues,
    pub sample: S,
    pub calls: u64,
}

impl<S> InvocationLog<S> {
    pub fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
            overflow: 0,
        }
    }

    /// Increments `calls` for an existing bucket. Otherwise, while fewer than
    /// `INVOCATION_LOG_BUCKETS` buckets exist, inserts `(values, sample())`.
    /// Beyond the cap the invocation is only counted in `overflow`.
    pub fn record(
        &mut self,
        bucket: InvocationBucket,
        values: &InvocationValues,
        sample: impl FnOnce() -> S,
    ) {
        if let Some(logged) = self.buckets.get_mut(&bucket) {
            logged.calls += 1;
        } else if self.buckets.len() < INVOCATION_LOG_BUCKETS {
            self.buckets.insert(
                bucket,
                LoggedInvocation {
                    values: values.clone(),
                    sample: sample(),
                    calls: 1,
                },
            );
        } else {
            self.overflow += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Invocations whose bucket was new after the log held
    /// `INVOCATION_LOG_BUCKETS` buckets.
    pub fn overflow(&self) -> u64 {
        self.overflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};

    fn contract(source: &str) -> InvocationContract {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "selection-fixture.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let entry = module
            .entry(module.entry_named("probe").unwrap(), &Default::default())
            .unwrap();
        InvocationContract::compile_entry(&entry)
    }

    fn bucket(coordinates: &[i64]) -> InvocationBucket {
        InvocationBucket(coordinates.into())
    }

    fn natural(value: u64) -> i64 {
        natural_coordinate(&BigUint::from(value))
    }

    #[test]
    fn natural_coordinates_are_exact_below_sixteen_then_quarter_octaves() {
        for value in 0..16 {
            assert_eq!(natural(value), value as i64);
        }
        assert_eq!(natural(16), 16);
        assert_eq!(natural(19), 16);
        assert_eq!(natural(20), 17);
        assert_eq!(natural(31), 19);
        assert_eq!(natural(32), 20);
        assert_eq!(natural(1 << 20), 16 + 4 * 16);
        assert_eq!(natural((1 << 20) + (3 << 18)), 16 + 4 * 16 + 3);
        let coordinates = (0..4096).map(natural).collect::<Vec<_>>();
        assert!(coordinates
            .windows(2)
            .all(|pair| pair[1] - pair[0] <= 1 && pair[1] >= pair[0]));
        let huge = BigUint::from(1u8) << 200u32;
        assert_eq!(natural_coordinate(&huge), 16 + 4 * 196);
    }

    #[test]
    fn signed_coordinates_separate_negative_values() {
        assert_eq!(signed_coordinate(false, &BigUint::from(0u8)), 0);
        assert_eq!(signed_coordinate(true, &BigUint::from(1u8)), -2);
        assert_eq!(
            signed_coordinate(true, &BigUint::from(40u8)),
            -natural(40) - 1
        );
    }

    #[test]
    fn axes_follow_schema_order_and_reads_cover_every_scalar_parameter() {
        let contract = contract(
            "fn probe[N](x: &tensor[N] f32, k: index[N], r: range[N], s: i32, u: u32, b: bool, f: f32) -> f32:\n    return f\n",
        );
        let axes = SelectionAxes::of(&contract);
        let schema = contract.schema();
        let dimension = schema.dimensions()[0].symbol;
        let symbol = |name: &str| match &schema
            .parameters()
            .iter()
            .find(|parameter| parameter.name == name)
            .unwrap()
            .kind
        {
            ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. } => *symbol,
            ParameterKind::Range { .. } | ParameterKind::Tensor { .. } => unreachable!(),
        };
        let ParameterKind::Range { start, end, .. } = schema
            .parameters()
            .iter()
            .find(|parameter| parameter.name == "r")
            .unwrap()
            .kind
        else {
            unreachable!()
        };
        assert_eq!(
            axes.axes
                .iter()
                .map(|(symbol, _)| *symbol)
                .collect::<Vec<_>>(),
            vec![
                dimension,
                symbol("k"),
                start,
                end,
                symbol("s"),
                symbol("u"),
                symbol("b")
            ]
        );
        assert_eq!(
            axes.reads(),
            &[
                symbol("k"),
                start,
                end,
                symbol("s"),
                symbol("u"),
                symbol("b"),
                symbol("f")
            ]
        );

        let mut values = InvocationValues::new();
        values.bind(dimension, SymbolValue::Nat(BigUint::from(100u32)));
        values.bind(symbol("k"), SymbolValue::Nat(BigUint::from(3u32)));
        values.bind(start, SymbolValue::Nat(BigUint::from(0u32)));
        values.bind(end, SymbolValue::Nat(BigUint::from(64u32)));
        values.bind(symbol("s"), SymbolValue::I32(-5));
        values.bind(symbol("u"), SymbolValue::U32(17));
        values.bind(symbol("b"), SymbolValue::Bool(true));
        values.bind(symbol("f"), SymbolValue::F32(1.5));
        assert_eq!(
            axes.bucket(&values),
            bucket(&[natural(100), 3, 0, natural(64), -6, natural(17), 1])
        );
    }

    #[test]
    #[should_panic(expected = "is unbound")]
    fn bucketing_an_unbound_axis_panics() {
        let contract = contract("fn probe(s: i32) -> i32:\n    return s\n");
        SelectionAxes::of(&contract).bucket(&InvocationValues::new());
    }

    #[test]
    fn zero_axis_entries_have_one_empty_bucket() {
        let contract = contract("fn probe(x: f32) -> f32:\n    return x\n");
        let axes = SelectionAxes::of(&contract);
        let mut values = InvocationValues::new();
        values.bind(axes.reads()[0], SymbolValue::F32(2.0));
        assert_eq!(axes.bucket(&values), bucket(&[]));
    }

    #[test]
    fn distance_is_l1() {
        assert_eq!(bucket(&[1, -2, 5]).distance(&bucket(&[4, 2, 5])), 7);
        assert_eq!(bucket(&[]).distance(&bucket(&[])), 0);
    }

    #[test]
    #[should_panic(expected = "outside its publication")]
    fn selection_table_rejects_an_out_of_range_member() {
        SelectionTable::new(2, vec![], vec![(bucket(&[0]), MemberIndex(2))]);
    }

    #[test]
    #[should_panic(expected = "always has its general member")]
    fn selection_table_rejects_an_empty_publication() {
        SelectionTable::new(0, vec![], vec![]);
    }

    #[test]
    #[should_panic(expected = "strictly sorted")]
    fn selection_table_rejects_unsorted_regions() {
        SelectionTable::new(
            2,
            vec![],
            vec![
                (bucket(&[5]), MemberIndex(1)),
                (bucket(&[1]), MemberIndex(0)),
            ],
        );
    }

    #[test]
    fn selection_prefers_the_nearest_applicable_region_then_defaults_then_general() {
        let values = InvocationValues::new();
        let table = SelectionTable::new(
            4,
            vec![
                (DefaultPreference::Always, MemberIndex(3)),
                (DefaultPreference::Always, MemberIndex(2)),
            ],
            vec![
                (bucket(&[10, 0]), MemberIndex(1)),
                (bucket(&[14, 0]), MemberIndex(2)),
                (bucket(&[30, 0]), MemberIndex(1)),
            ],
        );
        let all = |_: MemberIndex| true;
        assert_eq!(
            table.select(&values, &bucket(&[11, 0]), all),
            MemberIndex(1)
        );
        assert_eq!(
            table.select(&values, &bucket(&[13, 0]), all),
            MemberIndex(2)
        );
        // Equidistant from [10, 0] and [14, 0]: the smaller bucket wins.
        assert_eq!(
            table.select(&values, &bucket(&[12, 0]), all),
            MemberIndex(1)
        );
        // Outside every region's radius: the first applicable default.
        assert_eq!(
            table.select(&values, &bucket(&[20, 0]), all),
            MemberIndex(3)
        );
        let not_three = |member: MemberIndex| member != MemberIndex(3);
        assert_eq!(
            table.select(&values, &bucket(&[20, 0]), not_three),
            MemberIndex(2)
        );
        // An inapplicable region member falls to the defaults.
        let not_one = |member: MemberIndex| member != MemberIndex(1);
        assert_eq!(
            table.select(&values, &bucket(&[11, 0]), not_one),
            MemberIndex(3)
        );
        let only_general = |member: MemberIndex| member == MemberIndex(0);
        assert_eq!(
            table.select(&values, &bucket(&[11, 0]), only_general),
            MemberIndex(0)
        );
        let empty = SelectionTable::new(1, vec![], vec![]);
        assert_eq!(
            empty.select(&values, &bucket(&[11, 0]), all),
            MemberIndex(0)
        );
    }

    #[test]
    fn invocation_log_counts_known_buckets_and_caps_new_ones() {
        let values = InvocationValues::new();
        let mut log = InvocationLog::new();
        let mut samples = 0;
        for _ in 0..3 {
            log.record(bucket(&[7]), &values, || {
                samples += 1;
                samples
            });
        }
        assert_eq!(samples, 1, "the sample is taken only for a new bucket");
        assert_eq!(log.buckets[&bucket(&[7])].calls, 3);
        for coordinate in 0..INVOCATION_LOG_BUCKETS as i64 + 5 {
            log.record(bucket(&[100 + coordinate]), &values, || 0);
        }
        assert_eq!(log.len(), INVOCATION_LOG_BUCKETS);
        assert_eq!(log.overflow(), 6);
        log.record(bucket(&[7]), &values, || unreachable!());
        assert_eq!(log.buckets[&bucket(&[7])].calls, 4);
    }
}
