//! Row ranges: one tuning identity per range of the rows an entry serves.
//!
//! A unit's choice applies at every row count its launches take, which loses
//! wherever forms compete across row counts: a GEMV form against a matrix
//! form, each best at one end. [`Ranged`] splits the unit, and is the only
//! thing that does: its bindings carry the range, its points are the range's
//! rows, and a launch belongs to the unit whose range holds its rows. A case
//! that reads its range to choose its search's starts (attention's mix) is
//! handed it ([`EntryTuning::search_starts`]). The ranges are a partition of
//! the rows the entry serves ([`row_ranges`]), cut where its declaration
//! says the code a row count runs differs and, on a backend where a unit
//! forms no program, between every two rows the case is measured at
//! ([`row_cuts`]), each side holding rows the case is measured at; the
//! graph binds each range's kernel by its node's rows
//! ([`RangedKernels`](crate::native::RangedKernels)), whatever the cuts.
//!
//! A range's search is of what its rows can run and nothing else. Its
//! points are rows of the range, and Seismic's search measures a point once
//! for all the configurations that run the same launches with the same
//! values there (`seismic::LaunchKey`), compares two configurations over
//! the points that tell them apart, and keeps at its default every
//! parameter that only launches no point activates read. So a parameter no
//! launch of the range reads changes nothing its search measures and is no
//! part of its record's key, and a unit cut in two is measured no more than
//! whole: a point's measurements are those of the configurations that
//! differ there, whichever range the point lies in. A configuration is
//! placed, admitted and sampled a first round at every point that tells it
//! from the served choice, in one range or many. What the cut changes is
//! which of them take the further rounds that resolve a lead: whole, a
//! configuration ahead over the points' weighted sum takes them at every
//! point, those it trails at among them, and one behind over the sum takes
//! none, those it leads at among them; cut, a configuration takes them at
//! the points it leads at and nowhere else, which is the work that finds a
//! point's own best. What a range adds is a unit: a canonical program, a
//! record and a kernel of its own. Where a specialization forms a program
//! (`BackendName::specializations_form_programs`), that is why nothing but
//! the declaration cuts. Where it forms none (the CPU), a unit costs no
//! formation and neighbouring ranges of one choice are one kernel
//! (`Specializer::tuned_ranges`), so every measured row count is a range
//! and takes the configuration that is best there: a CPU projection's best
//! tier flips with its rows at row counts that follow its shape and the
//! machine, which no declaration states (a Q4_K 10752x2048 projection on
//! an AMX host, relative to the base tier at 8, 16, 32, 64 and 512 rows:
//! the AMX tier 1.538, 1.312, 1.066, 0.802 and 0.645, the VNNI tier 1.113,
//! 1.072, 0.835, 0.738 and 0.697).

use std::collections::BTreeMap;
use std::ops::Range;

use super::*;
use magnitude_kernels::constants::VERIFY_ROWS;
use seismic::{NativeCondition, NativeImplementation, NativeNatExpr};

/// Every row count: the rows of an entry whose launches take any.
pub(crate) const EVERY_ROW: Range<u64> = 1..u64::MAX;

/// Where `implementation`'s choice can differ by its rows (`row_dim`, e.g.
/// `"M"` or `"O"`) over `served` at the unit's `statics`, between the rows
/// `points` it is measured at, as its declaration states it:
///
/// - between two neighbouring points that can run different launches or
///   size different scratch buffers ([`Declared::reach`]), the first row
///   count past the lower one that differs from it ([`reach_cuts`]). What a
///   row count can run is read over every value the statics admit of the
///   parameters its conditions read: a dense Metal projection's batch
///   launch (`M >= BATCH_FROM and M <= 16`) cannot run below the least
///   `BATCH_FROM` and its row GEMV (`M < BATCH_FROM`) cannot from the
///   greatest on, so both are cuts, while between them the search places
///   the flip itself, within one range;
/// - within the decode rows, each row count at which a launch that runs on
///   both sides of it groups its configurations differently by the grid
///   they dispatch ([`tile_cuts`]): a parameter that tiles the rows
///   (`ceil_div(M, TOKENS)`) cuts past each of its values.
///
/// Where a specialization forms a program
/// (`BackendName::specializations_form_programs`), nothing else cuts. Two
/// neighbouring points that can run the same launches with the same buffers
/// have the same parameters to choose, of the same values (`where` reads no
/// rows), so they are one range's; the graph's row classes are no cut of
/// their own, since a node binds its kernel by its rows under any
/// partition. Where it forms none, two neighbouring points the declaration
/// does not part are parted past the lower one ([`point_cuts`]): each
/// measured row count is a range of its own.
pub(crate) fn row_cuts(
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    row_dim: &str,
    served: &Range<u64>,
    points: &[u64],
) -> Vec<u64> {
    let declared = Declared::of(implementation, statics, row_dim);
    let mut held = points
        .iter()
        .copied()
        .filter(|point| served.contains(point))
        .collect::<Vec<_>>();
    held.sort_unstable();
    held.dedup();
    let mut cuts = reach_cuts(&held, |row| declared.reach(row));
    let tiled = served.start + 1..served.end.min(VERIFY_ROWS + 2);
    for (ordinal, launch) in implementation.launches.iter().enumerate() {
        for axis in &launch.groups {
            let Some((names, assignments)) = declared.tiled(ordinal, axis) else {
                continue;
            };
            cuts.extend(tile_cuts(
                tiled.clone(),
                |row| declared.can_hold(launch.when.as_ref(), Some(ordinal), row),
                |row| tiling(axis, &names, &assignments, row_dim, row, statics),
            ));
        }
    }
    if !implementation.backend.specializations_form_programs() {
        cuts.extend(point_cuts(&held, &cuts));
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

/// Between each two neighbours of `held` (the measured rows, ascending)
/// that none of `cuts` parts, the first row past the lower one: with
/// `cuts`, every measured row count then lies in a range of its own. The
/// rows between two points go with the upper one, the point that times
/// them: a launch is represented at the least measured row count that
/// holds its rows ([`EntryTuning::invocation_rows`]), so the kernel that
/// serves a row count is the one tuned at the point its launches weigh.
/// The rows past the highest point stay with it, and those below the
/// lowest with the lowest.
fn point_cuts(held: &[u64], cuts: &[u64]) -> Vec<u64> {
    held.windows(2)
        .filter(|pair| !cuts.iter().any(|cut| pair[0] < *cut && *cut <= pair[1]))
        .map(|pair| pair[0] + 1)
        .collect()
}

/// Between each two neighbours of `held` (the measured rows, ascending)
/// whose `reach` differs, the first row past the lower one whose reach
/// differs from the lower one's. Neighbours of one reach are not cut,
/// whatever the rows between them run: no point measures those.
fn reach_cuts<R: PartialEq>(held: &[u64], reach: impl Fn(u64) -> R) -> Vec<u64> {
    held.windows(2)
        .filter_map(|pair| {
            let below = reach(pair[0]);
            (reach(pair[1]) != below).then(|| {
                (pair[0] + 1..=pair[1])
                    .find(|&row| reach(row) != below)
                    .expect("the upper point's reach differs")
            })
        })
        .collect()
}

/// The rows of `rows` at which a launch that `runs` at the row and at the
/// one below it groups its configurations differently by the grid axis
/// they dispatch (`tiling`: [`tiling`] of [`Declared::tiled`]). A
/// verification's rows are few, and at each tile value's own row count that
/// value fills its tile where the others leave theirs part empty, so each
/// value is the best somewhere only there: one row cannot be tiled, so
/// every configuration dispatches one grid, and from two rows on the tiles
/// part them. A prefill's rows fill every tile but their last, so `rows`
/// are the decode rows'.
fn tile_cuts(rows: Range<u64>, runs: impl Fn(u64) -> bool, tiling: impl Fn(u64) -> Vec<usize>) -> Vec<u64> {
    rows.filter(|&row| runs(row - 1) && runs(row) && tiling(row - 1) != tiling(row))
        .collect()
}

/// An implementation's declaration at a unit's statics: the values the
/// statics admit of each entry parameter, and with them what a row count
/// can run.
struct Declared<'a> {
    implementation: &'a NativeImplementation,
    statics: &'a NativeSpecialization,
    row_dim: &'a str,
    /// Each entry parameter's admitted values ([`admitted`]), by its name.
    admitted: BTreeMap<&'a str, Vec<u64>>,
}

impl<'a> Declared<'a> {
    fn of(
        implementation: &'a NativeImplementation,
        statics: &'a NativeSpecialization,
        row_dim: &'a str,
    ) -> Self {
        let mut conjuncts = Vec::new();
        if let Some(constraint) = &implementation.constraint {
            constraint.conjuncts(&mut conjuncts);
        }
        let admitted = implementation
            .params
            .iter()
            .map(|parameter| {
                (
                    parameter.name.as_str(),
                    admitted(&parameter.name, &parameter.values, &conjuncts, statics),
                )
            })
            .collect();
        Self {
            implementation,
            statics,
            row_dim,
            admitted,
        }
    }

    /// The values parameter `name` can take as launch `launch` reads it
    /// (none: as a scratch buffer does): the launch's own parameter of the
    /// name, which the statics do not restrict, else the entry's.
    fn values(&self, launch: Option<usize>, name: &str) -> Option<&[u64]> {
        launch
            .and_then(|launch| {
                self.implementation.launches[launch]
                    .params
                    .iter()
                    .find(|parameter| parameter.name == name)
            })
            .map(|parameter| parameter.values.as_slice())
            .or_else(|| self.admitted.get(name).map(Vec::as_slice))
    }

    /// Whether `condition`, of launch `launch` or of a scratch buffer, can
    /// hold at `row` rows ([`can_hold`] over [`Self::values`]).
    fn can_hold(&self, condition: Option<&NativeCondition>, launch: Option<usize>, row: u64) -> bool {
        condition.is_none_or(|condition| {
            can_hold(condition, self.row_dim, row, self.statics, &|name: &str| {
                let values = self.values(launch, name)?;
                values.iter().copied().min().zip(values.iter().copied().max())
            })
        })
    }

    /// What `row` rows can run: each launch, then each scratch buffer, by
    /// whether its condition can hold there. The parameters a range's
    /// search has to choose follow from it: those its launches read.
    fn reach(&self, row: u64) -> Vec<bool> {
        let launches = self
            .implementation
            .launches
            .iter()
            .enumerate()
            .map(|(ordinal, launch)| self.can_hold(launch.when.as_ref(), Some(ordinal), row));
        let scratch = self
            .implementation
            .scratch
            .iter()
            .map(|scratch| self.can_hold(scratch.when.as_ref(), None, row));
        launches.chain(scratch).collect()
    }

    /// The tuning parameters launch `launch`'s grid axis `axis` reads, with
    /// every assignment of the values they can take ([`Self::values`]),
    /// where the axis reads the rows too: what [`tiling`] groups. None for
    /// any other axis, which no parameter tiles the rows of.
    ///
    /// # Panics
    ///
    /// Where the axis reads an undeclared parameter: the checker admits
    /// none such.
    fn tiled(&self, launch: usize, axis: &NativeNatExpr) -> Option<(Vec<String>, Vec<Vec<u64>>)> {
        let (mut dimensions, mut names) = (Vec::new(), Vec::new());
        axis.dimensions(&mut dimensions);
        axis.parameters(&mut names);
        if names.is_empty() || !dimensions.iter().any(|name| name == self.row_dim) {
            return None;
        }
        let domains = names
            .iter()
            .map(|name| {
                self.values(Some(launch), name)
                    .unwrap_or_else(|| panic!("grid parameter `{name}` is declared"))
            })
            .collect::<Vec<_>>();
        let assignments = assignments(&domains);
        Some((names, assignments))
    }
}

/// The values of entry parameter `name`, declared with `values`, that
/// `statics` admit: the value they fix it at
/// (`NativeSpecialization::param`), else each declared value that no
/// conjunct of the declaration's `where` (`conjuncts`) rules out by the
/// statics and that value alone. A launch only an unadmitted value selects
/// cannot run.
///
/// # Panics
///
/// Where they admit none: such statics lie outside the declaration's
/// domain, which a unit's never do.
fn admitted(name: &str, values: &[u64], conjuncts: &[&NativeCondition], statics: &NativeSpecialization) -> Vec<u64> {
    let admitted = match statics.param(name) {
        Some(value) => vec![value],
        None => values
            .iter()
            .copied()
            .filter(|&value| {
                conjuncts.iter().all(|conjunct| {
                    conjunct.decided(&|dimension: &str| statics.static_value(dimension), &|parameter: &str| {
                        (parameter == name).then_some(value)
                    }) != Some(false)
                })
            })
            .collect(),
    };
    assert!(!admitted.is_empty(), "the statics admit a value of `{name}`");
    admitted
}

/// Whether `condition` can hold at `row` rows: unless it is false at every
/// value of the parameters it reads within `bounds` (the least and the
/// greatest each can take), at the rows and `statics`. A condition that
/// reads another dimension of the call is not decided by the rows, and can
/// hold at every row count.
fn can_hold(
    condition: &NativeCondition,
    row_dim: &str,
    row: u64,
    statics: &NativeSpecialization,
    bounds: &impl Fn(&str) -> Option<(u64, u64)>,
) -> bool {
    condition.may_hold(
        &|name: &str| {
            if name == row_dim {
                Some((row, row))
            } else {
                statics.static_value(name).map(|value| (value, value))
            }
        },
        bounds,
    )
}

/// Every assignment of one value from each of `domains`, in their order.
fn assignments(domains: &[&[u64]]) -> Vec<Vec<u64>> {
    domains.iter().fold(vec![Vec::new()], |partial, values| {
        partial
            .iter()
            .flat_map(|prefix| {
                values.iter().map(move |value| {
                    let mut assignment = prefix.clone();
                    assignment.push(*value);
                    assignment
                })
            })
            .collect()
    })
}

/// How the grid axis `axis` groups `assignments` of the parameters `names`
/// at `row` rows and `statics`: for each assignment, the first that
/// dispatches as many groups along the axis as it does. An axis that does
/// not evaluate (it reads another dimension of the call) groups them all
/// as one, at every row count.
fn tiling(
    axis: &NativeNatExpr,
    names: &[String],
    assignments: &[Vec<u64>],
    row_dim: &str,
    row: u64,
    statics: &NativeSpecialization,
) -> Vec<usize> {
    let groups = assignments
        .iter()
        .map(|assignment| {
            axis.evaluate(
                &|name: &str| {
                    if name == row_dim {
                        Some(row)
                    } else {
                        statics.static_value(name)
                    }
                },
                &|name: &str| {
                    names
                        .iter()
                        .position(|candidate| candidate == name)
                        .map(|position| assignment[position])
                },
            )
            .ok()
        })
        .collect::<Vec<_>>();
    groups
        .iter()
        .map(|count| {
            groups
                .iter()
                .position(|other| other == count)
                .expect("an assignment groups with itself")
        })
        .collect()
}

/// The ranges of `served`, the rows an entry takes, that it is tuned and
/// dispatched by: `served` cut at each of `cuts` ([`row_cuts`]) inside it,
/// where the case's points (`points`, the rows it is measured at) lie on
/// both sides. A range that would hold none of them joins the range below
/// it, or the first joins the one above: every range holds a point of its
/// own, and every served row a range.
///
/// # Panics
///
/// Where no point lies in `served`: such a case serves nothing it is
/// measured at.
pub(crate) fn row_ranges(served: Range<u64>, cuts: &[u64], points: &[u64]) -> Vec<Range<u64>> {
    let held = |rows: Range<u64>| points.iter().any(|point| rows.contains(point));
    assert!(held(served.clone()), "a case is measured at rows it serves");
    let mut cuts = cuts
        .iter()
        .copied()
        .filter(|cut| served.start < *cut && *cut < served.end)
        .collect::<Vec<_>>();
    cuts.sort_unstable();
    cuts.dedup();
    let mut starts = vec![served.start];
    for cut in cuts {
        let open = *starts.last().expect("a partition starts at the served rows' start");
        if held(open..cut) && held(cut..served.end) {
            starts.push(cut);
        }
    }
    let ends = starts.iter().skip(1).copied().chain([served.end]);
    starts.iter().zip(ends).map(|(&start, end)| start..end).collect()
}

/// A tuning case restricted to a range of the rows it serves. Everything
/// but identity, points and ownership of launches is the case's own.
#[derive(Clone)]
pub(crate) struct Ranged<T> {
    case: T,
    rows: Range<u64>,
}

impl<T> Ranged<T> {
    pub(crate) fn new(case: T, rows: Range<u64>) -> Self {
        Self { case, rows }
    }

    /// The rows the unit serves.
    pub(crate) fn rows(&self) -> &Range<u64> {
        &self.rows
    }
}

/// `bindings` with the range that identifies its unit; an unbounded range
/// is written open (`M=9..`).
fn identify(bindings: String, rows: &Range<u64>) -> String {
    if rows.end == u64::MAX {
        format!("{bindings},M={}..", rows.start)
    } else {
        format!("{bindings},M={}..{}", rows.start, rows.end)
    }
}

impl<T: EntryTuning> EntryTuning for Ranged<T> {
    type Entry = T::Entry;
    type Case = T::Case;

    fn bindings(&self) -> String {
        identify(self.case.bindings(), &self.rows)
    }
    fn launches(&self) -> usize {
        self.case.launches()
    }
    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.case.statics(inputs)
    }
    /// The case's points at the range's rows, weighed among themselves.
    ///
    /// # Panics
    ///
    /// Where the range holds none: [`row_ranges`] forms none such.
    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        let held = self
            .case
            .points(limits)
            .into_iter()
            .filter(|point| self.rows.contains(&point.rows))
            .collect::<Vec<_>>();
        assert!(!held.is_empty(), "row range {:?} holds none of its case's points", self.rows);
        normalized(held)
    }
    fn launch_rows(&self, launch: &Launch) -> u64 {
        self.case.launch_rows(launch)
    }
    fn row_dimension(&self) -> &'static str {
        self.case.row_dimension()
    }
    fn row_regime_cuts(&self) -> &'static [u64] {
        self.case.row_regime_cuts()
    }
    fn invocation_rows(&self, launch: &Launch, representatives: &[u64]) -> Option<u64> {
        if !self.rows.contains(&self.case.launch_rows(launch)) {
            return None;
        }
        self.case.invocation_rows(launch, representatives)
    }
    fn served_dimensions(&self, shapes: &[PointShape]) -> Vec<BTreeMap<String, u64>> {
        self.case.served_dimensions(shapes)
    }
    fn search_starts(
        &self,
        _rows: &Range<u64>,
        device: &Device,
        implementation: &seismic::OnDevice,
        statics: &NativeSpecialization,
        limits: TuningLimits,
    ) -> Vec<ParameterValues> {
        self.case.search_starts(&self.rows, device, implementation, statics, limits)
    }
    fn workload(&self, cases: usize) -> seismic::WorkloadRotation {
        self.case.workload(cases)
    }
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        self.case.rotation(inputs, point)
    }
    fn args<'a>(case: &'a mut Self::Case) -> <Self::Entry as seismic::Entry>::Args<'a> {
        T::args(case)
    }
    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        T::state(case)
    }
    #[cfg(feature = "tuning-survey")]
    fn tune(
        &self,
        charges: &seismic::Charges,
        statics: &NativeSpecialization,
        points: &mut dyn seismic::PointSource<'_, Self::Entry>,
        validation: seismic::NumericalRequirement,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError> {
        self.case.tune(charges, statics, points, validation, strategy)
    }
    fn open(
        &self,
        charges: &seismic::Charges,
        statics: &NativeSpecialization,
        validation: seismic::NumericalRequirement,
    ) -> Result<Box<dyn EntrySearch<Self::Entry>>, TuneError> {
        self.case.open(charges, statics, validation)
    }
    fn digest(&self, device: &Device, statics: &NativeSpecialization) -> Result<String, TuneError> {
        self.case.digest(device, statics)
    }
    fn stored_valid(&self, device: &Device, specialization: &NativeSpecialization) -> bool {
        self.case.stored_valid(device, specialization)
    }
    fn entry(&self) -> seismic::BoundEntry<Self::Entry> {
        self.case.entry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row points of a case served at every row, up to `max_rows`.
    fn points(max_rows: u64) -> Vec<u64> {
        TUNING_ROWS.into_iter().filter(|rows| *rows <= max_rows).collect()
    }

    #[test]
    fn a_range_holds_points_of_its_own() {
        // Cuts between a decode step's row, a verification's and a
        // prefill's.
        let classes = [2, VERIFY_ROWS + 1];
        assert_eq!(
            row_ranges(EVERY_ROW, &classes, &points(512)),
            vec![1..2, 2..VERIFY_ROWS + 1, VERIFY_ROWS + 1..u64::MAX]
        );
        // A model whose rows stop at the verification's measures nothing
        // past it: the verification's range takes every row above.
        assert_eq!(row_ranges(EVERY_ROW, &classes, &points(VERIFY_ROWS)), vec![1..2, 2..u64::MAX]);
        assert_eq!(row_ranges(EVERY_ROW, &classes, &points(1)), vec![EVERY_ROW]);
        // Without a cut, one range.
        assert_eq!(row_ranges(EVERY_ROW, &[], &points(512)), vec![EVERY_ROW]);
        // An entry of the decode rows alone, at its own cuts.
        let decode = 1..VERIFY_ROWS + 1;
        let cuts = [2, 4, 5];
        assert_eq!(row_ranges(decode.clone(), &cuts, &points(512)), vec![1..2, 2..4, 4..5, 5..VERIFY_ROWS + 1]);
        assert_eq!(row_ranges(decode.clone(), &cuts, &points(4)), vec![1..2, 2..4, 4..VERIFY_ROWS + 1]);
        // A cut no point lies past joins nothing; one below every point,
        // the first range to the second.
        assert_eq!(row_ranges(decode.clone(), &[2, 5, 7], &points(512)), vec![1..2, 2..5, 5..VERIFY_ROWS + 1]);
        assert_eq!(row_ranges(EVERY_ROW, &classes, &[16, 64]), vec![EVERY_ROW]);
        // An entry of the rows past the decode's alone: one range, also
        // where it is measured at its smallest row beyond the model's.
        let grouped = VERIFY_ROWS + 1..u64::MAX;
        assert_eq!(row_ranges(grouped.clone(), &classes, &[16]), vec![grouped]);
    }

    fn rows() -> Box<NativeNatExpr> {
        Box::new(NativeNatExpr::Dimension("M".into()))
    }

    fn constant(value: u64) -> NativeNatExpr {
        NativeNatExpr::Constant(value)
    }

    fn parameter(name: &str) -> NativeNatExpr {
        NativeNatExpr::Parameter(name.into())
    }

    fn compare(comparison: seismic::CmpOp, left: NativeNatExpr, right: NativeNatExpr) -> NativeCondition {
        NativeCondition::Compare { comparison, left, right }
    }

    fn and(left: NativeCondition, right: NativeCondition) -> NativeCondition {
        NativeCondition::And(Box::new(left), Box::new(right))
    }

    /// `BATCH_FROM in [5, 3, 9, 2]`, by its least and greatest value.
    fn batch_from(name: &str) -> Option<(u64, u64)> {
        (name == "BATCH_FROM").then_some((2, 9))
    }

    #[test]
    fn rows_are_cut_where_what_they_can_run_changes() {
        use seismic::CmpOp::{Eq, Ge, Gt, Le, Lt};
        let statics = NativeSpecialization::new().with_static("S", 4);
        // A dense projection's launches: the GEMV (`M < 2`), the pair
        // (`M == 2 and M < BATCH_FROM`), the row GEMV (`M >= 3 and
        // M < BATCH_FROM`), the batch (`M >= BATCH_FROM and M <= 16`) and
        // the matrix (`M > 16`).
        let launches = [
            compare(Lt, *rows(), constant(2)),
            and(compare(Eq, *rows(), constant(2)), compare(Lt, *rows(), parameter("BATCH_FROM"))),
            and(compare(Ge, *rows(), constant(3)), compare(Lt, *rows(), parameter("BATCH_FROM"))),
            and(compare(Ge, *rows(), parameter("BATCH_FROM")), compare(Le, *rows(), constant(16))),
            compare(Gt, *rows(), constant(16)),
        ];
        let reach = |row: u64| {
            launches
                .iter()
                .map(|launch| can_hold(launch, "M", row, &statics, &batch_from))
                .collect::<Vec<_>>()
        };
        assert_eq!(reach(1), [true, false, false, false, false]);
        // Two rows are the pair's or the batch's, as `BATCH_FROM` says.
        assert_eq!(reach(2), [false, true, false, true, false]);
        // From three rows to the greatest `BATCH_FROM` the search places
        // the flip between the row GEMV and the batch: one reach.
        assert_eq!(reach(3), [false, false, true, true, false]);
        assert_eq!(reach(8), reach(3));
        // From the greatest `BATCH_FROM` on no value selects the row GEMV.
        assert_eq!(reach(9), [false, false, false, true, false]);
        assert_eq!(reach(16), reach(9));
        assert_eq!(reach(17), [false, false, false, false, true]);
        assert_eq!(reach(512), reach(17));
        assert_eq!(reach_cuts(&points(512), &reach), vec![2, 3, 9, 17]);
        // Fewer measured rows, fewer cuts: none past the last point.
        assert_eq!(reach_cuts(&points(8), &reach), vec![2, 3]);
        assert!(reach_cuts(&points(1), &reach).is_empty());

        // Over a static: `M * S <= 16` holds through four rows of four
        // slots.
        let slots = compare(
            Le,
            NativeNatExpr::Mul(rows(), Box::new(NativeNatExpr::Dimension("S".into()))),
            constant(16),
        );
        assert_eq!(
            reach_cuts(&points(512), |row| can_hold(&slots, "M", row, &statics, &batch_from)),
            vec![5]
        );
        // A band no point lies in cuts nothing: the points on its two
        // sides run the same.
        let band = and(compare(Ge, *rows(), constant(20)), compare(Le, *rows(), constant(30)));
        assert!(can_hold(&band, "M", 24, &statics, &batch_from));
        assert!(reach_cuts(&points(512), |row| can_hold(&band, "M", row, &statics, &batch_from)).is_empty());
        // A condition a dimension of the call decides can hold at every
        // row count, and one over an unknown parameter likewise.
        let call = compare(Lt, *rows(), NativeNatExpr::Dimension("H".into()));
        let unknown = compare(Lt, *rows(), parameter("OTHER"));
        for condition in [&call, &unknown] {
            assert!(reach_cuts(&points(512), |row| can_hold(condition, "M", row, &statics, &batch_from)).is_empty());
        }
    }

    #[test]
    fn every_measured_row_is_a_range_where_a_specialization_forms_no_program() {
        // Past each point but the highest; the rows between two points go
        // with the upper one.
        assert_eq!(point_cuts(&points(512), &[]), vec![2, 3, 5, 9, 17, 33, 65, 129, 257]);
        assert_eq!(
            row_ranges(EVERY_ROW, &point_cuts(&points(16), &[]), &points(16)),
            vec![1..2, 2..3, 3..5, 5..9, 9..u64::MAX]
        );
        assert!(point_cuts(&points(1), &[]).is_empty());
        // Two points the declaration parts are cut where it says, at a row
        // between them (7) or past the lower one (17).
        assert_eq!(point_cuts(&points(32), &[7, 17]), vec![2, 3, 9]);

        // The CPU's projection states nothing by its rows and forms no
        // program: a range at each measured row count of the rows it
        // serves. Metal's is cut by its declaration alone.
        let project = |backend| {
            seismic::generated::native_implementation_for_backend::<magnitude_kernels::project_rows::Entry>(backend)
                .unwrap()
                .unwrap()
        };
        let cpu = project(seismic::BackendName::Cpu);
        assert!(cpu.has_tuning_parameters());
        let statics = NativeSpecialization::new().with_static("WS", 1);
        let ranges = |served: Range<u64>, points: &[u64]| {
            row_ranges(served.clone(), &row_cuts(&cpu, &statics, "M", &served, points), points)
        };
        assert_eq!(
            ranges(EVERY_ROW, &points(512)),
            vec![1..2, 2..3, 3..5, 5..9, 9..17, 17..33, 33..65, 65..129, 129..257, 257..u64::MAX]
        );
        assert_eq!(ranges(EVERY_ROW, &points(8)), vec![1..2, 2..3, 3..5, 5..u64::MAX]);
        assert_eq!(
            ranges(1..VERIFY_ROWS + 1, &points(512)),
            vec![1..2, 2..3, 3..5, 5..VERIFY_ROWS + 1]
        );
        assert_eq!(
            ranges(VERIFY_ROWS + 1..u64::MAX, &points(64)),
            vec![VERIFY_ROWS + 1..17, 17..33, 33..u64::MAX]
        );
        assert!(seismic::BackendName::Metal.specializations_form_programs());
        assert!(!seismic::BackendName::Cpu.specializations_form_programs());
    }

    #[test]
    fn the_statics_admit_the_values_no_restriction_rules_out_alone() {
        use seismic::CmpOp::{Eq, Le};
        // `(KS == 0 or TOKENS == 1) and TOKENS <= SIMDS`.
        let sinks = NativeCondition::Or(
            Box::new(compare(Eq, NativeNatExpr::Dimension("KS".into()), constant(0))),
            Box::new(compare(Eq, parameter("TOKENS"), constant(1))),
        );
        let lanes = compare(Le, parameter("TOKENS"), parameter("SIMDS"));
        let conjuncts = [&sinks, &lanes];
        let values = [1, 2, 4, 8];
        let plain = NativeSpecialization::new().with_static("KS", 0);
        // A restriction another parameter decides rules out nothing alone.
        assert_eq!(admitted("TOKENS", &values, &conjuncts, &plain), values);
        let sunk = NativeSpecialization::new().with_static("KS", 1);
        assert_eq!(admitted("TOKENS", &values, &conjuncts, &sunk), [1]);
        // A value the unit is fixed at is the one it can take.
        assert_eq!(admitted("TOKENS", &values, &conjuncts, &plain.with_param("TOKENS", 4)), [4]);
    }

    #[test]
    fn a_grid_that_tiles_the_rows_cuts_where_its_tiles_part_the_configurations() {
        let statics = NativeSpecialization::new().with_static("KV", 2).with_static("G", 4);
        let read = |axis: &NativeNatExpr| {
            let mut names = Vec::new();
            axis.parameters(&mut names);
            names
        };
        let decode = 2..VERIFY_ROWS + 2;
        // `KV * ceil_div(M, TOKENS)`, `TOKENS in [1, 2, 4, 8]`.
        let tokens = NativeNatExpr::Mul(
            Box::new(NativeNatExpr::Dimension("KV".into())),
            Box::new(NativeNatExpr::CeilDiv(rows(), Box::new(parameter("TOKENS")))),
        );
        let (names, values) = (read(&tokens), assignments(&[&[1, 2, 4, 8]]));
        let tiles = |row: u64| tiling(&tokens, &names, &values, "M", row, &statics);
        // One row is one tile under every value; two part the one-row tile
        // from the others.
        assert_eq!(tiles(1), [0, 0, 0, 0]);
        assert_eq!(tiles(2), [0, 1, 1, 1]);
        assert_eq!(tiles(4), tiles(3));
        assert_eq!(tiles(9), tiles(8));
        let cuts = tile_cuts(decode.clone(), |_| true, &tiles);
        assert_eq!(cuts, vec![2, 3, 5]);
        // The decode rows' points 1, 2, 4 and 8 each take a range of their
        // own.
        assert_eq!(
            row_ranges(1..VERIFY_ROWS + 1, &cuts, &points(512)),
            vec![1..2, 2..3, 3..5, 5..VERIFY_ROWS + 1]
        );
        // A launch that does not run on both sides of a row is cut there
        // by its condition, never by its tiles.
        assert_eq!(tile_cuts(decode.clone(), |row| row >= 2, &tiles), vec![3, 5]);

        // A tile a form selects and a parameter sizes:
        // `(1 - MATRIX) * M + MATRIX * ceil_div(M, max(1, WARPS * 16 / G))`,
        // `MATRIX in [0, 1]`, `WARPS in [4, 8, 1, 2]`, four query heads a
        // kv head: tiles of 16, 32, 4 and 8 rows.
        let binary = |make: fn(Box<NativeNatExpr>, Box<NativeNatExpr>) -> NativeNatExpr,
                      left: NativeNatExpr,
                      right: NativeNatExpr| make(Box::new(left), Box::new(right));
        let tile = binary(
            NativeNatExpr::Max,
            constant(1),
            binary(
                NativeNatExpr::Div,
                binary(NativeNatExpr::Mul, parameter("WARPS"), constant(16)),
                NativeNatExpr::Dimension("G".into()),
            ),
        );
        let formed = binary(
            NativeNatExpr::Add,
            binary(
                NativeNatExpr::Mul,
                binary(NativeNatExpr::Sub, constant(1), parameter("MATRIX")),
                *rows(),
            ),
            binary(
                NativeNatExpr::Mul,
                parameter("MATRIX"),
                binary(NativeNatExpr::CeilDiv, *rows(), tile),
            ),
        );
        let names = read(&formed);
        assert_eq!(names, ["MATRIX", "WARPS"]);
        let values = assignments(&[&[0, 1], &[4, 8, 1, 2]]);
        assert_eq!(values.len(), 8);
        assert_eq!(values[6], [1, 1]);
        // The matrix form parts from the row form at two rows, and its
        // four-row and eight-row tiles from the wider ones past their rows.
        assert_eq!(
            tile_cuts(decode.clone(), |_| true, |row| tiling(&formed, &names, &values, "M", row, &statics)),
            vec![2, 5, 9]
        );

        // A tile wider than the decode rows cuts nothing there; nor does
        // an axis the rows do not decide.
        let wide = assignments(&[&[32, 64, 128]]);
        let names = read(&tokens);
        assert!(tile_cuts(decode.clone(), |_| true, |row| tiling(&tokens, &names, &wide, "M", row, &statics)).is_empty());
        let columns = NativeNatExpr::Mul(
            rows(),
            Box::new(NativeNatExpr::CeilDiv(
                Box::new(NativeNatExpr::Dimension("N".into())),
                Box::new(parameter("TOKENS")),
            )),
        );
        assert!(tile_cuts(decode, |_| true, |row| tiling(&columns, &names, &wide, "M", row, &statics)).is_empty());
    }

    #[test]
    fn o_indexed_entries_cut_by_their_row_dimension() {
        let bf16 = Element::bf16();
        let expand = magnitude_kernels::dense_expand::native_entry_with(
            magnitude_kernels::dense_expand::Elements {
                NW: bf16,
                GW: bf16,
                UW: bf16,
                A: bf16,
            },
        );
        let cuda_expand = seismic::generated::native_implementation_at::<magnitude_kernels::dense_expand::Entry>(
            seismic::BackendName::Cuda,
            expand.elements(),
        )
        .unwrap()
        .unwrap();
        let cuda_statics = NativeSpecialization::new()
            .with_static("H", 2048)
            .with_static("F", 8192)
            .with_static("GS", 0)
            .with_static("US", 0)
            .with_static("LB", 0);
        // Binding `"M"` misses `"O"`-indexed launch conditions and leaves the
        // entry monolithic; binding `"O"` partitions its row regimes.
        assert!(row_cuts(&cuda_expand, &cuda_statics, "M", &EVERY_ROW, &points(512)).is_empty());
        let cuda_cuts = row_cuts(&cuda_expand, &cuda_statics, "O", &EVERY_ROW, &points(512));
        assert_eq!(cuda_cuts, vec![2, VERIFY_ROWS + 1, 17, 65]);
        assert_eq!(
            row_ranges(EVERY_ROW, &cuda_cuts, &points(512)),
            vec![1..2, 2..VERIFY_ROWS + 1, VERIFY_ROWS + 1..17, 17..65, 65..u64::MAX]
        );

        let metal_expand = seismic::generated::native_implementation_at::<magnitude_kernels::dense_expand::Entry>(
            seismic::BackendName::Metal,
            expand.elements(),
        )
        .unwrap()
        .unwrap();
        let metal_statics = cuda_statics.with_static("GB", 0).with_static("UB", 0);
        let metal_cuts = row_cuts(&metal_expand, &metal_statics, "O", &EVERY_ROW, &points(512));
        assert!(metal_cuts.contains(&2), "Metal dense_expand cuts O=1: {metal_cuts:?}");
        assert!(
            metal_cuts.contains(&(VERIFY_ROWS + 1)),
            "Metal dense_expand cuts verify from prefill: {metal_cuts:?}"
        );
        assert!(metal_cuts.contains(&65), "Metal dense_expand cuts tall/large GEMM: {metal_cuts:?}");
    }

    #[test]
    #[should_panic(expected = "a case is measured at rows it serves")]
    fn a_case_is_measured_where_it_serves() {
        row_ranges(1..VERIFY_ROWS + 1, &[2], &[16]);
    }

    #[test]
    fn a_range_identifies_its_unit() {
        assert_eq!(identify("A=bf16".into(), &(2..9)), "A=bf16,M=2..9");
        assert_eq!(identify("A=bf16".into(), &(9..u64::MAX)), "A=bf16,M=9..");
    }

    #[test]
    fn merged_pass_regime_cut_splits_grouped_experts_only_when_pass_points_exist_above_it() {
        use super::routed::{GROUPED_SERVED, MERGED_PASS_FROM, grouped_pass_points};
        let limits = |max_pass_rows| TuningLimits {
            max_rows: 512,
            max_pass_rows,
            max_projected_rows: 8,
            context_tokens: 65536,
            image_cells: 0,
        };
        let rows = |max_pass_rows| {
            grouped_pass_points(limits(max_pass_rows))
                .into_iter()
                .map(|point| point.rows)
                .collect::<Vec<_>>()
        };

        // Without merged passes > 1024, no point lies past MERGED_PASS_FROM,
        // so row_ranges keeps a single range 9.. identical to non-merged loads.
        let flat_512 = rows(512);
        assert_eq!(flat_512, vec![16, 32, 64, 128, 256, 512]);
        assert_eq!(
            row_ranges(GROUPED_SERVED, &[MERGED_PASS_FROM], &flat_512),
            vec![GROUPED_SERVED]
        );
        let flat_1024 = rows(1024);
        assert_eq!(flat_1024, vec![16, 32, 64, 128, 256, 512, 1024]);
        assert_eq!(
            row_ranges(GROUPED_SERVED, &[MERGED_PASS_FROM], &flat_1024),
            vec![GROUPED_SERVED]
        );

        // When max_pass_rows > 1024, points lie on both sides of
        // MERGED_PASS_FROM, splitting small-grouped (9..1025) from
        // merged-pass (1025..).
        let split_2048 = rows(2048);
        assert_eq!(split_2048, vec![16, 32, 64, 128, 256, 512, 1024, 2048]);
        assert_eq!(
            row_ranges(GROUPED_SERVED, &[MERGED_PASS_FROM], &split_2048),
            vec![VERIFY_ROWS + 1..MERGED_PASS_FROM, MERGED_PASS_FROM..u64::MAX]
        );
        let split_4096 = rows(4096);
        assert_eq!(split_4096, vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096]);
        let ranges = row_ranges(GROUPED_SERVED, &[MERGED_PASS_FROM], &split_4096);
        assert_eq!(
            ranges,
            vec![VERIFY_ROWS + 1..MERGED_PASS_FROM, MERGED_PASS_FROM..u64::MAX]
        );
        // Every range holds its own measured points and non-merged row_points
        // remain capped at max_rows = 512.
        assert_eq!(
            row_points(limits(4096))
                .into_iter()
                .map(|point| point.rows)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        );
        assert_eq!(
            split_4096
                .iter()
                .copied()
                .filter(|r| ranges[1].contains(r))
                .collect::<Vec<_>>(),
            vec![2048, 4096]
        );
    }
}
