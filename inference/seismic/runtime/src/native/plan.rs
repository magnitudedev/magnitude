//! The plan of a native tuning domain: which parameters are searched
//! together, which launches can run at each point, and how a configuration
//! the search asks for is produced. Everything here is derived from the
//! checked declaration. Formation and measurement consume this plan;
//! neither happens here.
//!
//! The declared domain is never produced. Its size is the product of its
//! parameters' value counts, and every form or launch parameter a
//! declaration gains multiplies it: a group is its parameters with their
//! declared values and form flags ([`Group`]), what admits an assignment
//! of it is asked of its domain ([`Domain`]), and an assignment exists only
//! when the search asks for it, one step from one it holds. Planning
//! therefore takes time and memory in the number of declared parameters and
//! values and in what the search measures, never in their product, and
//! nothing bounds it by a count or an instant: neither would be a property
//! of the declaration, and either would change which configurations exist.
//!
//! Every implementation is planned so. One whose launches are formed
//! separately has a group per set of launches that can run together
//! ([`partition`]); one whose source is formed whole is the case of one
//! group over the whole program ([`whole`]), since its source receives
//! every entry parameter and none can be shown to belong to one launch.

use super::search::PointKey;
use seismic_lang::checked::{
    NativeCondition, NativeEvalError, NativeImplementation, NativeParameter, NativeSpecialization,
    NativeSpecializationError,
};
use std::collections::{BTreeMap, BTreeSet};

/// What admits configurations of a planned domain, asked one configuration
/// at a time: the checked declaration for a search that forms and measures,
/// a survey's record for a replay of one. The search order asks nothing
/// else of where its configurations come from.
pub trait Domain {
    /// `base` with `parameters` at `values`, when that configuration is
    /// admissible as it stands; none when it is not.
    fn place(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError>;

    /// The admissible configuration with `parameters` at `values` nearest
    /// `base`; none when no configuration has them. Nearest is the one
    /// definition of [`TuningPartition::nearest`].
    fn nearest(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError>;

    /// The key `specialization` is expected to run under at timed point
    /// `point`, where the configuration it is a change of ran under
    /// `baseline`: what it would be measured as there, known before it is
    /// formed.
    fn forecast(
        &self,
        point: usize,
        baseline: &PointKey,
        specialization: &NativeSpecialization,
    ) -> PointKey;
}

/// How far `candidate` is from `base`: how many of `parameters` differ, and
/// over those the steps between the two values in the parameter's declared
/// values taken in numeric order ([`Group::steps`] moves along the same
/// order). Fewer parameters changed is nearer whatever the steps: a
/// parameter left alone keeps what was measured of it.
pub fn distance(
    parameters: &[(ParameterAddress, &[u64])],
    base: &NativeSpecialization,
    candidate: &NativeSpecialization,
) -> (usize, usize) {
    let mut distance = (0, 0);
    for (address, values) in parameters {
        let (from, to) = (address.value(base), address.value(candidate));
        if from != to {
            let between = values
                .iter()
                .filter(|value| (from.min(to)..from.max(to)).contains(*value))
                .count();
            distance = (distance.0 + 1, distance.1 + between);
        }
    }
    distance
}

/// A dynamic shape at which the consumer can measure the native entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointShape {
    pub label: String,
    pub dimensions: BTreeMap<String, u64>,
}

/// The lexical address of one tuning parameter.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParameterAddress {
    Entry(String),
    Launch { ordinal: usize, name: String },
}

impl ParameterAddress {
    pub(crate) fn value(&self, specialization: &NativeSpecialization) -> u64 {
        match self {
            Self::Entry(name) => specialization.param(name),
            Self::Launch { ordinal, name } => specialization.launch_param(*ordinal, name),
        }
        .expect("an admissible specialization values every parameter")
    }

    pub(crate) fn set(
        &self,
        specialization: NativeSpecialization,
        value: u64,
    ) -> NativeSpecialization {
        match self {
            Self::Entry(name) => specialization.with_param(name.clone(), value),
            Self::Launch { ordinal, name } => {
                specialization.with_launch_param(*ordinal, name.clone(), value)
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Entry(name) | Self::Launch { name, .. } => name,
        }
    }
}

/// Code arguments in the same order as the launch's template header:
/// received entry parameters first, then launch-local parameters.
pub fn code_values(
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    launch: usize,
) -> Vec<u64> {
    let reads = implementation.launches[launch].parameters();
    implementation
        .params
        .iter()
        .filter(|parameter| {
            parameter.code
                && (reads.contains(&parameter.name)
                    || !implementation
                        .launches
                        .iter()
                        .any(|candidate| candidate.parameters().contains(&parameter.name)))
        })
        .map(|parameter| {
            specialization
                .param(&parameter.name)
                .expect("validated entry code parameter")
        })
        .chain(
            implementation.launches[launch]
                .params
                .iter()
                .filter(|parameter| parameter.code)
                .map(|parameter| {
                    specialization
                        .launch_param(launch, &parameter.name)
                        .expect("validated launch code parameter")
                }),
        )
        .collect()
}

/// The instances of one launch a compile forms: those the configurations
/// about to be measured run, and no others. There is no set of a launch's
/// instances apart from a request for some.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LaunchSource {
    pub ordinal: usize,
    pub variants: Vec<LaunchVariant>,
}

/// One instance of a launch: the values of the code parameters it owns and,
/// when its group size reads only static dimensions and parameters, that
/// size, which a Metal pipeline is formed to admit.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LaunchVariant {
    pub code: Vec<u64>,
    pub group_size: Option<[u64; 3]>,
}

/// Every assignment of some value lists, the last varying fastest, produced
/// one at a time: the lists are held, never their product.
struct Assignments<'v> {
    values: Vec<&'v [u64]>,
    next: Option<Vec<usize>>,
}

impl<'v> Assignments<'v> {
    fn of(values: Vec<&'v [u64]>) -> Self {
        let next = values
            .iter()
            .all(|values| !values.is_empty())
            .then(|| vec![0; values.len()]);
        Self { values, next }
    }
}

impl Iterator for Assignments<'_> {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Vec<u64>> {
        let current = self.next.take()?;
        let assignment = current
            .iter()
            .zip(&self.values)
            .map(|(index, values)| values[*index])
            .collect();
        let mut following = current;
        for position in (0..following.len()).rev() {
            if following[position] + 1 < self.values[position].len() {
                following[position] += 1;
                self.next = Some(following);
                break;
            }
            following[position] = 0;
        }
        Some(assignment)
    }
}

/// Every subset of `count` of the positions `0..of`, in lexicographic
/// order, one at a time.
struct Subsets {
    of: usize,
    next: Option<Vec<usize>>,
}

impl Subsets {
    fn new(of: usize, count: usize) -> Self {
        Self {
            of,
            next: (count <= of).then(|| (0..count).collect()),
        }
    }
}

impl Iterator for Subsets {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let current = self.next.take()?;
        let count = current.len();
        // The last position that can still move right.
        if let Some(moved) = (0..count)
            .rev()
            .find(|&position| current[position] < self.of - (count - position))
        {
            let mut following = current.clone();
            following[moved] += 1;
            for position in moved + 1..count {
                following[position] = following[position - 1] + 1;
            }
            self.next = Some(following);
        }
        Some(current)
    }
}

/// The parameters a search moves together: those of launches that can run
/// in one call, or that a restriction or a shared parameter couples. The
/// group is its declaration: each parameter with its declared values and
/// whether they select a form. Its assignments are not held anywhere; the
/// search asks for the ones it measures, each one step from an assignment
/// it has ([`Group::steps`], [`Group::form_starts`],
/// [`Group::neighbours`]), and its domain says which of them the
/// declaration admits ([`Domain`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub launches: Vec<usize>,
    pub parameters: Vec<ParameterAddress>,
    /// Each parameter's declared values, in declaration order.
    pub values: Vec<Vec<u64>>,
    /// Which parameters' values select structurally different forms.
    pub forms: Vec<bool>,
}

impl Group {
    /// How many assignments the value lists declare: their product, which
    /// `constraint` can only reduce. A number alone; nothing is produced to
    /// count it.
    pub fn declared(&self) -> u128 {
        self.values
            .iter()
            .fold(1u128, |product, values| product.saturating_mul(values.len() as u128))
    }

    /// The group's assignment in `specialization`.
    pub fn assignment(&self, specialization: &NativeSpecialization) -> Vec<u64> {
        self.parameters
            .iter()
            .map(|address| address.value(specialization))
            .collect()
    }

    /// The values `assignment` gives the group's form parameters: the
    /// structural form it is an instance of.
    pub fn form(&self, assignment: &[u64]) -> Vec<u64> {
        assignment
            .iter()
            .zip(&self.forms)
            .filter(|(_, form)| **form)
            .map(|(value, _)| *value)
            .collect()
    }

    /// For one parameter, the assignments that move it alone from its value
    /// in `from`: toward lower declared values, nearest first, and toward
    /// higher ones, nearest first. The search takes the first of each that
    /// is admissible: the nearby change of that coordinate in each
    /// direction.
    pub fn steps(&self, from: &[u64], coordinate: usize) -> [Vec<Vec<u64>>; 2] {
        let moved = |value: u64| {
            let mut assignment = from.to_vec();
            assignment[coordinate] = value;
            assignment
        };
        let mut lower = self.values[coordinate]
            .iter()
            .copied()
            .filter(|value| *value < from[coordinate])
            .collect::<Vec<_>>();
        lower.sort_unstable_by(|left, right| right.cmp(left));
        let mut upper = self.values[coordinate]
            .iter()
            .copied()
            .filter(|value| *value > from[coordinate])
            .collect::<Vec<_>>();
        upper.sort_unstable();
        [
            lower.into_iter().map(moved).collect(),
            upper.into_iter().map(moved).collect(),
        ]
    }

    /// The forms one form parameter from `from`'s: each form parameter, in
    /// declaration order, and each of its other declared values, in
    /// declaration order. Their number is the sum of the form parameters'
    /// value counts, never the product: a form further off is reached from
    /// the form between, once that one leads. Each is the coordinate moved
    /// and its value; the configuration that represents the form is the
    /// domain's to give ([`Domain::nearest`]).
    pub fn form_starts<'g>(
        &'g self,
        from: &'g [u64],
    ) -> impl Iterator<Item = (usize, u64)> + 'g {
        (0..self.parameters.len())
            .filter(|coordinate| self.forms[*coordinate])
            .flat_map(move |coordinate| {
                self.values[coordinate]
                    .iter()
                    .filter(move |value| **value != from[coordinate])
                    .map(move |value| (coordinate, *value))
            })
    }

    /// The assignments of `from`'s form one parameter from it: each
    /// parameter that selects no form, in declaration order, at each of its
    /// other declared values, nearest first. One at a time.
    pub fn neighbours<'g>(&'g self, from: &'g [u64]) -> impl Iterator<Item = Vec<u64>> + 'g {
        (0..self.parameters.len())
            .filter(|coordinate| !self.forms[*coordinate])
            .flat_map(move |coordinate| {
                let [lower, upper] = self.steps(from, coordinate);
                // Nearest first, a lower value before the higher one as far.
                let mut moves = lower.into_iter().chain(upper).collect::<Vec<_>>();
                moves.sort_by_key(|assignment| {
                    (
                        assignment[coordinate].abs_diff(from[coordinate]),
                        assignment[coordinate],
                    )
                });
                moves
            })
    }
}

/// The partition of a declared domain: its independent groups, the
/// parameters that only choose which launch serves which rows, and the
/// launches each point can run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningPartition {
    pub groups: Vec<Group>,
    /// Entry parameters used solely to choose a launch's active row band,
    /// and each one's declared values.
    pub boundary: Vec<ParameterAddress>,
    pub boundary_values: Vec<Vec<u64>>,
    /// Which boundary parameters' values select structurally different
    /// forms, as against moving a row band's edge.
    pub boundary_forms: Vec<bool>,
    pub points: Vec<PlannedPoint>,
}

impl TuningPartition {
    /// The boundary's assignment in `specialization`.
    pub fn boundary_of(&self, specialization: &NativeSpecialization) -> Vec<u64> {
        self.boundary
            .iter()
            .map(|address| address.value(specialization))
            .collect()
    }

    /// The boundary assignments one parameter from `from`: each boundary
    /// parameter, in declaration order, at each of its other declared
    /// values. One at a time; their product is never produced.
    pub fn boundary_neighbours<'p>(
        &'p self,
        from: &'p [u64],
    ) -> impl Iterator<Item = Vec<u64>> + 'p {
        (0..self.boundary.len()).flat_map(move |coordinate| {
            self.boundary_values[coordinate]
                .iter()
                .filter(move |value| **value != from[coordinate])
                .map(move |value| {
                    let mut assignment = from.to_vec();
                    assignment[coordinate] = *value;
                    assignment
                })
        })
    }

    /// `base` with `parameters` at `values`, when the declaration admits
    /// that configuration as it stands; none when it does not. The checked
    /// declaration is the admissibility authority.
    pub fn place(
        implementation: &NativeImplementation,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        if parameters.len() != values.len() {
            return Err(PlanError::Choice);
        }
        let placed = parameters
            .iter()
            .zip(values)
            .fold(base.clone(), |placed, (address, value)| address.set(placed, *value));
        match implementation.validate(&placed) {
            Ok(()) => Ok(Some(placed)),
            Err(NativeSpecializationError::Inadmissible) => Ok(None),
            Err(error) => Err(PlanError::Declaration(error)),
        }
    }

    /// The admissible configuration that has `parameters` at `values`
    /// nearest `base`; none when no configuration has them.
    ///
    /// Nearest is defined, not chosen: of the admissible configurations
    /// with those values, the one that differs from `base` in the fewest
    /// other parameters; of those, the one whose changed parameters moved
    /// the fewest steps along their declared values in numeric order
    /// ([`distance`]); of those, the one that changes the parameters
    /// declared earliest, at their earliest declared values. So `base` so
    /// changed is its own nearest when it is admissible, and a change that
    /// needs another (a form that needs another launch's tile, a boundary a
    /// guard ties to a code parameter) moves that other and nothing else.
    ///
    /// A parameter no violated restriction couples to the change is never
    /// among the fewest: with it put back as `base` has it the
    /// configuration is still admissible. Only the coupled parameters are
    /// therefore tried, one changed, then two, to the first number that
    /// admits any: the work is the size of the restrictions the change
    /// violates, not of the domain.
    pub fn nearest(
        implementation: &NativeImplementation,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        if let Some(placed) = Self::place(implementation, base, parameters, values)? {
            return Ok(Some(placed));
        }
        let fixed = parameters
            .iter()
            .zip(values)
            .fold(base.clone(), |fixed, (address, value)| address.set(fixed, *value));
        let mut narrowed = implementation.clone();
        narrowed.constraint = match implementation
            .constraint
            .as_ref()
            .map(|condition| conditioned(condition, &fixed, parameters))
        {
            Some(Err(false)) => return Ok(None),
            Some(Ok(condition)) => Some(condition),
            Some(Err(true)) | None => None,
        };
        for parameter in &mut narrowed.params {
            let address = entry_address(parameter);
            if parameters.contains(&address) {
                parameter.values = vec![address.value(&fixed)];
            }
        }
        for (ordinal, launch) in narrowed.launches.iter_mut().enumerate() {
            for parameter in &mut launch.params {
                let address = local_address(ordinal, parameter);
                if parameters.contains(&address) {
                    parameter.values = vec![address.value(&fixed)];
                }
            }
        }
        let mut parts = Vec::new();
        if let Some(condition) = &narrowed.constraint {
            constraint_parts(condition, &mut parts);
        }
        // The parameters the violated restrictions leave free: every other
        // is held at one value.
        let coupled = restricted(&narrowed, &fixed, &parts);
        let free = coupled
            .params
            .iter()
            .map(entry_address)
            .chain(coupled.launches.iter().enumerate().flat_map(|(ordinal, launch)| {
                launch
                    .params
                    .iter()
                    .map(move |parameter| local_address(ordinal, parameter))
            }))
            .filter(|address| declared_values(&coupled, address).len() > 1)
            .map(|address| {
                let values = declared_values(implementation, &address);
                (address, values)
            })
            .collect::<Vec<_>>();
        for changed in 1..=free.len() {
            let mut nearest = None::<((usize, usize), NativeSpecialization)>;
            for moved in Subsets::new(free.len(), changed) {
                let others = moved
                    .iter()
                    .map(|position| {
                        let (address, values) = &free[*position];
                        let held = address.value(&fixed);
                        values
                            .iter()
                            .copied()
                            .filter(|value| *value != held)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for assignment in Assignments::of(others.iter().map(Vec::as_slice).collect()) {
                    let candidate = moved
                        .iter()
                        .zip(assignment)
                        .fold(fixed.clone(), |candidate, (position, value)| {
                            free[*position].0.set(candidate, value)
                        });
                    match implementation.validate(&candidate) {
                        Ok(()) => {
                            let distance = distance(&free, &fixed, &candidate);
                            if nearest.as_ref().is_none_or(|(least, _)| distance < *least) {
                                nearest = Some((distance, candidate));
                            }
                        }
                        Err(NativeSpecializationError::Inadmissible) => {}
                        Err(error) => return Err(PlanError::Declaration(error)),
                    }
                }
            }
            if let Some((_, nearest)) = nearest {
                return Ok(Some(nearest));
            }
        }
        Ok(None)
    }

    /// Every parameter the plan moves with its value in a configuration
    /// that has each group at its assignment in `groups` and the boundary
    /// at `boundary`: what a whole configuration the groups' choices
    /// generate is placed by ([`Domain::place`]), which remains the
    /// admissibility authority.
    pub fn assembled(
        &self,
        groups: &[Vec<u64>],
        boundary: &[u64],
    ) -> (Vec<ParameterAddress>, Vec<u64>) {
        self.boundary
            .iter()
            .cloned()
            .zip(boundary.iter().copied())
            .chain(self.groups.iter().zip(groups).flat_map(|(group, assignment)| {
                group.parameters.iter().cloned().zip(assignment.iter().copied())
            }))
            .unzip()
    }
}

/// The plan of an implementation whose source is formed whole: one group
/// of every launch and each of `parameters` (an entry parameter's name,
/// declared values and whether they select a form), and no boundary. Such
/// a source receives every entry parameter, so none is known to choose
/// only which launch serves which rows, and none to belong to one launch
/// alone: the whole program is what a change of any of them changes.
pub fn whole(
    parameters: &[(String, Vec<u64>, bool)],
    launches: usize,
    points: Vec<PlannedPoint>,
) -> TuningPartition {
    TuningPartition {
        groups: vec![Group {
            launches: (0..launches).collect(),
            parameters: parameters
                .iter()
                .map(|(name, _, _)| ParameterAddress::Entry(name.clone()))
                .collect(),
            values: parameters.iter().map(|(_, values, _)| values.clone()).collect(),
            forms: parameters.iter().map(|(_, _, form)| *form).collect(),
        }],
        boundary: Vec::new(),
        boundary_values: Vec::new(),
        boundary_forms: Vec::new(),
        points,
    }
}

/// One point as planned: the launches that can run there, each under some
/// assignment of the parameters its condition reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedPoint {
    pub label: String,
    pub active: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    Declaration(NativeSpecializationError),
    Choice,
    Evaluation {
        point: String,
        launch: usize,
        error: NativeEvalError,
    },
}

fn entry_address(parameter: &NativeParameter) -> ParameterAddress {
    ParameterAddress::Entry(parameter.name.clone())
}

fn local_address(launch: usize, parameter: &NativeParameter) -> ParameterAddress {
    ParameterAddress::Launch {
        ordinal: launch,
        name: parameter.name.clone(),
    }
}

fn root(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = root(parents, parents[index]);
    }
    parents[index]
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left = root(parents, left);
    let right = root(parents, right);
    parents[right] = left;
}

/// An `and` separates independent restrictions. Both arms of `or` must be
/// considered together because either arm can admit a combination that the
/// other excludes.
fn constraint_parts(condition: &NativeCondition, parts: &mut Vec<Vec<String>>) {
    match condition {
        NativeCondition::And(left, right) => {
            constraint_parts(left, parts);
            constraint_parts(right, parts);
        }
        other => {
            let mut names = Vec::new();
            other.parameters(&mut names);
            parts.push(names);
        }
    }
}

/// Launch-selection coordinates are boundaries even when selecting a form
/// changes scratch or constrains that form's local code parameters. Those
/// restrictions are conditioned on the boundary, not Cartesian group edges.
fn boundary_parameters(implementation: &NativeImplementation) -> Vec<ParameterAddress> {
    let mut conditions = Vec::new();
    let mut geometry = Vec::new();
    for launch in &implementation.launches {
        if let Some(condition) = &launch.when {
            condition.parameters(&mut conditions);
        }
        geometry.extend(launch.reads.iter().cloned());
        for expression in launch.groups.iter().chain(&launch.group_extent) {
            expression.parameters(&mut geometry);
        }
        launch.shared_bytes.parameters(&mut geometry);
    }
    if let Some(repeat) = &implementation.repeat {
        repeat.count.parameters(&mut geometry);
    }
    implementation
        .params
        .iter()
        .filter(|parameter| {
            conditions.contains(&parameter.name) && !geometry.contains(&parameter.name)
        })
        .map(entry_address)
        .collect()
}

/// Evaluate guards using the values `fixed` gives the entry parameters
/// among `pinned`, and its statics, while preserving unresolved local
/// restrictions. Short-circuiting removes inactive form constraints.
fn conditioned(
    condition: &NativeCondition,
    fixed: &NativeSpecialization,
    pinned: &[ParameterAddress],
) -> Result<NativeCondition, bool> {
    let parameter = |name: &str| {
        pinned.iter().find_map(|address| match address {
            ParameterAddress::Entry(key) if key == name => Some(address.value(fixed)),
            _ => None,
        })
    };
    match condition {
        NativeCondition::Compare { .. } => condition
            .holds(&|name| fixed.static_value(name), &parameter)
            .map_err(|_| ())
            .map_or_else(|_| Ok(condition.clone()), Err),
        NativeCondition::And(left, right) => match (
            conditioned(left, fixed, pinned),
            conditioned(right, fixed, pinned),
        ) {
            (Err(false), _) | (_, Err(false)) => Err(false),
            (Err(true), other) | (other, Err(true)) => other,
            (Ok(left), Ok(right)) => Ok(NativeCondition::And(Box::new(left), Box::new(right))),
        },
        NativeCondition::Or(left, right) => match (
            conditioned(left, fixed, pinned),
            conditioned(right, fixed, pinned),
        ) {
            (Err(true), _) | (_, Err(true)) => Err(true),
            (Err(false), other) | (other, Err(false)) => other,
            (Ok(left), Ok(right)) => Ok(NativeCondition::Or(Box::new(left), Box::new(right))),
        },
    }
}

/// `implementation` with every parameter that the restrictions violated at
/// `fixed` do not couple held at its value there. `parts` are the
/// independent restrictions of `where`: parameters one of them reads are
/// coupled, transitively. A restriction of the result reads either only
/// free parameters, and holds as it does in the whole domain, or only held
/// ones, and holds as it does at `fixed`.
fn restricted(
    implementation: &NativeImplementation,
    fixed: &NativeSpecialization,
    parts: &[Vec<String>],
) -> NativeImplementation {
    // The names restrictions couple to the violation. A name may be
    // several launches' parameters: a restriction couples them all.
    let mut reached = BTreeSet::<String>::new();
    if let Some(condition) = &implementation.constraint {
        let mut conjuncts = Vec::new();
        condition.conjuncts(&mut conjuncts);
        for conjunct in conjuncts {
            if !conjunct
                .holds(&|name| fixed.static_value(name), &|name| fixed.param(name))
                .unwrap_or(false)
            {
                let mut names = Vec::new();
                conjunct.parameters(&mut names);
                reached.extend(names);
            }
        }
    }
    loop {
        let before = reached.len();
        for part in parts {
            if part.iter().any(|name| reached.contains(name)) {
                reached.extend(part.iter().cloned());
            }
        }
        if reached.len() == before {
            break;
        }
    }
    let mut restricted = implementation.clone();
    for parameter in &mut restricted.params {
        let address = entry_address(parameter);
        if !reached.contains(address.name()) {
            parameter.values = vec![address.value(fixed)];
        }
    }
    for (ordinal, launch) in restricted.launches.iter_mut().enumerate() {
        for parameter in &mut launch.params {
            let address = local_address(ordinal, parameter);
            if !reached.contains(address.name()) {
                parameter.values = vec![address.value(fixed)];
            }
        }
    }
    restricted
}

/// The parameter a name in `where` is, as the declaration's validation
/// resolves it: the entry's, else the one launch's that declares it.
fn restricted_address(implementation: &NativeImplementation, name: &str) -> Option<ParameterAddress> {
    if implementation
        .params
        .iter()
        .any(|parameter| parameter.name == name)
    {
        return Some(ParameterAddress::Entry(name.to_owned()));
    }
    let mut owners = implementation
        .launches
        .iter()
        .enumerate()
        .filter(|(_, launch)| launch.params.iter().any(|parameter| parameter.name == name))
        .map(|(ordinal, _)| ordinal);
    let ordinal = owners.next()?;
    owners.next().is_none().then(|| ParameterAddress::Launch {
        ordinal,
        name: name.to_owned(),
    })
}

/// The declared values of the parameter at `address`.
fn declared_values<'i>(
    implementation: &'i NativeImplementation,
    address: &ParameterAddress,
) -> &'i [u64] {
    let parameters = match address {
        ParameterAddress::Entry(_) => &implementation.params,
        ParameterAddress::Launch { ordinal, .. } => &implementation.launches[*ordinal].params,
    };
    &parameters
        .iter()
        .find(|parameter| parameter.name == address.name())
        .expect("the address is of a declared parameter")
        .values
}

/// Partition the declared domain. The checked declaration gives launch-local
/// ownership; an entry parameter belongs to every launch whose geometry or
/// condition reads it. An otherwise unowned entry parameter conservatively
/// couples all launches unless it is a condition-only boundary parameter.
///
/// Which launches can run at a point, and which of them in one call, is
/// read from their conditions: a condition the point's dimensions decide is
/// decided; one that reads parameters holds if some assignment of the
/// parameters it reads satisfies it and is not excluded by the restrictions
/// `where` puts on those same parameters. Two launches are searched
/// together when their conditions can hold at one point at once. Only the
/// parameters conditions read are assigned for this, a launch or a pair at
/// a time; the domain's other parameters are never assigned at all.
pub fn partition(
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    points: &[PointShape],
) -> Result<TuningPartition, PlanError> {
    // The declaration is admissible somewhere, and its statics complete.
    implementation
        .default_specialization(statics)
        .map_err(PlanError::Declaration)?;
    let mut geometry_reads = Vec::with_capacity(implementation.launches.len());
    let mut condition_reads = Vec::with_capacity(implementation.launches.len());
    for launch in &implementation.launches {
        let mut geometry = launch.reads.clone();
        for expression in launch.groups.iter().chain(&launch.group_extent) {
            expression.parameters(&mut geometry);
        }
        launch.shared_bytes.parameters(&mut geometry);
        let mut condition = Vec::new();
        if let Some(when) = &launch.when {
            when.parameters(&mut condition);
        }
        geometry_reads.push(geometry);
        condition_reads.push(condition);
    }
    // How often a repeated launch is dispatched is part of its geometry.
    if let Some(repeat) = &implementation.repeat {
        for geometry in &mut geometry_reads[repeat.first..repeat.first + repeat.launches] {
            repeat.count.parameters(geometry);
        }
    }
    let boundary = boundary_parameters(implementation);
    let mut owners = BTreeMap::<ParameterAddress, Vec<usize>>::new();
    for parameter in &implementation.params {
        let address = entry_address(parameter);
        if boundary.contains(&address) {
            continue;
        }
        let mut reads = (0..implementation.launches.len())
            .filter(|&launch| {
                geometry_reads[launch].contains(&parameter.name)
                    || condition_reads[launch].contains(&parameter.name)
            })
            .collect::<Vec<_>>();
        if reads.is_empty() {
            reads = (0..implementation.launches.len()).collect();
        }
        owners.insert(address, reads);
    }
    for (launch, declaration) in implementation.launches.iter().enumerate() {
        for parameter in &declaration.params {
            owners.insert(local_address(launch, parameter), vec![launch]);
        }
    }
    let tuned = (0..implementation.launches.len())
        .filter(|launch| owners.values().any(|launches| launches.contains(launch)))
        .collect::<BTreeSet<_>>();
    let mut parents = (0..implementation.launches.len()).collect::<Vec<_>>();
    for launches in owners.values() {
        for pair in launches.windows(2) {
            union(&mut parents, pair[0], pair[1]);
        }
    }
    let mut conjuncts = Vec::new();
    if let Some(constraint) = &implementation.constraint {
        constraint.conjuncts(&mut conjuncts);
        let mut parts = Vec::new();
        constraint_parts(constraint, &mut parts);
        for names in parts {
            let launches = owners
                .iter()
                .filter(|(address, _)| names.iter().any(|name| name == address.name()))
                .flat_map(|(_, launches)| launches.iter().copied())
                .collect::<BTreeSet<_>>();
            let mut launches = launches.into_iter();
            if let Some(first) = launches.next() {
                for launch in launches {
                    union(&mut parents, first, launch);
                }
            }
        }
    }
    // A name a launch's condition reads resolves as its evaluation does:
    // the launch's own parameter first, then the entry's.
    let resolve = |ordinal: usize, name: &str| {
        let local = implementation.launches[ordinal]
            .params
            .iter()
            .any(|parameter| parameter.name == name);
        if local {
            Some(ParameterAddress::Launch {
                ordinal,
                name: name.to_owned(),
            })
        } else {
            implementation
                .params
                .iter()
                .any(|parameter| parameter.name == name)
                .then(|| ParameterAddress::Entry(name.to_owned()))
        }
    };
    // Whether the conditions of `launches` can hold together at `point`:
    // under some assignment of the parameters they read that `where` does
    // not exclude by those parameters alone.
    let together = |point: &PointShape, launches: &[usize]| -> Result<bool, PlanError> {
        let dimension = |name: &str| {
            point
                .dimensions
                .get(name)
                .copied()
                .or_else(|| statics.static_value(name))
        };
        let mut read = Vec::<ParameterAddress>::new();
        for &ordinal in launches {
            for address in condition_reads[ordinal]
                .iter()
                .filter_map(|name| resolve(ordinal, name))
            {
                if !read.contains(&address) {
                    read.push(address);
                }
            }
        }
        let mut undecided = None;
        for assignment in Assignments::of(
            read.iter()
                .map(|address| declared_values(implementation, address))
                .collect(),
        ) {
            let valued = |address: &ParameterAddress| {
                read.iter()
                    .position(|read| read == address)
                    .map(|position| assignment[position])
            };
            let excluded = conjuncts.iter().any(|conjunct| {
                conjunct.decided(&|name| statics.static_value(name), &|name| {
                    restricted_address(implementation, name)
                        .as_ref()
                        .and_then(valued)
                }) == Some(false)
            });
            if excluded {
                continue;
            }
            let mut holds = true;
            for &ordinal in launches {
                let Some(when) = &implementation.launches[ordinal].when else {
                    continue;
                };
                let parameter = |name: &str| resolve(ordinal, name).as_ref().and_then(valued);
                match when.holds(&dimension, &parameter) {
                    Ok(true) => {}
                    Ok(false) => {
                        holds = false;
                        break;
                    }
                    // This assignment cannot say; another may.
                    Err(error) => {
                        undecided.get_or_insert(PlanError::Evaluation {
                            point: point.label.clone(),
                            launch: ordinal,
                            error,
                        });
                        holds = false;
                        break;
                    }
                }
            }
            if holds {
                return Ok(true);
            }
        }
        undecided.map_or(Ok(false), Err)
    };
    let mut measured = BTreeSet::new();
    let mut planned = Vec::with_capacity(points.len());
    for point in points {
        let mut active = Vec::new();
        for ordinal in 0..implementation.launches.len() {
            if together(point, &[ordinal])? {
                active.push(ordinal);
                if tuned.contains(&ordinal) {
                    measured.insert(ordinal);
                }
            }
        }
        let tuned_active = active
            .iter()
            .copied()
            .filter(|ordinal| tuned.contains(ordinal))
            .collect::<Vec<_>>();
        for (position, &left) in tuned_active.iter().enumerate() {
            for &right in &tuned_active[position + 1..] {
                if root(&mut parents, left) != root(&mut parents, right)
                    && together(point, &[left, right])?
                {
                    union(&mut parents, left, right);
                }
            }
        }
        planned.push(PlannedPoint {
            label: point.label.clone(),
            active,
        });
    }
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for launch in tuned {
        components
            .entry(root(&mut parents, launch))
            .or_default()
            .push(launch);
    }
    // A bounded workload need not reach every declared launch. An entirely
    // unserved independent component keeps its declared default; there is no
    // measurement from which to choose a different configuration. Components
    // coupled to a served launch still participate in that launch's search.
    let mut groups = components
        .into_values()
        .filter(|launches| launches.iter().any(|launch| measured.contains(launch)))
        .map(|launches| {
            let parameters = owners
                .iter()
                .filter(|(_, owned)| owned.iter().any(|launch| launches.contains(launch)))
                .map(|(address, _)| address.clone())
                .collect::<Vec<_>>();
            Group {
                values: parameters
                    .iter()
                    .map(|address| declared_values(implementation, address).to_vec())
                    .collect(),
                forms: parameters
                    .iter()
                    .map(|address| match address {
                        ParameterAddress::Entry(name) => implementation
                            .params
                            .iter()
                            .any(|parameter| parameter.form && &parameter.name == name),
                        ParameterAddress::Launch { .. } => false,
                    })
                    .collect(),
                launches,
                parameters,
            }
        })
        .collect::<Vec<_>>();
    groups.sort_by_key(|group| group.launches[0]);
    Ok(TuningPartition {
        groups,
        boundary_values: boundary
            .iter()
            .map(|address| declared_values(implementation, address).to_vec())
            .collect(),
        boundary_forms: boundary
            .iter()
            .map(|address| {
                implementation
                    .params
                    .iter()
                    .any(|parameter| parameter.form && parameter.name == address.name())
            })
            .collect(),
        boundary,
        points: planned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{SourceFile, SourceSet, check_source};
    use seismic_lang::registry::BackendName;

    fn implementation(last_when: &str) -> NativeImplementation {
        checked(format!(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    launch gemv when N < BATCH_FROM:\n        params (SIMDGROUPS in [16, 8], code ROWS in [1, 2])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch batch when N >= BATCH_FROM and N <= 16:\n        params (SIMDGROUPS in [8, 4], code ROWS in [2, 1])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch gemm when {last_when}:\n        params (code TILE in [64, 128])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n"
        ))
    }

    fn points() -> Vec<PointShape> {
        [1, 4, 8, 16, 32]
            .into_iter()
            .map(|rows| PointShape {
                label: format!("m{rows}"),
                dimensions: BTreeMap::from([("N".into(), rows)]),
            })
            .collect()
    }

    fn checked(text: String) -> NativeImplementation {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text,
        }]))
        .unwrap();
        module
            .native_implementation(module.entries()[0].id, BackendName::Metal)
            .unwrap()
            .clone()
    }

    /// Structural forms whose launch parameters `where` holds at their
    /// defaults outside the form, as the projection entries declare them.
    fn forms() -> NativeImplementation {
        checked(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3], form TALL in [1, 0], form PACK in [0, 1])\n    where (PACK == 0 or TALL == 1) and (TALL == 1 or TILE == 64) and (PACK == 1 or (TOKENS == 4 and AHEAD == 0))\n    scratch staged bytes (N * 4 + TALL * 64) when N > 16\n    launch gemv when N < BATCH_FROM:\n        params (SIMDGROUPS in [16, 8, 4], code ROWS in [1, 2, 4])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch batch when N >= BATCH_FROM and N <= 16:\n        params (SIMDGROUPS in [8, 4], code ROWS in [2, 1])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch gemm when N > 16 and PACK == 0:\n        params (code TILE in [64, 128, 32])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n    launch packed when N > 16 and PACK == 1 and TALL == 1:\n        params (code TOKENS in [4, 2], AHEAD in [0, 1])\n        threadgroups (ceil_div(N, TOKENS), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n"
                .into(),
        )
    }

    fn constrained() -> NativeImplementation {
        checked("fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    where BATCH_FROM >= 3\n    launch gemv when N < BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch batch when N >= BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n".into())
    }

    /// The defaults with each group and the boundary as given, when the
    /// declaration admits that configuration.
    fn assemble(
        plan: &TuningPartition,
        implementation: &NativeImplementation,
        groups: &[Vec<u64>],
        boundary: &[u64],
    ) -> Option<NativeSpecialization> {
        let default = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let (parameters, values) = plan.assembled(groups, boundary);
        TuningPartition::place(implementation, &default, &parameters, &values).unwrap()
    }

    /// What the partition was taken from before it was derived: every
    /// admissible configuration, all of them produced. Per point, the sets
    /// of launches active under some configuration; and per set of
    /// parameters, their distinct admissible assignments in the order that
    /// set gave them (ascending).
    struct Materialised {
        admissible: Vec<NativeSpecialization>,
    }

    impl Materialised {
        fn of(implementation: &NativeImplementation) -> Self {
            Self {
                admissible: implementation
                    .admissible(&NativeSpecialization::new())
                    .unwrap(),
            }
        }

        fn active_sets(
            &self,
            implementation: &NativeImplementation,
            point: &PointShape,
        ) -> BTreeSet<Vec<usize>> {
            self.admissible
                .iter()
                .map(|configuration| {
                    (0..implementation.launches.len())
                        .filter(|&ordinal| {
                            implementation.launches[ordinal]
                                .when
                                .as_ref()
                                .is_none_or(|when| {
                                    when.holds(&|name| point.dimensions.get(name).copied(), &|name| {
                                        configuration
                                            .launch_param(ordinal, name)
                                            .or_else(|| configuration.param(name))
                                    })
                                    .unwrap()
                                })
                        })
                        .collect()
                })
                .collect()
        }

        fn candidates(&self, parameters: &[ParameterAddress]) -> Vec<Vec<u64>> {
            self.admissible
                .iter()
                .map(|configuration| {
                    parameters
                        .iter()
                        .map(|address| address.value(configuration))
                        .collect::<Vec<_>>()
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        }
    }

    /// The nearby changes of each coordinate as the materialised candidates
    /// gave them: for each coordinate, the nearest lower and higher value
    /// any candidate takes, at the candidate with the fewest other changes
    /// from `seed` (the first of equals).
    fn materialised_starts(candidates: &[Vec<u64>], seed: &[u64]) -> Vec<Vec<u64>> {
        let mut starts = Vec::<Vec<u64>>::new();
        for (coordinate, &default) in seed.iter().enumerate() {
            let lower = candidates
                .iter()
                .map(|values| values[coordinate])
                .filter(|&value| value < default)
                .max();
            let upper = candidates
                .iter()
                .map(|values| values[coordinate])
                .filter(|&value| value > default)
                .min();
            for value in [lower, upper].into_iter().flatten() {
                let candidate = candidates
                    .iter()
                    .filter(|values| values[coordinate] == value)
                    .min_by_key(|values| {
                        values
                            .iter()
                            .zip(seed)
                            .filter(|(value, default)| value != default)
                            .count()
                    })
                    .unwrap()
                    .clone();
                if !starts.contains(&candidate) {
                    starts.push(candidate);
                }
            }
        }
        starts
    }

    #[test]
    fn boundary_moves_points_without_coupling_launch_searches() {
        let implementation = implementation("N > 16");
        let plan = partition(&implementation, &NativeSpecialization::new(), &points()).unwrap();
        assert_eq!(
            plan.boundary,
            [ParameterAddress::Entry("BATCH_FROM".into())]
        );
        assert_eq!(plan.boundary_values, [vec![5, 3]]);
        assert_eq!(plan.boundary_forms, [false]);
        assert_eq!(
            plan.groups
                .iter()
                .map(|group| group.launches.as_slice())
                .collect::<Vec<_>>(),
            [&[0][..], &[1][..], &[2][..]]
        );
        assert_eq!(
            plan.groups
                .iter()
                .map(Group::declared)
                .collect::<Vec<_>>(),
            [4, 4, 2]
        );
        // Four rows are the first launch's under one boundary and the
        // second's under the other.
        assert_eq!(plan.points[1].active, [0, 1]);
        assert_eq!(plan.points[4].active, [2]);
        let defaults = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        assert_eq!(code_values(&implementation, &defaults, 0), [1]);
        assert_eq!(code_values(&implementation, &defaults, 1), [2]);
        assert_eq!(code_values(&implementation, &defaults, 2), [64]);
        // ROWS before SIMDGROUPS: a group's parameters are in address order.
        let assembled = assemble(
            &plan,
            &implementation,
            &[vec![1, 8], vec![2, 4], vec![64]],
            &[3],
        )
        .unwrap();
        assert_eq!(assembled.param("BATCH_FROM"), Some(3));
        assert_eq!(assembled.launch_param(0, "SIMDGROUPS"), Some(8));
        assert_eq!(assembled.launch_param(1, "SIMDGROUPS"), Some(4));
    }

    #[test]
    fn guarded_boundary_completes_with_a_legal_local_witness() {
        let mut implementation = forms();
        let constraint = implementation.constraint.take().unwrap();
        implementation.constraint = Some(NativeCondition::And(
            Box::new(constraint),
            Box::new(NativeCondition::Or(
                Box::new(NativeCondition::Compare {
                    left: seismic_lang::checked::NativeNatExpr::Parameter("PACK".into()),
                    comparison: seismic_lang::checked::NativeComparison::Eq,
                    right: seismic_lang::checked::NativeNatExpr::Constant(0),
                }),
                Box::new(NativeCondition::Compare {
                    left: seismic_lang::checked::NativeNatExpr::Parameter("TILE".into()),
                    comparison: seismic_lang::checked::NativeComparison::Eq,
                    right: seismic_lang::checked::NativeNatExpr::Constant(128),
                }),
            )),
        ));
        let statics = NativeSpecialization::new();
        let default = implementation.default_specialization(&statics).unwrap();
        let pack = [ParameterAddress::Entry("PACK".into())];
        // The defaults with the boundary moved are not admissible: the
        // packed form needs the other tile.
        assert_eq!(
            TuningPartition::place(&implementation, &default, &pack, &[1]),
            Ok(None)
        );
        let witness = TuningPartition::nearest(&implementation, &default, &pack, &[1])
            .unwrap()
            .unwrap();
        assert_eq!(witness.param("PACK"), Some(1));
        assert_eq!(witness.launch_param(2, "TILE"), Some(128));
        // Nothing the violated restriction does not couple moved.
        assert_eq!(witness.param("BATCH_FROM"), default.param("BATCH_FROM"));
        assert_eq!(
            witness.launch_param(0, "ROWS"),
            default.launch_param(0, "ROWS")
        );
        assert_eq!(
            witness.launch_param(1, "SIMDGROUPS"),
            default.launch_param(1, "SIMDGROUPS")
        );
        // A change no values of the coupled parameters satisfy has no
        // witness: the packed form does not exist without the tall one.
        assert_eq!(
            TuningPartition::nearest(
                &implementation,
                &default,
                &[
                    ParameterAddress::Entry("PACK".into()),
                    ParameterAddress::Entry("TALL".into()),
                ],
                &[1, 0],
            ),
            Ok(None)
        );
    }

    /// Nearest is the fewest other parameters changed, then the fewest
    /// steps along their values, then the earliest declared: a change that
    /// needs one of two parameters moved takes the one a step away, and
    /// from a base that is not the defaults it stays by the base.
    #[test]
    fn the_nearest_completion_is_defined_by_what_it_changes() {
        let implementation = checked(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (form WIDE in [0, 1], A in [1, 2, 4, 8], B in [1, 2, 4, 8], C in [1, 2])\n    where WIDE == 0 or A >= 4 or B >= 2\n    launch gemv:\n        threadgroups (ceil_div(N, A * B * C + WIDE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"
                .into(),
        );
        let default = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let wide = [ParameterAddress::Entry("WIDE".into())];
        let values = |specialization: &NativeSpecialization| {
            ["WIDE", "A", "B", "C"].map(|name| specialization.param(name).unwrap())
        };
        // One parameter must give way: B one step (to 2), not A two (to 4),
        // and never C, which no restriction couples to the change.
        let nearest = TuningPartition::nearest(&implementation, &default, &wide, &[1])
            .unwrap()
            .unwrap();
        assert_eq!(values(&nearest), [1, 1, 2, 1]);
        // From a base two steps along A and at the other C, the form is
        // admissible as the base stands: nothing else moves.
        let base = default.clone().with_param("A", 4).with_param("C", 2);
        let nearest = TuningPartition::nearest(&implementation, &base, &wide, &[1])
            .unwrap()
            .unwrap();
        assert_eq!(values(&nearest), [1, 4, 1, 2]);
        // Equally near completions: the earlier declared parameter moves.
        let tied = checked(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (form WIDE in [0, 1], A in [1, 2], B in [1, 2])\n    where WIDE == 0 or A == 2 or B == 2\n    launch gemv:\n        threadgroups (ceil_div(N, A * B + WIDE), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"
                .into(),
        );
        let default = tied
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let nearest = TuningPartition::nearest(&tied, &default, &wide, &[1])
            .unwrap()
            .unwrap();
        assert_eq!(
            ["WIDE", "A", "B"].map(|name| nearest.param(name).unwrap()),
            [1, 2, 1]
        );
    }

    /// The derived partition is the one every admissible configuration
    /// gives: a launch can run at a point exactly when some admissible
    /// configuration runs it there, launches some configuration runs
    /// together at a point are searched together, and a group's value lists
    /// and restriction admit exactly the assignments the configurations
    /// take, so that the nearby changes the search asks a group for are
    /// the ones the materialised candidates gave, in the same order.
    #[test]
    fn the_derived_partition_is_the_one_every_admissible_configuration_gives() {
        for implementation in [
            implementation("N > 16"),
            implementation("N > 64"),
            constrained(),
            forms(),
        ] {
            let statics = NativeSpecialization::new();
            let default = implementation.default_specialization(&statics).unwrap();
            let materialised = Materialised::of(&implementation);
            // Every served range, and ranges that leave launches unserved.
            for served in [&points()[..], &points()[..2], &points()[3..]] {
                let plan = partition(&implementation, &statics, served).unwrap();
                let group_of = |launch: usize| {
                    plan.groups
                        .iter()
                        .position(|group| group.launches.contains(&launch))
                };
                for (point, planned) in served.iter().zip(&plan.points) {
                    let sets = materialised.active_sets(&implementation, point);
                    assert_eq!(
                        planned.active,
                        sets.iter()
                            .flatten()
                            .copied()
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>(),
                        "{}",
                        point.label
                    );
                    for set in &sets {
                        let groups = set
                            .iter()
                            .filter_map(|launch| group_of(*launch))
                            .collect::<BTreeSet<_>>();
                        assert!(
                            groups.len() <= 1,
                            "launches {set:?} run together at {} and are searched apart",
                            point.label
                        );
                    }
                }
                for group in &plan.groups {
                    let candidates = materialised.candidates(&group.parameters);
                    // The value lists hold what the configurations take.
                    let produced = Assignments::of(
                        group.values.iter().map(Vec::as_slice).collect(),
                    )
                    .collect::<BTreeSet<_>>();
                    assert!(candidates.iter().all(|candidate| produced.contains(candidate)));
                    // The nearby change of each coordinate, as asked of
                    // the group and as the candidates gave it.
                    let seed = group.assignment(&default);
                    let mut asked = Vec::<Vec<u64>>::new();
                    for coordinate in 0..group.parameters.len() {
                        for ladder in group.steps(&seed, coordinate) {
                            let start = ladder.into_iter().find_map(|assignment| {
                                TuningPartition::nearest(
                                    &implementation,
                                    &default,
                                    &group.parameters[coordinate..=coordinate],
                                    &assignment[coordinate..=coordinate],
                                )
                                .unwrap()
                                .map(|admitted| group.assignment(&admitted))
                            });
                            if let Some(start) = start.filter(|start| !asked.contains(start)) {
                                asked.push(start);
                            }
                        }
                    }
                    assert_eq!(asked, materialised_starts(&candidates, &seed));
                }
            }
        }
        // Selection guards do not couple mutually exclusive code groups.
        let plan = partition(&forms(), &NativeSpecialization::new(), &points()).unwrap();
        let tall = plan
            .groups
            .iter()
            .find(|group| group.launches.contains(&2))
            .unwrap();
        assert_eq!(tall.launches, [2]);
        assert_eq!(tall.declared(), 3);
        let packed = plan
            .groups
            .iter()
            .find(|group| group.launches.contains(&3))
            .unwrap();
        assert_eq!(packed.launches, [3]);
        assert_eq!(packed.declared(), 4);
        assert_eq!(plan.groups.len(), 4);
    }

    #[test]
    fn independent_launches_are_partitioned_without_walking_their_product() {
        // Twelve launches, each serving one row count, sixteen candidates
        // each: 16^12 admissible configurations.
        let launches = (0..12)
            .map(|launch| {
                format!(
                    "    launch band{launch} when N == {}:\n        params (WIDTH in [1, 2, 4, 8], code ROWS in [1, 2, 4, 8])\n        threadgroups (ceil_div(N, WIDTH * ROWS), 1, 1)\n        threads_per_threadgroup (WIDTH * 32, 1, 1)\n",
                    launch + 1
                )
            })
            .collect::<String>();
        let implementation = checked(format!(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n{launches}"
        ));
        let points = (1..=12)
            .map(|rows| PointShape {
                label: format!("m{rows}"),
                dimensions: BTreeMap::from([("N".into(), rows)]),
            })
            .collect::<Vec<_>>();
        let plan = partition(&implementation, &NativeSpecialization::new(), &points).unwrap();
        assert_eq!(plan.groups.len(), 12);
        assert!(plan.groups.iter().all(|group| group.declared() == 16));
        assert!(
            plan.points
                .iter()
                .enumerate()
                .all(|(point, planned)| planned.active == [point])
        );
    }

    /// A group of 2^20 assignments is its twenty parameters and their forty
    /// values: planning produces no assignment, and the search is given the
    /// twenty one step from where it stands, whatever the group declares.
    #[test]
    fn a_group_of_a_million_assignments_is_its_parameters_and_their_values() {
        let coupled = |parameters: usize| {
            let declared = (0..parameters)
                .map(|parameter| format!("P{parameter} in [1, 2]"))
                .collect::<Vec<_>>()
                .join(", ");
            let product = (0..parameters)
                .map(|parameter| format!("P{parameter}"))
                .collect::<Vec<_>>()
                .join(" * ");
            checked(format!(
                "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    launch gemv:\n        params ({declared})\n        threadgroups (ceil_div(N, {product}), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"
            ))
        };
        let statics = NativeSpecialization::new();
        let plan = partition(&coupled(20), &statics, &points()).unwrap();
        let [group] = &plan.groups[..] else {
            panic!("one launch is one group");
        };
        assert_eq!(group.declared(), 1 << 20);
        // What the plan holds of the group is linear in its declaration.
        assert_eq!(group.parameters.len(), 20);
        assert_eq!(group.values.iter().map(Vec::len).sum::<usize>(), 40);
        let seed = vec![1; 20];
        assert_eq!(group.neighbours(&seed).count(), 20);
        assert_eq!(group.form_starts(&seed).count(), 0);
        // The same parameters as one program formed whole are one group of
        // the same size, held the same way; were all twenty forms, the
        // other forms asked for would be twenty, one parameter away each.
        let entries = (0..20)
            .map(|parameter| (format!("P{parameter}"), vec![1, 2], true))
            .collect::<Vec<_>>();
        let program = whole(&entries, 1, plan.points.clone());
        let [group] = &program.groups[..] else {
            panic!("a whole program is one group");
        };
        assert_eq!(group.declared(), 1 << 20);
        assert!(program.boundary.is_empty());
        assert_eq!(group.form_starts(&seed).count(), 20);
        assert_eq!(group.neighbours(&seed).count(), 0);
        // The same plan at half the parameters is the same but for their
        // number: nothing in it grew with the product.
        let smaller = partition(&coupled(10), &statics, &points()).unwrap();
        assert_eq!(smaller.groups[0].declared(), 1 << 10);
        assert_eq!(smaller.groups[0].neighbours(&[1; 10]).count(), 10);
        assert_eq!(smaller.points, plan.points);
    }

    #[test]
    fn unserved_independent_launch_keeps_its_default() {
        let implementation = implementation("N > 64");
        let plan = partition(&implementation, &NativeSpecialization::new(), &points()).unwrap();
        assert_eq!(
            plan.groups
                .iter()
                .map(|group| group.launches.as_slice())
                .collect::<Vec<_>>(),
            [&[0][..], &[1][..]]
        );
        let selected = assemble(&plan, &implementation, &[vec![1, 16], vec![2, 8]], &[5]).unwrap();
        assert_eq!(selected.launch_param(2, "TILE"), Some(64));
    }

    #[test]
    fn conditional_scratch_preserves_independent_row_bands() {
        let implementation = checked("fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    scratch staged bytes (N * 4) when N >= BATCH_FROM\n    launch gemv when N < BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch batch when N >= BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n".into());
        let plan = partition(&implementation, &NativeSpecialization::new(), &points()).unwrap();
        assert_eq!(
            plan.boundary,
            [ParameterAddress::Entry("BATCH_FROM".into())]
        );
        assert_eq!(plan.groups.len(), 2);
        assert_eq!(plan.groups[0].declared(), 2);
        assert_eq!(plan.groups[1].declared(), 2);
    }

    #[test]
    fn a_constrained_boundary_is_conditioned_per_assignment() {
        // A launch-selection coordinate stays a boundary when `where` also
        // constrains it: an inadmissible boundary value is never assembled,
        // no launch is taken to run where only that value would run it, and
        // the launches keep independent groups.
        let text = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    where BATCH_FROM >= 4\n    launch gemv when N < BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch batch when N >= BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let implementation = checked(text.into());
        let statics = NativeSpecialization::new();
        let plan = partition(&implementation, &statics, &points()).unwrap();
        assert_eq!(
            plan.boundary,
            [ParameterAddress::Entry("BATCH_FROM".into())]
        );
        assert_eq!(
            plan.groups
                .iter()
                .map(|group| group.launches.as_slice())
                .collect::<Vec<_>>(),
            [&[0][..], &[1][..]]
        );
        // Four rows are the second launch's only under the excluded value.
        assert_eq!(plan.points[1].active, [0]);
        let first = vec![vec![1]; plan.groups.len()];
        assert_eq!(
            assemble(&plan, &implementation, &first, &[5])
                .unwrap()
                .param("BATCH_FROM"),
            Some(5)
        );
        assert_eq!(assemble(&plan, &implementation, &first, &[3]), None);
        let default = implementation.default_specialization(&statics).unwrap();
        assert_eq!(
            TuningPartition::nearest(&implementation, &default, &plan.boundary, &[3]),
            Ok(None)
        );
    }
}
