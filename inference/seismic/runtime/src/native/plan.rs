//! Pure partition of a native tuning domain by launch ownership and activity.
//! Formation and measurement consume this plan; neither happens here.

use seismic_lang::checked::{
    NativeCondition, NativeEvalError, NativeImplementation, NativeParameter, NativeSpecialization,
    NativeSpecializationError,
};
use std::collections::{BTreeMap, BTreeSet};

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

    fn set(&self, specialization: NativeSpecialization, value: u64) -> NativeSpecialization {
        match self {
            Self::Entry(name) => specialization.with_param(name.clone(), value),
            Self::Launch { ordinal, name } => {
                specialization.with_launch_param(*ordinal, name.clone(), value)
            }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub launches: Vec<usize>,
    pub parameters: Vec<ParameterAddress>,
    /// Distinct admissible assignments of this group's parameters.
    pub candidates: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchSource {
    pub ordinal: usize,
    /// Distinct instances of this launch across the admissible
    /// configurations.
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningPartition {
    pub groups: Vec<Group>,
    /// Entry parameters used solely to choose a launch's active row band.
    pub boundary: Vec<ParameterAddress>,
    pub points: Vec<PlannedPoint>,
    pub sources: Vec<LaunchSource>,
}

impl TuningPartition {
    /// Combine one candidate per independent group with a boundary choice.
    /// The checked declaration remains the final admissibility authority.
    pub fn assemble(
        &self,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        groups: &[usize],
        boundary: &[u64],
    ) -> Result<NativeSpecialization, PlanError> {
        if groups.len() != self.groups.len() || boundary.len() != self.boundary.len() {
            return Err(PlanError::Choice);
        }
        let mut result = implementation
            .default_specialization(statics)
            .map_err(PlanError::Declaration)?;
        for (address, value) in self.boundary.iter().zip(boundary) {
            result = address.set(result, *value);
        }
        for (group, choice) in self.groups.iter().zip(groups) {
            let candidate = group.candidates.get(*choice).ok_or(PlanError::Choice)?;
            for (address, value) in group.parameters.iter().zip(candidate) {
                result = address.set(result, *value);
            }
        }
        implementation
            .validate(&result)
            .map_err(PlanError::Declaration)?;
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedPoint {
    pub label: String,
    pub active_sets: Vec<Vec<usize>>,
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

/// Visits admissible configurations that between them take every distinct
/// admissible assignment of the addresses given.
type Projection<'v> = dyn Fn(
        &[ParameterAddress],
        &mut dyn FnMut(&NativeSpecialization) -> Result<(), PlanError>,
    ) -> Result<(), PlanError>
    + 'v;

/// `implementation` with every parameter that `selected` neither names nor
/// is coupled to held at its value in `default`, an admissible
/// configuration. `parts` are the independent restrictions of `where`:
/// parameters one of them reads are coupled, transitively. The admissible
/// configurations of the result take exactly the admissible assignments of
/// `selected`: a restriction reads either only free parameters, and holds
/// as it does in the whole domain, or only held ones, and holds as it does
/// in `default`.
fn restricted(
    implementation: &NativeImplementation,
    default: &NativeSpecialization,
    parts: &[Vec<String>],
    selected: &[ParameterAddress],
) -> NativeImplementation {
    let name = |address: &ParameterAddress| match address {
        ParameterAddress::Entry(name) | ParameterAddress::Launch { name, .. } => name.clone(),
    };
    let named = selected.iter().map(name).collect::<BTreeSet<_>>();
    // The names restrictions couple to the selected parameters. A name may
    // be several launches' parameters: a restriction couples them all.
    let mut reached = BTreeSet::<String>::new();
    loop {
        let before = reached.len();
        for part in parts {
            if part
                .iter()
                .any(|name| named.contains(name) || reached.contains(name))
            {
                reached.extend(part.iter().cloned());
            }
        }
        if reached.len() == before {
            break;
        }
    }
    let free = |address: &ParameterAddress| {
        selected.contains(address) || reached.contains(&name(address))
    };
    let mut restricted = implementation.clone();
    for parameter in &mut restricted.params {
        let address = entry_address(parameter);
        if !free(&address) {
            parameter.values = vec![address.value(default)];
        }
    }
    for (ordinal, launch) in restricted.launches.iter_mut().enumerate() {
        for parameter in &mut launch.params {
            let address = local_address(ordinal, parameter);
            if !free(&address) {
                parameter.values = vec![address.value(default)];
            }
        }
    }
    restricted
}

/// Partition the declared domain. The checked declaration gives launch-local
/// ownership; an entry parameter belongs to every launch whose geometry or
/// condition reads it. An otherwise unowned entry parameter conservatively
/// couples all launches unless it is a condition-only boundary parameter.
///
/// The domain is never enumerated: its size is the product of its
/// parameters' value counts, and every form or launch parameter a
/// declaration gains multiplies it. What the partition needs of it are
/// distinct assignments of a few parameters at a time (those the launch
/// conditions read, each group's, each launch's code), and each is walked
/// with every parameter it is not coupled to held at its default
/// ([`restricted`]): time and memory follow the sizes of the groups, not
/// their product.
pub fn partition(
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    points: &[PointShape],
) -> Result<TuningPartition, PlanError> {
    let default = implementation
        .default_specialization(statics)
        .map_err(PlanError::Declaration)?;
    let mut parts = Vec::new();
    if let Some(constraint) = &implementation.constraint {
        constraint_parts(constraint, &mut parts);
    }
    partition_over(implementation, points, &|selected, visit| {
        let mut failure = None;
        restricted(implementation, &default, &parts, selected)
            .walk_admissible(statics, |configuration| match visit(configuration) {
                Ok(()) => true,
                Err(error) => {
                    failure = Some(error);
                    false
                }
            })
            .map_err(PlanError::Declaration)?;
        failure.map_or(Ok(()), Err)
    })
}

/// [`partition`] over the admissible configurations `visit` supplies.
fn partition_over(
    implementation: &NativeImplementation,
    points: &[PointShape],
    visit: &Projection<'_>,
) -> Result<TuningPartition, PlanError> {
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
    let mut scratch_reads = Vec::new();
    for scratch in &implementation.scratch {
        scratch.bytes.parameters(&mut scratch_reads);
        if let Some(when) = &scratch.when {
            when.parameters(&mut scratch_reads);
        }
    }
    let mut constraint_reads = Vec::new();
    if let Some(constraint) = &implementation.constraint {
        constraint.parameters(&mut constraint_reads);
    }
    let boundary = implementation
        .params
        .iter()
        .filter(|parameter| {
            let name = &parameter.name;
            condition_reads.iter().any(|reads| reads.contains(name))
                && !geometry_reads.iter().any(|reads| reads.contains(name))
                && !scratch_reads.contains(name)
                && !constraint_reads.contains(name)
        })
        .map(entry_address)
        .collect::<Vec<_>>();
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
    if let Some(constraint) = &implementation.constraint {
        let mut parts = Vec::new();
        constraint_parts(constraint, &mut parts);
        for names in parts {
            let launches = owners
                .iter()
                .filter(|(address, _)| match address {
                    ParameterAddress::Entry(name) | ParameterAddress::Launch { name, .. } => {
                        names.contains(name)
                    }
                })
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
    let addresses = implementation
        .params
        .iter()
        .map(entry_address)
        .chain(
            implementation
                .launches
                .iter()
                .enumerate()
                .flat_map(|(launch, declaration)| {
                    declaration
                        .params
                        .iter()
                        .map(move |parameter| local_address(launch, parameter))
                }),
        )
        .collect::<Vec<_>>();
    // A name a launch reads resolves as its evaluation does: the launch's
    // own parameter first, then the entry's.
    let resolve = |ordinal: usize, name: &str| {
        [
            ParameterAddress::Launch {
                ordinal,
                name: name.to_owned(),
            },
            ParameterAddress::Entry(name.to_owned()),
        ]
        .into_iter()
        .find(|address| addresses.contains(address))
    };
    let mut condition_addresses = Vec::new();
    for (ordinal, names) in condition_reads.iter().enumerate() {
        for address in names.iter().filter_map(|name| resolve(ordinal, name)) {
            if !condition_addresses.contains(&address) {
                condition_addresses.push(address);
            }
        }
    }
    let mut measured = BTreeSet::new();
    let mut point_sets = vec![BTreeSet::<Vec<usize>>::new(); points.len()];
    // A point's active launches depend only on the values the launch
    // conditions read: each distinct assignment of those is decided once.
    let mut decided = vec![BTreeSet::<Vec<u64>>::new(); points.len()];
    visit(&condition_addresses, &mut |specialization| {
        let read = condition_addresses
            .iter()
            .map(|address| address.value(specialization))
            .collect::<Vec<_>>();
        for (index, point) in points.iter().enumerate() {
            if !decided[index].insert(read.clone()) {
                continue;
            }
            let dimension = |name: &str| {
                point
                    .dimensions
                    .get(name)
                    .copied()
                    .or_else(|| specialization.static_value(name))
            };
            let mut active = Vec::new();
            let mut tuned_active = Vec::new();
            for (ordinal, launch) in implementation.launches.iter().enumerate() {
                let parameter = |name: &str| {
                    specialization
                        .launch_param(ordinal, name)
                        .or_else(|| specialization.param(name))
                };
                let holds = launch
                    .when
                    .as_ref()
                    .map(|condition| condition.holds(&dimension, &parameter))
                    .transpose()
                    .map_err(|error| PlanError::Evaluation {
                        point: point.label.clone(),
                        launch: ordinal,
                        error,
                    })?
                    .unwrap_or(true);
                if holds {
                    active.push(ordinal);
                    if tuned.contains(&ordinal) {
                        measured.insert(ordinal);
                        tuned_active.push(ordinal);
                    }
                }
            }
            for pair in tuned_active.windows(2) {
                union(&mut parents, pair[0], pair[1]);
            }
            point_sets[index].insert(active);
        }
        Ok(())
    })?;
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
            let mut candidates = BTreeSet::new();
            visit(&parameters, &mut |specialization| {
                candidates.insert(
                    parameters
                        .iter()
                        .map(|address| address.value(specialization))
                        .collect::<Vec<_>>(),
                );
                Ok(())
            })?;
            Ok(Group {
                launches,
                parameters,
                candidates: candidates.into_iter().collect(),
            })
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    groups.sort_by_key(|group| group.launches[0]);
    let sources = implementation
        .launches
        .iter()
        .enumerate()
        .map(|(ordinal, launch)| {
            let code = implementation
                .params
                .iter()
                .filter(|parameter| {
                    parameter.code
                        && owners
                            .get(&entry_address(parameter))
                            .is_some_and(|reads| reads.contains(&ordinal))
                })
                .map(entry_address)
                .chain(
                    launch
                        .params
                        .iter()
                        .filter(|parameter| parameter.code)
                        .map(|parameter| local_address(ordinal, parameter)),
                )
                .collect::<Vec<_>>();
            // A variant is its code values and its group size, which reads
            // the parameters the launch's group extents name.
            let mut read = code.clone();
            let mut extent = Vec::new();
            for expression in &launch.group_extent {
                expression.parameters(&mut extent);
            }
            for address in extent.iter().filter_map(|name| resolve(ordinal, name)) {
                if !read.contains(&address) {
                    read.push(address);
                }
            }
            let mut variants = BTreeSet::new();
            visit(&read, &mut |specialization| {
                variants.insert(LaunchVariant {
                    code: code
                        .iter()
                        .map(|address| address.value(specialization))
                        .collect(),
                    group_size: implementation.static_group_size(specialization, ordinal),
                });
                Ok(())
            })?;
            Ok(LaunchSource {
                ordinal,
                variants: variants.into_iter().collect(),
            })
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    let points = points
        .iter()
        .zip(point_sets)
        .map(|(point, active_sets)| PlannedPoint {
            label: point.label.clone(),
            active_sets: active_sets.into_iter().collect(),
        })
        .collect();
    Ok(TuningPartition {
        groups,
        boundary,
        points,
        sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::registry::BackendName;

    fn implementation(last_when: &str) -> NativeImplementation {
        let text = format!(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    launch gemv when N < BATCH_FROM:\n        params (SIMDGROUPS in [16, 8], code ROWS in [1, 2])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch batch when N >= BATCH_FROM and N <= 16:\n        params (SIMDGROUPS in [8, 4], code ROWS in [2, 1])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch gemm when {last_when}:\n        params (code TILE in [64, 128])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n"
        );
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

    fn points() -> Vec<PointShape> {
        [1, 4, 8, 16, 32]
            .into_iter()
            .map(|rows| PointShape {
                label: format!("m{rows}"),
                dimensions: BTreeMap::from([("N".into(), rows)]),
            })
            .collect()
    }

    #[test]
    fn boundary_moves_points_without_coupling_launch_searches() {
        let plan = partition(
            &implementation("N > 16"),
            &NativeSpecialization::new(),
            &points(),
        )
        .unwrap();
        assert_eq!(
            plan.boundary,
            [ParameterAddress::Entry("BATCH_FROM".into())]
        );
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
                .map(|group| group.candidates.len())
                .collect::<Vec<_>>(),
            [4, 4, 2]
        );
        assert_eq!(
            plan.sources
                .iter()
                .map(|source| source.variants.len())
                .collect::<Vec<_>>(),
            // Two code instances each; the first two launches' group sizes
            // read a launch parameter, so each instance has two sizes.
            [4, 4, 2]
        );
        assert_eq!(plan.points[1].active_sets, [vec![0], vec![1]]);
        let implementation = implementation("N > 16");
        let defaults = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        assert_eq!(code_values(&implementation, &defaults, 0), [1]);
        assert_eq!(code_values(&implementation, &defaults, 1), [2]);
        assert_eq!(code_values(&implementation, &defaults, 2), [64]);
        let assembled = plan
            .assemble(
                &implementation,
                &NativeSpecialization::new(),
                &[0, 0, 0],
                &[3],
            )
            .unwrap();
        assert_eq!(assembled.param("BATCH_FROM"), Some(3));
        assert_eq!(assembled.launch_param(0, "SIMDGROUPS"), Some(8));
        assert_eq!(assembled.launch_param(1, "SIMDGROUPS"), Some(4));
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

    /// The partition taken from every admissible configuration: what
    /// [`partition`] must equal without enumerating them.
    fn enumerated(implementation: &NativeImplementation, points: &[PointShape]) -> TuningPartition {
        let admissible = implementation
            .admissible(&NativeSpecialization::new())
            .unwrap();
        partition_over(implementation, points, &|_, visit| {
            admissible.iter().try_for_each(|configuration| visit(configuration))
        })
        .unwrap()
    }

    /// Structural forms whose launch parameters `where` holds at their
    /// defaults outside the form, as the projection entries declare them.
    fn forms() -> NativeImplementation {
        checked(
            "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3], form TALL in [1, 0], form PACK in [0, 1])\n    where (PACK == 0 or TALL == 1) and (TALL == 1 or TILE == 64) and (PACK == 1 or (TOKENS == 4 and AHEAD == 0))\n    launch gemv when N < BATCH_FROM:\n        params (SIMDGROUPS in [16, 8, 4], code ROWS in [1, 2, 4])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch batch when N >= BATCH_FROM and N <= 16:\n        params (SIMDGROUPS in [8, 4], code ROWS in [2, 1])\n        threadgroups (ceil_div(N, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch gemm when N > 16 and PACK == 0:\n        params (code TILE in [64, 128, 32])\n        threadgroups (ceil_div(N, TILE), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n    launch packed when N > 16 and PACK == 1 and TALL == 1:\n        params (code TOKENS in [4, 2], AHEAD in [0, 1])\n        threadgroups (ceil_div(N, TOKENS), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n"
                .into(),
        )
    }

    #[test]
    fn the_partition_equals_the_one_taken_from_every_admissible_configuration() {
        let constrained = checked("fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    where BATCH_FROM >= 3\n    launch gemv when N < BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch batch when N >= BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n".into());
        for implementation in [
            implementation("N > 16"),
            implementation("N > 64"),
            constrained,
            forms(),
        ] {
            // Every served range, and ranges that leave launches unserved.
            for served in [&points()[..], &points()[..2], &points()[3..]] {
                let plan =
                    partition(&implementation, &NativeSpecialization::new(), served).unwrap();
                assert_eq!(plan, enumerated(&implementation, served));
            }
        }
        // The forms couple their launches through `where`: one group holds
        // the tall launches, with the candidates of each form and not their
        // product.
        let plan = partition(&forms(), &NativeSpecialization::new(), &points()).unwrap();
        let tall = plan
            .groups
            .iter()
            .find(|group| group.launches.contains(&2))
            .unwrap();
        assert_eq!(tall.launches, [2, 3]);
        // TALL 0 with the default tile; TALL 1 unpacked with three tiles;
        // packed with three tiles and four token and lookahead choices.
        assert_eq!(tall.candidates.len(), 1 + 3 + 12);
        assert_eq!(plan.groups.len(), 3);
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
        let began = std::time::Instant::now();
        let plan = partition(&implementation, &NativeSpecialization::new(), &points).unwrap();
        assert!(began.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(plan.groups.len(), 12);
        assert!(plan.groups.iter().all(|group| group.candidates.len() == 16));
        assert!(plan
            .sources
            .iter()
            .all(|source| source.variants.len() == 16));
        assert!(plan
            .points
            .iter()
            .enumerate()
            .all(|(point, planned)| planned.active_sets == [vec![point]]));
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
        let selected = plan
            .assemble(&implementation, &NativeSpecialization::new(), &[0, 0], &[5])
            .unwrap();
        assert_eq!(selected.launch_param(2, "TILE"), Some(64));
    }

    #[test]
    fn a_constrained_boundary_stays_in_the_joint_group() {
        // A boundary that also participates in `where` is a choice in the
        // admissible domain, not an independently movable band selector.
        let text = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (BATCH_FROM in [5, 3])\n    where BATCH_FROM >= 3\n    launch gemv when N < BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch batch when N >= BATCH_FROM:\n        params (code ROWS in [1, 2])\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text: text.into(),
        }]))
        .unwrap();
        let implementation = module
            .native_implementation(module.entries()[0].id, BackendName::Metal)
            .unwrap();
        let plan = partition(implementation, &NativeSpecialization::new(), &points()).unwrap();
        assert!(plan.boundary.is_empty());
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].launches, [0, 1]);
    }
}
