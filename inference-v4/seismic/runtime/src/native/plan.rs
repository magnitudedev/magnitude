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
    /// Distinct assignments of the code parameters this launch owns.
    pub code_variants: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningPartition {
    pub groups: Vec<Group>,
    /// Entry parameters used solely to choose a launch's active row band.
    pub boundary: Vec<ParameterAddress>,
    pub points: Vec<PlannedPoint>,
    pub sources: Vec<LaunchSource>,
    /// Distinct candidate/point/active-set timings, including points with no
    /// active tuned launch.
    pub measurements: usize,
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

/// Partition the declared domain. The checked declaration gives launch-local
/// ownership; an entry parameter belongs to every launch whose geometry or
/// condition reads it. An otherwise unowned entry parameter conservatively
/// couples all launches unless it is a condition-only boundary parameter.
pub fn partition(
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    points: &[PointShape],
) -> Result<TuningPartition, PlanError> {
    let admissible = implementation
        .admissible(statics)
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
    let mut measured = BTreeSet::new();
    let mut activity = vec![Vec::<Vec<usize>>::with_capacity(points.len()); admissible.len()];
    let mut point_sets = vec![BTreeSet::<Vec<usize>>::new(); points.len()];
    for (configuration, specialization) in admissible.iter().enumerate() {
        for (index, point) in points.iter().enumerate() {
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
            point_sets[index].insert(active.clone());
            activity[configuration].push(active);
        }
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
            let candidates = admissible
                .iter()
                .map(|specialization| {
                    parameters
                        .iter()
                        .map(|address| address.value(specialization))
                        .collect::<Vec<_>>()
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            Group {
                launches,
                parameters,
                candidates,
            }
        })
        .collect::<Vec<_>>();
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
            let code_variants = admissible
                .iter()
                .map(|specialization| {
                    code.iter()
                        .map(|address| address.value(specialization))
                        .collect::<Vec<_>>()
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            LaunchSource {
                ordinal,
                code_variants,
            }
        })
        .collect();
    let mut measurements = 0usize;
    for group in &groups {
        for candidate in &group.candidates {
            for point in 0..points.len() {
                let sets = admissible
                    .iter()
                    .enumerate()
                    .filter_map(|(configuration, specialization)| {
                        let values = group
                            .parameters
                            .iter()
                            .map(|address| address.value(specialization))
                            .collect::<Vec<_>>();
                        if &values != candidate {
                            return None;
                        }
                        let active = &activity[configuration][point];
                        active
                            .iter()
                            .any(|launch| group.launches.contains(launch))
                            .then(|| active.clone())
                    })
                    .collect::<BTreeSet<_>>();
                measurements += sets.len();
            }
        }
    }
    for active_sets in &point_sets {
        measurements += active_sets
            .iter()
            .filter(|active| {
                !active
                    .iter()
                    .any(|launch| groups.iter().any(|group| group.launches.contains(launch)))
            })
            .count();
    }
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
        measurements,
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
                .map(|source| source.code_variants.len())
                .collect::<Vec<_>>(),
            [2, 2, 2]
        );
        assert_eq!(plan.measurements, 22);
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
