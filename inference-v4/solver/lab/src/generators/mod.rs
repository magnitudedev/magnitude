use crate::LabResult;
use magnitude_solver::model::{
    Constraint, Cost, Domain, Fragment, LinearTerm, Literal, Model, ModelBuilder, VarId,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, time::Instant};

mod kernels;
mod schedules;

pub const FAMILIES: &[&str] = &[
    "independent",
    "chain",
    "separator",
    "shared-producer",
    "process-plan",
    "process-plan-lazy",
    "schedule",
    "repeated",
    "coverage-gap",
    "pipeline",
    "packing-gap",
    "repeated-coupled",
    "kernel-contraction",
    "kernel-shared",
    "kernel-schedule",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameters {
    pub n: usize,
    pub d: usize,
    pub width: usize,
    pub depth: usize,
    pub repeat: u64,
    pub capacity: u64,
    pub horizon: u64,
    pub outputs: u64,
    pub input_length: u64,
    pub overhead: u64,
    pub element_bytes: u64,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            n: 4,
            d: 2,
            width: 1,
            depth: 2,
            repeat: 16,
            capacity: 16,
            horizon: 64,
            outputs: 4,
            input_length: 4,
            overhead: 2,
            element_bytes: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Reference {
    /// Exact multiple-choice capacity dynamic program, conditioned on global modes.
    ResourceChoices {
        capacity: u64,
        modes: Vec<(u64, u64, Vec<Vec<(u64, u64)>>)>,
    },
    /// Separate implementation and finite-start enumeration; metadata is never
    /// passed to a search implementation.
    KernelSchedule {
        modes: Vec<VarId>,
        starts: Vec<VarId>,
        ends: Vec<VarId>,
        durations: Vec<(VarId, [u64; 2])>,
        demands: Vec<(VarId, [u64; 2])>,
        completion: VarId,
        horizon: u64,
        capacity: u64,
    },
    Exact(u64),
    Chain {
        local: Vec<Vec<u64>>,
        transitions: Vec<Vec<Vec<u64>>>,
    },
    Incomplete,
    FixedSchedule {
        starts: Vec<VarId>,
        ends: Vec<VarId>,
        durations: Vec<u64>,
        completion: VarId,
        horizon: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub schema: u32,
    pub name: String,
    pub family: String,
    pub seed: u64,
    pub parameters: Parameters,
    pub model: Model,
    pub reference: Option<Reference>,
    pub notes: String,
    #[serde(default)]
    pub generation_ms: Option<f64>,
}
impl Instance {
    pub fn fingerprint(&self) -> LabResult<String> {
        // Deterministic identity for comparison, not a proof-authorizing memo key.
        let bytes = serde_json::to_vec(&self.model)?;
        let hash = bytes
            .into_iter()
            .fold(0xcbf29ce484222325_u64, |state, byte| {
                (state ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            });
        Ok(format!("fnv1a64:{hash:016x}"))
    }
    pub fn read(path: &Path) -> LabResult<Self> {
        let instance: Self = serde_json::from_slice(&fs::read(path)?)?;
        if instance.schema != 1 {
            return Err(format!("unsupported instance schema {}", instance.schema).into());
        }
        instance.model.validate()?;
        Ok(instance)
    }
    pub fn save(&self, directory: &Path) -> LabResult<std::path::PathBuf> {
        fs::create_dir_all(directory)?;
        let path = directory.join(format!("{}.instance.json", self.name));
        fs::write(&path, serde_json::to_vec_pretty(self)?)?;
        Ok(path)
    }
    pub fn rectangular_log10(&self) -> f64 {
        self.model
            .variables()
            .iter()
            .map(|v| (v.domain.cardinality() as f64).log10())
            .sum()
    }
}

pub struct Random(u64);
impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn cost(&mut self) -> u64 {
        1 + self.next_u64() % 19
    }
}

fn variable(builder: &mut ModelBuilder, name: impl Into<String>, d: usize) -> LabResult<VarId> {
    Ok(builder.variable(
        name,
        Domain::interval(
            0,
            i64::try_from(d.checked_sub(1).ok_or("empty choice domain")?)?,
        )?,
    ))
}
fn unary(builder: &mut ModelBuilder, var: VarId, costs: &[u64]) {
    builder.cost(Cost::Table {
        variables: vec![var],
        entries: costs
            .iter()
            .enumerate()
            .map(|(i, &c)| (vec![i as i64], c))
            .collect(),
    });
}

pub fn generate(family: &str, p: Parameters, seed: u64) -> LabResult<Instance> {
    let generation_start = Instant::now();
    if p.n == 0 || p.d == 0 || p.d > i64::MAX as usize {
        return Err("n and d must be positive, representable finite sizes".into());
    }
    if p.n > 100_000 || p.d > 1_000_000 {
        return Err("generator size exceeds the explicit construction safety envelope".into());
    }
    if p.n.checked_mul(p.d).ok_or("generator size overflow")? > 2_000_000 {
        return Err("generator exceeds two million explicit unary alternatives".into());
    }
    if family.starts_with("kernel-")
        && (p.d > 32
            || p.n > 4096
            || p.input_length > 1_000_000
            || p.outputs > 1_000_000
            || p.overhead > 1_000_000
            || p.capacity > 1_000_000
            || p.horizon > 1_000_000)
    {
        return Err("kernel fixture construction envelope exceeded".into());
    }
    let mut builder = ModelBuilder::new();
    let mut rng = Random::new(seed);
    let (reference, notes) = match family {
        "kernel-contraction" => (Some(kernels::contraction(&mut builder, &p, &mut rng)?), "Separate tile, staging, layout and arithmetic live-storage choices. Primitive transfer/compute/tail costs and one shared storage capacity. Synthetic service objective, not schedule makespan."),
        "kernel-shared" => (Some(kernels::shared(&mut builder, &p, &mut rng)?), "Separate sharing, retention, consumer tile/layout/fusion and live-storage choices. Shared preparation and per-consumer reload/materialization work under a common capacity. Synthetic service objective."),
        "kernel-schedule" => (Some(kernels::scheduled(&mut builder, &p, &mut rng)?), "Exposed instruction-mode duration/resource tradeoff with free bounded timestamps and shared capacity; independent per-implementation schedule reference."),
        "independent" => {
            let mut expected = 0_u64;
            for i in 0..p.n {
                let v = variable(&mut builder, format!("stage-{i}"), p.d)?;
                let costs: Vec<_> = (0..p.d).map(|_| rng.cost()).collect();
                expected = expected.checked_add(*costs.iter().min().unwrap()).ok_or("objective overflow")?;
                unary(&mut builder, v, &costs);
            }
            (Some(Reference::Exact(expected)), "Independent additive choices: known sum of minima; nominal assignments d^n.")
        }
        "chain" => {
            let vars: Vec<_> = (0..p.n).map(|i| variable(&mut builder, format!("representation-{i}"), p.d)).collect::<LabResult<_>>()?;
            let mut local = Vec::new(); let mut transitions = Vec::new();
            for (i, &v) in vars.iter().enumerate() {
                let costs: Vec<_> = (0..p.d).map(|_| rng.cost()).collect(); unary(&mut builder, v, &costs); local.push(costs);
                if i > 0 {
                    let matrix: Vec<Vec<u64>> = (0..p.d).map(|_| (0..p.d).map(|_| rng.cost()).collect()).collect();
                    builder.cost(Cost::Table { variables: vec![vars[i - 1], v], entries: matrix.iter().enumerate().flat_map(|(a, row)| row.iter().enumerate().map(move |(b, &cost)| (vec![a as i64, b as i64], cost))).collect() });
                    transitions.push(matrix);
                }
            }
            (Some(Reference::Chain { local, transitions }), "Pairwise additive chain: independent O(n*d^2) dynamic-programming reference.")
        }
        "separator" => {
            let vars: Vec<_> = (0..p.n).map(|i| variable(&mut builder, format!("bag-var-{i}"), p.d)).collect::<LabResult<_>>()?;
            let width = p.width.min(p.n.saturating_sub(1));
            let arity = width + 1;
            let tuples = assignments(p.d, arity, 100_000)?;
            for first in 0..=p.n - arity {
                let entries = tuples.iter().cloned().map(|tuple| (tuple, rng.cost())).collect();
                builder.cost(Cost::Table { variables: vars[first..first + arity].to_vec(), entries });
            }
            (None, "Sliding bags with adjustable constructed width; this parameter is not a measured minimum treewidth.")
        }
        "shared-producer" => {
            let shared = variable(&mut builder, "production-policy", 2)?;
            let rep = variable(&mut builder, "representation", p.d)?;
            // One activity identity is charged once under sharing; recomputation is charged per occurrence.
            builder.guarded_cost(vec![Literal::new(shared, 1)], Cost::Constant(6));
            builder.guarded_cost(vec![Literal::new(shared, 0)], Cost::Constant(6_u64.checked_mul(p.n as u64).ok_or("objective overflow")?));
            for i in 0..p.n {
                let consumer = variable(&mut builder, format!("consumer-{i}"), p.d)?;
                builder.constraint(Constraint::Equal { left: consumer, right: rep });
                unary(&mut builder, consumer, &(0..p.d).map(|r| 1 + r as u64).collect::<Vec<_>>());
            }
            let sharing_legal = p.capacity >= p.n as u64;
            if !sharing_legal { builder.constraint(Constraint::LinearLe { terms: vec![LinearTerm::new(shared, 1)], rhs: 0 }); }
            let expected = (if sharing_legal { 6 } else { 6 * p.n as u64 }) + p.n as u64;
            (Some(Reference::Exact(expected)), "Explicit shared preparation cost versus distinct recomputation occurrences; capacity can forbid sharing.")
        }
        "process-plan" | "process-plan-lazy" => {
            if p.depth > 12 { return Err("process-plan depth exceeds explicit construction envelope".into()); }
            let mut count = 0;
            let best = process_node(&mut builder, &mut rng, p.depth, p.d, &mut count, family=="process-plan-lazy")?;
            (Some(Reference::Exact(best)), "Guarded process realizations introduce more choices from reusable immutable definitions; inactive branches contribute no cost. Eager/lazy forms have the same finite family.")
        }
        "coverage-gap" => {
            let choice = variable(&mut builder, "covered-or-unresolved", 2)?;
            builder.guarded_cost(vec![Literal::new(choice, 0)], Cost::Constant(10));
            builder.unresolved(vec![Literal::new(choice, 1)], "declared implementation coverage missing");
            (Some(Reference::Incomplete), "Competitive unresolved coverage must prevent a claim of optimality.")
        }
        "repeated" => {
            let mut minimum = 0_u64;
            for i in 0..p.n {
                let v = variable(&mut builder, format!("repeated-body-choice-{i}"), p.d)?;
                let costs: Vec<_> = (0..p.d).map(|_| rng.cost().checked_mul(p.repeat).ok_or("repeated cost overflow")).collect::<Result<_, _>>()?;
                minimum = minimum.checked_add(*costs.iter().min().unwrap()).ok_or("objective overflow")?;
                unary(&mut builder, v, &costs);
            }
            (Some(Reference::Exact(minimum)), "Mandatory independent serial occurrences share one implementation decision; cost multiplication is exact and model size does not depend on repetition count. No parallel-overlap claim.")
        }
        "schedule" => { schedules::schedule(&mut builder, &p, &mut rng)?; (None, "Optional modes with precedence and capacity over half-open intervals; makespan objective.") }
        "pipeline" => { schedules::pipeline(&mut builder, &p)?; (None, "Abstract preparation/consumption pipeline, alternative group/window widths, retain/recompute, explicit tails and event lifetimes. Synthetic durations only.") }
        "packing-gap" => (Some(schedules::packing(&mut builder)?), "Durations 2,3,4 each demand 2 of capacity 3: aggregate work bound 6, mandatory serialization optimum 9."),
        "repeated-coupled" => (Some(schedules::repeated_coupled(&mut builder)?), "Two explicit occurrences of a transfer/compute body share resources; overlap improves makespan from serial 10 to 8. Joint finite schedule, not isolated-body reuse."),
        _ => return Err(format!("unknown family {family}; choose {}", FAMILIES.join(",")).into()),
    };
    let model = builder.build()?;
    let name = format!(
        "{family}-s{seed}-n{}-d{}-w{}-h{}-r{}-b{}-t{}-o{}-k{}-a{}-e{}",
        p.n,
        p.d,
        p.width,
        p.depth,
        p.repeat,
        p.capacity,
        p.horizon,
        p.outputs,
        p.input_length,
        p.overhead,
        p.element_bytes
    );
    Ok(Instance {
        schema: 1,
        name,
        family: family.into(),
        seed,
        parameters: p,
        model,
        reference,
        notes: notes.into(),
        generation_ms: Some(generation_start.elapsed().as_secs_f64() * 1000.0),
    })
}

fn process_node(
    builder: &mut ModelBuilder,
    rng: &mut Random,
    depth: usize,
    d: usize,
    count: &mut usize,
    lazy: bool,
) -> LabResult<u64> {
    *count += 1;
    if *count > 10_000 {
        return Err("process-plan exceeds 10,000-node eager construction envelope".into());
    }
    let v = variable(builder, format!("process-{}", *count), d)?;
    let child = if depth > 0 {
        let mut child_builder = ModelBuilder::new();
        let optimum = process_node(&mut child_builder, rng, depth - 1, d, count, lazy)?;
        Some((Fragment::new(child_builder.build()?, vec![])?, optimum))
    } else {
        None
    };
    let mut best = u64::MAX;
    for option in 0..d {
        let active = vec![Literal::new(v, option as i64)];
        let mut cost = rng.cost();
        builder.guarded_cost(active.clone(), Cost::Constant(cost));
        if let Some((fragment, minimum)) = &child {
            if lazy {
                builder.instantiate_lazy(format!("alternative-{option}"), fragment, &[], active)?;
            } else {
                builder.instantiate(format!("alternative-{option}"), fragment, &[], active)?;
            }
            cost = cost.checked_add(*minimum).ok_or("objective overflow")?;
        }
        best = best.min(cost);
    }
    Ok(best)
}

fn assignments(d: usize, n: usize, cap: usize) -> LabResult<Vec<Vec<i64>>> {
    let size = (0..n)
        .try_fold(1_usize, |a, _| a.checked_mul(d))
        .ok_or("factor table size overflow")?;
    if size > cap {
        return Err(format!(
            "explicit factor table needs {size} rows; construction limit is {cap}"
        )
        .into());
    }
    Ok((0..size)
        .map(|mut index| {
            let mut tuple = vec![0; n];
            for value in tuple.iter_mut().rev() {
                *value = (index % d) as i64;
                index /= d;
            }
            tuple
        })
        .collect())
}

pub fn conditional_fixture() -> LabResult<Instance> {
    let generation_start = Instant::now();
    let mut builder = ModelBuilder::new();
    let format = variable(&mut builder, "format", 2)?;
    unary(&mut builder, format, &[2, 8]);
    unary(&mut builder, format, &[9, 3]);
    unary(&mut builder, format, &[9, 4]);
    Ok(Instance {
        schema: 1,
        name: "conditional-composition".into(),
        family: "fixture".into(),
        seed: 0,
        parameters: Parameters::default(),
        model: builder.build()?,
        reference: Some(Reference::Exact(15)),
        notes: "A=2+9+9=20; B=8+3+4=15. Local producer minimization is invalid.".into(),
        generation_ms: Some(generation_start.elapsed().as_secs_f64() * 1000.0),
    })
}

pub fn suite(name: &str, seeds: &[u64]) -> LabResult<Vec<Instance>> {
    let mut instances = Vec::new();
    match name {
        "lns-development" | "lns-held-out" => {
            let held = name == "lns-held-out";
            for &seed in seeds {
                for n in [2, 8] {
                    for (capacity, overhead, input_length) in if held {
                        [(n as u64, 5, 11), (3 * n as u64, 11, 7)]
                    } else {
                        [(n as u64, 0, 8), (2 * n as u64, 8, 12)]
                    } {
                        for family in ["kernel-contraction", "kernel-shared"] {
                            instances.push(generate(
                                family,
                                Parameters {
                                    n,
                                    d: 3,
                                    capacity,
                                    overhead,
                                    input_length,
                                    outputs: if held { 11 } else { 8 },
                                    ..Parameters::default()
                                },
                                seed,
                            )?);
                        }
                    }
                }
                instances.push(generate(
                    "kernel-schedule",
                    Parameters {
                        n: 3,
                        horizon: 12,
                        capacity: if held { 3 } else { 2 },
                        ..Parameters::default()
                    },
                    seed,
                )?);
                instances.push(generate(
                    "chain",
                    Parameters {
                        n: 32,
                        d: 4,
                        ..Parameters::default()
                    },
                    seed,
                )?);
            }
        }
        "lns-scaling" => {
            for &seed in seeds {
                for n in [2, 4, 8, 16, 32, 64] {
                    for family in ["kernel-contraction", "kernel-shared", "chain"] {
                        instances.push(generate(
                            family,
                            Parameters {
                                n,
                                d: 3,
                                capacity: n as u64 * 2,
                                input_length: 11,
                                outputs: 11,
                                ..Parameters::default()
                            },
                            seed,
                        )?);
                    }
                }
                for d in [2, 4, 8, 16, 32] {
                    instances.push(generate(
                        "kernel-contraction",
                        Parameters {
                            n: 8,
                            d,
                            capacity: 8 * d as u64 / 2,
                            input_length: 33,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
                for capacity in [0, 4, 8, 16, 32] {
                    instances.push(generate(
                        "kernel-shared",
                        Parameters {
                            n: 8,
                            d: 4,
                            capacity,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
                for repeat in [1, 16, 256, 4096, 1_000_000] {
                    instances.push(generate(
                        "repeated",
                        Parameters {
                            n: 8,
                            d: 4,
                            repeat,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
            }
        }
        "oracle-small" => {
            instances.push(conditional_fixture()?);
            for &seed in seeds {
                for family in FAMILIES.iter().filter(|f| !f.starts_with("kernel-")) {
                    let p = Parameters {
                        n: 2,
                        d: if *family == "schedule" || *family == "pipeline" {
                            1
                        } else {
                            2
                        },
                        depth: 1,
                        horizon: 4,
                        outputs: 1,
                        input_length: 1,
                        overhead: 0,
                        ..Parameters::default()
                    };
                    instances.push(generate(family, p, seed)?);
                }
            }
        }
        "structure-v1" => {
            for &seed in seeds {
                for n in [8, 32, 128, 512] {
                    for family in ["independent", "chain", "separator"] {
                        instances.push(generate(
                            family,
                            Parameters {
                                n,
                                ..Parameters::default()
                            },
                            seed,
                        )?);
                    }
                }
                for width in [1, 2, 4, 6, 8] {
                    instances.push(generate(
                        "separator",
                        Parameters {
                            n: 16,
                            width,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
                for repeat in [1, 16, 256, 65_536, 1_000_000] {
                    instances.push(generate(
                        "repeated",
                        Parameters {
                            repeat,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
                for d in [2, 4, 8, 16, 64] {
                    instances.push(generate(
                        "chain",
                        Parameters {
                            n: 16,
                            d,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
            }
        }
        "scheduling-v1" => {
            for &seed in seeds {
                for n in [4, 8, 16, 32] {
                    instances.push(generate(
                        "schedule",
                        Parameters {
                            n,
                            horizon: 128,
                            capacity: 2,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
            }
        }
        "corrections-v1" => {
            for family in ["packing-gap", "repeated-coupled"] {
                instances.push(generate(family, Parameters::default(), 0)?);
            }
        }
        "pipeline-v1" => {
            for &seed in seeds {
                for (outputs, input_length, capacity) in
                    [(2, 2, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]
                {
                    instances.push(generate(
                        "pipeline",
                        Parameters {
                            outputs,
                            input_length,
                            capacity,
                            horizon: 4096,
                            ..Parameters::default()
                        },
                        seed,
                    )?);
                }
            }
        }
        _ => return Err(format!("unknown suite {name}").into()),
    }
    Ok(instances)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generation_is_reproducible_and_seed_changes_data() {
        let a = generate("chain", Parameters::default(), 7).unwrap();
        let b = generate("chain", Parameters::default(), 7).unwrap();
        let c = generate("chain", Parameters::default(), 8).unwrap();
        assert_eq!(a.fingerprint().unwrap(), b.fingerprint().unwrap());
        assert_ne!(a.fingerprint().unwrap(), c.fingerprint().unwrap());
    }
    #[test]
    fn repetition_does_not_expand_variables() {
        let a = generate(
            "repeated",
            Parameters {
                repeat: 1,
                ..Parameters::default()
            },
            0,
        )
        .unwrap();
        let b = generate(
            "repeated",
            Parameters {
                repeat: 1_000_000,
                ..Parameters::default()
            },
            0,
        )
        .unwrap();
        assert_eq!(a.model.variables().len(), b.model.variables().len());
        assert_eq!(a.model.factors().len(), b.model.factors().len());
    }
    #[test]
    fn lazy_and_eager_realizations_agree() {
        let p = Parameters {
            depth: 1,
            ..Parameters::default()
        };
        let eager = generate("process-plan", p.clone(), 4).unwrap();
        let lazy = generate("process-plan-lazy", p, 4).unwrap();
        let a =
            crate::reference::exhaustive(&eager, 10000, std::time::Duration::from_secs(2)).unwrap();
        let b =
            crate::reference::exhaustive(&lazy, 10000, std::time::Duration::from_secs(2)).unwrap();
        assert_eq!(a.outcome, b.outcome);
    }
}
