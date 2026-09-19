//! `select`, `emit`, `analyze-search`: joint selection on one target and its inspection output.
use crate::{load_program, options, Options};
use seismic_compiler::selection::{self, Backend, Budget, ProofStatus, Selected, Strategy};
use seismic_cpu::mapping::Cpu;
use seismic_cuda::mapping::Cuda;
use seismic_lang::family::{
    self, CandidateRef, Family, OccurrenceId, Requirement, SiteKind, UnitKind, Witness,
};
use seismic_lang::sir::{DefId, DefKind, Program};
use seismic_metal::mapping::{EstimateModel, Limits, Metal};
use std::collections::BTreeMap;

const FLAGS: &[&str] = &[
    "--fn",
    "--shape",
    "--element",
    "--precision",
    "--atol",
    "--rtol",
    "--relative-floor",
    "--ulps",
    "--evidence",
    "--output-tolerance",
    "--input-range",
    "--allow-special-changes",
    "--target",
    "--strategy",
];

/// Capacities used when no Metal device can be opened: the limits every Apple-silicon GPU
/// family reports (1024 threads per threadgroup, 32 KiB of threadgroup memory).
const DEFAULT_MAX_THREADS_PER_THREADGROUP: u64 = 1024;
const DEFAULT_MAX_THREADGROUP_BYTES: u64 = 32 * 1024;

/// The Metal backend with the real device's capacities when one opens, else the documented
/// defaults. The second value says which, for the report.
fn metal() -> Result<(Metal, String), String> {
    #[cfg(target_os = "macos")]
    if let Ok(device) = seismic_metal::runtime::Device::open() {
        let info = device.info();
        let origin = format!(
            "device `{}` (max_threads_per_threadgroup={}, max_threadgroup_bytes={})",
            info.name, info.max_threads_per_threadgroup, info.max_threadgroup_bytes
        );
        return Ok((
            Metal::from_device(&info).map_err(|e| e.to_string())?,
            origin,
        ));
    }
    let limits = Limits {
        max_threads_per_threadgroup: DEFAULT_MAX_THREADS_PER_THREADGROUP,
        max_threadgroup_bytes: DEFAULT_MAX_THREADGROUP_BYTES,
        max_private_bytes: seismic_metal::mapping::PRIVATE_BYTES,
    };
    Ok((
        Metal::new(limits, EstimateModel::default()).map_err(|e| e.to_string())?,
        format!(
            "documented defaults, no Metal device (max_threads_per_threadgroup={DEFAULT_MAX_THREADS_PER_THREADGROUP}, max_threadgroup_bytes={DEFAULT_MAX_THREADGROUP_BYTES}, default estimate coefficients)"
        ),
    ))
}

/// The CPU backend for this host's worker count.
fn cpu() -> Result<(Cpu, String), String> {
    let workers = std::thread::available_parallelism()
        .map_err(|e| format!("host parallelism: {e}"))?
        .get() as u64;
    let backend = Cpu::host(workers).map_err(|e| e.to_string())?;
    let origin = format!(
        "host ({} workers, {} scratch bytes per worker, unqualified estimate coefficients)",
        backend.limits().workers,
        backend.limits().max_scratch_bytes
    );
    Ok((backend, origin))
}

/// The CUDA backend with the queried device's limits when the driver opens one, else the
/// documented GB10 limits.
fn cuda() -> Result<(Cuda, String), String> {
    if let Ok(device) = seismic_cuda::Device::open(0) {
        let origin = format!(
            "device `{}` (max_threads_per_block={}, max_grid_x={}, warp_size={})",
            device.info.name,
            device.info.max_threads_per_block,
            device.info.max_grid_x,
            device.info.warp_size
        );
        return Ok((
            Cuda::from_device(&device.info).map_err(|e| e.to_string())?,
            origin,
        ));
    }
    let backend = Cuda::new(
        seismic_cuda::mapping::Limits::gb10(),
        seismic_cuda::mapping::EstimateModel::default(),
    )
    .map_err(|e| e.to_string())?;
    Ok((
        backend,
        "documented GB10 limits, no CUDA device (unqualified estimate coefficients)".into(),
    ))
}

fn run<B: Backend>(
    o: &Options,
    program: &Program,
    backend: &B,
) -> Result<Selected<B::Execution>, String> {
    selection::select(
        program,
        o.entry()?,
        &o.workload,
        backend,
        Budget {
            strategy: o.strategy,
            ..Budget::default()
        },
    )
    .map_err(|e| e.to_string())
}

/// The selection report of `backend`: the same finite site domains selection searched over.
fn selected_report<B: Backend>(
    o: &Options,
    program: &Program,
    backend: &B,
    capacities: &str,
) -> Result<String, String> {
    let selected = run(o, program, backend)?;
    let domains = backend
        .bind_structure(program, &selected.family)
        .map_err(|e| e.to_string())?;
    Ok(report(program, &selected, &domains, capacities))
}

pub fn select(args: &[String]) -> Result<(), String> {
    let o = options(args, FLAGS)?;
    let (_, program) = load_program(&o)?;
    let text = match o.target.as_str() {
        "cpu" => {
            let (backend, capacities) = cpu()?;
            selected_report(&o, &program, &backend, &capacities)?
        }
        "cuda" => {
            let (backend, capacities) = cuda()?;
            selected_report(&o, &program, &backend, &capacities)?
        }
        _ => {
            let (backend, capacities) = metal()?;
            selected_report(&o, &program, &backend, &capacities)?
        }
    };
    print!("{text}");
    Ok(())
}

pub fn emit(args: &[String]) -> Result<(), String> {
    let o = options(args, FLAGS)?;
    let (_, program) = load_program(&o)?;
    let text = match o.target.as_str() {
        // The CPU has no textual source: the listing is the scalar instruction IR that the
        // native compiler receives, one function per phase.
        "cpu" => run(&o, &program, &cpu()?.0)?.execution.listing(),
        "cuda" => run(&o, &program, &cuda()?.0)?
            .execution
            .ptx()
            .iter()
            .enumerate()
            .map(|(launch, text)| format!("; launch {launch}\n{text}\n"))
            .collect(),
        _ => seismic_metal::msl::emit_execution(&run(&o, &program, &metal()?.0)?.execution)?.source,
    };
    print!("{text}");
    Ok(())
}

pub fn analyze_search(args: &[String]) -> Result<(), String> {
    let o = options(args, FLAGS)?;
    let (_, program) = load_program(&o)?;
    let target = o.target.as_str();
    let family = family::construct(&program, o.entry()?, target, &o.workload)?;
    let a = selection::analyze(&family);
    println!(
        "search structure of `{}` on {target} for {} (counts, not a time prediction)",
        family.entry,
        workload(&family)
    );
    println!("  templates                 {}", a.templates);
    println!("  occurrences               {}", a.occurrences);
    println!("  occurrences with a choice {}", a.choice_occurrences);
    println!("  max alternatives          {}", a.max_alternatives);
    println!("  numerical sites           {}", a.sites);
    println!("  sequences                 {}", a.sequences);
    println!("  fusion intervals          {}", a.intervals);
    println!("  log10 raw assignments     {:.2}", a.log10_raw_assignments);
    println!("  independent components    {}", a.independent_components);
    println!("  unexplored obligations    {}", a.obligations);
    Ok(())
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn workload(family: &Family) -> String {
    let shapes = family
        .workload
        .shapes
        .iter()
        .map(|(k, v)| format!("{k}={v}"));
    let elems = family
        .workload
        .elems
        .iter()
        .map(|(k, v)| format!("{k}={v}"));
    shapes.chain(elems).collect::<Vec<_>>().join(",")
}

fn definition(program: &Program, id: DefId) -> String {
    let d = program.definition(id);
    let kind = match &d.kind {
        DefKind::Body { target: None } => "fn".to_string(),
        DefKind::Body {
            target: Some(target),
        } => format!("fn for {target}"),
        DefKind::Lower { target } => format!("lower for {target}"),
    };
    let (path, text) = &program.files[d.file];
    let (line, _) = seismic_lang::span::line_col(text, d.span.start);
    format!("{kind} `{}` ({path}:{line})", d.name)
}

fn candidate_ref(r: CandidateRef) -> String {
    format!("occurrence {} candidate {}", r.occurrence.0, r.candidate)
}

fn requirement(r: &Requirement) -> String {
    match r {
        Requirement::Multiple { site, unit } => format!("site {} multiple of {unit}", site.0),
        Requirement::AtLeast { site, value } => format!("site {} >= {value}", site.0),
        Requirement::AtMost { site, value } => format!("site {} <= {value}", site.0),
        Requirement::Equal { site, value } => format!("site {} == {value}", site.0),
        Requirement::Divides { site, extent } => format!("site {} divides {extent}", site.0),
    }
}

fn domain(values: &[i64]) -> String {
    const SHOWN: usize = 32;
    let list = |v: &[i64]| v.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    match values {
        v if v.len() <= SHOWN => format!("{{{}}}", list(v)),
        [head @ .., last] => format!(
            "{{{},…,{last}}} ({} values)",
            list(&head[..8]),
            values.len()
        ),
        [] => "{}".into(),
    }
}

fn choice(witness: &Witness, occurrence: OccurrenceId) -> String {
    witness
        .choices
        .get(&occurrence)
        .map_or("inactive".into(), |c| format!("candidate {c}"))
}

fn cover(cover: Option<&Vec<(u32, u32)>>) -> String {
    cover.map_or("inactive".into(), |c| {
        c.iter()
            .map(|(s, e)| format!("[{s},{e})"))
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// Inspection output (spec section 13.4): every occurrence with its candidates and
/// rejections, every site with its domain and value, every cover, both witnesses with
/// their estimates, and the proof status. Unselected candidates are listed without any
/// claim that they are inferior.
fn report<E>(
    program: &Program,
    selected: &Selected<E>,
    domains: &BTreeMap<family::SiteId, Vec<i64>>,
    capacities: &str,
) -> String {
    macro_rules! say {
        ($out:expr, $($format:tt)*) => { $out.push(format!($($format)*)) };
    }
    let (family, witness, seed) = (&*selected.family, &selected.witness, &selected.seed);
    let mut out: Vec<String> = Vec::new();
    say!(
        out,
        "entry `{}` on {} for {}",
        family.entry,
        family.target,
        workload(family)
    );
    say!(out, "backend capacities: {capacities}");
    say!(out, "\noccurrences:");
    for occurrence in &family.occurrences {
        let origin = match occurrence.parent {
            None => "entry".to_string(),
            Some(parent) => format!(
                "call {} in {}",
                occurrence.call.map_or("?".into(), |c| c.0.to_string()),
                candidate_ref(parent)
            ),
        };
        say!(
            out,
            "  occurrence {} `{}` ({origin}): selected {}, seed {}",
            occurrence.id.0,
            program.families[occurrence.family].name,
            choice(witness, occurrence.id),
            choice(seed, occurrence.id)
        );
        for (ordinal, candidate) in occurrence.candidates.iter().enumerate() {
            let mark = if witness.choices.get(&occurrence.id) == Some(&(ordinal as u32)) {
                '*'
            } else {
                ' '
            };
            let template = family.template(candidate.template);
            let bound = template
                .shapes
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .chain(template.elems.iter().map(|(k, v)| format!("{k}={v}")))
                .chain(
                    candidate
                        .structural
                        .iter()
                        .map(|(k, s)| format!("{k}=site {}", s.0 .0)),
                )
                .collect::<Vec<_>>()
                .join(",");
            say!(
                out,
                "  {mark} candidate {ordinal}{}: {} via {}, template {} [{bound}]",
                if candidate.reference { " [reference]" } else { " [alternative]" },
                definition(program, template.definition),
                definition(program, candidate.via),
                candidate.template.0
            );
            if !candidate.requirements.is_empty() {
                say!(
                    out,
                    "      requires {}",
                    candidate
                        .requirements
                        .iter()
                        .map(requirement)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !candidate.numerical_effects.is_empty() {
                say!(out, "      numerical effects {:?}", candidate.numerical_effects);
            }
        }
        if occurrence.candidates.is_empty() {
            say!(
                out,
                "    no applicable implementation: missing coverage on this path"
            );
        }
        for (rejected, reason) in &occurrence.rejected {
            say!(
                out,
                "    rejected {}: {reason}",
                definition(program, *rejected)
            );
        }
    }
    say!(out, "\nnumerical sites:");
    for site in &family.sites {
        let kind = match &site.kind {
            SiteKind::Width { region, slice } => {
                format!("width of slice {} in region {}", slice.0, region.0)
            }
            SiteKind::Parts { region, slice } => {
                format!("parts of slice {} in region {}", slice.0, region.0)
            }
        };
        let value = |w: &Witness| {
            w.sites
                .get(&site.id)
                .map_or("inactive".into(), i64::to_string)
        };
        say!(
            out,
            "  site {} ({kind}, extent {}, owner {}): domain {}, selected {}, seed {}",
            site.id.0,
            site.extent,
            candidate_ref(site.owner),
            domains
                .get(&site.id)
                .map_or("unbound".into(), |d| domain(d)),
            value(witness),
            value(seed)
        );
    }
    say!(out, "\nsequences:");
    for sequence in &family.sequences {
        let units = sequence
            .units
            .iter()
            .map(|u| {
                let kind = match &u.kind {
                    UnitKind::Elementwise => "elementwise".to_string(),
                    UnitKind::Local => "local".to_string(),
                    UnitKind::Call(o) => format!("call occurrence {}", o.0),
                    UnitKind::Publish => "publish".to_string(),
                    UnitKind::Region(r) => format!("region {}", r.0),
                    UnitKind::Stage(s) => format!("stage {s}"),
                };
                format!("{kind}{}", if u.completion_after { "|" } else { "" })
            })
            .collect::<Vec<_>>()
            .join(", ");
        say!(
            out,
            "  sequence {} (owner {}, scope {:?}): units [{units}]",
            sequence.id.0,
            candidate_ref(sequence.owner),
            sequence.scope
        );
        say!(
            out,
            "    cover selected {}, seed {}",
            cover(witness.covers.get(&sequence.id)),
            cover(seed.covers.get(&sequence.id))
        );
    }
    say!(
        out,
        "\nestimate model: {} (estimates, not measurements)",
        selected.estimate_model
    );
    say!(out, "  seed estimate     {}", selected.seed_estimate);
    say!(out, "  selected estimate {}", selected.estimate);
    say!(out, "  proved lower bound {}", selected.lower_bound);
    say!(out, "numerical evidence: {:?}", selected.numerical_assessment.evidence);
    if let Some(policy) = &selected.numerical_assessment.validated_policy {
        say!(out, "  validated policy: {policy:?}");
    }
    for output in &selected.numerical_assessment.outputs {
        let metrics = output.metrics;
        say!(
            out,
            "  output `{}` {:?}: max_abs={:.9e}, max_rel={:.9e}, max_ulps={}, differing={}/{}, special mismatches nan={} inf={} signed_zero={} subnormal={}, worst={:?}",
            output.output,
            output.dtype,
            metrics.maximum_absolute,
            metrics.maximum_relative,
            metrics.maximum_ulps,
            metrics.differing,
            metrics.compared,
            metrics.nan_mismatches,
            metrics.infinity_mismatches,
            metrics.signed_zero_mismatches,
            metrics.subnormal_mismatches,
            output.worst_element,
        );
        for attribution in &output.attribution {
            say!(out, "    {attribution}");
        }
    }
    if let Some(qualification) = &selected.qualification {
        say!(out, "  qualification environment: {}", qualification.numerical_environment);
        say!(out, "  qualification corpus: {}", qualification.corpus);
        say!(out, "  qualification method: {}", qualification.method);
    }
    for reason in &selected.numerical_assessment.reasons {
        say!(out, "  {reason}");
    }
    say!(out,"proof status: {}", match selected.status {
        ProofStatus::Feasible => "feasible (checked complete execution; search ended by budget or with open obligations)",
        ProofStatus::ModelOptimal => "model-optimal over the stated family under the stated estimate model",
    });
    let (t, s) = (&selected.timings, &selected.search);
    say!(
        out,
        "strategy: {:?} over {} variables, {} factors",
        s.strategy,
        s.variables,
        s.factors
    );
    match s.strategy {
        Strategy::Greedy => say!(
            out,
            "  greedy: {} sweeps, {} trial witnesses",
            s.greedy_sweeps,
            s.greedy_trials
        ),
        Strategy::Exact => {
            say!(
                out,
                "  exact phase: {:.3} ms, work {}, nodes {}",
                ms(s.exact.time),
                s.exact.work,
                s.exact.nodes
            );
            match s.neighborhood {
                Some(n) => say!(
                    out,
                    "  neighborhood phase: {:.3} ms, work {}, nodes {}",
                    ms(n.time),
                    n.work,
                    n.nodes
                ),
                None => say!(out, "  neighborhood phase: not run"),
            }
        }
    }
    say!(out,"selection time (ms): family {:.3}, backend hooks {:.3}, export {:.3}, seed {:.3}, search {:.3}; solve total {:.3}", ms(t.family), ms(t.backend_hooks), ms(t.export), ms(t.seed), ms(t.search), ms(t.solve()));
    say!(out,"after selection (ms): instantiate {:.3}, realize {:.3} (emission and native compilation happen in the runtime)", ms(t.instantiate), ms(t.realize));
    if family.obligations.is_empty() && selected.unresolved.is_empty() {
        say!(out, "unresolved obligations: none");
    } else {
        say!(out, "unresolved obligations:");
        for o in &family.obligations {
            say!(
                out,
                "  occurrence {} {}: {}",
                o.occurrence.0,
                definition(program, o.definition),
                o.reason
            );
        }
        for u in &selected.unresolved {
            say!(out, "  {u}");
        }
    }
    out.push(String::new());
    out.join("\n")
}
