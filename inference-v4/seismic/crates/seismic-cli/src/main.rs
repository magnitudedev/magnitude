//! `seismic` command-line tool over the structured pipeline:
//! sources -> `sir::Program` -> joint selection -> the selected witness's MSL.

mod bindings;
mod select;

use seismic_lang::family::{Numerics, Workload};
use seismic_lang::program::{collect_files, compile};
use seismic_lang::sir::Program;
use seismic_lang::syntax;
use seismic_lang::types::{DType, Elem};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  seismic check <file|dir>...
  seismic print <file|dir>...
  seismic select <file|dir>... --fn <name> --shape K=V,... [--element NAME=TYPE,...] [--numerics exact|admitted] [--target cpu|cuda|metal] [--strategy exact|greedy]
  seismic emit <file|dir>... --fn <name> --shape K=V,... [--element NAME=TYPE,...] [--numerics exact|admitted] [--target cpu|cuda|metal] [--strategy exact|greedy]
  seismic analyze-search <file|dir>... --fn <name> --shape K=V,... [--element NAME=TYPE,...] [--numerics exact|admitted] [--target cpu|cuda|metal]
  seismic bindings <file|dir>... --fn <name> [--element NAME=TYPE,...]
`select`, `emit` and `analyze-search` target Metal unless `--target` is given. `emit` prints
what the selected witness compiles to: MSL on Metal, the scalar instruction listing of every
phase on the CPU, the PTX text of every launch on CUDA. An unselected candidate cannot be emitted. Implementation choices are
compiler-owned.";

/// The default target of `select`, `emit` and `analyze-search`.
pub const TARGET: &str = "metal";
/// Targets with a backend on the structured pipeline. A ported backend adds its name here
/// and one arm in `select::Target`.
pub const TARGETS: &[&str] = &["metal", "cpu", "cuda"];

pub struct Options {
    pub paths: Vec<PathBuf>,
    /// The backend `select`, `emit` and `analyze-search` run on.
    pub target: String,
    pub function: Option<String>,
    pub workload: Workload,
    /// How `select` and `emit` improve the seed.
    pub strategy: seismic_compiler::selection::Strategy,
}

impl Options {
    pub fn entry(&self) -> Result<&str, String> {
        self.function
            .as_deref()
            .ok_or_else(|| "--fn is required".to_string())
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or(&[]);
    let result = match args.first().map(String::as_str) {
        Some("check") => check(rest),
        Some("print") => print_files(rest),
        Some("select") => select::select(rest),
        Some("emit") => select::emit(rest),
        Some("analyze-search") => select::analyze_search(rest),
        Some("bindings") => bindings::bindings(rest),
        Some("help") | Some("--help") | Some("-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(format!("unknown command `{other}`\n{USAGE}")),
        None => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

/// Parse `args`, accepting only the flags in `allowed`.
pub fn options(args: &[String], allowed: &[&str]) -> Result<Options, String> {
    let mut o = Options {
        paths: Vec::new(),
        target: TARGET.into(),
        function: None,
        workload: Workload::default(),
        strategy: Default::default(),
    };
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if !arg.starts_with("--") {
            o.paths.push(PathBuf::from(arg));
            continue;
        }
        if !allowed.contains(&arg.as_str()) {
            return Err(format!(
                "unsupported option `{arg}`; implementation choices are compiler-owned"
            ));
        }
        let value = rest
            .next()
            .ok_or_else(|| format!("{arg} requires a value"))?;
        match arg.as_str() {
            "--target" => {
                if !TARGETS.contains(&value.as_str()) {
                    return Err(format!(
                        "unknown target `{value}`; expected one of {}",
                        TARGETS.join(", ")
                    ));
                }
                o.target = value.clone();
            }
            "--fn" => o.function = Some(value.clone()),
            "--shape" => {
                for binding in value.split(',') {
                    let (name, extent) = binding
                        .split_once('=')
                        .ok_or_else(|| format!("bad shape binding `{binding}`; expected K=V"))?;
                    let extent: i64 = extent
                        .trim()
                        .parse()
                        .map_err(|_| format!("bad shape value `{extent}`"))?;
                    if extent < 0 {
                        return Err(format!("shape `{name}` must be nonnegative"));
                    }
                    if o.workload
                        .shapes
                        .insert(name.trim().to_string(), extent)
                        .is_some()
                    {
                        return Err(format!("duplicate shape binding {name}"));
                    }
                }
            }
            "--element" => {
                for binding in value.split(',') {
                    let (name, element) = binding.split_once('=').ok_or_else(|| {
                        format!("bad element binding `{binding}`; expected NAME=TYPE")
                    })?;
                    let element = element.trim();
                    let element = if let Some(dtype) = DType::from_name(element) {
                        Elem::Dtype(dtype)
                    } else if seismic_lang::repr::lookup(element).is_some() {
                        Elem::Repr(element.into())
                    } else {
                        return Err(format!("unknown concrete element type {element}"));
                    };
                    if o.workload
                        .elems
                        .insert(name.trim().to_string(), element)
                        .is_some()
                    {
                        return Err(format!("duplicate element binding {name}"));
                    }
                }
            }
            "--strategy" => {
                o.strategy = match value.as_str() {
                    "exact" => seismic_compiler::selection::Strategy::Exact,
                    "greedy" => seismic_compiler::selection::Strategy::Greedy,
                    other => {
                        return Err(format!(
                            "bad --strategy `{other}`; expected exact or greedy"
                        ))
                    }
                }
            }
            "--numerics" => {
                o.workload.numerics = match value.as_str() {
                    "exact" => Numerics::Exact,
                    "admitted" => Numerics::Admitted,
                    other => {
                        return Err(format!(
                            "bad --numerics `{other}`; expected exact or admitted"
                        ))
                    }
                }
            }
            other => return Err(format!("option `{other}` has no parser")),
        }
    }
    if o.paths.is_empty() {
        return Err(format!("no files given\n{USAGE}"));
    }
    Ok(o)
}

/// Compile every collected file as one closed program; diagnostics are rendered in full.
pub fn load_program(o: &Options) -> Result<(usize, Program), String> {
    let files = collect_files(&o.paths)?;
    let program = compile(&files).map_err(|diagnostics| {
        let mut out: Vec<String> = diagnostics.iter().map(|d| d.render()).collect();
        out.push(format!("{} error(s)", diagnostics.len()));
        out.join("\n")
    })?;
    Ok((files.len(), program))
}

fn check(args: &[String]) -> Result<(), String> {
    let o = options(args, &[])?;
    let (files, program) = load_program(&o)?;
    println!(
        "ok: {files} file(s), {} definition(s), {} linked function family(ies)",
        program.definitions.len(),
        program.families.len(),
    );
    Ok(())
}

fn print_files(args: &[String]) -> Result<(), String> {
    let o = options(args, &[])?;
    for f in collect_files(&o.paths)? {
        let file = syntax::parse(&f.text).map_err(|d| d.render(&f.path, &f.text))?;
        print!("{}", syntax::print(&file));
    }
    Ok(())
}
