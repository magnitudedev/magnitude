//! `seismic` command-line tool.

mod source;

use seismic_lang::program::{collect_files, compile, Program};
use seismic_lang::{parse, print};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  seismic check <file|dir>... [--lib <dir>]... [--backends a,b]
  seismic print <file>...
  seismic plan <file|dir>... --fn <name> --shape K=V,...
  seismic bindings <file|dir>... --fn <name>
Native execution requires completed automatic selection through seismic-runtime.
Explicit candidate selection and native replay commands are not supported.";

pub struct Options {
    pub paths: Vec<PathBuf>,
    pub backends: Vec<String>,
    pub function: Option<String>,
    pub shapes: HashMap<String, i64>,
    pub scalars: HashMap<String, f64>,
    pub elements: HashMap<String,seismic_lang::types::Elem>,
    pub target: String,

}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str);
    let result = match command {
        Some("check") => check(&args[1..]),
        Some("print") => print_files(&args[1..]),
        Some("plan") => source::plan(&args[1..]),
        Some("bindings") => source::bindings(&args[1..]),
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

pub fn options(args: &[String]) -> Result<Options, String> {
    let mut o = Options { paths: Vec::new(), backends: vec!["metal".into(), "cpu".into()], function: None, shapes: HashMap::new(), elements: HashMap::new(), scalars: HashMap::new(), target: "metal".into() };
    let mut i = 0;
    let value = |i: &mut usize, what: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{what} requires a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--lib" => o.paths.push(PathBuf::from(value(&mut i, "--lib")?)),
            "--backends" => o.backends = value(&mut i, "--backends")?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            "--fn" => o.function = Some(value(&mut i, "--fn")?),
            "--target" => o.target = value(&mut i, "--target")?,
            "--element" => {
                for binding in value(&mut i,"--element")?.split(',') {
                    let (name,element)=binding.split_once('=').ok_or("expected element binding NAME=TYPE")?;
                    let element=if let Some(dtype)=seismic_lang::types::DType::from_name(element) {seismic_lang::types::Elem::Dtype(dtype)}
                        else if seismic_lang::repr::lookup(element).is_some() {seismic_lang::types::Elem::Repr(element.into())}
                        else {return Err(format!("unknown concrete element type {element}"));};
                    if o.elements.insert(name.into(),element).is_some() {return Err(format!("duplicate element binding {name}"));}
                }
            }
            "--shape" => {
                for kv in value(&mut i, "--shape")?.split(',') {
                    let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad shape binding `{kv}`"))?;
                    o.shapes.insert(k.trim().to_string(), v.trim().parse().map_err(|_| format!("bad shape value `{v}`"))?);
                }
            }
            "--scalar" => {
                for kv in value(&mut i, "--scalar")?.split(',') {
                    let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad scalar binding `{kv}`"))?;
                    o.scalars.insert(k.trim().to_string(), v.trim().parse().map_err(|_| format!("bad scalar value `{v}`"))?);
                }
            }
            other if other.starts_with("--") => return Err(format!("unsupported option `{other}`; implementation choices are compiler-owned")),
            other => o.paths.push(PathBuf::from(other)),
        }
        i += 1;
    }
    if o.paths.is_empty() {
        return Err(format!("no files given\n{USAGE}"));
    }
    if o.shapes.values().any(|n| *n < 0) {
        return Err("shape dimensions must be nonnegative".into());
    }
    Ok(o)
}

pub fn load_program(o: &Options) -> Result<Program, String> {
    let files = collect_files(&o.paths)?;
    let mut backends = o.backends.clone();
    if o.target == "cuda" && !backends.iter().any(|b| b == "cuda") { backends.push("cuda".into()); }
    compile(&files, &backends).map_err(|errors| {
        let mut out: Vec<String> = errors.iter().map(|e| e.render()).collect();
        out.push(format!("{} error(s)", errors.len()));
        out.join("\n")
    })
}

fn check(args: &[String]) -> Result<(), String> {
    let o = options(args)?;
    let files = collect_files(&o.paths)?;
    let program = load_program(&o)?;
    println!("ok: {} file(s), {} function(s), {} lowering(s), backends {}", files.len(), program.functions.len(), program.lowerings.len(), o.backends.join(","));
    for l in &program.lowerings {
        if !l.residual.is_empty() {
            println!("  {}.{}: applies where {}", l.construct, l.backend, l.residual.iter().map(|r| format!("{r} >= 0")).collect::<Vec<_>>().join(" and "));
        }
    }
    Ok(())
}

fn print_files(args: &[String]) -> Result<(), String> {
    let o = options(args)?;
    for f in collect_files(&o.paths)? {
        let file = parse(&f.text).map_err(|d| d.render(&f.path.display().to_string(), &f.text))?;
        print!("{}", print(&file));
    }
    Ok(())
}
