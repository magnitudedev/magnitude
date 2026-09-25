//! Source-facing Seismic tooling.
//!
//! This binary deliberately stops at the checked-module boundary. Target
//! preparation and execution belong to generated Rust bindings plus the
//! public `seismic` API; exposing plan-space, solver, frozen-plan, or native
//! schedule internals here would recreate the public escape hatch W9 removes.

use seismic_lang::checked::{CheckedModule, SourceSet};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  seismic check [--no-std] <file|dir>...
  seismic entries [--no-std] <file|dir>...

`check` parses and semantically checks one closed module.
`entries` prints the generated-binding surface of that checked module.";

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let mut include_std = true;
    let mut paths = Vec::new();
    for argument in arguments {
        if argument == "--no-std" {
            include_std = false;
        } else if argument.starts_with('-') {
            eprintln!("unknown option `{argument}`\n{USAGE}");
            return ExitCode::from(2);
        } else {
            paths.push(PathBuf::from(argument));
        }
    }

    let result = load(paths, include_std).and_then(|module| match command.as_str() {
        "check" => {
            println!("checked {} exported entries", module.entries().len());
            Ok(())
        }
        "entries" => {
            print_entries(&module);
            Ok(())
        }
        other => Err(format!("unknown command `{other}`\n{USAGE}")),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

fn load(paths: Vec<PathBuf>, include_std: bool) -> Result<CheckedModule, String> {
    let prelude = if include_std {
        seismic_std::sources()
    } else {
        SourceSet::default()
    };
    seismic_lang::source::load(&paths, prelude)
        .map(|loaded| loaded.module)
        .map_err(|e| e.to_string())
}

fn print_entries(module: &CheckedModule) {
    for entry in module.entries() {
        if entry.element_parameters.is_empty() {
            println!("{}", entry.name);
        } else {
            println!("{}<{}>", entry.name, entry.element_parameters.join(", "));
        }
        for parameter in &entry.parameters {
            println!("  argument {}: {:?}", parameter.name, parameter.kind);
        }
        for result in &entry.results {
            println!("  result {:?}: {:?}", result.path, result.kind);
        }
    }
}
