//! Source-facing Seismic tooling.
//!
//! This binary deliberately stops at the checked-module boundary. Target
//! preparation and execution belong to generated Rust bindings plus the
//! public `seismic` API; exposing plan-space, solver, frozen-plan, or native
//! schedule internals here would recreate the public escape hatch W9 removes.

use seismic_lang::checked::{check_source, CheckedModule, SourceFile, SourceSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
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
    if paths.is_empty() && !include_std {
        return Err("no .seismic source was provided".to_owned());
    }
    let mut files = Vec::new();
    for path in paths {
        collect(&path, &mut files)?;
    }
    files.sort();
    files.dedup();

    let mut sources = if include_std {
        seismic_std::sources()
    } else {
        SourceSet::default()
    };
    for path in files {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        sources.push(SourceFile {
            path: path.to_string_lossy().replace('\\', "/"),
            text,
        });
    }
    check_source(sources).map_err(|error| error.to_string())
}

fn collect(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata =
        std::fs::metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if metadata.is_file() {
        if path.extension() != Some(OsStr::new("seismic")) {
            return Err(format!("{} is not a .seismic file", path.display()));
        }
        files.push(path.to_path_buf());
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(format!(
            "{} is neither a file nor a directory",
            path.display()
        ));
    }
    let mut children = std::fs::read_dir(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    children.sort_by_key(std::fs::DirEntry::path);
    for child in children {
        let child_path = child.path();
        if child
            .file_type()
            .map_err(|error| format!("{}: {error}", child_path.display()))?
            .is_dir()
            || child_path.extension() == Some(OsStr::new("seismic"))
        {
            collect(&child_path, files)?;
        }
    }
    Ok(())
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
