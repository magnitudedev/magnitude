//! Library invocation files (`library/{std,engine}.invocations`, design A10 §2.8).
//!
//! ```text
//! # origin: H-24                      (before the first line: applies to the file)
//! # origin: H-27 H-28                 (applies to the following line)
//! rms_norm[T=f32]: f32[3,64]=rand(1) ; f32[64]=rand(2) ; f32[3,64]=zero ; f32:0.000001
//! ```
//!
//! Other `#` lines are comments. Anything else panics `<file>:<line>: <reason>`.
use crate::scenario::{origin_line, parse_arguments, ArgumentSpec, Origin};
use seismic_lang::registry;
use std::path::Path;

pub struct LibraryFile {
    /// Origins stated before the first invocation line.
    pub origins: Vec<Origin>,
    pub invocations: Vec<LibraryInvocation>,
}

pub struct LibraryInvocation {
    /// 1-based line in its file, for failure reports.
    pub line: usize,
    pub entry: String,
    /// `<Name>=<element>` bindings of generic element parameters.
    pub elements: Vec<(String, String)>,
    pub arguments: Vec<ArgumentSpec>,
    pub origins: Vec<Origin>,
}

pub fn load(path: &Path) -> LibraryFile {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse(&path.display().to_string(), &text)
}

pub fn parse(file: &str, text: &str) -> LibraryFile {
    let mut library = LibraryFile {
        origins: Vec::new(),
        invocations: Vec::new(),
    };
    let mut pending: Option<(usize, Vec<Origin>)> = None;
    for (index, line) in text.lines().enumerate() {
        let fail = |reason: String| -> ! { panic!("{file}:{}: {reason}", index + 1) };
        let line = line.trim();
        if let Some(origins) = origin_line(line) {
            let origins = origins.unwrap_or_else(|e| fail(e));
            if library.invocations.is_empty() && pending.is_none() {
                library.origins.extend(origins);
            } else if pending.replace((index + 1, origins)).is_some() {
                fail("two `# origin:` lines precede one invocation".into());
            }
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (head, arguments) = line
            .split_once(':')
            .unwrap_or_else(|| fail(format!("`{line}` is not `entry[bindings]: args`")));
        let (entry, elements) = parse_head(head).unwrap_or_else(|e| fail(e));
        library.invocations.push(LibraryInvocation {
            line: index + 1,
            entry,
            elements,
            arguments: parse_arguments(arguments).unwrap_or_else(|e| fail(e)),
            origins: pending.take().map_or_else(Vec::new, |(_, origins)| origins),
        });
    }
    if let Some((line, _)) = pending {
        panic!("{file}:{line}: `# origin:` is not followed by an invocation");
    }
    library
}

/// `entry` or `entry[Name=element,...]`.
fn parse_head(head: &str) -> Result<(String, Vec<(String, String)>), String> {
    let Some((entry, bindings)) = head.split_once('[') else {
        return Ok((head.to_owned(), Vec::new()));
    };
    let bindings = bindings
        .strip_suffix(']')
        .ok_or_else(|| format!("`{head}` has an unclosed binding list"))?;
    let elements = bindings
        .split(',')
        .map(|binding| {
            let (name, element) = binding
                .split_once('=')
                .ok_or_else(|| format!("`{binding}` is not `Name=element`"))?;
            if registry::representation(element).is_none() {
                return Err(format!("`{element}` is not a registered representation"));
            }
            Ok((name.to_owned(), element.to_owned()))
        })
        .collect::<Result<_, _>>()?;
    Ok((entry.to_owned(), elements))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_origins_bindings_and_arguments() {
        let library = parse(
            "engine.invocations",
            "# origin: H-24\n\
             # Tier 0 geometry\n\
             argmax_row: f32[1,8]=rand(1) ; i32[1]=zero\n\
             # origin: H-27 H-28\n\
             rms_norm[T=f32,W=bf16]: f32[3,64]=rand(1) ; f32:0.000001\n",
        );
        assert_eq!(library.origins.len(), 1);
        assert_eq!(library.invocations.len(), 2);
        assert!(library.invocations[0].origins.is_empty());
        let second = &library.invocations[1];
        assert_eq!((second.line, second.entry.as_str()), (5, "rms_norm"));
        assert_eq!(
            second.elements,
            [("T".into(), "f32".into()), ("W".into(), "bf16".into())]
        );
        assert_eq!(second.origins.len(), 2);
        assert_eq!(second.arguments.len(), 2);
    }

    #[test]
    #[should_panic(expected = "std.invocations:1: `# origin:H-24` is not `# origin: <origin> ...`")]
    fn origin_without_space_panics() {
        parse(
            "std.invocations",
            "# origin:H-24\nargmax_row: i32[1]=zero\n",
        );
    }

    #[test]
    #[should_panic(expected = "std.invocations:1: `f33` is not a registered representation")]
    fn unknown_binding_element_panics() {
        parse("std.invocations", "rms_norm[T=f33]: f32:0.5\n");
    }

    #[test]
    #[should_panic(expected = "engine.invocations:2: `# origin:` is not followed by an invocation")]
    fn dangling_origin_panics() {
        parse(
            "engine.invocations",
            "argmax_row: i32[1]=zero\n# origin: H-27\n",
        );
    }
}
