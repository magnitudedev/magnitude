//! What a native source reads of its declaration's tuning parameters,
//! read lexically from the source with every include inlined: checking a
//! module derives from them which parameters change a launch's generated
//! code, and refuses a source whose template headers or reads disagree
//! with its declaration.

use crate::checked::{NativeImplementation, NativeLaunch};
use crate::native_name::native_macro;
use crate::registry::{BackendName, SourceDialect};
use crate::syntax::ast::RowsKind;

/// Every identifier outside comments and literals, with its offset.
pub fn identifiers(source: &str) -> Vec<(&str, usize)> {
    tokens(source)
        .filter(|token| token.kind == TokenKind::Identifier)
        .map(|token| (token.text, token.offset))
        .collect()
}

/// One preprocessing token of a native source: an identifier, a number or a
/// punctuator, outside comments and string and character literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token<'s> {
    pub kind: TokenKind,
    pub text: &'s str,
    pub offset: usize,
    /// Whether it begins its logical line (lines joined by a trailing
    /// backslash): a `#` that does is a directive's.
    pub first: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Identifier,
    /// A preprocessing number (`1u`, `0x1Fu`, `1.5e-3f`), whose letters are
    /// no identifier.
    Number,
    /// `##` and `::` are one punctuator each; any other is one character.
    Punctuator,
}

/// The preprocessing tokens of `source`, in order.
pub fn tokens(source: &str) -> Tokens<'_> {
    Tokens {
        source,
        index: 0,
        started: false,
    }
}

/// The preprocessing tokens of a source ([`tokens`]).
pub struct Tokens<'s> {
    source: &'s str,
    index: usize,
    /// Whether the current logical line has had a token (or a literal).
    started: bool,
}

impl<'s> Iterator for Tokens<'s> {
    type Item = Token<'s>;

    fn next(&mut self) -> Option<Token<'s>> {
        let source = self.source;
        let bytes = source.as_bytes();
        let mut index = self.index;
        while index < bytes.len() {
            let start = index;
            let kind = match bytes[index] {
                b'\\' if bytes.get(index + 1) == Some(&b'\n') => {
                    index += 2;
                    continue;
                }
                b'\\' if bytes.get(index + 1) == Some(&b'\r') && bytes.get(index + 2) == Some(&b'\n') => {
                    index += 3;
                    continue;
                }
                b'\n' => {
                    self.started = false;
                    index += 1;
                    continue;
                }
                byte if byte.is_ascii_whitespace() => {
                    index += 1;
                    continue;
                }
                b'/' if bytes.get(index + 1) == Some(&b'/') => {
                    index += 2;
                    while index < bytes.len() && bytes[index] != b'\n' {
                        index += 1;
                    }
                    continue;
                }
                b'/' if bytes.get(index + 1) == Some(&b'*') => {
                    index += 2;
                    while index + 1 < bytes.len()
                        && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                    {
                        index += 1;
                    }
                    index = (index + 2).min(bytes.len());
                    continue;
                }
                b'"' | b'\'' => {
                    let quote = bytes[index];
                    index += 1;
                    while index < bytes.len() {
                        if bytes[index] == b'\\' {
                            index = (index + 2).min(bytes.len());
                        } else if bytes[index] == quote {
                            index += 1;
                            break;
                        } else {
                            index += 1;
                        }
                    }
                    self.started = true;
                    continue;
                }
                byte if byte == b'_' || byte.is_ascii_alphabetic() => {
                    index += 1;
                    while index < bytes.len()
                        && (bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric())
                    {
                        index += 1;
                    }
                    TokenKind::Identifier
                }
                byte if byte.is_ascii_digit()
                    || (byte == b'.' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit)) =>
                {
                    index += 1;
                    while index < bytes.len() {
                        match bytes[index] {
                            b'+' | b'-' if matches!(bytes[index - 1], b'e' | b'E' | b'p' | b'P') => index += 1,
                            byte if byte == b'_' || byte == b'.' || byte.is_ascii_alphanumeric() => index += 1,
                            _ => break,
                        }
                    }
                    TokenKind::Number
                }
                b'#' if bytes.get(index + 1) == Some(&b'#') => {
                    index += 2;
                    TokenKind::Punctuator
                }
                b':' if bytes.get(index + 1) == Some(&b':') => {
                    index += 2;
                    TokenKind::Punctuator
                }
                _ => {
                    // One character, whole if it is not ASCII.
                    index += source[index..].chars().next().map_or(1, char::len_utf8);
                    TokenKind::Punctuator
                }
            };
            self.index = index;
            let first = !self.started;
            self.started = true;
            return Some(Token {
                kind,
                text: &source[start..index],
                offset: start,
                first,
            });
        }
        self.index = index;
        None
    }
}

/// The template parameters of launch `launch`'s kernel definition in a
/// launch-scoped Metal or CUDA `source` (`tokens` its [`identifiers`]):
/// `None` when the definition has no well-formed `template <..>` header of
/// integer parameters. `grid_blocks`: the backend authors a written-grid
/// launch as a block (`BackendName::authors_written_grid_blocks`).
pub fn launch_template<'s>(
    dialect: SourceDialect,
    grid_blocks: bool,
    launch: &NativeLaunch,
    source: &'s str,
    tokens: &[(&'s str, usize)],
) -> Result<Option<Vec<&'s str>>, String> {
    // A written-grid launch of a backend that authors it as a block
    // is that block's body, which the runtime's generated kernel
    // runs.
    let (qualified, defined) = match dialect {
        SourceDialect::Metal => ("kernel", launch.kernel.clone()),
        SourceDialect::Cuda if grid_blocks && launch.grid.is_some() => {
            ("__device__", crate::registry::written_grid_block(&launch.kernel))
        }
        SourceDialect::Cuda => ("__global__", launch.kernel.clone()),
        SourceDialect::Vulkan => unreachable!("a Vulkan source declares no launch templates"),
    };
    let definitions = tokens
        .windows(3)
        .filter_map(|triple| {
            let [(qualifier, offset), (void, _), (name, _)] = triple else {
                unreachable!()
            };
            (*qualifier == qualified && *void == "void" && *name == defined)
                .then_some(*offset)
        })
        .collect::<Vec<_>>();
    let [kernel_at] = definitions.as_slice() else {
        return Err(format!("expected one GPU kernel definition for `{defined}`"));
    };
    let guard = format!("SEISMIC_FORMING_{}", native_macro(&launch.kernel));
    let mut guards = Vec::<(Option<&str>, usize)>::new();
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        if offset >= *kernel_at {
            break;
        }
        let directive = line.trim_start();
        if let Some(name) = directive.strip_prefix("#ifdef ") {
            guards.push((Some(name.trim()), offset));
        } else if directive.starts_with("#if ") || directive.starts_with("#ifndef ") {
            guards.push((None, offset));
        } else if directive.starts_with("#else") || directive.starts_with("#elif") {
            if let Some(top) = guards.last_mut() {
                top.0 = None;
            }
        } else if directive.starts_with("#endif") {
            guards.pop();
        }
        offset += line.len();
    }
    let Some((_, guarded_at)) = guards
        .iter()
        .rev()
        .find(|(name, _)| *name == Some(guard.as_str()))
    else {
        return Err(format!("definition must be inside `#ifdef {guard}`"));
    };
    let template = tokens
        .iter()
        .rev()
        .find(|(name, at)| *name == "template" && *at > *guarded_at && *at < *kernel_at)
        .map(|(_, at)| *at);
    let Some(at) = template else {
        return Ok(None);
    };
    let header = source[at + "template".len()..*kernel_at].trim();
    // A CUDA kernel may declare its launch bounds between its template
    // header and `__global__`.
    let header = match (dialect, header.rfind("__launch_bounds__")) {
        (SourceDialect::Cuda, Some(bounds)) if header.ends_with(')') => header[..bounds].trim_end(),
        _ => header,
    };
    // Text between the last `template` and the kernel that is not one
    // `<...>` header is a templated helper's: the kernel has none.
    let Some(body) = header.strip_prefix('<').and_then(|header| header.strip_suffix('>')) else {
        return Ok(None);
    };
    if body.trim().is_empty() {
        return Ok(Some(Vec::new()));
    }
    body.split(',')
        .map(|field| {
            let mut words = field.split_whitespace();
            match (words.next(), words.next(), words.next()) {
                (Some("int" | "uint" | "unsigned"), Some(name), None) => Ok(name),
                _ => Err(format!(
                    "template parameter `{}` of `{defined}` is not one `int` or `uint` code parameter",
                    field.trim()
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// The kind of rows a CUDA kernel's prologue forms: `projection::Strided`
/// stages them (`staged`), `Rms`, `GroupRms` (normalizing each part of a
/// row, `parts`) and `Input` (normalizing, or summing partials) form them in
/// the kernel (`rms`), and every other prologue passes them as given
/// (`plain`).
fn prologue_rows(source: &str) -> RowsKind {
    let names = identifiers(source);
    let named = |wanted: &[&str]| {
        names.windows(2).any(|pair| pair[0].0 == "projection" && wanted.contains(&pair[1].0))
    };
    if named(&["Strided"]) {
        RowsKind::Staged
    } else if named(&["Rms", "GroupRms", "Input", "OrderedRms", "OrderedInput"]) {
        RowsKind::Rms
    } else {
        RowsKind::Plain
    }
}

/// A projecting CUDA entry's declared kind of rows (`rows KIND`) against its
/// kernel's prologue.
pub fn check_rows_kind(kind: RowsKind, source: &str) -> Result<(), String> {
    let formed = prologue_rows(source);
    if formed == kind {
        Ok(())
    } else {
        Err(format!(
            "it declares `rows {}`, but its prologue forms `{}` rows (`projection::Strided`: staged; `Rms`, `GroupRms` or `Input`: rms; any other: plain)",
            kind.name(),
            formed.name()
        ))
    }
}

/// What a native source fails to read of its declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeReadError {
    /// Launch `kernel`'s definition does not match its forming guard, or
    /// its template parameters are not the launch's code parameters in
    /// declaration order.
    Template { kernel: String, detail: String },
    /// No launch's source or declaration reads declared parameter
    /// `parameter`.
    Unread { parameter: String },
    /// The implementation's launches take code parameters from its source
    /// (a launch-scoped Metal or CUDA implementation), and none was given.
    Unsourced,
    /// The implementation's sub-tile cannot be derived from its source.
    SubTile(&'static str),
}

impl std::fmt::Display for NativeReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Template { kernel, detail } => write!(f, "launch `{kernel}`: {detail}"),
            Self::Unread { parameter } => write!(
                f,
                "native parameter `{parameter}` is never read: no launch's source or declaration reads it"
            ),
            Self::SubTile(detail) => write!(f, "sub-tile domain: {detail}"),
            Self::Unsourced => f.write_str(
                "its launches take their code parameters from its source, and no source was given",
            ),
        }
    }
}

/// The sub-tile the Vulkan projection GEMM template (`projection.glsl`)
/// states it cannot run: it defines `PROJECTION_GEMM_SPILLING_SUB_M` and
/// `PROJECTION_GEMM_SPILLING_SUB_N`, and every implementation whose source
/// includes the template and reads the tuned sub-tile
/// (`SEISMIC_TUNE_SUB_M`, `SEISMIC_TUNE_SUB_N`) inherits that exclusion, so
/// the template alone states its domain and no declaration can omit it. A
/// source that reads the tuned sub-tile without the template, or the
/// template without the tuned sub-tile (constant sub-tiles), is not
/// constrained. A spilling pipeline is also the slowest to compile, which a
/// compile that cannot be interrupted turns into a tuning budget overrun.
fn inherit_projection_gemm_domain(
    implementation: &mut NativeImplementation,
    source: Option<&str>,
    tokens: &[(&str, usize)],
) -> Result<(), NativeReadError> {
    use crate::checked::{NativeComparison, NativeCondition, NativeNatExpr};
    let declares = ["SUB_M", "SUB_N"]
        .iter()
        .all(|name| implementation.params.iter().any(|parameter| parameter.name == *name));
    if implementation.backend != BackendName::Vulkan || !declares {
        return Ok(());
    }
    let Some(source) = source else {
        return Err(NativeReadError::SubTile(
            "it declares a sub-tile, whose domain the source states, and no source was given",
        ));
    };
    let reads = |name: &str| tokens.iter().any(|(token, _)| *token == name);
    if !(reads("SEISMIC_TUNE_SUB_M") && reads("SEISMIC_TUNE_SUB_N")) {
        return Ok(());
    }
    let spilling = |name: &str| -> Option<u64> {
        source.lines().find_map(|line| {
            let mut words = line.trim_start().strip_prefix("#define")?.split_whitespace();
            if words.next()? != name {
                return None;
            }
            words.next()?.trim_end_matches(['u', 'U']).parse().ok()
        })
    };
    // A define the source carries must parse as `#define NAME <integer>`: the
    // rule is never skipped for one that does not.
    let carried = |name: &str| reads(name);
    let (m, n) = match (
        carried("PROJECTION_GEMM_SPILLING_SUB_M"),
        carried("PROJECTION_GEMM_SPILLING_SUB_N"),
    ) {
        (false, false) => return Ok(()),
        (true, true) => match (
            spilling("PROJECTION_GEMM_SPILLING_SUB_M"),
            spilling("PROJECTION_GEMM_SPILLING_SUB_N"),
        ) {
            (Some(m), Some(n)) => (m, n),
            _ => {
                return Err(NativeReadError::SubTile(
                    "PROJECTION_GEMM_SPILLING_SUB_M and _N must each be defined as `#define NAME <integer>`",
                ));
            }
        },
        _ => {
            return Err(NativeReadError::SubTile(
                "the template defines one of PROJECTION_GEMM_SPILLING_SUB_M and _N without the other",
            ));
        }
    };
    let differs = |name: &str, value: u64| NativeCondition::Compare {
        comparison: NativeComparison::Ne,
        left: NativeNatExpr::Parameter(name.to_owned()),
        right: NativeNatExpr::Constant(value),
    };
    let rule = NativeCondition::Or(Box::new(differs("SUB_M", m)), Box::new(differs("SUB_N", n)));
    implementation.constraint = Some(match implementation.constraint.take() {
        Some(declared) => NativeCondition::And(Box::new(declared), Box::new(rule)),
        None => rule,
    });
    Ok(())
}

/// Derive which of the implementation's parameters change generated code
/// from its source (with every include inlined): in a launch-scoped Metal
/// or CUDA source, those a launch's kernel takes as template parameters,
/// which must be the launch's code parameters in declaration order
/// ([`NativeImplementation::code_parameters`]). Every declared parameter
/// must be read: by the source (a template parameter, its
/// `SEISMIC_RUNTIME_` or `SEISMIC_TUNE_` macro, or a CPU form's accessor)
/// or by the declaration's launch geometry, conditions or scratch. Without
/// a source no parameter changes code, which only an implementation whose
/// launches take no template parameters admits.
pub fn derive_codes(
    implementation: &mut NativeImplementation,
    source: Option<&str>,
) -> Result<(), NativeReadError> {
    derive_code_parameters(implementation, source)?;
    check_alike(implementation)
}

/// A launch declared `alike` the native's one-row launch is claimed to
/// compute each row as that launch computes it alone. This checks only what
/// the declaration can show: both launches read the same reduction-order
/// settings (`arithmetic` parameters) and neither declares an order of its
/// own, so the orders are the native's, one setting each for every row class.
/// It says nothing about the bodies' arithmetic, which the row-equality tests
/// hold. A launch's reads are its dispatch's (not its `when` condition, which
/// selects the launch by rows and is no reduction) and, in a templated
/// source, its code parameters.
fn check_alike(implementation: &NativeImplementation) -> Result<(), NativeReadError> {
    let orders = |ordinal: usize| {
        let mut names = implementation.launches[ordinal].dispatch_parameters();
        names.extend(
            implementation.code_parameters(ordinal).into_iter().map(|(_, name)| name.to_owned()),
        );
        names.retain(|name| implementation.params.iter().any(|own| &own.name == name && own.arithmetic));
        names.sort();
        names.dedup();
        names
    };
    for (ordinal, launch) in implementation.launches.iter().enumerate() {
        let Some(alone) = launch.alike else { continue };
        let one_row = &implementation.launches[alone];
        let refuse = |detail: String| NativeReadError::Template { kernel: launch.kernel.clone(), detail };
        for own in [launch, one_row] {
            if let Some(order) = own.params.iter().find(|parameter| parameter.arithmetic) {
                return Err(refuse(format!(
                    "`alike` launches share the native's reduction orders; `{}` declares `arithmetic` parameter `{}` of its own",
                    own.kernel, order.name
                )));
            }
        }
        let (mine, theirs) = (orders(ordinal), orders(alone));
        if mine != theirs {
            return Err(refuse(format!(
                "`alike` `{}` must read the same reduction orders; reads {mine:?}, `{}` reads {theirs:?}",
                one_row.kernel, one_row.kernel
            )));
        }
    }
    Ok(())
}

fn derive_code_parameters(
    implementation: &mut NativeImplementation,
    source: Option<&str>,
) -> Result<(), NativeReadError> {
    let backend = implementation.backend;
    let templated = match backend.source_dialect() {
        Some(dialect @ (SourceDialect::Metal | SourceDialect::Cuda))
            if implementation.launch_scoped() =>
        {
            Some(dialect)
        }
        _ => None,
    };
    let Some(source) = source else {
        if templated.is_some() {
            return Err(NativeReadError::Unsourced);
        }
        inherit_projection_gemm_domain(implementation, None, &[])?;
        for parameter in implementation
            .params
            .iter_mut()
            .chain(implementation.launches.iter_mut().flat_map(|launch| &mut launch.params))
        {
            parameter.code = false;
        }
        return Ok(());
    };
    let tokens = match backend.source_dialect() {
        Some(_) => identifiers(source),
        None => Vec::new(),
    };
    inherit_projection_gemm_domain(implementation, Some(source), &tokens)?;
    // Each launch's template header, where it has one.
    let mut headers: Vec<Option<Vec<String>>> = vec![None; implementation.launches.len()];
    if let Some(dialect) = templated {
        for (ordinal, launch) in implementation.launches.iter().enumerate() {
            headers[ordinal] = launch_template(
                dialect,
                backend.authors_written_grid_blocks(),
                launch,
                source,
                &tokens,
            )
            .map_err(|detail| NativeReadError::Template {
                kernel: launch.kernel.clone(),
                detail,
            })?
            .map(|names| names.into_iter().map(str::to_owned).collect());
        }
    }
    let templated_by = |ordinal: usize, name: &str| {
        headers[ordinal]
            .as_ref()
            .is_some_and(|names| names.iter().any(|candidate| candidate == name))
    };
    // A CPU form is Rust, whose lifetimes the C-family tokenizer reads as
    // character literals: its accessor calls are found in the text.
    let named = |name: &str| match backend {
        BackendName::Cpu => source.contains(&format!("param_{}(", name.to_lowercase())),
        _ => {
            let runtime = format!("SEISMIC_RUNTIME_{}", native_macro(name));
            let tune = format!("SEISMIC_TUNE_{}", native_macro(name));
            tokens.iter().any(|(token, _)| *token == runtime || *token == tune)
        }
    };
    let mut declared = Vec::new();
    for scratch in &implementation.scratch {
        scratch.bytes.parameters(&mut declared);
        if let Some(when) = &scratch.when {
            when.parameters(&mut declared);
        }
    }
    if let Some(repeat) = &implementation.repeat {
        repeat.count.parameters(&mut declared);
    }
    let launch_reads = implementation
        .launches
        .iter()
        .map(NativeLaunch::parameters)
        .collect::<Vec<_>>();
    let unread = |name: &str| NativeReadError::Unread {
        parameter: name.to_owned(),
    };
    let launches = implementation.launches.len();
    for parameter in implementation.params.iter_mut() {
        if parameter.role != crate::checked::NativeParameterRole::Declared {
            continue;
        }
        parameter.code = (0..launches).any(|ordinal| templated_by(ordinal, &parameter.name));
        let read = parameter.code
            || named(&parameter.name)
            || declared.contains(&parameter.name)
            || launch_reads.iter().any(|reads| reads.contains(&parameter.name));
        if !read {
            return Err(unread(&parameter.name));
        }
    }
    for (ordinal, launch) in implementation.launches.iter_mut().enumerate() {
        let reads = &launch_reads[ordinal];
        for parameter in launch.params.iter_mut() {
            parameter.code = templated_by(ordinal, &parameter.name);
            let read = parameter.code || named(&parameter.name) || reads.contains(&parameter.name);
            if !read {
                return Err(unread(&parameter.name));
            }
        }
    }
    // Each header states exactly its launch's code parameters, in
    // declaration order: an entry parameter in the header of every launch
    // that reads it (of every launch, when none does).
    for (ordinal, header) in headers.iter().enumerate() {
        let expected = implementation
            .code_parameters(ordinal)
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>();
        let matches = match header {
            Some(names) => names.iter().map(String::as_str).eq(expected.iter().copied()),
            None => expected.is_empty(),
        };
        if !matches {
            return Err(NativeReadError::Template {
                kernel: implementation.launches[ordinal].kernel.clone(),
                detail: format!(
                    "template parameters must be `{expected:?}` in declaration order; found {header:?}"
                ),
            });
        }
    }
    Ok(())
}

/// A native source for `implementation` whose launches take the declared
/// parameters `codes` names as template parameters, in the order a header
/// states them ([`NativeImplementation::code_parameters`]), and that reads
/// every other declared parameter as a runtime word: what a test reads for
/// a declaration it has no source file of.
#[cfg(any(test, feature = "fixtures"))]
pub fn fixture_source(implementation: &NativeImplementation, codes: &[&str]) -> String {
    use crate::checked::NativeParameterRole;
    let mut coded = implementation.clone();
    for parameter in coded
        .params
        .iter_mut()
        .chain(coded.launches.iter_mut().flat_map(|launch| &mut launch.params))
    {
        parameter.code =
            parameter.role == NativeParameterRole::Declared && codes.contains(&parameter.name.as_str());
    }
    let backend = coded.backend;
    let scoped = coded.launch_scoped();
    let word = |name: &str| match backend {
        BackendName::Cpu => format!("let _ = cx.param_{}();", name.to_lowercase()),
        _ if scoped => format!("(void)SEISMIC_RUNTIME_{};", native_macro(name)),
        _ => format!("(void)SEISMIC_TUNE_{};", native_macro(name)),
    };
    let mut source = String::new();
    for (ordinal, launch) in coded.launches.iter().enumerate() {
        let header = coded
            .code_parameters(ordinal)
            .into_iter()
            .map(|(_, name)| format!("unsigned {name}"))
            .collect::<Vec<_>>();
        let words = coded
            .params
            .iter()
            .chain(&launch.params)
            .filter(|parameter| parameter.role == NativeParameterRole::Declared && !parameter.code)
            .map(|parameter| word(&parameter.name))
            .collect::<String>();
        let (qualifier, name) = match backend {
            BackendName::Cuda if backend.authors_written_grid_blocks() && launch.grid.is_some() => {
                ("__device__", crate::registry::written_grid_block(&launch.kernel))
            }
            BackendName::Cuda => ("__global__", launch.kernel.clone()),
            _ => ("kernel", launch.kernel.clone()),
        };
        let template = match header.is_empty() {
            true => String::new(),
            false => format!("template <{}>\n", header.join(", ")),
        };
        source.push_str(&format!(
            "#ifdef SEISMIC_FORMING_{}\n{template}{qualifier} void {name}() {{ {words} }}\n#endif\n",
            native_macro(&launch.kernel)
        ));
    }
    source
}

/// Check `sources`, reading for every native implementation its
/// [`fixture_source`] with `codes` as its template parameters.
#[cfg(any(test, feature = "fixtures"))]
pub fn check_fixture(
    sources: crate::checked::SourceSet,
    codes: &[&str],
) -> Result<crate::checked::CheckedModule, crate::checked::SourceError> {
    crate::checked::check_source_with(sources, &mut |_entry: &str, implementation: &NativeImplementation| {
        Ok::<_, std::convert::Infallible>(Some(fixture_source(implementation, codes).into()))
    })
    .map_err(|failure| match failure {
        crate::checked::CheckFailure::Source(error) => error,
        crate::checked::CheckFailure::Read(never) => match never {},
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checked::{check_source_with, CheckFailure, CheckedModule, SourceFile, SourceSet};

    /// Numbers, `##` and `::` are one token each, and a backslash joins
    /// lines: a directive's `#` is the first token of its logical line.
    #[test]
    fn tokens_are_preprocessing_tokens_of_logical_lines() {
        let source = "#define CAT(a, b) a ## b /* x */ \\\n  ::q\n  # if 0x1Fu + 1.5e-3f // y\n'#' \"z\" #\n";
        let tokens: Vec<(TokenKind, &str, bool)> =
            tokens(source).map(|token| (token.kind, token.text, token.first)).collect();
        use TokenKind::{Identifier as I, Number as N, Punctuator as P};
        assert_eq!(
            tokens,
            [
                (P, "#", true),
                (I, "define", false),
                (I, "CAT", false),
                (P, "(", false),
                (I, "a", false),
                (P, ",", false),
                (I, "b", false),
                (P, ")", false),
                (I, "a", false),
                (P, "##", false),
                (I, "b", false),
                (P, "::", false),
                (I, "q", false),
                (P, "#", true),
                (I, "if", false),
                (N, "0x1Fu", false),
                (P, "+", false),
                (N, "1.5e-3f", false),
                (P, "#", false),
            ]
        );
        assert_eq!(identifiers("x = 0x1Fu;"), [("x", 0)]);
    }

    /// A projecting CUDA kernel's prologue is exactly the kind its declaration names.
    #[test]
    fn a_declared_kind_of_rows_must_be_its_prologues() {
        let rms = "using Raw = projection::Rms<float, projection::SelectedRows>;";
        let input = "using Raw = projection::Input<float, projection::AllRows, NM, PT>;";
        let staged = "using Raw = projection::Strided<element::F32, projection::AllRows>;";
        let plain = "using Raw = projection::Plain<float, projection::AllRows>;";
        for (kind, source) in [
            (RowsKind::Rms, rms),
            (RowsKind::Rms, input),
            (RowsKind::Staged, staged),
            (RowsKind::Plain, plain),
        ] {
            assert_eq!(check_rows_kind(kind, source), Ok(()), "{kind:?} over {source}");
        }
        // Each kind refuses each other kind's prologue, `staged` and `rms` included.
        for (kind, source) in [
            (RowsKind::Staged, rms),
            (RowsKind::Staged, input),
            (RowsKind::Rms, staged),
            (RowsKind::Plain, rms),
            (RowsKind::Plain, staged),
            (RowsKind::Rms, plain),
            (RowsKind::Staged, plain),
        ] {
            assert!(check_rows_kind(kind, source).is_err(), "{kind:?} over {source}");
        }
        // A comment does not name a prologue.
        assert_eq!(check_rows_kind(RowsKind::Plain, "// projection::Rms rows\n"), Ok(()));
    }

    fn sources(native: &str) -> SourceSet {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!(
                "fn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\n{native}"
            ),
        });
        sources
    }

    /// `native` checked with `source` as its implementation's source.
    fn checked(native: &str, source: &str) -> Result<CheckedModule, String> {
        check_source_with(sources(native), &mut |_entry: &str, _native: &NativeImplementation| {
            Ok::<_, std::convert::Infallible>(Some(source.to_owned().into()))
        })
        .map_err(|failure| match failure {
            CheckFailure::Source(error) => error.to_string(),
            CheckFailure::Read(never) => match never {},
        })
    }

    const DECLARATION: &str = "native scale for metal from \"scale.metal\":\n    params (SPLIT in [1, 2] default 1)\n    launch small when N < 8:\n        params (ROWS in [1, 2] default 1, WIDTH in [32, 64] default 32)\n        threadgroups (ceil_div(N, ROWS), SPLIT, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n";

    const SOURCE: &str = "#ifdef SEISMIC_FORMING_SMALL\ntemplate <uint ROWS>\nkernel void small() { (void)SEISMIC_RUNTIME_SPLIT; }\n#endif\n";

    /// A parameter a launch's kernel takes as a template parameter changes
    /// its code; one the source reads as a runtime word, or that only the
    /// launch geometry reads, does not; one nothing reads is refused.
    #[test]
    fn codes_are_derived_when_checked_and_reads_are_required() {
        let module = checked(DECLARATION, SOURCE).expect("the source reads its declaration");
        let entry = module.entries()[0].id;
        let native = module.native_implementation(entry, BackendName::Metal).unwrap();
        assert!(!native.params[0].code, "SPLIT is a runtime word");
        assert!(native.launches[0].params[0].code, "ROWS is a template parameter");
        assert!(!native.launches[0].params[1].code, "WIDTH only shapes the launch");
        assert_eq!(module.native_asset(entry, BackendName::Metal).unwrap().text, SOURCE);

        let unread = DECLARATION.replace(
            "params (SPLIT in [1, 2] default 1)\n",
            "params (SPLIT in [1, 2] default 1, UNUSED in [1, 2] default 1)\n",
        );
        let error = checked(&unread, SOURCE).unwrap_err();
        assert!(error.contains("native parameter `UNUSED` is never read"), "{error}");
    }

    /// The projection GEMM template states the sub-tile it cannot run
    /// (`PROJECTION_GEMM_SPILLING_SUB_M`, `_SUB_N`); an implementation whose
    /// source includes the template and tunes its sub-tile admits no such
    /// sub-tile whatever its declaration lists. One that tunes the sub-tile
    /// without the template, or is not Vulkan's, admits it; a Vulkan one
    /// declaring the sub-tile without a source does not check.
    #[test]
    fn a_source_including_the_projection_gemm_inherits_its_sub_tile_domain() {
        /// The test's directory, removed however the test ends.
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let scratch = Scratch(std::env::temp_dir().join(format!("sub-tile-domain-{}", std::process::id())));
        let directory = scratch.0.clone();
        std::fs::create_dir_all(directory.join("lib")).unwrap();
        std::fs::write(
            directory.join("lib/template.glsl"),
            "#define PROJECTION_GEMM_SPILLING_SUB_M 64\n#define PROJECTION_GEMM_SPILLING_SUB_N 64\nvoid projection_gemm(uint tm, uint tn, uint wm, uint wn) {}\n",
        )
        .unwrap();
        let body = "void main() { projection_gemm(SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N, SEISMIC_TUNE_SUB_M, SEISMIC_TUNE_SUB_N); }\n";
        std::fs::write(directory.join("template.comp"), format!("#include \"lib/template.glsl\"\n{body}")).unwrap();
        std::fs::write(directory.join("own.comp"), body).unwrap();
        // Metal's includes are its own `.h` files.
        std::fs::write(
            directory.join("lib/template.h"),
            "#define PROJECTION_GEMM_SPILLING_SUB_M 64\n#define PROJECTION_GEMM_SPILLING_SUB_N 64\n",
        )
        .unwrap();
        std::fs::write(directory.join("template.metal"), format!("#include \"lib/template.h\"\n{body}")).unwrap();
        let declaration = |backend: &str, file: &str| {
            format!(
                "native scale for {backend} from \"{file}\":\n    params (TILE_M in [32, 64] default 32, TILE_N in [128, 64] default 128, SUB_M in [32, 64] default 32, SUB_N in [32, 64] default 32)\n    where SUB_M <= TILE_M and SUB_N <= TILE_N\n    launch scale:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"
            )
        };
        let tiles = |backend: BackendName, file: &str| {
            let module = crate::source::check_beside(sources(&declaration(backend.as_str(), file)), &directory)
                .expect("the tile declaration checks");
            module
                .native_implementation(module.entries()[0].id, backend)
                .unwrap()
                .admissible(&crate::checked::NativeSpecialization::new())
                .unwrap()
                .iter()
                .map(|point| (point.param("SUB_M").unwrap(), point.param("SUB_N").unwrap()))
                .collect::<Vec<_>>()
        };
        let vulkan = tiles(BackendName::Vulkan, "template.comp");
        assert!(!vulkan.is_empty());
        assert!(!vulkan.contains(&(64, 64)));
        assert!(vulkan.contains(&(64, 32)) && vulkan.contains(&(32, 64)));
        assert!(tiles(BackendName::Vulkan, "own.comp").contains(&(64, 64)), "not the template");
        assert!(tiles(BackendName::Metal, "template.metal").contains(&(64, 64)), "not Vulkan's");
        let error = crate::checked::check_source(sources(&declaration("vulkan", "template.comp"))).unwrap_err().to_string();
        assert!(error.contains("no source was given"), "{error}");
        std::fs::write(
            directory.join("lib/template.glsl"),
            "#define PROJECTION_GEMM_SPILLING_SUB_M (64)\n#define PROJECTION_GEMM_SPILLING_SUB_N 64\n",
        )
        .unwrap();
        let error = crate::source::check_beside(sources(&declaration("vulkan", "template.comp")), &directory)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must each be defined"), "{error}");
    }

    /// A header states its launch's code parameters in declaration order;
    /// a launch-scoped implementation without a source does not check.
    #[test]
    fn template_headers_state_the_code_parameters_in_order_and_need_a_source() {
        let declaration = DECLARATION.replace(
            "params (ROWS in [1, 2] default 1, WIDTH in [32, 64] default 32)",
            "params (ROWS in [1, 2] default 1, WIDTH in [32, 64] default 32, LANES in [1, 2] default 1)",
        ).replace("threads_per_threadgroup (WIDTH, 1, 1)", "threads_per_threadgroup (WIDTH * LANES, 1, 1)");
        let ordered = SOURCE.replace("template <uint ROWS>", "template <uint ROWS, uint LANES>");
        checked(&declaration, &ordered).expect("declaration order");
        let reversed = SOURCE.replace("template <uint ROWS>", "template <uint LANES, uint ROWS>");
        let error = checked(&declaration, &reversed).unwrap_err();
        assert!(error.contains("in declaration order"), "{error}");
        let error = crate::checked::check_source(sources(DECLARATION)).unwrap_err().to_string();
        assert!(error.contains("no source was given"), "{error}");
    }

    /// A fixture source states the codes it is given and checks.
    #[test]
    fn a_fixture_source_checks_with_the_codes_it_names() {
        let module = check_fixture(sources(DECLARATION), &["ROWS"]).expect("a fixture source checks");
        let entry = module.entries()[0].id;
        let native = module.native_implementation(entry, BackendName::Metal).unwrap();
        assert!(native.launches[0].params[0].code);
        assert!(!native.launches[0].params[1].code);
    }
}
