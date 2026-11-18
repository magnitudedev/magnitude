//! The library boundary: an engine's native sources name only the public
//! surface of Seismic's native library (operand-system-clean.md §D1.7).
//!
//! Every authored Metal, CUDA and Vulkan file of a module (each asset and the
//! files it includes by relative path; never a `<seismic/…>` library file)
//! obeys four textual rules, and spells no ABI symbol of library visibility
//! (`Visibility`, assigned where the build generates each symbol):
//! 1. it names no `detail` of a library namespace (`packets::detail`), nor a
//!    library prefix's (`packets_detail_`, `PACKETS_DETAIL_`);
//! 2. it opens no library namespace, and `#define`s no name of a library macro
//!    prefix but the ones an includer sets ([`ENGINE_SET_PREFIXES`],
//!    [`ENGINE_SET_NAMES`]);
//! 3. it brings no library namespace in whole (`using namespace`);
//! 4. it pastes no ABI name: it names none of the library's paste macros
//!    ([`PASTE_MACROS`]); no `##` chain ends or begins with a literal of a
//!    library-visibility suffix family ([`TENSOR_FAMILIES`]), whatever else it
//!    holds; and every other chain either has a literal operand with a
//!    lowercase letter (ABI names are all caps) or begins with a literal no
//!    ABI prefix can begin, save the two segment stems ([`SEGMENT_STEMS`]).
//!    The suffix table names what a paste may not reach; the prefix and
//!    lowercase conditions close the routes it cannot name: a chain built
//!    from the front (`SEISMIC##X`), one of parameters alone (`A##B`), and
//!    suffixes of no tensor family (conversions', `C##_WORDS`, and library
//!    macros', `W##_MATRIX`).
//!
//! Rule 4 closes the route around the visibility check: a macro could paste
//! a library-visibility suffix onto a binding prefix it is passed.

use crate::BuildError;
use seismic_lang::native_source::{Token, TokenKind, tokens};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

/// Who may spell a generated ABI symbol in an authored native source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Visibility {
    /// Any source: the names the ABI fixes, an entry's own binding names
    /// (the bare prefixes entries pass as binding arguments) and facts.
    Surface,
    /// Seismic's native library alone: how a representation is laid out and
    /// decoded, and how a conversion runs.
    Library,
}

/// The suffix families of a tensor's, element's or result's prefix
/// (`SEISMIC_<NAME>`, `SEISMIC_PARAM_<n>`, `SEISMIC_ELEMENT_<E>`,
/// `SEISMIC_RESULT_<n>`): a suffix is visible as the first family it begins
/// with. The bare prefix is surface. A suffix of no family is an invariant
/// violation: every generated suffix is classified here.
pub(crate) const TENSOR_FAMILIES: &[(&str, Visibility)] = &[
    ("_RANK", Visibility::Surface),
    ("_EXTENT_", Visibility::Surface),
    ("_STRIDE_", Visibility::Surface),
    ("_KIND_", Visibility::Surface),
    ("_DECODED_", Visibility::Surface),
    ("_REPRESENTATION", Visibility::Library),
    ("_OPERANDS_", Visibility::Library),
    ("_LAYOUT_", Visibility::Library),
    ("_PACKET_", Visibility::Library),
    ("_ROW_", Visibility::Library),
    ("_GRID_", Visibility::Library),
    ("_MMA_", Visibility::Library),
    ("_PLANE_", Visibility::Library),
    ("_RECORD_", Visibility::Library),
    ("_LOGICAL_GROUP", Visibility::Library),
    ("_GROUP_MULTIPLE", Visibility::Library),
    ("_TILE_ROWS", Visibility::Library),
];

/// The suffix families of a conversion's prefix
/// (`SEISMIC_CONVERSION_<FROM>_<TO>`): every suffix is the library's.
pub(crate) const CONVERSION_FAMILIES: &[(&str, Visibility)] = &[("_", Visibility::Library)];

/// The visibility of `<prefix><suffix>` under `families`.
pub(crate) fn visibility(suffix: &str, families: &[(&str, Visibility)]) -> Visibility {
    if suffix.is_empty() {
        return Visibility::Surface;
    }
    families
        .iter()
        .find(|(family, _)| suffix.starts_with(family))
        .map(|(_, visibility)| *visibility)
        .unwrap_or_else(|| {
            panic!("ABI suffix `{suffix}` belongs to no family of the visibility table")
        })
}

/// The library's namespaces (Metal, CUDA) and, as `<name>_`, its
/// identifier prefixes (Vulkan).
const LIBRARY_NAMESPACES: [&str; 9] = [
    "packets",
    "element",
    "slab",
    "producers",
    "codebooks",
    "grids",
    "rows16",
    "repack",
    "nvfp4",
];

/// The library's macro prefixes, which an engine source does not define.
const LIBRARY_MACROS: [&str; 4] = ["ELEMENT_", "PACKETS_", "SLAB_", "ROWS16_"];

/// The library macros an includer sets: configuration, by prefix
/// (`PACKETS_CONFIG_*`)...
const ENGINE_SET_PREFIXES: [&str; 1] = ["PACKETS_CONFIG_"];

/// ...and the rows16 binding, the table parameter's prefix, bound like
/// `KERNEL_Wn`.
const ENGINE_SET_NAMES: [&str; 1] = ["ROWS16_T"];

/// The library's paste macros, which reach any ABI name.
const PASTE_MACROS: [&str; 5] = [
    "ELEMENT_CAT",
    "ELEMENT_HAS",
    "ELEMENT_IS_ONE",
    "ELEMENT_WHEN",
    "PACKETS_CAT",
];

/// The prefixes of the names a paste could reach: ABI symbols and the
/// library's binding and slot macros.
const PASTED_PREFIXES: [&str; 5] = ["SEISMIC_", "KERNEL_", "ELEMENT_", "PACKETS_", "SLAB_"];

/// The ABI stems a projection pastes a segment's number onto
/// (`SEISMIC_SEGMENT_WEIGHTS_##s`), the one admitted ABI paste.
const SEGMENT_STEMS: [&str; 2] = ["SEISMIC_SEGMENT_WEIGHTS_", "SEISMIC_SEGMENT_ORDERED_"];

/// A boundary rule a source breaks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Rule {
    /// Rule 1: a library namespace's or prefix's `detail` name.
    Detail(String),
    /// Rule 2: a library namespace opened.
    Opened(String),
    /// Rule 2: a library macro defined.
    Defined(String),
    /// Rule 3: a library namespace brought in whole.
    UsingNamespace(String),
    /// Rule 4: a library paste macro named.
    PasteMacro(String),
    /// Rule 4: a `##` chain that could form an ABI name.
    Paste(String),
    /// A generated ABI symbol of library visibility spelled.
    LibrarySymbol(String),
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Detail(name) => write!(f, "`{name}` is the library's internal `detail`"),
            Self::Opened(name) => write!(f, "`namespace {name}` opens a library namespace"),
            Self::Defined(name) => write!(
                f,
                "`#define {name}` defines a library macro, which an includer does not set"
            ),
            Self::UsingNamespace(name) => {
                write!(
                    f,
                    "`using namespace {name}` brings a library namespace in whole; name its surface (`using {name}::X`)"
                )
            }
            Self::PasteMacro(name) => write!(
                f,
                "`{name}` pastes onto ABI names, which only the library may"
            ),
            Self::Paste(chain) => write!(
                f,
                "`{chain}` could paste an ABI name: a chain ends in no library-visibility suffix, and needs a literal operand with a lowercase letter or a leading literal no ABI prefix can begin"
            ),
            Self::LibrarySymbol(symbol) => {
                write!(
                    f,
                    "`{symbol}` is a library-visibility ABI symbol; reach it through the library's surface"
                )
            }
        }
    }
}

/// The boundary check of one build: each authored file's rules once,
/// however many entries include it.
#[derive(Default)]
pub(crate) struct Boundary {
    checked: HashSet<PathBuf>,
}

impl Boundary {
    /// Rules 1-4 over the authored file `path`, once per build: the first
    /// violation fails the build.
    pub(crate) fn check(&mut self, path: &Path, source: &str) -> Result<(), BuildError> {
        if !self.checked.insert(path.to_path_buf()) {
            return Ok(());
        }
        match violations(source).into_iter().next() {
            Some((offset, rule)) => Err(violation(path, source, offset, rule)),
            None => Ok(()),
        }
    }
}

/// The build's failure for `rule`, broken at `offset` of `path`.
pub(crate) fn violation(path: &Path, source: &str, offset: usize, rule: Rule) -> BuildError {
    let (line, column) = crate::internals::location(source, offset);
    BuildError::LibraryBoundary {
        path: path.to_path_buf(),
        line,
        column,
        rule: rule.to_string(),
    }
}

/// Every place `source` breaks rules 1-4, with the rule, in source order.
pub(crate) fn violations(source: &str) -> Vec<(usize, Rule)> {
    let tokens: Vec<Token<'_>> = tokens(source).collect();
    let mut found = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let before = |back: usize| index.checked_sub(back).map(|at| &tokens[at]);
        if token.kind != TokenKind::Identifier {
            continue;
        }
        let name = token.text;
        // Rule 1.
        if name == "detail" && before(1).is_some_and(|token| token.text == "::") {
            if let Some(namespace) = library(before(2)) {
                found.push((
                    namespace.offset,
                    Rule::Detail(format!("{}::detail", namespace.text)),
                ));
            }
        }
        if LIBRARY_NAMESPACES
            .iter()
            .any(|namespace| detail_prefixed(name, namespace))
        {
            found.push((token.offset, Rule::Detail(name.to_owned())));
        }
        // Rules 2 and 3: the namespace a `namespace` names, past a leading `::`.
        if name == "namespace" {
            let mut next = index + 1;
            if tokens.get(next).is_some_and(|token| token.text == "::") {
                next += 1;
            }
            if let Some(namespace) = library(tokens.get(next)) {
                let rule = if before(1).is_some_and(|token| token.text == "using") {
                    Rule::UsingNamespace(namespace.text.to_owned())
                } else {
                    Rule::Opened(namespace.text.to_owned())
                };
                found.push((token.offset, rule));
            }
        }
        // Rule 2: a definition's name.
        let defined = (name == "define"
            && before(1).is_some_and(|hash| hash.first && hash.text == "#"))
        .then(|| tokens.get(index + 1))
        .flatten()
        .filter(|defined| defined.kind == TokenKind::Identifier && library_macro(defined.text));
        if let Some(defined) = defined {
            found.push((defined.offset, Rule::Defined(defined.text.to_owned())));
        }
        // Rule 4: the paste macros.
        if PASTE_MACROS.contains(&name) {
            found.push((token.offset, Rule::PasteMacro(name.to_owned())));
        }
    }
    // Rule 4: every `##` chain of every definition.
    let mut index = 0;
    while index < tokens.len() {
        let directive = tokens[index].first
            && tokens[index].text == "#"
            && tokens
                .get(index + 1)
                .is_some_and(|token| token.text == "define");
        if !directive {
            index += 1;
            continue;
        }
        let end = tokens[index + 1..]
            .iter()
            .position(|token| token.first)
            .map_or(tokens.len(), |at| index + 1 + at);
        found.extend(definition_pastes(&tokens[index + 2..end]));
        index = end;
    }
    found.sort_by_key(|(offset, _)| *offset);
    found
}

/// Whether `text` is, or begins, a suffix of a library-visibility family of
/// the visibility table.
fn library_suffix(text: &str) -> bool {
    TENSOR_FAMILIES
        .iter()
        .any(|(family, visibility)| *visibility == Visibility::Library && text.starts_with(family))
}

/// `token`, if it names a library namespace.
fn library<'t, 's>(token: Option<&'t Token<'s>>) -> Option<&'t Token<'s>> {
    token.filter(|token| {
        token.kind == TokenKind::Identifier && LIBRARY_NAMESPACES.contains(&token.text)
    })
}

/// Whether `name` is the library prefix `namespace`'s `detail`
/// (`packets_detail_*`, `PACKETS_DETAIL_*`).
fn detail_prefixed(name: &str, namespace: &str) -> bool {
    let (Some(head), Some(rest)) = (name.get(..namespace.len()), name.get(namespace.len()..))
    else {
        return false;
    };
    (head == namespace && rest.starts_with("_detail_"))
        || (head
            .bytes()
            .zip(namespace.bytes())
            .all(|(h, n)| h == n.to_ascii_uppercase())
            && rest.starts_with("_DETAIL_"))
}

/// Whether `#define name` defines a library macro an includer does not set.
fn library_macro(name: &str) -> bool {
    LIBRARY_MACROS.iter().any(|prefix| name.starts_with(prefix))
        && !ENGINE_SET_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        && !ENGINE_SET_NAMES.contains(&name)
}

/// The `##` chains of a definition (`tokens` its name and the rest of its
/// logical line) that could form an ABI name.
fn definition_pastes(tokens: &[Token<'_>]) -> Vec<(usize, Rule)> {
    let Some(name) = tokens.first() else {
        return Vec::new();
    };
    // A function-like macro's parameters: its `(` touches its name.
    let mut parameters = Vec::new();
    let mut body = 1;
    if tokens
        .get(1)
        .is_some_and(|open| open.text == "(" && open.offset == name.offset + name.text.len())
    {
        let Some(close) = tokens.iter().position(|token| token.text == ")") else {
            return Vec::new();
        };
        for token in &tokens[2..close] {
            match (token.kind, token.text) {
                (TokenKind::Identifier, text) => parameters.push(text),
                // `...`: the tokenizer's three `.`.
                (TokenKind::Punctuator, ".") => parameters.push("__VA_ARGS__"),
                _ => {}
            }
        }
        body = close + 1;
    }
    let body = &tokens[body..];
    let mut found = Vec::new();
    let mut at = 0;
    while at < body.len() {
        if body[at].text != "##" || at == 0 {
            at += 1;
            continue;
        }
        // The chain: operand (## operand)+.
        let first = at - 1;
        let mut last = at + 1;
        while body.get(last + 1).is_some_and(|token| token.text == "##") {
            last += 2;
        }
        let last = last.min(body.len() - 1);
        let operands: Vec<&Token<'_>> = body[first..=last].iter().step_by(2).collect();
        let literal = |token: &Token<'_>| !parameters.contains(&token.text);
        let lowercase = operands.iter().any(|token| {
            literal(token) && token.text.bytes().any(|byte| byte.is_ascii_lowercase())
        });
        let leading = operands[0];
        // A literal end that is, or begins, a library-visibility suffix.
        let library_end = [leading, operands[operands.len() - 1]]
            .into_iter()
            .any(|end| literal(end) && library_suffix(end.text));
        let unprefixed = literal(leading)
            && !PASTED_PREFIXES
                .iter()
                .any(|prefix| leading.text.starts_with(prefix) || prefix.starts_with(leading.text));
        let stem = literal(leading) && operands.len() == 2 && SEGMENT_STEMS.contains(&leading.text);
        if library_end || !(lowercase || unprefixed || stem) {
            let chain: Vec<&str> = operands.iter().map(|token| token.text).collect();
            found.push((leading.offset, Rule::Paste(chain.join("##"))));
        }
        at = last + 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(source: &str) -> Vec<Rule> {
        violations(source)
            .into_iter()
            .map(|(_, rule)| rule)
            .collect()
    }

    #[test]
    fn rule_1_refuses_library_detail_names() {
        assert_eq!(
            rules("float x = packets::detail::Planes<W>::floats(p);\n"),
            [Rule::Detail("packets::detail".to_owned())]
        );
        assert_eq!(
            rules("typedef ::element :: detail::Raw R;\n"),
            [Rule::Detail("element::detail".to_owned())]
        );
        assert_eq!(
            rules("uint w = packets_detail_word(0u);\n#if PACKETS_DETAIL_HELD\n#endif\n"),
            [
                Rule::Detail("packets_detail_word".to_owned()),
                Rule::Detail("PACKETS_DETAIL_HELD".to_owned()),
            ]
        );
        // Every library prefix's, in both cases.
        assert_eq!(
            rules(
                "element_detail_x(); repack_detail_y(); nvfp4_detail_z(); SLAB_DETAIL_A; ROWS16_DETAIL_B; NVFP4_DETAIL_C;\n"
            ),
            [
                Rule::Detail("element_detail_x".to_owned()),
                Rule::Detail("repack_detail_y".to_owned()),
                Rule::Detail("nvfp4_detail_z".to_owned()),
                Rule::Detail("SLAB_DETAIL_A".to_owned()),
                Rule::Detail("ROWS16_DETAIL_B".to_owned()),
                Rule::Detail("NVFP4_DETAIL_C".to_owned()),
            ]
        );
        // An engine's own `detail`, comments and strings are not the library's.
        assert!(rules("namespace projection { namespace detail { int x; } }\nint y = projection::detail::x;\n").is_empty());
        assert!(rules("// packets::detail::Fields\n/* PACKETS_DETAIL_X */ const char *s = \"packets::detail\";\n").is_empty());
    }

    #[test]
    fn rule_2_refuses_opened_namespaces_and_library_defines() {
        assert_eq!(
            rules("namespace packets { typedef ProgressiveTop W0; }\n"),
            [Rule::Opened("packets".to_owned())]
        );
        assert_eq!(
            rules("namespace readout { namespace nvfp4 { } }\n"),
            [Rule::Opened("nvfp4".to_owned())]
        );
        assert_eq!(
            rules(
                "#define ELEMENT_A ELEMENT_F32\n  #  define PACKETS_STAGED_TABLES 1\n#define SLAB_X 1\n#define ROWS16_U SEISMIC_TABLE\n"
            ),
            [
                Rule::Defined("ELEMENT_A".to_owned()),
                Rule::Defined("PACKETS_STAGED_TABLES".to_owned()),
                Rule::Defined("SLAB_X".to_owned()),
                Rule::Defined("ROWS16_U".to_owned()),
            ]
        );
        // Configuration, the rows16 binding, slot bindings, engine names and
        // reads of library macros are an includer's.
        assert!(
            rules(
                "#define PACKETS_CONFIG_STAGED_TABLES 1\n#define ROWS16_T SEISMIC_TABLE\n#define KERNEL_W0 SEISMIC_WEIGHT\n\
                 #define ACTIVATION_A SEISMIC_ELEMENT_A\n#if ELEMENT_KIND(KERNEL_A) == ELEMENT_F32\n#endif\n\
                 namespace readout { using packets::Matrix; }\nnamespace pk = packets;\nint define = 0;\n"
            )
            .is_empty()
        );
    }

    #[test]
    fn rule_3_refuses_using_whole_library_namespaces() {
        assert_eq!(
            rules("using namespace packets;\nusing namespace ::element;\n"),
            [
                Rule::UsingNamespace("packets".to_owned()),
                Rule::UsingNamespace("element".to_owned())
            ]
        );
        assert!(rules("using namespace metal;\nusing packets::Same;\n").is_empty());
    }

    #[test]
    fn rule_4_refuses_paste_macros_and_abi_pastes() {
        assert_eq!(
            rules("#define ELEMENT_OF_A ELEMENT_CAT(SEISMIC, _ELEMENT_A)\n"),
            [
                Rule::Defined("ELEMENT_OF_A".to_owned()),
                Rule::PasteMacro("ELEMENT_CAT".to_owned())
            ]
        );
        assert_eq!(
            rules(
                "#if ELEMENT_HAS(KERNEL_W0, _LAYOUT_PACKET) || ELEMENT_IS_ONE(X) || PACKETS_CAT(A, B)\n#endif\n"
            ),
            [
                Rule::PasteMacro("ELEMENT_HAS".to_owned()),
                Rule::PasteMacro("ELEMENT_IS_ONE".to_owned()),
                Rule::PasteMacro("PACKETS_CAT".to_owned()),
            ]
        );
        // A parameter pasted onto an all-caps suffix, or a literal that
        // begins an ABI prefix pasted onto anything.
        for (source, chain) in [
            ("#define ROUTED_MATRIX(W) W##_MATRIX\n", "W##_MATRIX"),
            ("#define WORDS(C) uint(C##_WORDS)\n", "C##_WORDS"),
            ("#define A_OF(X) SEISMIC##X\n", "SEISMIC##X"),
            (
                "#define B_OF(X) SEISMIC_ELEMENT_ ## X\n",
                "SEISMIC_ELEMENT_##X",
            ),
            ("#define C_OF(X) S##X##_PACKET_SIZE\n", "S##X##_PACKET_SIZE"),
            (
                "#define STEM(s, t) SEISMIC_SEGMENT_WEIGHTS_##s##t\n",
                "SEISMIC_SEGMENT_WEIGHTS_##s##t",
            ),
            (
                "#define LONG(P) \\\n    P ## \\\n    _ROW_GROUPS\n",
                "P##_ROW_GROUPS",
            ),
            (
                "#define VARIADIC(...) __VA_ARGS__##_KIND_DENSE\n",
                "__VA_ARGS__##_KIND_DENSE",
            ),
            ("#define GROUP(S) S##_LOGICAL_GROUP\n", "S##_LOGICAL_GROUP"),
            ("#define SIZE(P) P##_PACKET_SIZE\n", "P##_PACKET_SIZE"),
            // A library-visibility suffix at either end, whatever else the
            // chain holds.
            (
                "#define OUT_ROWS(s) out##s##_ROW_GROUPS\n",
                "out##s##_ROW_GROUPS",
            ),
            (
                "#define N_ROWS PROJECTION_N##_ROW_STRIDE_BYTES\n",
                "PROJECTION_N##_ROW_STRIDE_BYTES",
            ),
            ("#define LAYOUT(s) _LAYOUT_##s\n", "_LAYOUT_##s"),
        ] {
            assert_eq!(rules(source), [Rule::Paste(chain.to_owned())], "{source}");
        }
    }

    #[test]
    fn rule_4_admits_engine_pastes_and_the_segment_stems() {
        for source in [
            // A literal with a lowercase letter: the result is no ABI name.
            "#define SEG(s) if (segment == s##u) { out##s.store(w##s); }\n",
            "#define STAGE(name) threadgroup float4 name##_words[4];\n",
            "#define WORDS(name) name##_words\n",
            // `_ROWS` is no `_ROW_` suffix.
            "#define EACH_ROWS(COUNT) PROJECTION_FOR_##COUNT##_ROWS\n",
            // A leading literal no ABI prefix begins.
            "#define PROJECTION_PACKET(s) PROJECTION_PACKET_##s\n",
            "#define EACH(COUNT, X) PROJECTION_FOR_##COUNT##_ROWS(X)\n",
            "#define PLANAR(flag) REPACK_PLANAR_##flag\n",
            // The segment stems.
            "#define PROJECTION_PAIRED(s) (SEISMIC_SEGMENT_WEIGHTS_##s == 2)\n",
            "#define ORDERED(s) (SEISMIC_SEGMENT_ORDERED_ ## s != 0)\n",
            // `##` in comments and strings, and stringizing.
            "// SEISMIC_##X\n#define NAME(x) #x \"S##X\"\n",
        ] {
            assert!(rules(source).is_empty(), "{source}: {:?}", rules(source));
        }
    }

    #[test]
    fn visibility_follows_the_suffix_family() {
        assert_eq!(visibility("", TENSOR_FAMILIES), Visibility::Surface);
        for suffix in [
            "_RANK",
            "_EXTENT_1",
            "_STRIDE_0",
            "_KIND_PACKED",
            "_DECODED_F32",
        ] {
            assert_eq!(
                visibility(suffix, TENSOR_FAMILIES),
                Visibility::Surface,
                "{suffix}"
            );
        }
        for suffix in [
            "_REPRESENTATION",
            "_REPRESENTATION_F32",
            "_OPERANDS_NVFP4",
            "_LAYOUT_ROWS16",
            "_PACKET_SIZE",
            "_ROW_STRIDE_BYTES",
            "_GRID_PLANAR",
            "_MMA_KBLOCK",
            "_PLANE_COUNT",
            "_RECORD_ENTRY",
            "_LOGICAL_GROUP",
            "_GROUP_MULTIPLE",
            "_TILE_ROWS",
        ] {
            assert_eq!(
                visibility(suffix, TENSOR_FAMILIES),
                Visibility::Library,
                "{suffix}"
            );
        }
        assert_eq!(visibility("", CONVERSION_FAMILIES), Visibility::Surface);
        assert_eq!(
            visibility("_WORDS", CONVERSION_FAMILIES),
            Visibility::Library
        );
    }

    #[test]
    #[should_panic(expected = "belongs to no family")]
    fn an_unclassified_suffix_is_an_invariant_violation() {
        visibility("_UNHEARD_OF", TENSOR_FAMILIES);
    }
}
