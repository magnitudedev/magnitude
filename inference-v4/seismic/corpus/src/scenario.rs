//! Scenario model and header parser (design A10 §2.2.1-§2.2.2).
//!
//! A scenario is one `.seismic` file under `scenarios/`. Its header is a run of
//! Seismic comment lines before the first source line; each header line has
//! exactly one of the forms below, and anything else panics with
//! `<name>:<line>: <reason>`, so a malformed header can never become a
//! silently skipped case.
use crate::inputs;
use crate::matrix::MatrixCell;
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry::{self, BackendName, RepresentationKind};
use seismic_lang::types::DType;
use std::fmt;
use std::path::Path;
use std::time::Duration;

/// "<directory path under scenarios/>/<file stem>", e.g. "history/h42-varying-snapshot".
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScenarioName(String);

impl ScenarioName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// The first path component: the scenario's source family (`history`, `areas`, ...).
    pub fn top_directory(&self) -> &str {
        self.0
            .split('/')
            .next()
            .expect("split yields one component")
    }
}

impl fmt::Display for ScenarioName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub struct Scenario {
    pub name: ScenarioName,
    /// The whole file, header included (headers are comments).
    pub source: String,
    pub include_std: bool,
    pub origins: Vec<Origin>,
    pub cells: Vec<MatrixCell>,
    pub backends: BackendSet,
    pub policies: PolicySet,
    pub undecided: Option<UndecidedAllowance>,
    pub general_launch_bound: Option<u32>,
    pub preparation_bound: Option<Duration>,
    pub class: ScenarioClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// "H-26c"
    History { number: u8, suffix: Option<char> },
    /// "C11:loop_tensor_carry"
    Fact { collector: String, case: String },
    /// "X1:P15"
    Review { reviewer: String, case: String },
    /// A deleted unit test (rule T1): "test:<crate>:<path>:<function>".
    Test {
        crate_name: String,
        path: String,
        function: String,
    },
    /// "A2:C8-1"
    Area { area: u8, gap: String },
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::History { number, suffix } => {
                write!(f, "H-{number}")?;
                suffix.map_or(Ok(()), |suffix| write!(f, "{suffix}"))
            }
            Self::Fact { collector, case } => write!(f, "{collector}:{case}"),
            Self::Review { reviewer, case } => write!(f, "{reviewer}:{case}"),
            Self::Test {
                crate_name,
                path,
                function,
            } => write!(f, "test:{crate_name}:{path}:{function}"),
            Self::Area { area, gap } => write!(f, "A{area}:{gap}"),
        }
    }
}

/// The only relation verdicts a scenario may accept as a pass besides `Match`
/// (A5 `UndecidedReason`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UndecidedAllowance {
    Reassociated,
    AtomicSubset,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BackendSet {
    pub cpu: bool,
    pub metal: bool,
}

impl BackendSet {
    pub fn contains(self, backend: BackendName) -> bool {
        match backend {
            BackendName::Cpu => self.cpu,
            BackendName::Metal => self.metal,
            BackendName::Cuda | BackendName::Vulkan => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PolicySet {
    pub exact: bool,
    pub unconstrained: bool,
}

impl PolicySet {
    pub const BOTH: PolicySet = PolicySet {
        exact: true,
        unconstrained: true,
    };
    /// The selected precision policies, `exact` first.
    pub fn policies(self) -> Vec<PrecisionPolicy> {
        let mut policies = Vec::new();
        if self.exact {
            policies.push(PrecisionPolicy::Exact);
        }
        if self.unconstrained {
            policies.push(PrecisionPolicy::Unconstrained);
        }
        policies
    }
}

pub enum ScenarioClass {
    /// Never empty: the parser panics on a scenario with no invocation.
    Executes(Vec<Invocation>),
    /// `# rejected: "<substring>"`
    Rejected { diagnostic: String },
}

pub struct Invocation {
    /// `# invoke <entry>:`; `probe` when no entry is named.
    pub entry: String,
    pub arguments: Vec<ArgumentSpec>,
    /// `# expect:` following the invoke.
    pub termination: Option<Termination>,
    /// `# pin:` lines following the invoke.
    pub pins: Vec<Pin>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Termination {
    Returned,
    Failed,
    InvalidInvocation,
    /// A7 `ResourceRefusal::IndexWidth` (R2-X3-1): an in-kernel quantity of this invocation
    /// exceeds the target's signed 64-bit quantity word. A real device limit (DoD-3), not a source outcome.
    RefusedIndexWidth,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ArgumentSpec {
    Tensor {
        element: String,
        shape: Vec<u64>,
        fill: Fill,
    },
    Scalar {
        dtype: ScalarDtype,
        value: Literal,
    },
    Index(u64),
    Range {
        start: u64,
        end: u64,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarDtype {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    Bool,
}

impl ScalarDtype {
    pub fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::I32 => DType::I32,
            Self::U32 => DType::U32,
            Self::Bool => DType::Bool,
        }
    }
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "f32" => Self::F32,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "i32" => Self::I32,
            "u32" => Self::U32,
            "bool" => Self::Bool,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Fill {
    Zero,
    Sequence,
    Uniform { seed: u64 },
    Bits { seed: u64 },
    Values(Vec<Literal>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    Decimal(f64),
    Integer(i128),
    /// A raw bit pattern of the element dtype (`0x...`).
    Bits(u64),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pin {
    pub subject: PinSubject,
    pub element: String,
    pub shape: Vec<u64>,
    pub values: Vec<Literal>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PinSubject {
    Parameter(String),
    /// `result` is the empty path; `result.0.1` is `[0, 1]`.
    Result(Vec<u32>),
}

/// Every scenario under `root`, sorted by name.
pub fn load_all(root: &Path) -> Vec<Scenario> {
    let mut names = Vec::new();
    collect_names(root, root, &mut names);
    names.sort();
    names.into_iter().map(|name| load(root, name)).collect()
}

/// The scenario `<root>/<name>.seismic`.
pub fn load(root: &Path, name: ScenarioName) -> Scenario {
    let path = root.join(format!("{}.seismic", name.as_str()));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse(name, &text)
}

fn collect_names(root: &Path, directory: &Path, names: &mut Vec<ScenarioName>) {
    let entries =
        std::fs::read_dir(directory).unwrap_or_else(|e| panic!("{}: {e}", directory.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
            .path();
        if path.is_dir() {
            collect_names(root, &path, names);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "seismic")
        {
            let relative = path
                .with_extension("")
                .strip_prefix(root)
                .expect("walked path lies under the root")
                .components()
                .map(|c| c.as_os_str().to_str().expect("scenario paths are UTF-8"))
                .collect::<Vec<_>>()
                .join("/");
            names.push(ScenarioName::new(relative));
        }
    }
}

/// Panics `"<name>:<line>: <reason>"` on anything but the header grammar.
pub fn parse(name: ScenarioName, text: &str) -> Scenario {
    let mut header = Header::default();
    let mut in_header = true;
    for (index, line) in text.lines().enumerate() {
        let fail = |reason: String| -> ! { panic!("{name}:{}: {reason}", index + 1) };
        if in_header && line.trim().is_empty() {
            continue;
        }
        if in_header && line.starts_with('#') {
            let Some((keyword, rest)) = header_line(line) else {
                fail(format!("`{line}` is not a header line"));
            };
            header.apply(keyword, rest).unwrap_or_else(|e| fail(e));
            continue;
        }
        in_header = false;
        if header_line(line.trim_start()).is_some() {
            fail(format!("header line `{}` follows source", line.trim()));
        }
    }
    let end = text.lines().count();
    header
        .finish(name.clone(), text)
        .unwrap_or_else(|e| panic!("{name}:{end}: {e}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Keyword<'a> {
    Std,
    Origin,
    Cell,
    Backends,
    Policies,
    Undecided,
    GeneralLaunchesAtMost,
    PrepareWithinMs,
    Rejected,
    Invoke(Option<&'a str>),
    Expect,
    Pin,
}

/// Splits `# <keyword>: <rest>`; `None` when the line is not a header line.
fn header_line(line: &str) -> Option<(Keyword<'_>, &str)> {
    let (key, rest) = line.strip_prefix("# ")?.split_once(':')?;
    let keyword = match key {
        "std" => Keyword::Std,
        "origin" => Keyword::Origin,
        "cell" => Keyword::Cell,
        "backends" => Keyword::Backends,
        "policies" => Keyword::Policies,
        "undecided" => Keyword::Undecided,
        "general-launches-at-most" => Keyword::GeneralLaunchesAtMost,
        "prepare-within-ms" => Keyword::PrepareWithinMs,
        "rejected" => Keyword::Rejected,
        "invoke" => Keyword::Invoke(None),
        "expect" => Keyword::Expect,
        "pin" => Keyword::Pin,
        _ => {
            let entry = key.strip_prefix("invoke ")?;
            if !is_identifier(entry) {
                return None;
            }
            Keyword::Invoke(Some(entry))
        }
    };
    let rest = match rest.strip_prefix(' ') {
        Some(rest) => rest,
        None if rest.is_empty() => rest,
        None => return None,
    };
    Some((keyword, rest))
}

#[derive(Default)]
struct Header {
    include_std: Option<bool>,
    origins: Vec<Origin>,
    cells: Vec<MatrixCell>,
    backends: Option<BackendSet>,
    policies: Option<PolicySet>,
    undecided: Option<UndecidedAllowance>,
    general_launch_bound: Option<u32>,
    preparation_bound: Option<Duration>,
    rejected: Option<String>,
    invocations: Vec<Invocation>,
}

fn set_once<T>(slot: &mut Option<T>, value: T, what: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("`# {what}:` appears twice"));
    }
    Ok(())
}

impl Header {
    fn apply(&mut self, keyword: Keyword<'_>, rest: &str) -> Result<(), String> {
        match keyword {
            Keyword::Std => {
                let value = match rest {
                    "yes" => true,
                    "no" => false,
                    _ => return Err(format!("`std` is `yes` or `no`, not `{rest}`")),
                };
                set_once(&mut self.include_std, value, "std")
            }
            Keyword::Origin => {
                self.origins.extend(parse_origins(rest)?);
                Ok(())
            }
            Keyword::Cell => {
                let cell = MatrixCell::parse(rest)
                    .ok_or_else(|| format!("`{rest}` is not `row/column`"))?;
                self.cells.push(cell);
                Ok(())
            }
            Keyword::Backends => {
                let mut set = BackendSet {
                    cpu: false,
                    metal: false,
                };
                for word in words(rest)? {
                    let member = match word {
                        "cpu" => &mut set.cpu,
                        "metal" => &mut set.metal,
                        _ => return Err(format!("unknown backend `{word}`")),
                    };
                    if std::mem::replace(member, true) {
                        return Err(format!("backend `{word}` is listed twice"));
                    }
                }
                set_once(&mut self.backends, set, "backends")
            }
            Keyword::Policies => {
                let mut set = PolicySet {
                    exact: false,
                    unconstrained: false,
                };
                for word in words(rest)? {
                    let member = match word {
                        "exact" => &mut set.exact,
                        "unconstrained" => &mut set.unconstrained,
                        _ => return Err(format!("unknown policy `{word}`")),
                    };
                    if std::mem::replace(member, true) {
                        return Err(format!("policy `{word}` is listed twice"));
                    }
                }
                set_once(&mut self.policies, set, "policies")
            }
            Keyword::Undecided => {
                let allowance = match rest {
                    "reassociated" => UndecidedAllowance::Reassociated,
                    "atomic-subset" => UndecidedAllowance::AtomicSubset,
                    _ => return Err(format!("unknown undecided reason `{rest}`")),
                };
                set_once(&mut self.undecided, allowance, "undecided")
            }
            Keyword::GeneralLaunchesAtMost => {
                let bound = parse_number(rest)?;
                set_once(
                    &mut self.general_launch_bound,
                    bound,
                    "general-launches-at-most",
                )
            }
            Keyword::PrepareWithinMs => {
                let bound = Duration::from_millis(u64::from(parse_number::<u32>(rest)?));
                set_once(&mut self.preparation_bound, bound, "prepare-within-ms")
            }
            Keyword::Rejected => {
                let diagnostic = rest
                    .strip_prefix('"')
                    .and_then(|rest| rest.strip_suffix('"'))
                    .filter(|diagnostic| !diagnostic.is_empty())
                    .ok_or("`rejected` takes a non-empty quoted diagnostic substring")?;
                set_once(&mut self.rejected, diagnostic.to_owned(), "rejected")
            }
            Keyword::Invoke(entry) => {
                self.invocations.push(Invocation {
                    entry: entry.unwrap_or("probe").to_owned(),
                    arguments: parse_arguments(rest)?,
                    termination: None,
                    pins: Vec::new(),
                });
                Ok(())
            }
            Keyword::Expect => {
                let termination = match rest {
                    "returned" => Termination::Returned,
                    "failed" => Termination::Failed,
                    "invalid-invocation" => Termination::InvalidInvocation,
                    "refused-index-width" => Termination::RefusedIndexWidth,
                    _ => return Err(format!("unknown termination `{rest}`")),
                };
                let invocation = self
                    .invocations
                    .last_mut()
                    .ok_or("`expect` must follow an `invoke`")?;
                set_once(&mut invocation.termination, termination, "expect")
            }
            Keyword::Pin => {
                let pin = parse_pin(rest)?;
                self.invocations
                    .last_mut()
                    .ok_or("`pin` must follow an `invoke`")?
                    .pins
                    .push(pin);
                Ok(())
            }
        }
    }

    fn finish(self, name: ScenarioName, text: &str) -> Result<Scenario, String> {
        let class = match (self.rejected, self.invocations.is_empty()) {
            (Some(diagnostic), true) => ScenarioClass::Rejected { diagnostic },
            (None, false) => {
                let refused_with_pins = self.invocations.iter().any(|invocation| {
                    invocation.termination == Some(Termination::RefusedIndexWidth)
                        && !invocation.pins.is_empty()
                });
                if refused_with_pins {
                    return Err("a `refused-index-width` invocation has no values to pin".into());
                }
                ScenarioClass::Executes(self.invocations)
            }
            (Some(_), false) => return Err("a rejected scenario has no invocations".into()),
            (None, true) => {
                return Err("a scenario needs `# rejected:` or at least one `# invoke:`".into())
            }
        };
        Ok(Scenario {
            name,
            source: text.to_owned(),
            include_std: self.include_std.unwrap_or(false),
            origins: self.origins,
            cells: self.cells,
            backends: self.backends.unwrap_or(BackendSet {
                cpu: true,
                metal: true,
            }),
            policies: self.policies.unwrap_or(PolicySet::BOTH),
            undecided: self.undecided,
            general_launch_bound: self.general_launch_bound,
            preparation_bound: self.preparation_bound,
            class,
        })
    }
}

fn words(text: &str) -> Result<std::str::SplitAsciiWhitespace<'_>, String> {
    if text.trim().is_empty() {
        return Err("expected at least one word".into());
    }
    Ok(text.split_ascii_whitespace())
}

fn parse_number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("`{text}` is not an unsigned integer"))
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The origins of a `# origin: ...` line in the scenario header form, which
/// library invocation files share. `None` when the line does not start with
/// `# origin`; a line that does but is not exactly that form is an error.
pub fn origin_line(line: &str) -> Option<Result<Vec<Origin>, String>> {
    if !line.starts_with("# origin") {
        return None;
    }
    Some(match header_line(line) {
        Some((Keyword::Origin, rest)) => parse_origins(rest),
        _ => Err(format!("`{line}` is not `# origin: <origin> ...`")),
    })
}

/// One or more space-separated origins (`# origin: H-27 H-28`).
pub fn parse_origins(text: &str) -> Result<Vec<Origin>, String> {
    words(text)?.map(parse_origin).collect()
}

fn parse_origin(text: &str) -> Result<Origin, String> {
    let malformed = || format!("`{text}` is not an origin");
    let non_empty = |part: &str| {
        (!part.is_empty())
            .then(|| part.to_owned())
            .ok_or_else(malformed)
    };
    if let Some(rest) = text.strip_prefix("test:") {
        let mut parts = rest.split(':');
        let (Some(crate_name), Some(path), Some(function), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(malformed());
        };
        return Ok(Origin::Test {
            crate_name: non_empty(crate_name)?,
            path: non_empty(path)?,
            function: non_empty(function)?,
        });
    }
    if let Some(rest) = text.strip_prefix("H-") {
        let digits = rest.trim_end_matches(|c: char| c.is_ascii_lowercase());
        let suffix = &rest[digits.len()..];
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) || suffix.len() > 1 {
            return Err(malformed());
        }
        return Ok(Origin::History {
            number: digits.parse().map_err(|_| malformed())?,
            suffix: suffix.chars().next(),
        });
    }
    let (source, case) = text.split_once(':').ok_or_else(malformed)?;
    let case = non_empty(case)?;
    let letter = source.get(..1).ok_or_else(malformed)?;
    let number = &source[1..];
    let digits = number.trim_end_matches(['a', 'b']);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed());
    }
    match letter {
        "C" => Ok(Origin::Fact {
            collector: source.to_owned(),
            case,
        }),
        "X" if digits.len() == 1 && digits == number => Ok(Origin::Review {
            reviewer: source.to_owned(),
            case,
        }),
        "A" if digits == number => Ok(Origin::Area {
            area: digits.parse().map_err(|_| malformed())?,
            gap: case,
        }),
        _ => Err(malformed()),
    }
}

/// `arg (" ; " arg)*`, or nothing.
pub fn parse_arguments(text: &str) -> Result<Vec<ArgumentSpec>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    text.split(';')
        .map(|arg| parse_argument(arg.trim()))
        .collect()
}

fn parse_argument(text: &str) -> Result<ArgumentSpec, String> {
    if let Some(value) = text.strip_prefix("index:") {
        return Ok(ArgumentSpec::Index(parse_number(value)?));
    }
    if let Some(range) = text.strip_prefix("range:") {
        let (start, end) = range
            .split_once("..")
            .ok_or_else(|| format!("`{text}` is not `range:<start>..<end>`"))?;
        return Ok(ArgumentSpec::Range {
            start: parse_number(start)?,
            end: parse_number(end)?,
        });
    }
    if text.contains('[') {
        let (element, shape, fill) = split_tensor(text)?;
        let fill = parse_fill(fill)?;
        check_fill(element, &shape, &fill)?;
        return Ok(ArgumentSpec::Tensor {
            element: element.to_owned(),
            shape,
            fill,
        });
    }
    let (dtype, value) = text
        .split_once(':')
        .ok_or_else(|| format!("`{text}` is not an argument"))?;
    let dtype =
        ScalarDtype::from_name(dtype).ok_or_else(|| format!("`{dtype}` is not a scalar dtype"))?;
    let value = parse_literal(value)?;
    inputs::literal_bits(dtype.dtype(), value)?;
    Ok(ArgumentSpec::Scalar { dtype, value })
}

/// `element "[" dims "]=" rest`
fn split_tensor(text: &str) -> Result<(&str, Vec<u64>, &str), String> {
    let malformed = || format!("`{text}` is not `element[dims]=...`");
    let (element, rest) = text.split_once('[').ok_or_else(malformed)?;
    let (dims, rest) = rest.split_once("]=").ok_or_else(malformed)?;
    if registry::representation(element).is_none() {
        return Err(format!("`{element}` is not a registered representation"));
    }
    let shape = if dims.is_empty() {
        Vec::new()
    } else {
        dims.split(',')
            .map(parse_number)
            .collect::<Result<_, _>>()?
    };
    Ok((element, shape, rest))
}

fn parse_fill(text: &str) -> Result<Fill, String> {
    let seed = |prefix: &str| {
        text.strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(')'))
            .map(parse_number)
    };
    Ok(match text {
        "zero" => Fill::Zero,
        "seq" => Fill::Sequence,
        _ => {
            if let Some(seed) = seed("rand(") {
                Fill::Uniform { seed: seed? }
            } else if let Some(seed) = seed("bits(") {
                Fill::Bits { seed: seed? }
            } else {
                Fill::Values(
                    text.split(',')
                        .map(parse_literal)
                        .collect::<Result<_, _>>()?,
                )
            }
        }
    })
}

/// Logical element count of a shape; `None` when it does not fit in `u64`.
pub fn element_count(shape: &[u64]) -> Option<u64> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
}

fn check_fill(element: &str, shape: &[u64], fill: &Fill) -> Result<(), String> {
    let representation = registry::representation(element).expect("checked by split_tensor");
    let dtype = match &registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => *dtype,
        RepresentationKind::Packed(_) | RepresentationKind::External(_) => {
            return match fill {
                Fill::Zero | Fill::Uniform { .. } | Fill::Bits { .. } => Ok(()),
                Fill::Sequence | Fill::Values(_) => Err(format!(
                    "`seq` and literal fills are not defined for the packed element `{element}`"
                )),
            };
        }
        // A row layout places packet planes per row at tensor-dependent
        // offsets, so bounded random values (`uniform`) are not defined
        // bytewise; its bytes are zero or random.
        RepresentationKind::PackedRows(_) => {
            return match fill {
                Fill::Zero | Fill::Bits { .. } => Ok(()),
                Fill::Uniform { .. } | Fill::Sequence | Fill::Values(_) => Err(format!(
                    "only `zero` and `bits` fills are defined for the row-layout element `{element}`"
                )),
            };
        }
    };
    let count = element_count(shape);
    match fill {
        Fill::Bits { .. } if dtype == DType::Bool => Err(format!(
            "`bits` is not defined for `{element}`: random bytes are not bool values"
        )),
        Fill::Zero | Fill::Bits { .. } => Ok(()),
        Fill::Uniform { .. } => count
            .map(drop)
            .ok_or_else(|| format!("`rand` over {shape:?} has no element count")),
        Fill::Sequence => {
            let last = count.map(|count| count.saturating_sub(1));
            let fits = match dtype {
                DType::I32 => last.is_some_and(|last| last <= i32::MAX as u64),
                DType::U32 => last.is_some_and(|last| last <= u64::from(u32::MAX)),
                _ => count.is_some(),
            };
            if fits {
                Ok(())
            } else {
                Err(format!("`seq` over {shape:?} overflows `{element}`"))
            }
        }
        Fill::Values(values) => {
            if count != Some(values.len() as u64) {
                return Err(format!(
                    "{} literal(s) for a tensor of {shape:?} elements",
                    values.len()
                ));
            }
            values
                .iter()
                .try_for_each(|value| inputs::literal_bits(dtype, *value).map(drop))
        }
    }
}

/// `decimal | integer | 0x<hex> | true | false | nan | inf | -inf | -0`
pub fn parse_literal(text: &str) -> Result<Literal, String> {
    Ok(match text {
        "true" => Literal::Bool(true),
        "false" => Literal::Bool(false),
        "nan" => Literal::Decimal(f64::NAN),
        "inf" => Literal::Decimal(f64::INFINITY),
        "-inf" => Literal::Decimal(f64::NEG_INFINITY),
        "-0" => Literal::Decimal(-0.0),
        _ => {
            let malformed = || format!("`{text}` is not a literal");
            if let Some(hex) = text.strip_prefix("0x") {
                Literal::Bits(u64::from_str_radix(hex, 16).map_err(|_| malformed())?)
            } else if text.contains(['.', 'e', 'E']) {
                Literal::Decimal(text.parse().map_err(|_| malformed())?)
            } else {
                Literal::Integer(text.parse().map_err(|_| malformed())?)
            }
        }
    })
}

/// `subject " = " element "[" dims "]=" values`
fn parse_pin(text: &str) -> Result<Pin, String> {
    let (subject, value) = text
        .split_once(" = ")
        .ok_or_else(|| format!("`{text}` is not `subject = element[dims]=values`"))?;
    let subject = if subject == "result" {
        PinSubject::Result(Vec::new())
    } else if let Some(path) = subject.strip_prefix("result.") {
        PinSubject::Result(
            path.split('.')
                .map(parse_number)
                .collect::<Result<_, _>>()?,
        )
    } else if is_identifier(subject) {
        PinSubject::Parameter(subject.to_owned())
    } else {
        return Err(format!("`{subject}` is not a pin subject"));
    };
    let (element, shape, values) = split_tensor(value)?;
    let representation = registry::representation(element).expect("checked by split_tensor");
    if !matches!(
        registry::representation_info(representation).kind,
        RepresentationKind::Dense(_)
    ) {
        return Err(format!("pins compare dense elements, not `{element}`"));
    }
    let fill = parse_fill(values)?;
    check_fill(element, &shape, &fill)?;
    let Fill::Values(values) = fill else {
        return Err("a pin states literal values".into());
    };
    Ok(Pin {
        subject,
        element: element.to_owned(),
        shape,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(text: &str) -> Scenario {
        parse(ScenarioName::new("test/scenario"), text)
    }

    #[test]
    fn parses_every_header_form() {
        let s = scenario(
            "# std: yes\n\
             # origin: H-26c X1:P15\n\
             # origin: test:seismic-runtime:runtime/src/feedback/cohort_tests.rs:lanes\n\
             # cell: if/join\n\
             # backends: metal\n\
             # policies: exact\n\
             # undecided: reassociated\n\
             # general-launches-at-most: 1\n\
             # prepare-within-ms: 30000\n\
             # invoke: f32[4096]=zero ; f32[48]=seq ; i32:100\n\
             # invoke second: q4k[2,256]=rand(4) ; index:3 ; range:0..0\n\
             # expect: failed\n\
             # pin: x = f32[3]=1,1,2\n\
             # pin: result.0.1 = f32[]=0x5f000000\n\
             fn probe(x: &tensor[1] f32):\n    x[0] = 1.0\n",
        );
        assert!(s.include_std);
        assert_eq!(
            s.origins,
            [
                Origin::History {
                    number: 26,
                    suffix: Some('c')
                },
                Origin::Review {
                    reviewer: "X1".into(),
                    case: "P15".into()
                },
                Origin::Test {
                    crate_name: "seismic-runtime".into(),
                    path: "runtime/src/feedback/cohort_tests.rs".into(),
                    function: "lanes".into()
                },
            ]
        );
        assert_eq!(s.cells[0].to_string(), "if/join");
        assert_eq!(
            s.backends,
            BackendSet {
                cpu: false,
                metal: true
            }
        );
        assert_eq!(s.policies.policies().len(), 1);
        assert_eq!(s.undecided, Some(UndecidedAllowance::Reassociated));
        assert_eq!(s.general_launch_bound, Some(1));
        assert_eq!(s.preparation_bound, Some(Duration::from_secs(30)));
        let ScenarioClass::Executes(invocations) = &s.class else {
            panic!("executes");
        };
        assert_eq!(invocations[0].entry, "probe");
        assert_eq!(
            invocations[0].arguments[2],
            ArgumentSpec::Scalar {
                dtype: ScalarDtype::I32,
                value: Literal::Integer(100)
            }
        );
        assert_eq!(invocations[1].entry, "second");
        assert_eq!(invocations[1].termination, Some(Termination::Failed));
        assert_eq!(
            invocations[1].arguments[2],
            ArgumentSpec::Range { start: 0, end: 0 }
        );
        assert_eq!(
            invocations[1].pins[1].subject,
            PinSubject::Result(vec![0, 1])
        );
        assert_eq!(invocations[1].pins[1].values, [Literal::Bits(0x5f000000)]);
    }

    #[test]
    fn parses_origin_forms() {
        assert_eq!(
            parse_origins("C11:loop_tensor_carry C4a:x A2:C8-1 H-1").unwrap(),
            [
                Origin::Fact {
                    collector: "C11".into(),
                    case: "loop_tensor_carry".into()
                },
                Origin::Fact {
                    collector: "C4a".into(),
                    case: "x".into()
                },
                Origin::Area {
                    area: 2,
                    gap: "C8-1".into()
                },
                Origin::History {
                    number: 1,
                    suffix: None
                },
            ]
        );
        assert!(parse_origins("H-").is_err());
        assert!(parse_origins("X12:P1").is_err());
        assert!(parse_origins("Q1:x").is_err());
    }

    #[test]
    fn rejected_scenario_and_defaults() {
        let s = scenario("# rejected: \"early return\"\nfn probe():\n    return\n");
        assert!(
            matches!(&s.class, ScenarioClass::Rejected { diagnostic } if diagnostic == "early return")
        );
        assert!(!s.include_std);
        assert_eq!(s.policies, PolicySet::BOTH);
        assert!(s.backends.contains(BackendName::Cpu) && s.backends.contains(BackendName::Metal));
    }

    #[test]
    fn parses_every_termination() {
        let s = scenario(
            "# invoke: index:0\n\
             # expect: returned\n\
             # invoke: index:1\n\
             # expect: failed\n\
             # invoke: index:2\n\
             # expect: invalid-invocation\n\
             # invoke: index:3\n\
             # expect: refused-index-width\n\
             fn probe(i: index[4]):\n    return\n",
        );
        let ScenarioClass::Executes(invocations) = &s.class else {
            panic!("executes");
        };
        let terminations: Vec<_> = invocations.iter().map(|i| i.termination).collect();
        assert_eq!(
            terminations,
            [
                Some(Termination::Returned),
                Some(Termination::Failed),
                Some(Termination::InvalidInvocation),
                Some(Termination::RefusedIndexWidth),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "test/scenario:2: unknown termination `refused`")]
    fn unknown_termination_panics() {
        scenario("# invoke:\n# expect: refused\nfn probe():\n    return\n");
    }

    #[test]
    #[should_panic(expected = "a `refused-index-width` invocation has no values to pin")]
    fn pin_on_refused_invocation_panics() {
        scenario(
            "# invoke: f32[2]=zero\n# expect: refused-index-width\n# pin: x = f32[2]=0,0\n\
             fn probe(x: &tensor[2] f32):\n    return\n",
        );
    }

    #[test]
    #[should_panic(expected = "`bits` is not defined for `bool`")]
    fn bits_over_bool_panics() {
        scenario("# invoke: bool[4]=bits(1)\nfn probe():\n    return\n");
    }

    #[test]
    #[should_panic(expected = "test/scenario:1: `# invokes: f32[1]=zero` is not a header line")]
    fn unknown_header_line_panics() {
        scenario("# invokes: f32[1]=zero\nfn probe():\n    return\n");
    }

    #[test]
    #[should_panic(
        expected = "test/scenario:3: header line `# invoke: f32[1]=zero` follows source"
    )]
    fn header_line_after_source_panics() {
        scenario("# invoke:\nfn probe():\n    # invoke: f32[1]=zero\n    return\n");
    }

    #[test]
    #[should_panic(expected = "not defined for the packed element `q4k`")]
    fn sequence_over_packed_panics() {
        scenario("# invoke: q4k[1,256]=seq\nfn probe():\n    return\n");
    }

    #[test]
    #[should_panic(expected = "a scenario needs")]
    fn scenario_without_class_panics() {
        scenario("# std: no\nfn probe():\n    return\n");
    }
}
