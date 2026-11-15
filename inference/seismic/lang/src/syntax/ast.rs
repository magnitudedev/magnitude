//! Abstract syntax of structured Seismic source files. See `docs/seismic/language.md`.

use crate::span::Span;

/// `value`'s structure: its derived `Debug` rendering with every source
/// position (`Span { .. }`) removed, so two parses are equal in it exactly
/// when they are equal but for where their text sat. Every field of the
/// syntax tree is in it by construction. Free text that could hold `Span {`
/// is only a native declaration's asset path; identifiers cannot.
pub(crate) fn structure(value: &impl std::fmt::Debug) -> String {
    let text = format!("{value:?}");
    let mut out = String::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("Span {") {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        rest = &rest[rest.find('}').map_or(rest.len(), |j| j + 1)..];
    }
    out + rest
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
}

impl AssignOp {
    pub fn text(self) -> &'static str {
        match self {
            AssignOp::Assign => "=",
            AssignOp::Add => "+=",
            AssignOp::Sub => "-=",
            AssignOp::Mul => "*=",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

impl UnaryOp {
    pub fn text(self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "not ",
            UnaryOp::BitNot => "~",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BitOr,
    BitXor,
    BitAnd,
    Shl,
    Shr,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl BinaryOp {
    pub fn text(self) -> &'static str {
        match self {
            BinaryOp::Or => "or",
            BinaryOp::And => "and",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::BitAnd => "&",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
        }
    }

    /// Binding power; higher binds tighter. All binary operators are left-associative.
    pub fn precedence(self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => 4,
            BinaryOp::BitOr => 5,
            BinaryOp::BitXor => 6,
            BinaryOp::BitAnd => 7,
            BinaryOp::Shl | BinaryOp::Shr => 8,
            BinaryOp::Add | BinaryOp::Sub => 9,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => 10,
        }
    }
}

/// Precedence of `not`, between `and` and comparisons.
pub const NOT_PRECEDENCE: u8 = 3;
/// Precedence of unary `-` and `~`, above every binary operator.
pub const UNARY_PRECEDENCE: u8 = 11;

#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct File {
    pub decls: Vec<Decl>,
    pub constants: Vec<ConstDecl>,
}

/// `const NAME = INTEGER`: a module constant. Every file of a source set
/// sees it, and an expression naming it is that integer literal; natives
/// read it as `SEISMIC_CONST_NAME`, and the generated Rust API exports it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstDecl {
    pub name: Ident,
    pub value: u64,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decl {
    Fn(FnDecl),
    Lower(LowerDecl),
    Native(NativeDecl),
    Projection(ProjectionDecl),
}

impl Decl {
    pub fn name(&self) -> &Ident {
        match self {
            Decl::Fn(f) => &f.name,
            Decl::Lower(l) => &l.name,
            Decl::Native(n) => &n.function,
            Decl::Projection(p) => &p.name,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Decl::Fn(f) => f.span,
            Decl::Lower(l) => l.span,
            Decl::Native(n) => n.span,
            Decl::Projection(p) => p.span,
        }
    }
}

/// A direct top-level backend implementation of an existing portable function.
///
/// ```text
/// native NAME for TARGET from "SOURCE":
///     static (DIM, ..)
///     params ([arithmetic] [form] NAME in [V, ..] default DEFAULT, ..)
///     elements (ELEMENT in [DTYPE, ..], ..)
///     let NAME = EXPR
///     where CONDITION
///     scratch NAME bytes (EXPR) [sync] [when CONDITION]
///     project PROJECTION VARIANT (WEIGHT, ..)
///     scratch NAME bytes (EXPR) [sync] [when CONDITION]
///     launch KERNEL [before LAUNCH] [when CONDITION]:
///         params ([arithmetic] [form] NAME in [V, ..] default DEFAULT, ..)
///         where CONDITION
///         reads (PARAMETER, ..)
///         threadgroups (X, Y, Z)
///         grid (PARAMETER)
///         threads_per_threadgroup (X, Y, Z)
///         shared_bytes (EXPR)
///     repeat (EXPR):
///         launch ..
/// ```
///
/// A `CONDITION` is comparisons of natural-number expressions joined by
/// `and` and `or`, grouped with parentheses; a capability `device.NAME`
/// stands as a condition. Every expression may read `device.NAME`, a
/// property the target backend provides. A `DEFAULT` is `EXPR`, `least
/// where CONDITION` or `greatest where CONDITION`, over the device, the
/// static dimensions and the bound elements.
///
/// `project` splices a projection's forms in (`crate::projection`): its
/// parameters after the declaration's, its conditions after its `where`,
/// its scratch at the line, and its launches in their order, each launch of
/// the declaration's own before the projection launch it names
/// (`before LAUNCH`) or after them all.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeDecl {
    pub function: Ident,
    pub target: Ident,
    pub source: String,
    /// Entry dimensions fixed at preparation.
    pub statics: Vec<Ident>,
    pub params: Vec<NativeParamDecl>,
    /// The dense representations a build-time compiled (CPU) form covers,
    /// per element parameter it monomorphizes.
    pub elements: Vec<NativeElementsDecl>,
    /// `records (PORT, ..)`: tensor ports the native's launches reach a
    /// record (a leading-axis row) at a time, each bound through a table of
    /// its records' device addresses; a launch's own `records` replaces it.
    pub records: Vec<NativeRecordDecl>,
    /// Named natural-number expressions, in declaration order.
    pub terms: Vec<NativeTermDecl>,
    /// The `where` condition restricting admissible configurations.
    pub constraint: Option<Expr>,
    pub scratch: Vec<NativeScratchDecl>,
    /// The projection whose forms the declaration takes; checking reads the
    /// declaration only once it is expanded, which leaves this `None`.
    pub project: Option<NativeProjectDecl>,
    /// The kind of rows (`rows KIND`) the declaration was expanded over: set
    /// by the expansion, which the checker compares with its kernel's
    /// prologue.
    pub projected_rows: Option<RowsKind>,
    /// The weights each segment serves, set by the expansion of a `segmented`
    /// projection: the implementation's `segments`.
    pub projected_segments: Option<Vec<usize>>,
    /// Whether each segment of a `segmented` projection accumulates in the
    /// reference order (`ordered`), parallel to `projected_segments`.
    pub projected_ordered: Vec<bool>,
    /// The CPU operator's formation of the operands the `project` line
    /// names, set by the expansion.
    pub formation: Option<NativeFormationDecl>,
    /// Every launch in dispatch order, those of the `repeat` block included.
    pub launches: Vec<NativeLaunchDecl>,
    pub repeat: Option<NativeRepeatDecl>,
    pub span: Span,
}

/// The kind of rows a projecting entry projects (`rows KIND`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowsKind {
    /// The prologue passes the rows as given.
    Plain,
    /// The prologue forms the rows, normalizing them: a stage launch
    /// materializes them for more than one row.
    Rms,
    /// A stage launch always forms the rows, whatever the row count.
    Staged,
}

impl RowsKind {
    pub const ALL: [RowsKind; 3] = [RowsKind::Plain, RowsKind::Rms, RowsKind::Staged];

    pub fn name(self) -> &'static str {
        match self {
            RowsKind::Plain => "plain",
            RowsKind::Rms => "rms",
            RowsKind::Staged => "staged",
        }
    }

    pub fn named(name: &str) -> Option<RowsKind> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

/// `project PROJECTION VARIANT (WEIGHT, ..) [shared (WEIGHT, ..)] [down (WEIGHT)] [groups G] [parts P] [gathered PORT] [rows KIND [publishes (result N)]] [ordered (WEIGHT, ..)]`:
/// the entry's projection of its weight ports, by the named projection's
/// `VARIANT` (`paired`: two weights one activation row serves, `single`:
/// one), over `G` groups of each weight's rows where it names them, over `P`
/// parts of its columns where it names them (each weight then `[N, P * X]`,
/// each part's rows of their own, `rms` rows normalized per part), over the
/// weight rows a `tensor[N]` port selects where it is `gathered` (its extent
/// is the projection's `N`; a `tensor[M, I, N]` port selects them per
/// activation row), over rows of the kind `KIND` (`plain` by default), whose
/// formed rows the entry's result `N` is (`publishes`), and the products of
/// the weights it `ordered` sum in the reference order. Lines of `if (WORD)`
/// or `unless (WORD)` after it hold every launch the line expands to.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeProjectDecl {
    pub projection: Ident,
    pub variant: Ident,
    pub weights: Vec<Ident>,
    /// `over (ROWS, WIDTH)` after a weight, parallel to `weights`: the rows
    /// and width the weight multiplies where it has its own
    /// (`NativeProjectOver`).
    pub weight_overs: Vec<Option<NativeProjectOver>>,
    /// `planes (PLANE, ..) over (ROWS, WIDTH)` in place of a weight,
    /// parallel to `weights`: the ports of a plane set (one weight stored as
    /// planes, its rows and width those of its `over`), empty for a weight
    /// that is one port.
    pub weight_planes: Vec<Vec<Ident>>,
    /// `over (ROWS, WIDTH)` after the list: the activation rows and width
    /// the entry's weights multiply, for an entry whose weights share no
    /// `[N, K]` shape (the CPU's): `weights` then name element parameters.
    pub over: Option<NativeProjectOver>,
    /// `segmented` only: the weights each segment serves, in order, where
    /// the line groups them (`((b, x), c)`: a segment of two weights one
    /// activation row serves, then one of a single weight); empty where it
    /// does not (a weight a segment).
    pub served: Vec<usize>,
    /// A routed projection's (`choice`, `summed`): the shared expert's
    /// weights (`shared (..)`), projected beside the chosen experts'.
    pub shared: Vec<Ident>,
    /// A routed `grouped` projection's: the weight its expansion's products
    /// project by (`down (..)`).
    pub down: Option<Ident>,
    pub groups: Option<Ident>,
    pub parts: Option<Ident>,
    /// `restate NAME` lines after it: the projection's named `mapping`s, taken as
    /// if their `restate` lines were the entry's own.
    pub mappings: Vec<Ident>,
    pub gathered: Option<Ident>,
    pub rows: Option<Ident>,
    /// `publishes (result N)`: the rows the operator forms (`rows rms` or
    /// `staged`) are the entry's result `N`, stored by the launch that forms
    /// them in place of the projection's `staged` scratch (`PUBLISHED`).
    pub publishes: Option<NativePublishDecl>,
    /// `ordered (WEIGHT, ..)`: the weights whose output sums accumulate in
    /// the reference order (`ORDERED`), because they feed a discrete choice.
    pub ordered: Vec<Ident>,
    /// `if (WORD)` and `unless (WORD)` lines after the `project` line: held
    /// by every launch it expands to, as a launch's own are (a launch of the
    /// entry's keeps only its own).
    pub guard: Vec<NativeConditionDecl>,
    /// `restate [LAUNCH] PARAM ..` lines after it: a projection's or one of its
    /// launches' parameter taking the entry's range and default.
    pub restated: Vec<NativeRestateDecl>,
    /// `operand (PARAMETER [normed (WEIGHT, EPSILON)], ..)`: the tensor
    /// parameters whose operands the projection's operand buffers (their Q8
    /// blocks) and the entry's own `operand rows` buffers (their formed rows)
    /// hold, each one for one in the order the expansion places them (a CPU
    /// consumer whose input a producer may form), each with the norm its
    /// rows are formed under, where they are.
    pub operands: Vec<NativeOperandDecl>,
    /// `emits (RESULT over (ROWS, WIDTH), ..)`: the results whose operands the
    /// entry's launches write, each `ROWS` rows of `WIDTH` values (a CPU
    /// producer); a line of these alone (`project NAME emits (..)`) takes
    /// nothing else of the projection.
    pub emits: Vec<NativeEmitDecl>,
    /// The declaration's scratch buffers before the line: the projection's
    /// follow them, and its later ones follow the projection's.
    pub scratch_before: usize,
    pub span: Span,
}

/// The result whose rows the operator forms (`publishes (result N)`).
#[derive(Clone, Debug, PartialEq)]
pub struct NativePublishDecl {
    pub result: u64,
    pub span: Span,
}

/// One result a producer emits the operand of (`emits (result N over
/// (ROWS, WIDTH) [normed (WEIGHT, EPSILON)])`).
#[derive(Clone, Debug, PartialEq)]
pub struct NativeEmitDecl {
    pub result: u64,
    pub over: NativeProjectOver,
    pub norm: Option<OperandNormDecl>,
    pub span: Span,
}

/// One operand a consumer's buffers hold (`operand (SOURCE [tiles
/// (COND)], ..)`): the rows its source gives, their Q8 blocks tiles of a
/// `tiled` projection only where `COND` holds (absent: always).
#[derive(Clone, Debug, PartialEq)]
pub struct NativeOperandDecl {
    pub source: OperandSourceDecl,
    pub tiles: Option<Expr>,
}

/// Where an operand's rows come from.
#[derive(Clone, Debug, PartialEq)]
pub enum OperandSourceDecl {
    /// A tensor parameter's rows (`PARAMETER [at (INDEX)] [within (FIRST,
    /// COUNT)] [normed (WEIGHT, EPSILON)]`): each the row of the parameter
    /// that `INDEX` (a `tensor[ROWS] i32` port) names where it names one,
    /// only rows `FIRST[0]` to `FIRST[0] + COUNT[0]` (`tensor[1] i32`
    /// ports) where it has a window, formed under the norm where it has one.
    Parameter {
        parameter: Ident,
        at: Option<Ident>,
        within: Option<[Ident; 2]>,
        norm: Option<OperandNormDecl>,
    },
    /// Rows whose values the entry's function computes (`values
    /// (FUNCTION)`), a row at a time.
    Values(Ident),
    /// Rows of a weight's own (`W over (ROWS, WIDTH)`) that one of the
    /// entry's launches stores and forms itself (`emitted`, through the
    /// operator's `projection::emit`).
    Emitted(Span),
}

/// The CPU operator's formation of a consumer's operands, set by the
/// expansion (`project .. operand (..)`): its `form` launch, whose body the
/// build generates, and each operand it forms, in the order the line names
/// them.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeFormationDecl {
    pub launch: Ident,
    pub operands: Vec<FormedOperandDecl>,
}

/// One operand a formation forms: `rows` rows of `width` values from
/// `values` (in `groups` runs of `width` columns of each row, group-major);
/// its formed rows in buffer `staged` and, where `formed` holds, its Q8
/// blocks in `quantized`, as tiles where the condition `tiles` holds and the weights take them, for
/// the products over `weights` (ports).
#[derive(Clone, Debug, PartialEq)]
pub struct FormedOperandDecl {
    pub values: FormedValuesDecl,
    pub rows: Expr,
    pub width: Expr,
    pub groups: Expr,
    pub staged: Ident,
    pub quantized: Ident,
    pub formed: Expr,
    pub weights: Vec<Ident>,
    pub tiles: Expr,
}

/// The values a formation forms an operand from: a parameter's rows (each
/// the row `at` names where it names one, only the window `within` where it
/// has one), or the entry's function's.
#[derive(Clone, Debug, PartialEq)]
pub enum FormedValuesDecl {
    Parameter { parameter: Ident, at: Option<Ident>, within: Option<[Ident; 2]> },
    Function(Ident),
}

/// The norm an operand's rows are formed under (`normed (WEIGHT,
/// EPSILON)`): the entry's norm weight port and its epsilon scalar, by
/// name. A consumer's operand and its producer's emitted one pair only
/// under one norm, bound to the same weight and epsilon.
#[derive(Clone, Debug, PartialEq)]
pub struct OperandNormDecl {
    pub weight: Ident,
    pub epsilon: Ident,
    pub span: Span,
}

/// The activation rows (`M`) and their width (`K`) a `project` line gives.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeProjectOver {
    pub rows: Expr,
    pub width: Expr,
}

/// `restate [LAUNCH] NAME in [..] default ..`: the entry's range and default for
/// a parameter of its projection (entry-wide) or of one of the projection's
/// launches (`LAUNCH`).
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRestateDecl {
    pub launch: Option<Ident>,
    pub param: NativeParamDecl,
}

/// One consumer projection for a backend: the forms of projecting
/// activation rows by weights, written once over formals an entry binds
/// with `project` (`crate::projection`).
///
/// ```text
/// projection NAME for TARGET:
///     let NAME = EXPR [for VARIANT]
///     params ([arithmetic] [form] NAME in [V, ..] default DEFAULT [for VARIANT], ..)
///     where CONDITION [for VARIANT]
///     scratch NAME bytes (EXPR) [sync] [when CONDITION] [for VARIANT]
///     each W:
///         scratch NAME bytes (EXPR) [sync] [when CONDITION] [for VARIANT]
///     launch NAME [for VARIANT] [when CONDITION]:
///         params (.. [for VARIANT], ..)
///         ..
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionDecl {
    pub name: Ident,
    pub target: Ident,
    /// Named expressions over the formals, in order.
    pub terms: Vec<Gated<NativeTermDecl>>,
    pub params: Vec<Gated<NativeParamDecl>>,
    /// The `where` lines, conjoined in order.
    pub constraints: Vec<Gated<Expr>>,
    pub scratch: Vec<ProjectionScratchDecl>,
    pub launches: Vec<ProjectionLaunchDecl>,
    /// The `mapping NAME:` blocks: restates an entry takes by name.
    pub mappings: Vec<ProjectionMapping>,
    pub span: Span,
}

/// `mapping NAME:` and its `restate` lines: parameter ranges and defaults several
/// entries share, owned once by the projection and taken by an entry's
/// `restate NAME`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionMapping {
    pub name: Ident,
    pub restated: Vec<NativeRestateDecl>,
    pub span: Span,
}

/// A projection item and the variant it belongs to (`for VARIANT`); `None`
/// belongs to every variant.
#[derive(Clone, Debug, PartialEq)]
pub struct Gated<T> {
    pub item: T,
    /// `for VARIANT | ..`: the variants it belongs to; empty, every one.
    pub variants: Vec<Ident>,
}

/// A projection's scratch: one buffer, or (`each W:`) buffers per weight,
/// named `<stem>_<name>` for each weight port `<stem>_weight`.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionScratchDecl {
    One(Gated<NativeScratchDecl>),
    Each(Vec<Gated<NativeScratchDecl>>),
}

/// A projection launch, named `<entry>_<name>` in the entry it expands
/// into; `param_variants` gates its `params` one for one.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionLaunchDecl {
    pub launch: NativeLaunchDecl,
    pub variants: Vec<Ident>,
    pub param_variants: Vec<Vec<Ident>>,
}

/// `let NAME = EXPR`: a name for a natural-number expression over entry
/// dimensions, entry parameters and the terms before it. The declaration's
/// later expressions read it as that expression.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeTermDecl {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

/// `repeat (EXPR):` over consecutive launches: a call dispatches the block's
/// launches in order, `count` times.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRepeatDecl {
    pub count: Expr,
    /// Ordinal of the block's first launch.
    pub first: usize,
    /// Number of launches in the block.
    pub launches: usize,
    pub span: Span,
}

/// One tuning parameter, its finite domain and its default:
/// `[arithmetic] [form] NAME in [V, ..] default DEFAULT`.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeParamDecl {
    pub name: Ident,
    /// The parameter changes the arithmetic order of a row's result.
    pub arithmetic: bool,
    /// The parameter's values select structurally different algorithms.
    pub form: bool,
    pub values: Vec<u64>,
    pub default: NativeDefaultDecl,
    pub span: Span,
}

/// A parameter's default, over the device (`device.NAME`) and the static
/// dimensions: an expression whose value is one of the listed values, or
/// the least or greatest listed value at which a condition over the
/// parameter itself holds.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeDefaultDecl {
    /// `default EXPR`
    Value(Expr),
    /// `default least where CONDITION`
    Least(Expr),
    /// `default greatest where CONDITION`
    Greatest(Expr),
}

/// `NAME in [DTYPE, ..]`: the dense element types one element parameter of a
/// CPU form is compiled for.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeElementsDecl {
    pub name: Ident,
    pub dtypes: Vec<Ident>,
    pub span: Span,
}

/// A record port of a native declaration: a parameter by name, or `result N`
/// (the entry's result leaf `N`).
#[derive(Clone, Debug, PartialEq)]
pub enum NativeRecordDecl {
    Parameter(Ident),
    Result { ordinal: u64, span: Span },
}

/// Call-private device memory shared by the launches of one native call.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeScratchDecl {
    pub name: Ident,
    pub bytes: Expr,
    /// Arrival counters: every buffer of this kind is zero when a launch
    /// starts, and the call's kernels restore it to zero before it ends.
    pub sync: bool,
    /// The buffer is sized only when this condition holds.
    pub when: Option<Expr>,
    /// The buffer is a projection's operand (`ScratchOperandDecl`).
    pub operand: Option<ScratchOperandDecl>,
    /// Expanded on a producer (`project .. emits (result N ..)`): the result
    /// `N` whose emitted operand the buffer serves, one of the projection's
    /// `for emits` buffers (its forms and their arrival counters alike).
    pub emission: Option<u64>,
    pub span: Span,
}

/// A buffer that holds one form of an operand of the entry: the activation
/// rows' INT8 form (`projection quantized for cpu`'s Q8 blocks) or its F32
/// rows (`rows`: the formed rows exact products read, under the operand's
/// norm where it has one), which either the entry forms from one of its
/// tensor parameters or the producer of that tensor writes in its epilogue.
#[derive(Clone, Debug, PartialEq)]
pub enum ScratchOperandDecl {
    /// `operand [rows] [when PAIRS] [formed (COND)] [nvfp4 (COND)]`, as a
    /// projection or a consumer writes it on a buffer. On a consumer's
    /// buffer: `when`, the condition under which a producer's operand stands
    /// for the one the entry forms (absent: always); `formed`, when the
    /// entry's formation writes this form (absent: always); `nvfp4`, on a
    /// rows form, when the formation applies the NVFP4 operand transform to
    /// the rows (absent: never). On an `emits` buffer, none: the form a
    /// producer writes.
    Declared { rows: bool, when: Option<Expr>, formed: Option<Expr>, nvfp4: Option<Expr> },
    /// Expanded on a consumer (`project .. operand (PARAMETER, ..)`): the
    /// operand of tensor parameter `parameter`, which the entry's launch
    /// `formation` forms unless a producer emits it and `pairs` holds; the
    /// launch writes this form where `formed` holds, its rows as NVFP4
    /// operands where `nvfp4` holds (a Q8 form's is `false`).
    Formed {
        parameter: Ident,
        formation: Ident,
        pairs: Expr,
        rows: bool,
        norm: Option<OperandNormDecl>,
        formed: Expr,
        nvfp4: Expr,
    },
    /// Expanded on a producer (`project .. emits (RESULT over (..), ..)`): the
    /// operand of its result (the buffer's `emission`), which its launches
    /// write.
    Emitted { rows: bool, norm: Option<OperandNormDecl> },
}

/// One ordered dispatch of a kernel function from the native source.
/// A launch condition's word: `PARAMETER` (word 0 of a `&tensor[1] i32`)
/// or `PARAMETER[INDEX]` (word INDEX of a `&tensor[N] i32`, INDEX a constant
/// below N).
#[derive(Clone, Debug, PartialEq)]
pub struct NativeConditionDecl {
    pub encoded: crate::checked::Encoded,
    pub word: Ident,
    /// The word's index, as written (a constant, which checking requires):
    /// none for the plain form.
    pub index: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeLaunchDecl {
    pub kernel: Ident,
    /// `before LAUNCH`: in a declaration that projects, the projection
    /// launch this launch dispatches before.
    pub before: Option<Ident>,
    /// `resident (BARRIER)`: every threadgroup of the launch is co-resident,
    /// its grid the backend's (`RESIDENT`, at most `threadgroups`' first
    /// axis), and its threadgroups meet at grid barriers held in the `sync`
    /// scratch buffer BARRIER.
    pub resident: Option<Ident>,
    /// The launch runs only when this condition holds.
    pub when: Option<Expr>,
    /// Parameters local to this launch; names may be reused by other launches.
    pub params: Vec<NativeParamDecl>,
    /// The launch's `where`: restricts the admissible configurations of the
    /// parameters it reads, the launch's own and the entry's.
    pub constraint: Option<Expr>,
    /// Entry parameters read by the kernel but absent from launch expressions.
    pub reads: Vec<Ident>,
    /// `alike (LAUNCH)`: the launch computes each row as LAUNCH computes
    /// that row alone (the launch that serves one row, or the launch whose
    /// row the others reproduce): a row's bits do not depend on how many
    /// rows its call carries. Neither declares an `arithmetic` parameter of
    /// its own, and both read the same `arithmetic` settings of the native,
    /// so one reduction order serves every row count.
    pub alike: Option<Ident>,
    /// `staging`: the launch forms operands for another launch and computes
    /// no row's result, so it is no row class: the row classes a row count
    /// selects up to the verification rows (`alike`) leave staging passes
    /// out. A launch is not both.
    pub staging: bool,
    /// `records (PORT, ..)`: the record ports of this launch in place of the
    /// native's; `None` takes the native's.
    pub records: Option<Vec<NativeRecordDecl>>,
    /// The most groups the launch dispatches on each axis; with `grid`, the
    /// bound of the grid it reads.
    pub threadgroups: [Expr; 3],
    /// `grid (PARAMETER)`: the launch dispatches the grid an earlier launch
    /// wrote into this entry parameter (three `i32` group counts, each at
    /// most `threadgroups`'), read when the launch runs (indirect dispatch).
    pub grid: Option<Ident>,
    /// `if (WORD)` or `unless (WORD)` lines, a conjunction: the launch is
    /// encoded only while each such word of a host-shared `&tensor[N] i32`
    /// entry parameter the host writes between submissions is nonzero, or
    /// zero, read when the run is encoded.
    pub conditions: Vec<NativeConditionDecl>,
    pub threads_per_threadgroup: [Expr; 3],
    pub shared_bytes: Option<Expr>,
    pub span: Span,
}

/// `fn name[shape](params) [-> result] [requires capability] [where pred]: body`
#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    pub signature: Signature,
    pub name: Ident,
    /// Capability namespaces named by the declaration; a portable function
    /// requires none, and the resolver rejects any.
    pub requires: Vec<CapabilityPath>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub shape: Vec<Ident>,
    pub params: Vec<Param>,
    /// `None` means `void`.
    pub result: Option<TypeExpr>,
    /// Conjuncts of the `where` clause.
    pub predicates: Vec<Expr>,
}

/// `lower name[..](..) [-> r] for target [where pred]: body`.
#[derive(Clone, Debug, PartialEq)]
pub struct LowerDecl {
    pub name: Ident,
    pub signature: Signature,
    pub target: Ident,
    /// Capability namespaces explicitly required by this lowering body.
    pub requires: Vec<CapabilityPath>,
    /// Conjuncts following `for target where`; for the long form these are also stored here,
    /// not in `signature.predicates`.
    pub predicates: Vec<Expr>,
    pub body: Block,
    pub span: Span,
}

/// A backend capability namespace such as `metal.matrix`.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityPath {
    pub backend: Ident,
    pub capability: Ident,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeExpr {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    /// `f32`, or an element parameter used as a scalar type (`T`).
    Scalar(Ident),
    /// `index[N]`
    Index(Box<Expr>),
    /// `range[N]`: a bounded logical half-open range.
    Range(Box<Expr>),
    /// `tensor[shape] elem`, `&tensor[shape] elem`, or `&mut tensor[shape] elem`.
    Shaped {
        head: ShapedHead,
        shape: Vec<Expr>,
        elem: Ident,
    },
    Tuple(Vec<TypeExpr>),
    Void,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapedHead {
    /// Owned logical tensor.
    Tensor,
    /// Shared logical tensor borrow (`&tensor`).
    SharedTensor,
    /// Exclusive mutable logical tensor borrow (`&mut tensor`).
    MutTensor,
    /// A host port (`&host tensor`, `&host mut tensor`): a tensor the host
    /// reads (`write` false) or rewrites once it sees the launch, which the
    /// kernel never accesses.
    HostTensor { write: bool },
    /// An awaited port (`&await tensor`, `&await mut tensor`): a shared or
    /// (`write`) mutable borrow holding words the host writes while the
    /// launch waits for them. The launch parks: it reads them only once its
    /// host has answered, which each backend realizes its own way (a
    /// backend whose devices see host writes while work runs spins in the
    /// kernel; one whose devices do not ends the submission before the
    /// launch and resumes it on the host's signal).
    AwaitTensor { write: bool },
    /// A slab port (`&slab tensor`, `&slab mut tensor`): a shared or
    /// (`write`) mutable borrow whose binding is a slab view, its rows held
    /// by the slabs its address table names. Its natives reach its rows
    /// through that table, so it binds no other storage
    /// ([`crate::entry::PortBinding::Slab`]).
    SlabTensor { write: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

/// `let` target: `a`, `a, b`, `(a, b)`, nested.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Name(Ident),
    Tuple(Vec<Pattern>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Let {
        mutable: bool,
        pattern: Pattern,
        value: Expr,
    },
    /// `target op= value`; `target` may be a tuple of places for tuple assignment.
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// `for targets in iter`: `iter` is a bounded range or a range value.
    For {
        /// `false` is ordered `for`; `true` is independent `parallel for`.
        parallel: bool,
        targets: Vec<Ident>,
        iter: Expr,
        body: Block,
    },
    If {
        cond: Expr,
        then: Block,
        els: Option<Block>,
    },
    Return(Vec<Expr>),
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(u64),
    Float(f64),
    Inf,
    Bool(bool),
    Name(Ident),
    Tuple(Vec<Expr>),
    /// `lo..hi`
    Range {
        lo: Box<Expr>,
        hi: Box<Expr>,
    },
    /// `tensor[shape] elem`: uninitialized owned logical tensor storage.
    Tensor {
        shape: Vec<Expr>,
        elem: Ident,
    },
    /// `f[R = 64](args)`
    Call {
        callee: Box<Expr>,
        bindings: Vec<(Ident, Expr)>,
        args: Vec<Arg>,
    },
    Index {
        base: Box<Expr>,
        indices: Vec<Index>,
    },
    /// `t.T`, `t.words`, `metal.name`
    Attr {
        base: Box<Expr>,
        name: Ident,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub name: Option<Ident>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    /// A point, a slice binder or a tile coordinate; the checker distinguishes them.
    Expr(Expr),
    /// `lo:hi`, `lo:`, `:hi`, `:`
    Slice {
        start: Option<Expr>,
        end: Option<Expr>,
    },
}
