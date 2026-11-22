//! Projections: a backend's consumer forms of projecting activation rows by
//! weights, written once and taken by every entry that projects.
//!
//! # Decision
//!
//! A form is two things: a property of the weight (whether the backend's
//! producer delivers its operand, decided in [`crate::forms`]) and a
//! pipeline (its parameter, admission, shape guards, scratch and launches).
//! The first reaches every caller through `form(W, NAME)`; the second was
//! restated by each entry. A projection owns the second, in the language,
//! so that it is checked, tuned and formed exactly as a hand-written native
//! declaration is:
//!
//! - `projection NAME for TARGET:` declares the forms over formals: `M`
//!   activation rows, `K` columns, `N` rows of each weight, `G` the groups
//!   of those rows (an entry's `project` line names them with `groups G`,
//!   each weight then `[G * N, K]`; else 1), `P` the parts of the columns
//!   (`parts P`, each weight then `[N, P * X]`, each part's rows its own;
//!   else 1) and `PARTED` whether the line names them, `WEIGHTS` the
//!   weights one activation row serves, and `W` a weight inside a
//!   quantifier. A `let` names an expression over them, per variant where
//!   it carries `for VARIANT` (or `for A | B`, either), and so may any
//!   parameter, `where` line, scratch buffer or launch. A `segmented`
//!   projection's weights have rows of their own, so it reads `N` only per
//!   weight.
//! - Admission quantifies over the weights: `each(W, C)` is C of every
//!   weight, `any(W, C)` of some weight, and `agree(W, E)` that every
//!   weight's E equals the first's, `sum(W, E)` adds E over the weights and
//!   `tiles(W, E)` is E over the projection's output rows (the shared `N`
//!   once, or each segment's `N` summed); `bound(W)` is a weight's packing-bound
//!   extent. Scratch in an `each W:` block is one buffer per weight.
//! - An entry's native declaration takes a projection with one line,
//!   `project NAME VARIANT (WEIGHT, ..)`, naming its weight ports (a
//!   `segmented` one may group them into segments serving a pair, `((b, x), c)`;
//!   `GATHERED` is 1 where the weight rows are selected by a port (`gathered`),
//!   `SERVED` is the most weights a segment serves, and within a weight
//!   quantifier the weights of that weight's segment, so `tiles` counts each
//!   segment's rows once, at its own tile width; `gathered PORT` takes the
//!   weight rows a `tensor[N]` port selects, `N` its extent). Each port
//!   is a `tensor[N, K] E` named by its stem (`<stem>_weight`, `weight` with
//!   an empty stem and no `_`, or any other name, its own stem), its bound
//!   port (if any; `bound(W)` is 0 without one)
//!   `<stem>_bound: tensor[B] f32` (`weight_bound` for `weight`),
//!   and the entry returns its `M` rows first (`[M, ..]`, or a tuple whose
//!   first result is; its epilogue may widen them past `N`): the formals
//!   are read from the signature, any dimension expression it writes,
//!   never restated. A projection
//!   launch `NAME` is the entry's kernel `<entry>_NAME`, and a per-weight
//!   buffer `NAME` is `<stem>_NAME` (`NAME` for `weight`).
//! - An entry may take its own range and default for a projection's parameter
//!   (`restate NAME in [..] default D`) or for a parameter of one of its
//!   launches (`restate LAUNCH NAME in [..] default D`), in lines after the
//!   `project` line. It keeps the parameter's flags and admits only values
//!   the projection admits. An entry-wide restate of a parameter the
//!   projection declares only on launches makes it one entry-wide parameter
//!   that every one of its launches declaring it reads, so those launches
//!   share its value (`restate KSPLIT in [..] default D` over two GEMV
//!   launches that each declare `KSPLIT`). A projection may name a set of such restates
//!   several entries share (`mapping NAME:` and its `restate` lines, after its
//!   launches), which an entry takes with `restate NAME`; its own restates win.
//! - A weight of a `segmented` projection may be one row, a `tensor[K]` (a
//!   segment of `N` = 1: a router's shared-expert gate column).
//! - An entry whose `rms` or `staged` rows are one of its results says so
//!   (`rows rms publishes (result N)`, the result a `tensor[M, K]`): the
//!   launch that forms the rows stores them in that result, so the fact
//!   `PUBLISHED` is 1 and the projection declares no `staged` scratch for
//!   them (`when PUBLISHED == 0`). An entry whose outputs feed a discrete
//!   choice (a router's top-k) names those weights (`ordered (W, ..)`): the
//!   weight's fact `ORDERED` lets the projection offer, for them alone, the
//!   forms that accumulate every output in the CPU reference's order. `if (WORD)` and `unless (WORD)` lines
//!   after a `project` line are held by every launch it expands to, as a
//!   launch's own are.
//! - A form (`form NAME in [..]`) is the projection's alone: no entry, and
//!   no mapping, restates one. Its `where` lines state its admission once,
//!   over the weights (`form(W, NAME)`, `operands(W)`) and the facts the
//!   `project` line gives (`GATHERED`, `PLANES`, `PARTED`, `PUBLISHED`, `ORDERED`, a routed entry's
//!   `STACKED`, `SHARED`, ..), and its default may be an expression over the
//!   same facts. Where an entry's facts reduce a conjunct to a condition on
//!   one parameter alone, that is the parameter's range for the entry
//!   ([`admit`]): the values it rejects are not the entry's, so a form never
//!   reaches an entry whose facts its admission excludes, and one whose
//!   admission names no fact reaches every entry. A launch's own `where`
//!   does the same for the launch's parameters.
//! - [`expand`] rewrites every such declaration into the ordinary native
//!   declaration it stands for, before checking. The checker, coverage,
//!   the tuner's parameter addresses and formation see only that
//!   declaration, so moving an entry onto a projection changes none of
//!   them: its parameters follow the entry's, its conditions follow its
//!   `where`, its scratch sits at the `project` line, and its launches keep
//!   their order, an entry launch placed by `before LAUNCH`. A launch or
//!   scratch buffer whose `when` needs more rows than the entry's contract
//!   admits (its function's `M <= c` or `M < c`) is not part of it, and
//!   neither is a projection parameter that nothing kept reads: its
//!   conditions hold for some value of it.
//!
//! A plane set (`planes (PLANE, ..) over (ROWS, WIDTH)` in place of a
//! `single` projection's weight, on a GPU) is one weight stored as planes:
//! each `PLANE` is a port of the entry, its rows `N` and width `K` are those
//! its `over` gives, and its planes take none of the forms a weight's
//! producer delivers, so `form(W, ..)`, `operands(W)` and `bound(W)` are 0.
//! The fact `PLANES` is 1 for a plane set and 0 for every other weight; the
//! projection's items that project a plane set, or cannot, are gated on it,
//! so a projection an entry takes a plane set by must read `PLANES`.
//!
//! A `gathered` port of rank 3, `tensor[M, I, N]` with `I` a static
//! dimension, selects rows per activation row: row m projects by the `N`
//! weight rows its own selections name where `I` is 1, and by the first `N`
//! (the weight's prefix) where `I` is 0. The entry then takes two
//! projections, one for each value of `I`, every item of each gated by it:
//! the line's own (a `paired` or `single` one, `GATHERED` 0, `N` the prefix),
//! its items under `I == 0`; and the routed projection's `choice` variant
//! (`PER_ROW`), whose experts are the activation rows, each one choice (`I`)
//! of the `N` rows it selects, its items under `I > 0`. A condition of either
//! holds where the other's value of `I` is in force. The routed expansion's
//! parameters are `GATHER_<NAME>` (an entry restates one as such), its
//! launches `<entry>_gather_<NAME>` and its buffers `gather_<NAME>`, so the
//! two never meet.
//!
//! A routed projection (variants `choice`, `summed`, `grouped`, the
//! mixture's forms: `ROUTED_VARIANTS`) projects by expert-stacked weights
//! (`tensor[E, N, K]`) or dense ones: `project NAME VARIANT (WEIGHT, ..)`
//! names the one or two weights that expand (`SERVED`), then the shared
//! expert's (`shared (..)`, `choice` and `summed`) or the down weight
//! (`down (..)`, `grouped`). Its extents other than its rows and choices are
//! the routed entries' own names, which they share, so they are not formals; its
//! formals are its rows `M` and choices `K` (a routed entry's own `M` and
//! `K`; a per-row gathered entry's rows and `I`), each weight's `N` and
//! `WIDTH`, `STACKED` and `FIRST` (of the first list), and the entry's facts
//! `SHARED`, `RECORDS` (a `records` line,
//! which a launch's `records (W)` lists), `WHOLE` (a `policy` port) and
//! `PER_ROW` (a per-row gathered entry's: each row its own expert);
//! `most(W, E)` is the largest of E's distinct values over the weights. A
//! launch or buffer whose `when` the facts decide false is not part of the
//! entry, and one they decide true has no condition; a launch whose grid is
//! read from a buffer of the projection's that the entry does not have runs
//! its declared groups.
//!
//! The rows an entry projects are of one kind, fixed by the `project` line's
//! `rows KIND` (`plain`, the default: its prologue passes them as given; `rms`:
//! its prologue forms them, normalizing, so a stage launch materializes them
//! for more than one row; `staged`: a stage launch alone forms them, at every
//! row count), and `for rms` / `for plain` gates an item like a
//! variant. The kind is the prologue's: a CUDA declaration of one kind over a
//! kernel whose prologue is of another is refused when the module is checked.
//!
//! A variant (`paired`, `single`, `segmented`) is fixed by the entry and
//! folds away:
//! formals and terms substitute, literal arithmetic and comparisons
//! evaluate, `x * 1` and `x / 1` reduce, and a condition that becomes true
//! drops out of its conjunction. The expansion is therefore exactly the
//! text an entry would write by hand.
//!
//! A CPU consumer's activation operand is the operator's
//! (`projection quantized for cpu`): the line names where each operand's
//! rows come from (`operand (SOURCE [tiles (COND)], ..)`, a `tiled`
//! consumer's Q8 blocks tiles only where its condition `COND` holds), and the
//! projection's buffers hold its forms (the Q8 blocks, `operand`; the formed
//! F32 rows, `operand rows`, NVFP4 operands where the weights take them). A
//! source is a tensor parameter (`PARAMETER [at (INDEX)] [within (FIRST,
//! COUNT)] [normed (WEIGHT, EPSILON)]`: the rows `INDEX` names, only the
//! window from `FIRST[0]` of `COUNT[0]` rows, formed under the norm), the
//! entry's function of a row (`values (FUNCTION)`), or, for a weight's own
//! rows, the entry's launch that stores them (`emitted`). The `form` launch
//! forms the first two: `<entry>_form`, its body generated from the
//! declaration's formation, one work item a part of whole Q8 blocks of a
//! row (a function's rows whole). A producer of a parameter may write the
//! same forms in its epilogue (`emits`), and a graph then leaves the launch
//! out; an `emitted` operand's launch forms it in its own epilogue
//! (`projection::emit`). No entry writes a stage launch of its own.
//!
//! The native libraries own the matching kernel bodies, per backend
//! (`metal/lib/projection/operator.h`); an entry's source keeps only its
//! prologue, epilogue and weights.

use std::collections::{BTreeMap, BTreeSet};

use crate::checked::{DiagnosticRule, PROJECTION_SEGMENTS};
use crate::span::{Diagnostic, Span};
use crate::syntax::ast::*;

/// The variants an entry projects by, with the number of weights each takes:
/// `paired`, two weights one activation row serves, of the same rows;
/// `single`, one weight; `segmented`, one to `PROJECTION_SEGMENTS` segments
/// of their own rows each (any of them possibly empty), every one projecting
/// the same activation rows into its own columns of the result, in order.
const VARIANTS: [(&str, usize); 3] = [("paired", 2), ("single", 1), ("segmented", PROJECTION_SEGMENTS)];

/// The variants of a routed projection, over a mixture's experts, with the
/// most weights each takes in its first list: `choice`, each choice (m, k)
/// of an activation row projected by its expert's weights into its own
/// product row; `summed`, a row's choices projected and summed in slot
/// order; `grouped`, the rows grouped by expert (`order`, `blocks`, `used`),
/// each block expanded by its expert's weights and its products projected by
/// the `down` weight. One or two weights expand (a gate and an up weight
/// one activation row serves, combined by the entry's epilogue, or an up
/// weight alone); `summed` projects by one.
const ROUTED_VARIANTS: [(&str, usize); 3] = [("choice", 2), ("summed", 1), ("grouped", 2)];

/// The projection and variant a per-row gathered entry (a `gathered` port
/// `tensor[M, I, N]`) projects its selected rows by where `I` is 1: each
/// activation row an expert of one choice, its selected rows the expert's.
const PER_ROW: (&str, &str) = ("routed", "choice");

/// What a per-row gathered entry's routed expansion's parameters, launches
/// (`<entry>_gather_<NAME>`) and buffers (`gather_<NAME>`, the parameters
/// upper case) are named with.
const GATHER: &str = "gather_";

/// The variants of a projection for the CPU, whose weights share no
/// `[N, K]` shape and an entry gives the activation rows it quantizes: the
/// rows in tiles of the tile kernel's rows (`tiled`) or each row alone
/// (`rows`); or none projected (`emits`: a producer's line, `project NAME
/// emits (..)`, which takes only the buffers the projection gates `for
/// emits`). An entry whose dot is its own takes no CPU projection but, as a
/// producer, its `emits` line: its rows are split by the pool
/// (`device.units`), not by the operator's `ROWS`. `plain` names only a kind
/// of rows ([`RowsKind`]). The rows the line gives (`project .. (W, ..) over
/// (ROWS, WIDTH)`) are those every weight without rows of its own
/// multiplies, the projection's buffers of no weight quantizing them; a
/// weight that multiplies rows of its own (`W over (ROWS, WIDTH)`) has the
/// buffers of an `each W:` block, and no other weight has them, so one line
/// mixes weights of the entry's rows and of their own. Outside a weight `M`
/// and `K` are the rows the line gives, 0 where it gives none; a weight's are
/// its own where it has them, and `OWN` is 1 where it does, else 0.
const CPU_VARIANTS: [&str; 3] = ["tiled", "rows", "emits"];

/// The CPU projection's launch that forms a consumer's operands
/// (`<entry>_form`): taken by an entry whose `project` line names them,
/// its body generated from the declaration's formation
/// (`NativeFormationDecl`), never written by an entry.
const FORM: &str = "form";

/// Rewrite every native declaration of the parsed files that projects into
/// the declaration it stands for, and remove the projections. The
/// diagnostics name the file (its index in `parsed`'s numbering) they are
/// in.
pub(crate) fn expand(parsed: &mut [(usize, File)]) -> Vec<(usize, Diagnostic)> {
    let mut diagnostics = Vec::new();
    let mut projections: BTreeMap<(String, String), (usize, ProjectionDecl)> = BTreeMap::new();
    for (file, ast) in parsed.iter_mut() {
        let mut kept = Vec::with_capacity(ast.decls.len());
        for decl in std::mem::take(&mut ast.decls) {
            match decl {
                Decl::Projection(projection) => {
                    if let Err(diagnostic) = gates(&projection) {
                        diagnostics.push((*file, diagnostic));
                    }
                    let key = (projection.name.name.clone(), projection.target.name.clone());
                    if projections.contains_key(&key) {
                        diagnostics.push((
                            *file,
                            error(
                                projection.name.span,
                                format!(
                                    "projection `{}` is declared twice for `{}`",
                                    projection.name.name, projection.target.name
                                ),
                            ),
                        ));
                    }
                    projections.insert(key, (*file, projection));
                }
                other => kept.push(other),
            }
        }
        ast.decls = kept;
    }
    let mut signatures: BTreeMap<String, Vec<Signature>> = BTreeMap::new();
    let mut constants: BTreeMap<String, u64> = BTreeMap::new();
    for (_, ast) in parsed.iter() {
        for constant in &ast.constants {
            constants.insert(constant.name.name.clone(), constant.value);
        }
        for decl in &ast.decls {
            if let Decl::Fn(function) = decl {
                signatures
                    .entry(function.name.name.clone())
                    .or_default()
                    .push(function.signature.clone());
            }
        }
    }
    for (file, ast) in parsed.iter_mut() {
        for decl in &mut ast.decls {
            let Decl::Native(native) = decl else {
                continue;
            };
            let result = match native.project.take() {
                Some(project) => {
                    expand_native(native, project, *file, &projections, &signatures, &constants, None)
                }
                None => match native.launches.iter().find_map(|launch| launch.before.as_ref()) {
                    Some(before) => Err(error(
                        before.span,
                        "`before` places a launch among a projection's; this declaration does not project",
                    )),
                    None => Ok(()),
                },
            };
            if let Err(diagnostic) = result {
                diagnostics.push((*file, diagnostic));
            }
        }
    }
    diagnostics
}

fn error(span: Span, message: impl Into<String>) -> Diagnostic {
    Diagnostic::with_rule(DiagnosticRule::NativeDeclaration, span, message)
}

/// Every `for VARIANT` of a projection names a variant.
fn gates(projection: &ProjectionDecl) -> Result<(), Diagnostic> {
    let mut variants = Vec::new();
    variants.extend(projection.terms.iter().flat_map(|term| &term.variants));
    variants.extend(projection.params.iter().flat_map(|param| &param.variants));
    variants.extend(projection.constraints.iter().flat_map(|constraint| &constraint.variants));
    for scratch in &projection.scratch {
        match scratch {
            ProjectionScratchDecl::One(one) => variants.extend(&one.variants),
            ProjectionScratchDecl::Each(each) => variants.extend(each.iter().flat_map(|one| &one.variants)),
        }
    }
    for launch in &projection.launches {
        variants.extend(&launch.variants);
        variants.extend(launch.param_variants.iter().flatten());
    }
    for variant in variants {
        if !VARIANTS.iter().any(|(name, _)| *name == variant.name)
            && !ROUTED_VARIANTS.iter().any(|(name, _)| *name == variant.name)
            && !CPU_VARIANTS.contains(&variant.name.as_str())
            && RowsKind::named(&variant.name).is_none()
            && variant.name != "ordered"
        {
            return Err(error(
                variant.span,
                format!(
                    "`{}` is not a projection variant (`paired`, `single`, `segmented`; routed `choice`, `summed`, `grouped`; the CPU's `tiled`, `rows`, `emits`), kind of rows (`plain`, `rms`, `staged`), or `ordered`",
                    variant.name
                ),
            ));
        }
    }
    Ok(())
}

/// One weight an entry projects by.
struct Weight {
    /// Its element parameter: what `W` stands for.
    element: Ident,
    /// Its port's stem with a trailing `_` (`gate_` for `gate_weight` or
    /// for `gate`; empty for `weight`): its bound port and per-weight
    /// buffers take the same.
    prefix: String,
    /// Its bound port's extent (`bound(W)`; 0 without one).
    bound: Option<Ident>,
    /// Its rows (`N` of this weight).
    n: Expr,
    /// Its columns (`WIDTH` of this weight, a routed projection's).
    width: Expr,
    /// Whether its port stacks one matrix an expert (`tensor[E, N, K]`, a
    /// routed projection's), the expert's weights then addressed by its
    /// source: `STACKED`.
    stacked: bool,
    /// The segment it belongs to (a `segmented` projection's weights are in
    /// segments of one weight, or of a served pair).
    segment: usize,
    /// The rows and width it multiplies where it has its own
    /// (`project .. (W over (ROWS, WIDTH))`).
    over: Option<NativeProjectOver>,
    /// Whether one of the entry's launches stores and forms its own rows'
    /// operand (`operand (.., emitted)`): `EMITTED`.
    emitted: bool,
    /// Whether it is a plane set (`planes (..)`): no element, and no form a
    /// producer delivers.
    planes: bool,
    /// Whether its output sums accumulate in the reference order
    /// (`ordered (..)`): `ORDERED`.
    ordered: bool,
}

/// One projection instantiated for one entry.
struct Instance<'a> {
    entry: &'a str,
    variant: &'a str,
    /// The kind of its rows (`rows KIND`).
    rows: RowsKind,
    /// The formals, as the signature writes them; `g` the groups the
    /// `project` line names, else 1, and `p` the parts of the columns it
    /// names (`parts P`), else 1.
    g: Expr,
    p: Expr,
    /// Whether the line names parts (`PARTED`): an item every parted entry
    /// takes, at every part count, one included.
    parted: bool,
    m: Expr,
    k: Expr,
    /// The weights' shared rows; `None` where each has its own
    /// (`segmented`), read through a weight.
    n: Option<Expr>,
    weights: Vec<Weight>,
    /// The most weights one segment serves.
    served: usize,
    /// Whether the weight rows are selected by a port (`gathered`): forms that
    /// stage or code whole weight rows in storage order do not apply.
    gathered: bool,
    /// Whether its weight is a plane set (`PLANES`).
    planes: bool,
    /// Whether the rows the operator forms are an entry result
    /// (`publishes`): `PUBLISHED`.
    published: bool,
    /// The conditions every launch of the line holds (`if`/`unless` lines).
    guard: &'a [NativeConditionDecl],
    /// The projection's terms in force for the variant.
    terms: BTreeMap<String, Expr>,
    /// Every term name of the projection, in force or not, but those that
    /// name a parameter of this variant.
    named: BTreeSet<String>,
    /// Where the projection's text sits: its spans where it shares the
    /// entry's file, else the `project` line.
    same_file: bool,
    at: Span,
    /// A routed projection's facts about the entry (`None` for the others).
    routed: Option<Routed>,
    /// Module constants (`const NAME = INTEGER`), for row bounds and `sources`.
    constants: &'a BTreeMap<String, u64>,
    /// The entry's `restate` lines.
    restated: &'a [NativeRestateDecl],
    /// The launch parameters the entry restates entry-wide: one parameter
    /// of the entry that the launches declaring them read.
    lifted: BTreeSet<String>,
}

/// What a routed projection reads of its entry, beyond its weights. Its
/// extents other than its rows and choices are routed.seismic's own names (`H`,
/// `F`, `S`, `XS`, `B`, `T`), which every routed entry shares, so its text
/// names them as they are rather than as formals.
struct Routed {
    /// `K`: the choices of each of its rows; a routed entry's own `K`.
    choices: Expr,
    /// `XS`: the source rows of `sources`, folded where it names a constant.
    xs: Option<Expr>,
    /// `SHARED`: the shared expert's weights are projected too (`shared (..)`).
    shared: bool,
    /// `RECORDS`: the entry reads its experts' weights through record ports
    /// (its `records` line), which a launch's `records (W)` lists.
    records: Vec<NativeRecordDecl>,
    /// `WHOLE`: the entry is dispatched by its bank's policy (a `policy`
    /// port), whose whole-bank word chooses its launch.
    whole: bool,
    /// `PER_ROW`: each activation row is its own expert, the weight rows it
    /// selects (a per-row gathered entry's), so no two rows share one.
    per_row: bool,
}

/// The value of a per-row gathered entry's static dimension `I` an
/// expansion's items are in force at: each launch and buffer is one where
/// `on` holds, and each condition holds where `off` does.
#[derive(Clone)]
struct Gate {
    on: Expr,
    off: Expr,
}

/// What a per-row gathered entry's routed expansion (`PER_ROW`) projects:
/// `m` rows (the entry's), each of `choices` (`I`) choices of `n` selected
/// weight rows, under `gate` (`I > 0`).
struct PerRow {
    m: Expr,
    choices: Expr,
    n: Expr,
    gate: Gate,
}

/// Expand `native`'s `project` line; `per_row` is given for the routed
/// expansion of a per-row gathered entry, whose line it makes. The phases
/// run in the order an entry's declaration is read: what the line takes
/// ([`taking`]), the formals its signature binds ([`bind_formals`]), the
/// instance and its terms, then the items the instance expands
/// ([`parameters`], [`buffers`], [`form_operands`], [`launches`]), and last a
/// per-row gathered entry's selected rows ([`expand_selected`]). Each phase
/// refuses what it alone can tell is wrong; one never repairs another's.
fn expand_native(
    native: &mut NativeDecl,
    project: NativeProjectDecl,
    file: usize,
    projections: &BTreeMap<(String, String), (usize, ProjectionDecl)>,
    signatures: &BTreeMap<String, Vec<Signature>>,
    constants: &BTreeMap<String, u64>,
    per_row: Option<&PerRow>,
) -> Result<(), Diagnostic> {
    let Some((projection_file, projection)) =
        projections.get(&(project.projection.name.clone(), native.target.name.clone()))
    else {
        return Err(error(
            project.projection.span,
            format!("no projection `{}` for `{}`", project.projection.name, native.target.name),
        ));
    };
    let taking = taking(native, &project)?;
    let signature = match signatures.get(&native.function.name).map(Vec::as_slice) {
        Some([signature]) => signature,
        _ => {
            return Err(error(
                project.span,
                format!("`{}` is not one portable function to project", native.function.name),
            ));
        }
    };
    let Formals { mut weights, n, k, m, selected } = bind_formals(native, &project, signature, &taking, per_row)?;
    for (weight, port) in weights.iter_mut().zip(&project.weights) {
        weight.ordered = project.ordered.iter().any(|name| name.name == port.name);
    }
    let mut segment_ordered = Vec::with_capacity(taking.served.len());
    let mut cursor = 0;
    for &count in &taking.served {
        let slice = &weights[cursor..cursor + count];
        if slice.len() == 2 && slice[0].ordered != slice[1].ordered {
            return Err(error(
                project.span,
                "a served pair's weights either both accumulate in the reference order (`ordered`) or neither does",
            ));
        }
        segment_ordered.push(slice.first().is_some_and(|weight| weight.ordered));
        cursor += count;
    }
    let gate = selection_gate(&project, taking.segmented, &selected, &m, per_row)?;
    let published = published_rows(&project, signature, &m, &k)?;
    let g = match &project.groups {
        Some(groups) => Expr { kind: ExprKind::Name(groups.clone()), span: groups.span },
        None => Expr { kind: ExprKind::Int(1), span: project.span },
    };
    let p = match &project.parts {
        Some(parts) => Expr { kind: ExprKind::Name(parts.clone()), span: parts.span },
        None => Expr { kind: ExprKind::Int(1), span: project.span },
    };
    // The entry's own restates first (they win), then those of the mappings it takes.
    // A per-row gathered entry's `GATHER_` restates are its routed expansion's.
    let gathers = GATHER.to_uppercase();
    let (gathered_restated, mut restated): (Vec<_>, Vec<_>) = project
        .restated
        .iter()
        .cloned()
        .partition(|restated| selected.is_some() && restated.param.name.name.starts_with(&gathers));
    for taken in &project.mappings {
        let Some(mapping) = projection.mappings.iter().find(|mapping| mapping.name.name == taken.name) else {
            return Err(error(taken.span, format!("projection `{}` has no mapping `{}`", projection.name.name, taken.name)));
        };
        restated.extend(mapping.restated.iter().cloned());
    }
    // The line's operands name the entry's rows first, where it gives them,
    // then each weight's own rows in order; an entry's launch emits only a
    // weight's own rows' operand.
    let entry_operands = usize::from(project.over.is_some()).min(project.operands.len());
    if let Some(operand) = project.operands[..entry_operands].first() {
        if let OperandSourceDecl::Emitted(span) = operand.source {
            return Err(error(span, "the entry's rows are formed by the operator or their producer; only a weight's own rows are `emitted`"));
        }
    }
    for (weight, operand) in weights.iter_mut().filter(|weight| weight.over.is_some()).zip(&project.operands[entry_operands..]) {
        weight.emitted = matches!(operand.source, OperandSourceDecl::Emitted(_));
    }
    // A routed entry's rows are its own prologue's (`routed`, below).
    native.projected_rows = (!taking.routed).then_some(taking.rows);
    native.projected_segments = taking.segmented.then(|| taking.served.clone());
    native.projected_ordered = if taking.segmented && segment_ordered.iter().any(|&b| b) { segment_ordered } else { Vec::new() };
    let routed = taking.routed.then(|| Routed {
        choices: match per_row {
            Some(per_row) => per_row.choices.clone(),
            None => Expr { kind: ExprKind::Name(Ident { name: "K".to_owned(), span: project.span }), span: project.span },
        },
        xs: port(signature, "sources")
            .and_then(|sources| dims(&sources.ty))
            .and_then(|(extents, _)| extents.first().cloned())
            .map(|mut xs| {
                if let ExprKind::Name(name) = &xs.kind {
                    if let Some(value) = constants.get(&name.name) {
                        xs.kind = ExprKind::Int(*value);
                    }
                }
                xs
            }),
        shared: !project.shared.is_empty(),
        records: native.records.clone(),
        whole: signature.params.iter().any(|param| param.name.name == "policy"),
        per_row: per_row.is_some(),
    });
    let ordered_any = weights.iter().any(|w| w.ordered);
    let mut instance = Instance {
        entry: &native.function.name,
        variant: taking.variant,
        rows: taking.rows,
        g,
        p,
        parted: project.parts.is_some(),
        m,
        k,
        n,
        served: if routed.is_some() {
            project.weights.len()
        } else {
            taking.served.iter().copied().max().expect("a variant takes at least one weight")
        },
        // A per-row gathered entry's rows here are the weight's prefix.
        gathered: project.gathered.is_some() && selected.is_none(),
        planes: taking.planes,
        published,
        guard: &project.guard,
        weights,
        terms: BTreeMap::new(),
        // A name a term gives for some variants and a parameter for this one
        // is the parameter here.
        named: projection
            .terms
            .iter()
            .map(|term| term.item.name.name.clone())
            .filter(|name| {
                !projection.params.iter().any(|param| {
                    param.item.name.name == *name
                        && (param.variants.is_empty()
                            || param.variants.iter().any(|named| {
                                named.name == taking.variant
                                    || (!taking.cpu
                                        && (named.name == taking.rows.name()
                                            || (named.name == "ordered" && ordered_any)))
                            }))
                })
            })
            .collect(),
        same_file: *projection_file == file,
        at: project.span,
        routed,
        constants,
        restated: &restated,
        lifted: BTreeSet::new(),
    };
    // A projection that never reads `PLANES` gates none of its forms for a
    // plane set.
    if taking.planes && !reads_name(projection, "PLANES") {
        return Err(error(
            project.span,
            format!(
                "projection `{}` takes a plane set but never reads `PLANES`: none of its forms is gated for one",
                projection.name.name
            ),
        ));
    }
    let lifted = lifted_parameters(projection, &instance, &restated)?;
    // The rows the entry's contract binds it at: a launch or buffer whose
    // `when` needs more rows is never emitted.
    let rows = row_bound(&signature.predicates, &instance.m, constants);
    for term in &projection.terms {
        if !instance.takes(&term.variants) {
            continue;
        }
        if instance.formal(&term.item.name.name) {
            return Err(error(instance.span(term.item.name.span), format!("`{}` is a projection formal", term.item.name.name)));
        }
        let value = instance.expr(&term.item.value)?;
        instance.terms.insert(term.item.name.name.clone(), value);
    }

    let emitted = emitted_buffers(&project, projection, &instance)?;
    if taking.emitting {
        // A producer alone takes the emitted buffers and nothing else.
        if !restated.is_empty() {
            return Err(error(project.span, "an `emits` line takes no parameter of its projection to restate"));
        }
        place_at_line(&mut native.scratch, project.scratch_before, emitted);
        if let Some(before) = native.launches.iter().find_map(|launch| launch.before.as_ref()) {
            return Err(error(before.span, "`before` places a launch among a projection's; an `emits` line has none"));
        }
        return Ok(());
    }

    let mut projection_params =
        parameters(&mut native.params, &mut native.constraint, projection, &mut instance, lifted, gate.as_ref())?;
    let scratch = buffers(&project, projection, &taking, &instance, rows, gate.as_ref())?;
    let (forms_any, clause_reads) = form_operands(
        &mut native.scratch,
        &mut native.formation,
        &project,
        projection,
        signature,
        &instance,
        scratch,
        emitted,
    )?;
    let dropped =
        launches(&mut native.launches, &native.scratch, projection, &instance, forms_any, rows, gate.as_ref())?;
    // A parameter no launch or buffer of the projection reads is read by the
    // entry's source alone (a CPU form's accessor): it is always kept.
    let declared = item_reads(projection);
    projection_params.retain(|name| declared.contains(name));
    let (m, n) = (instance.m.clone(), instance.n.clone());
    fix_unread(native, &projection_params, &dropped, clause_reads);

    // A per-row gathered entry's selected rows: the routed expansion of its
    // weights, under the other value of `I`, named apart.
    let (Some((_, choices)), Some(gate)) = (selected, gate) else {
        return Ok(());
    };
    expand_selected(native, &project, file, projections, signatures, constants, choices, gate, gathered_restated, &gathers, m, n)
}

/// What a `project` line takes its projection by: the variant and kind of
/// rows it names and what they fix.
struct Taking {
    cpu: bool,
    variant: &'static str,
    rows: RowsKind,
    /// The variant is a routed projection's.
    routed: bool,
    segmented: bool,
    /// A producer's line alone (`project NAME emits (..)`).
    emitting: bool,
    /// The line's weight is a plane set.
    planes: bool,
    /// The weights each segment serves.
    served: Vec<usize>,
}

/// What `project` takes of its projection, from the line alone. Invariant:
/// the variant is one of the declaration's target, the line's clauses are
/// those the variant takes (`served`, `shared`, `down`, `groups`, `parts`,
/// `gathered`, `rows`, `over`, operands, a plane set) and its weights are as
/// many as the variant projects by, so no later phase meets a clause its
/// variant has no meaning for.
fn taking(native: &NativeDecl, project: &NativeProjectDecl) -> Result<Taking, Diagnostic> {
    let cpu = native.target.name == "cpu";
    let routed_variant = (!cpu).then(|| ROUTED_VARIANTS.iter().find(|(name, _)| *name == project.variant.name)).flatten();
    let routed = routed_variant.is_some();
    let found = if cpu {
        CPU_VARIANTS.iter().find(|name| **name == project.variant.name).map(|name| (*name, 0))
    } else {
        VARIANTS.iter().chain(&ROUTED_VARIANTS).find(|(name, _)| *name == project.variant.name).copied()
    };
    let Some((variant, count)) = found else {
        return Err(error(
            project.variant.span,
            if cpu {
                format!("`{}` is not a CPU projection variant (`tiled`, `rows`, `emits`)", project.variant.name)
            } else {
                format!("`{}` is not a projection variant (`paired`, `single`, `segmented`, `choice`, `summed`, `grouped`)", project.variant.name)
            },
        ));
    };
    let rows = match &project.rows {
        None => RowsKind::Plain,
        Some(kind) => match RowsKind::named(&kind.name) {
            Some(rows) => rows,
            None => {
                return Err(error(kind.span, format!("`{}` is not a kind of rows (`plain`, `rms`, `staged`)", kind.name)));
            }
        },
    };
    let segmented = variant == "segmented";
    // A producer's line alone (`project NAME emits (..)`).
    let emitting = variant == "emits";
    if !cpu && (!project.operands.is_empty() || !project.emits.is_empty()) {
        return Err(error(project.span, "only a CPU projection holds operands its producers emit (`operand`, `emits`)"));
    }
    if emitting && project.emits.is_empty() {
        return Err(error(project.span, "an `emits` line names the results it emits the operands of"));
    }
    if project.publishes.is_some() && (cpu || routed || emitting || !matches!(rows, RowsKind::Rms | RowsKind::Staged)) {
        return Err(error(
            project.span,
            "only a GPU projection of `rms` or `staged` rows publishes them (`publishes (result N)`)",
        ));
    }
    if !project.ordered.is_empty() && (cpu || routed || emitting || project.weight_planes.iter().any(|planes| !planes.is_empty())) {
        return Err(error(
            project.span,
            "only the weights of a GPU projection are `ordered` (the CPU's sums always are)",
        ));
    }
    if let Some(name) = project.ordered.iter().find(|name| !project.weights.iter().any(|weight| weight.name == name.name)) {
        return Err(error(name.span, format!("`{}` is not a weight of the line", name.name)));
    }
    if !project.guard.is_empty() && (cpu || emitting) {
        return Err(error(project.span, "only a GPU projection's launches take `if` and `unless` lines"));
    }
    if !project.served.is_empty() && !segmented {
        return Err(error(project.span, "only a `segmented` projection groups its weights into served pairs"));
    }
    if !project.shared.is_empty() && !matches!(variant, "choice" | "summed") {
        return Err(error(project.span, "only a routed `choice` or `summed` projection projects a shared expert"));
    }
    if project.down.is_some() != (variant == "grouped") {
        return Err(error(project.span, "a routed `grouped` projection, and only it, names its `down` weight"));
    }
    if routed && (project.groups.is_some() || project.gathered.is_some() || project.rows.is_some() || project.over.is_some()) {
        return Err(error(project.span, "a routed projection's rows are its entry's: it takes no `groups`, `gathered`, `rows` or `over`"));
    }
    if project.parts.is_some() && (routed || cpu) {
        return Err(error(project.span, "only a GPU projection of its entry's rows splits its columns into `parts`"));
    }
    // A plane set is a GPU `single` projection's one weight, which gives its
    // rows and width; no other GPU weight gives them.
    let planes = project.weight_planes.iter().any(|planes| !planes.is_empty());
    if planes
        && (cpu || variant != "single" || project.groups.is_some() || project.parts.is_some() || project.gathered.is_some())
    {
        return Err(error(
            project.span,
            "a plane set (`planes (..)`) is the one weight of a GPU `single` projection, without `groups`, `parts` or `gathered`",
        ));
    }
    if !cpu {
        for ((weight, over), planes) in project.weights.iter().zip(&project.weight_overs).zip(&project.weight_planes) {
            if over.is_some() == planes.is_empty() {
                return Err(error(
                    weight.span,
                    if planes.is_empty() {
                        "only a CPU projection's weights and a plane set give their rows and width with `over`"
                    } else {
                        "a plane set gives its rows and width (`planes (..) over (ROWS, WIDTH)`)"
                    },
                ));
            }
        }
    }
    // The weights each segment serves.
    let served: Vec<usize> = if !segmented {
        vec![project.weights.len()]
    } else if project.served.is_empty() {
        vec![1; project.weights.len()]
    } else {
        project.served.clone()
    };
    if segmented && served.iter().any(|count| !(1..=2).contains(count)) {
        return Err(error(project.span, "a segment serves one weight or a pair"));
    }
    if project.weights.is_empty() && !emitting {
        return Err(error(project.span, "a projection takes at least one weight"));
    }
    if (segmented && !(1..=count).contains(&served.len()))
        || (routed && !(1..=count).contains(&project.weights.len()))
        || (!segmented && !routed && !cpu && project.weights.len() != count)
    {
        return Err(error(
            project.span,
            if segmented {
                format!("a `segmented` projection takes 1 to {count} segments, not {}", served.len())
            } else if routed {
                format!("a routed `{variant}` projection takes 1 to {count} weight(s), not {}", project.weights.len())
            } else {
                format!("a `{variant}` projection takes {count} weight(s), not {}", project.weights.len())
            },
        ));
    }
    if project.shared.len() > project.weights.len() {
        return Err(error(project.span, "the shared expert takes at most the experts' weights"));
    }
    if native.repeat.is_some() {
        return Err(error(project.span, "a projecting declaration has no `repeat` block"));
    }
    Ok(Taking { cpu, variant, rows, routed, segmented, emitting, planes, served })
}

/// A shaped type's dimensions and element.
fn dims(ty: &TypeExpr) -> Option<(Vec<Expr>, Ident)> {
    let TypeKind::Shaped { shape, elem, .. } = &ty.kind else {
        return None;
    };
    Some((shape.clone(), elem.clone()))
}

/// The port of `signature` named `name`.
fn port<'s>(signature: &'s Signature, name: &str) -> Option<&'s Param> {
    signature.params.iter().find(|param| param.name.name == name)
}

/// The projection's formals as the entry's signature binds them.
struct Formals {
    weights: Vec<Weight>,
    /// The weights' shared rows; `None` where each has its own.
    n: Option<Expr>,
    k: Expr,
    m: Expr,
    /// A per-row gathered port's rows and choices (`tensor[M, I, N]`'s `M`
    /// and `I`).
    selected: Option<(Expr, Expr)>,
}

/// The formals and weights of the line, read from the entry's signature.
/// Invariant: every weight is a port (or, on the CPU, an element) of the
/// function with its stem, bound port, rows and width; the weights that
/// share a shape agree in it (`N` and `K`, a served pair's `N`, a
/// segmented projection's `K`); and `M`, `K` and `N` are expressions of the
/// signature, never restated.
fn bind_formals(
    native: &NativeDecl,
    project: &NativeProjectDecl,
    signature: &Signature,
    taking: &Taking,
    per_row: Option<&PerRow>,
) -> Result<Formals, Diagnostic> {
    let (cpu, routed, segmented, emitting) = (taking.cpu, taking.routed, taking.segmented, taking.emitting);
    let served = &taking.served;
    let mut weights = Vec::new();
    let mut shape: Option<(Expr, Expr)> = None;
    // A per-row gathered port's rows and choices (`tensor[M, I, N]`'s `M`
    // and `I`).
    let mut selected: Option<(Expr, Expr)> = None;
    let segment_of: Vec<usize> =
        served.iter().enumerate().flat_map(|(segment, count)| std::iter::repeat(segment).take(*count)).collect();
    // An entry whose weights share no shape gives the activation rows and
    // width instead; its weights name element parameters.
    let formals = cpu;
    // A routed projection's experts' weights are its first segment, the
    // shared expert's or the down weight its second.
    let listed: Vec<(&Ident, usize)> = if routed {
        let second = project.shared.iter().chain(project.down.iter());
        project.weights.iter().map(|weight| (weight, 0)).chain(second.map(|weight| (weight, 1))).collect()
    } else {
        project.weights.iter().zip(segment_of.iter().copied()).collect()
    };
    for (ordinal, (weight, segment)) in listed.iter().copied().enumerate().filter(|_| !formals) {
        let segment = &segment;
        // A plane set's rows and width are its `over`'s, each plane a port.
        if let (Some(planes), Some(Some(over))) = (
            project.weight_planes.get(ordinal).filter(|planes| !planes.is_empty()),
            project.weight_overs.get(ordinal),
        ) {
            if let Some(plane) = planes.iter().find(|plane| port(signature, &plane.name).is_none()) {
                return Err(error(plane.span, format!("`{}` is not a port of `{}`", plane.name, native.function.name)));
            }
            shape = Some((over.rows.clone(), over.width.clone()));
            weights.push(Weight {
                element: weight.clone(),
                prefix: String::new(),
                bound: None,
                n: over.rows.clone(),
                width: over.width.clone(),
                stacked: false,
                segment: *segment,
                over: None,
                emitted: false,
                planes: true,
                ordered: false,
            });
            continue;
        }
        let Some(param) = port(signature, &weight.name) else {
            return Err(error(weight.span, format!("`{}` is not a port of `{}`", weight.name, native.function.name)));
        };
        // `weight`'s stem is empty and `<stem>_weight`'s is `<stem>`; any
        // other port is its own stem.
        let prefix = match weight.name.strip_suffix("weight").filter(|prefix| prefix.is_empty() || prefix.ends_with('_')) {
            Some(prefix) => prefix.to_owned(),
            None => format!("{}_", weight.name),
        };
        let prefix = prefix.as_str();
        // A routed projection's weight may stack one matrix an expert.
        // A segment of a segmented projection may be one row, a `tensor[K]`.
        let ranks: &[usize] = if routed {
            &[2, 3]
        } else if segmented {
            &[1, 2]
        } else {
            &[2]
        };
        let Some((dims_of, element)) = dims(&param.ty).filter(|(dims_of, _)| ranks.contains(&dims_of.len())) else {
            return Err(error(
                weight.span,
                format!(
                    "weight `{}` is not a `tensor[N, K]`{}",
                    weight.name,
                    if routed {
                        " or `tensor[E, N, K]`"
                    } else if segmented {
                        " or `tensor[K]`"
                    } else {
                        ""
                    }
                ),
            ));
        };
        let stacked = dims_of.len() == 3;
        let (n, k) = match dims_of.as_slice() {
            [k] => (Expr { kind: ExprKind::Int(1), span: weight.span }, k.clone()),
            [.., n, k] => (n.clone(), k.clone()),
            [] => unreachable!("a weight has a rank from one to three"),
        };
        // A weight over `P` parts of its columns is a `tensor[N, P * X]`.
        if let Some(parts) = &project.parts {
            if !matches!(&k.kind, ExprKind::Binary { op: BinaryOp::Mul, lhs, .. }
                if matches!(&lhs.kind, ExprKind::Name(name) if name.name == parts.name))
            {
                return Err(error(
                    weight.span,
                    format!("a weight over `{}` parts is a `tensor[N, {} * X]`", parts.name, parts.name),
                ));
            }
        }
        let width = k.clone();
        // A gathered weight's rows are the extent of the port that selects
        // them; per activation row (`tensor[M, I, N]`), its last.
        let n = match (&project.gathered, per_row) {
            (_, Some(per_row)) => per_row.n.clone(),
            (Some(selects), None) => match port(signature, &selects.name).and_then(|selects| dims(&selects.ty)) {
                Some((extent, _)) if extent.len() == 1 => extent[0].clone(),
                Some((extent, _)) if extent.len() == 3
                    && matches!(&extent[1].kind, ExprKind::Name(choices) if native.statics.iter().any(|name| name.name == choices.name)) =>
                {
                    selected = Some((extent[0].clone(), extent[1].clone()));
                    extent[2].clone()
                }
                _ => {
                    return Err(error(
                        selects.span,
                        format!(
                            "`{}` is not a `tensor[N]` port of `{}`, or a `tensor[M, I, N]` one of a static `I`",
                            selects.name, native.function.name
                        ),
                    ));
                }
            },
            (None, None) => n,
        };
        // A grouped weight's rows are `G * N`: N each group's.
        let n = match &project.groups {
            None => n,
            Some(groups) => match &n.kind {
                ExprKind::Binary { op: BinaryOp::Mul, lhs, rhs }
                    if matches!(&lhs.kind, ExprKind::Name(name) if name.name == groups.name) =>
                {
                    (**rhs).clone()
                }
                _ => {
                    return Err(error(
                        weight.span,
                        format!("a weight over `{}` groups is a `tensor[{} * N, K]`", groups.name, groups.name),
                    ));
                }
            },
        };
        if let (true, Some((first_n, first_k))) = (!routed || *segment == 0, &shape) {
            if structure(first_k) != structure(&k) || (!segmented && structure(first_n) != structure(&n)) {
                return Err(error(
                    weight.span,
                    if segmented { "the segments of a projection share `K`" } else { "the weights of a projection share `N` and `K`" },
                ));
            }
        } else {
            shape = Some((n.clone(), k));
        }
        // `weight`'s bound port is `weight_bound`; `<stem>_weight`'s `<stem>_bound`.
        let bound_port = if prefix.is_empty() { "weight_bound".to_owned() } else { format!("{prefix}bound") };
        let bound = match port(signature, &bound_port) {
            Some(bound) => match dims(&bound.ty) {
                Some((extent, _)) => match extent.as_slice() {
                    [Expr { kind: ExprKind::Name(extent), .. }] => Some(extent.clone()),
                    _ => return Err(error(weight.span, format!("`{bound_port}` is not a `tensor[B]`"))),
                },
                _ => return Err(error(weight.span, format!("`{bound_port}` is not a `tensor[B]`"))),
            },
            None => None,
        };
        // The weights of a served pair share their rows.
        if let Some(first) = weights.iter().rev().find(|other: &&Weight| other.segment == *segment).filter(|_| !routed) {
            if structure(&first.n) != structure(&n) {
                return Err(error(weight.span, "the weights of a served pair share `N`"));
            }
        }
        weights.push(Weight {
            element,
            prefix: prefix.to_owned(),
            bound,
            n,
            width,
            stacked,
            segment: *segment,
            over: None,
            emitted: false,
            planes: false,
            ordered: false,
        });
    }
    let (n, k, m) = if formals {
        if !cpu {
            return Err(error(project.span, "only a CPU projection gives its rows and width with `over`"));
        }
        for (weight, over) in project.weights.iter().zip(&project.weight_overs) {
            let element = signature.params.iter().find_map(|param| match &param.ty.kind {
                TypeKind::Shaped { elem, .. } if elem.name == weight.name => Some(elem.clone()),
                _ => None,
            });
            let Some(element) = element else {
                return Err(error(
                    weight.span,
                    format!("`{}` is the element of no port of `{}`", weight.name, native.function.name),
                ));
            };
            weights.push(Weight {
                element,
                prefix: format!("{}_", weight.name.to_lowercase()),
                bound: None,
                n: Expr { kind: ExprKind::Int(1), span: weight.span },
                width: Expr { kind: ExprKind::Int(1), span: weight.span },
                stacked: false,
                segment: 0,
                over: over.clone(),
                emitted: false,
                planes: false,
                ordered: false,
            });
        }
        // Every weight of a projection that quantizes multiplies the entry's
        // rows or its own; an `emits` line projects none.
        if !emitting && project.over.is_none() {
            if let Some((weight, _)) = project.weights.iter().zip(&project.weight_overs).find(|(_, over)| over.is_none()) {
                return Err(error(
                    weight.span,
                    format!("`{}` multiplies no rows: give them after the weights or on it (`{} over (ROWS, WIDTH)`)", weight.name, weight.name),
                ));
            }
        }
        // The entry's rows and width; none where the line gives none (every
        // weight multiplies rows of its own, read through it).
        match project.over.as_ref() {
            Some(over) => (None, over.width.clone(), over.rows.clone()),
            None => {
                let none = Expr { kind: ExprKind::Int(0), span: project.span };
                (None, none.clone(), none)
            }
        }
    } else if routed {
        // The routed entries' rows are `M`, by name; a per-row gathered
        // entry's are its own.
        let (n, k) = shape.expect("a variant takes at least one weight");
        let m = match per_row {
            Some(per_row) => per_row.m.clone(),
            None => Expr { kind: ExprKind::Name(Ident { name: "M".to_owned(), span: project.span }), span: project.span },
        };
        (Some(n), k, m)
    } else {
        let (n, k) = shape.expect("a variant takes at least one weight");
        // A segmented projection's `N` is each weight's own.
        let n = (!segmented).then_some(n);
        // The rows of the result, or of a tuple's first result.
        let first = |result: &TypeExpr| match &result.kind {
            TypeKind::Tuple(results) => results.first().and_then(|first| dims(first)),
            _ => dims(result),
        };
        let rows_tensor = match &signature.result {
            Some(result) => first(result),
            None => signature.params.first().and_then(|param| dims(&param.ty)),
        };
        let Some(m) = rows_tensor.and_then(|(result, _)| result.first().cloned()) else {
            return Err(error(project.span, "a projecting entry returns its `M` projected rows' tensor, or takes it first when returning nothing"));
        };
        (n, k, m)
    };
    Ok(Formals { weights, n, k, m, selected })
}

/// The value of `I` the expansion's items are in force at, for a per-row
/// gathered entry (`selected`: its port's rows and choices) or its routed
/// expansion (`per_row`); `None` for every other entry. Invariant: a per-row
/// gathered projection is `paired` or `single`, without `groups` or `parts`,
/// over the entry's own rows.
fn selection_gate(
    project: &NativeProjectDecl,
    segmented: bool,
    selected: &Option<(Expr, Expr)>,
    m: &Expr,
    per_row: Option<&PerRow>,
) -> Result<Option<Gate>, Diagnostic> {
    // A per-row gathered entry selects rows for each of its own, of one
    // weight list: this expansion is its `I == 0` one, and the routed one
    // its `I > 0` one.
    Ok(match (selected, per_row) {
        (_, Some(per_row)) => Some(per_row.gate.clone()),
        (Some((rows, choices)), None) => {
            if structure(rows) != structure(m) || segmented || project.groups.is_some() || project.parts.is_some() {
                return Err(error(
                    project.span,
                    "a per-row gathered projection (`tensor[M, I, N]`) is `paired` or `single`, without `groups` or `parts`, its port's `M` the entry's rows",
                ));
            }
            let zero = Expr { kind: ExprKind::Int(0), span: project.span };
            Some(Gate {
                on: binary(BinaryOp::Eq, choices.clone(), zero.clone()),
                off: binary(BinaryOp::Gt, choices.clone(), zero),
            })
        }
        (None, None) => None,
    })
}

/// Whether the line's formed rows are an entry result (`publishes (result
/// N)`). Invariant: that result is a `tensor[M, K]` of the line's rows and
/// width, so the launch that forms the rows stores them in it whole.
fn published_rows(project: &NativeProjectDecl, signature: &Signature, m: &Expr, k: &Expr) -> Result<bool, Diagnostic> {
    let Some(publishes) = &project.publishes else {
        return Ok(false);
    };
    let result = signature.result.as_ref().and_then(|result| match &result.kind {
        TypeKind::Tuple(items) => items.get(publishes.result as usize),
        _ => (publishes.result == 0).then_some(result),
    });
    match result.and_then(dims) {
        Some((extent, _)) if extent.len() == 2 && structure(&extent[0]) == structure(m) && structure(&extent[1]) == structure(k) => {
            Ok(true)
        }
        _ => Err(error(
            publishes.span,
            format!("`result {}` is not a `tensor[M, K]` of the line's rows and width", publishes.result),
        )),
    }
}

/// An item of the expansion is one where its gate holds (a condition holds
/// where the other's does).
fn gated(gate: Option<&Gate>, when: Option<Expr>) -> Option<Expr> {
    match (gate, when) {
        (None, when) => when,
        (Some(gate), None) => Some(gate.on.clone()),
        (Some(gate), Some(when)) => Some(fold(binary(BinaryOp::And, gate.on.clone(), when))),
    }
}

/// Whether a launch or buffer of condition `when` is no part of an entry of
/// rows `m` bound at `rows`: it needs more rows than the contract admits, or
/// the entry's facts decide it false (a routed projection's `SHARED`,
/// `STACKED`, ..).
fn beyond(when: &Option<Expr>, instance: &Instance, rows: Option<u64>) -> bool {
    when.as_ref().is_some_and(|when| needs_more_rows(when, &instance.m, rows, instance.constants) || matches!(when.kind, ExprKind::Bool(false)))
}

/// The launch parameters the entry restates entry-wide, each with the
/// projection's declaration of it. Invariant: every restate names a
/// parameter the projection declares for the variant, entry-wide or on the
/// launch the restate names.
fn lifted_parameters<'r>(
    projection: &ProjectionDecl,
    instance: &Instance,
    restated: &'r [NativeRestateDecl],
) -> Result<Vec<(&'r NativeRestateDecl, NativeParamDecl)>, Diagnostic> {
    let variant = instance.variant;
    // The first declaration of launch parameter `name` among the launches
    // the entry takes.
    let launch_param = |instance: &Instance, name: &str| {
        projection
            .launches
            .iter()
            .filter(|candidate| instance.takes(&candidate.variants))
            .flat_map(|candidate| candidate.launch.params.iter().zip(&candidate.param_variants))
            .find(|(param, variant)| param.name.name == name && instance.takes(variant))
            .map(|(param, _)| param.clone())
    };
    let mut lifted = Vec::new();
    for restated in restated {
        let name = &restated.param.name.name;
        let known = match &restated.launch {
            None => {
                projection.params.iter().any(|param| param.item.name.name == *name && instance.takes(&param.variants))
                    || launch_param(instance, name).is_some_and(|declared| {
                        lifted.push((restated, declared));
                        true
                    })
            }
            Some(launch) => projection.launches.iter().any(|candidate| {
                instance.takes(&candidate.variants)
                    && candidate.launch.kernel.name == launch.name
                    && candidate.launch.params.iter().zip(&candidate.param_variants).any(|(param, variant)| {
                        param.name.name == *name && instance.takes(variant)
                    })
            }),
        };
        if !known {
            let (span, place) = match &restated.launch {
                None => (restated.param.name.span, format!("parameter `{name}`")),
                Some(launch) => {
                    (launch.span, format!("`{variant}` launch `{}` with a parameter `{name}`", launch.name))
                }
            };
            return Err(error(span, format!("projection `{}` has no {place}", projection.name.name)));
        }
    }
    Ok(lifted)
}

/// The buffers each emitted result's operand takes (`emits (..)`).
/// Invariant: every emitted result has exactly one `for emits` buffer of each
/// form, unconditioned.
fn emitted_buffers(
    project: &NativeProjectDecl,
    projection: &ProjectionDecl,
    instance: &Instance,
) -> Result<Vec<NativeScratchDecl>, Diagnostic> {
    // The buffers each emitted result's operand takes: the projection's
    // `for emits` buffers over its rows and width, named for the result
    // (`result_N_<name>`), each serving it (`emission`), its forms among
    // them its `Emitted` ones: the Q8 blocks and the formed rows, one buffer
    // each, under the norm the line names where it names one.
    let mut emitted = Vec::new();
    for emit in &project.emits {
        let result = Weight {
            element: Ident { name: format!("result_{}", emit.result), span: emit.span },
            prefix: format!("result_{}_", emit.result),
            bound: None,
            n: Expr { kind: ExprKind::Int(1), span: emit.span },
            width: Expr { kind: ExprKind::Int(1), span: emit.span },
            stacked: false,
            segment: 0,
            over: Some(emit.over.clone()),
            emitted: false,
            planes: false,
            ordered: false,
        };
        let mut forms = [false; 2];
        for item in &projection.scratch {
            let ProjectionScratchDecl::One(one) = item else { continue };
            if !one.variants.iter().any(|variant| variant.name == "emits") {
                continue;
            }
            let mut buffer = instance.scratch(&one.item, Some(&result))?;
            buffer.emission = Some(emit.result);
            buffer.operand = match buffer.operand {
                None => None,
                Some(ScratchOperandDecl::Declared { rows, when: None, formed: None, nvfp4: None })
                    if !forms[usize::from(rows)] =>
                {
                    forms[usize::from(rows)] = true;
                    Some(ScratchOperandDecl::Emitted { rows, norm: emit.norm.clone() })
                }
                Some(_) => {
                    return Err(error(
                        instance.span(one.item.span),
                        "an emitted operand's form is one `for emits` buffer marked `operand` (its Q8 blocks) or `operand rows`, with no condition: its consumer's forms decide what it writes",
                    ));
                }
            };
            emitted.push(buffer);
        }
        if forms != [true; 2] {
            return Err(error(
                emit.span,
                format!(
                    "projection `{}` has no `for emits` buffer of an operand's {}",
                    projection.name.name,
                    if forms[0] { "formed rows (`operand rows`)" } else { "Q8 blocks (`operand`)" }
                ),
            ));
        }
    }
    Ok(emitted)
}

/// The projection's parameters and conditions, added to the entry's
/// (`entry_params`, `entry_constraint`); returns the names of the parameters
/// added. Invariant: each parameter is the projection's for the variant, in
/// the entry's restated range where it restates one and never a form's; a
/// condition on one parameter alone is its range ([`admit`]), and the rest
/// hold wherever the gate's other value is in force.
fn parameters(
    entry_params: &mut Vec<NativeParamDecl>,
    entry_constraint: &mut Option<Expr>,
    projection: &ProjectionDecl,
    instance: &mut Instance,
    lifted: Vec<(&NativeRestateDecl, NativeParamDecl)>,
    gate: Option<&Gate>,
) -> Result<BTreeSet<String>, Diagnostic> {
    let restated = instance.restated;
    let mut params = std::mem::take(entry_params);
    let mut projection_params = projection
        .params
        .iter()
        .filter(|param| instance.takes(&param.variants))
        .map(|param| param.item.name.name.clone())
        .collect::<BTreeSet<_>>();
    for param in &projection.params {
        if instance.takes(&param.variants) {
            let theirs = instance.param(&param.item)?;
            let restated = restated
                .iter()
                .find(|restated| restated.launch.is_none() && restated.param.name.name == param.item.name.name);
            params.push(match restated {
                Some(restated) => restate(&restated.param, theirs)?,
                None => theirs,
            });
        }
    }
    for (restated, declared) in lifted {
        let theirs = instance.param(&declared)?;
        params.push(restate(&restated.param, theirs)?);
        projection_params.insert(declared.name.name.clone());
        instance.lifted.insert(declared.name.name);
    }

    let mut conjuncts = Vec::new();
    for constraint in &projection.constraints {
        if instance.takes(&constraint.variants) {
            flatten(instance.expr(&constraint.item)?, BinaryOp::And, &mut conjuncts);
        }
    }
    // The admission's conditions on one parameter alone are its range.
    let admitted = params.iter_mut().filter(|param| projection_params.contains(&param.name.name)).collect();
    let conjuncts = admit(admitted, conjuncts)?;
    *entry_params = params;
    for conjunct in conjuncts {
        let conjunct = match gate {
            Some(gate) => fold(binary(BinaryOp::Or, gate.off.clone(), conjunct)),
            None => conjunct,
        };
        *entry_constraint = Some(match entry_constraint.take() {
            None => conjunct,
            Some(constraint) => binary(BinaryOp::And, constraint, conjunct),
        });
    }
    Ok(projection_params)
}

/// The projection's buffers the entry takes, each with the weight it is one
/// of (an `each W:` buffer). Invariant: a buffer is taken only where the
/// variant, the entry's rows and its facts reach it, under the gate.
fn buffers(
    project: &NativeProjectDecl,
    projection: &ProjectionDecl,
    taking: &Taking,
    instance: &Instance,
    rows: Option<u64>,
    gate: Option<&Gate>,
) -> Result<Vec<(NativeScratchDecl, Option<usize>)>, Diagnostic> {
    let (cpu, emitting) = (taking.cpu, taking.emitting);
    // Each buffer, with the weight it is one of (an `each W:` buffer). On the
    // CPU the projection's buffers of no weight are of the rows the line gives
    // (or, an `emits` line's, of the results it emits), and an `each W:`
    // block is of the weights with rows of their own.
    let shared_rows = !cpu || project.over.is_some() || emitting;
    let mut scratch = Vec::new();
    for item in &projection.scratch {
        match item {
            ProjectionScratchDecl::One(one) => {
                if shared_rows && instance.takes(&one.variants) {
                    let mut one = instance.scratch(&one.item, None)?;
                    if !beyond(&one.when, instance, rows) {
                        one.when = gated(gate, one.when);
                        scratch.push((one, None));
                    }
                }
            }
            ProjectionScratchDecl::Each(each) => {
                for (index, weight) in instance.weights.iter().enumerate().filter(|(_, weight)| !cpu || weight.over.is_some()) {
                    for one in each {
                        if instance.takes(&one.variants) {
                            let mut one = instance.scratch(&one.item, Some(weight))?;
                            if !beyond(&one.when, instance, rows) {
                                one.when = gated(gate, one.when);
                                scratch.push((one, Some(index)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(scratch)
}

/// The operands the line names, bound to the projection's operand buffers
/// (`scratch`), and the formation that forms them; the buffers, with the
/// `emitted` ones, placed at the line (`entry_scratch`, `entry_formation`).
/// Returns whether the entry forms any operand, and the parameters the kept
/// buffers' operand clauses read. Invariant: the k-th buffer of each form
/// holds the k-th operand the line names, every formed operand has both
/// forms, and its sources are tensor ports of the entry.
#[allow(clippy::too_many_arguments)]
fn form_operands(
    entry_scratch: &mut Vec<NativeScratchDecl>,
    entry_formation: &mut Option<NativeFormationDecl>,
    project: &NativeProjectDecl,
    projection: &ProjectionDecl,
    signature: &Signature,
    instance: &Instance,
    scratch: Vec<(NativeScratchDecl, Option<usize>)>,
    emitted: Vec<NativeScratchDecl>,
) -> Result<(bool, BTreeSet<String>), Diagnostic> {
    let variant = instance.variant;
    // The operands the line names, in order, are the projection's: the k-th
    // buffer of each form (`operand`, the Q8 blocks; `operand rows`, the
    // formed rows) holds the k-th operand's. The projection's `form` launch
    // (`<entry>_form`, its body the build's, from the declaration's
    // formation) forms a parameter's rows and an entry function's, unless,
    // for a parameter's, a producer's form stands for it: where the buffer's
    // `when` holds (absent: always), of the parameter's own rows (no `at`),
    // and under a norm that is bound (its extent past 0). The launch writes a
    // form where its `formed` holds (absent: always), the rows as NVFP4
    // operands where `nvfp4` holds (absent: never). An `emitted` operand's
    // buffers are the entry's launch's to write (`projection::emit`), and an
    // entry function's never pair. An entry that names none takes the Q8
    // blocks alone, which its own launches form.
    let function = instance.entry;
    let formation = Ident { name: format!("{}_{FORM}", instance.entry), span: project.span };
    let tensor_port = |name: &Ident| -> Result<Vec<Expr>, Diagnostic> {
        match port(signature, &name.name).and_then(|port| dims(&port.ty)) {
            Some((extents, _)) => Ok(extents),
            None => Err(error(name.span, format!("`{}` is not a tensor port of `{function}`", name.name))),
        }
    };
    let one_row = |name: &Ident, what: &str| -> Result<(), Diagnostic> {
        match tensor_port(name)?.len() {
            1 => Ok(()),
            _ => Err(error(name.span, format!("`{}` is not a `{what}` port of `{function}`", name.name))),
        }
    };
    for operand in &project.operands {
        if let OperandSourceDecl::Parameter { parameter, at, within, .. } = &operand.source {
            tensor_port(parameter)?;
            if let Some(at) = at {
                one_row(at, "tensor[ROWS]")?;
            }
            for bound in within.iter().flatten() {
                one_row(bound, "tensor[1]")?;
            }
        }
    }
    let forms_any = project.operands.iter().any(|operand| !matches!(operand.source, OperandSourceDecl::Emitted(_)));
    if forms_any
        && !projection.launches.iter().any(|launch| launch.launch.kernel.name == FORM && instance.takes(&launch.variants))
    {
        return Err(error(project.span, format!("projection `{}` has no `{FORM}` launch to form operands", projection.name.name)));
    }
    let holds = |value: bool, span: Span| Expr { kind: ExprKind::Bool(value), span };
    let pairing = |at: &Option<Ident>, norm: &Option<OperandNormDecl>, declared: Option<Expr>, span: Span| -> Result<Expr, Diagnostic> {
        if at.is_some() {
            return Ok(holds(false, span));
        }
        let declared = declared.unwrap_or_else(|| holds(true, span));
        Ok(match norm {
            None => declared,
            Some(norm) => {
                let [extent] = tensor_port(&norm.weight)?.try_into().map_err(|_| {
                    error(norm.weight.span, format!("`{}` is not a one-row norm, `tensor[N]`", norm.weight.name))
                })?;
                let bound = binary(BinaryOp::Gt, extent, Expr { kind: ExprKind::Int(0), span });
                fold(binary(BinaryOp::And, declared, bound))
            }
        })
    };
    // The k-th operand's buffers ([Q8 blocks, rows]), the weight they are
    // of where they are one weight's, and where its Q8 blocks are written.
    let mut forms: Vec<([Option<Ident>; 2], Option<usize>, Option<Expr>)> = vec![([None, None], None, None); project.operands.len()];
    let mut counts = [0usize; 2];
    let mut kept = Vec::new();
    // The parameters the kept buffers' operand clauses read, whatever the
    // entry makes of the buffers (`fix_unread`).
    let mut clause_reads = BTreeSet::new();
    for (mut buffer, owner) in scratch {
        let Some(ScratchOperandDecl::Declared { rows, when, formed: writes, nvfp4 }) = buffer.operand.take() else {
            kept.push(buffer);
            continue;
        };
        if project.operands.is_empty() && rows {
            // The formed rows are the formation's alone.
            continue;
        }
        for clause in [&when, &writes, &nvfp4].into_iter().flatten() {
            names(clause, &mut clause_reads);
        }
        if project.operands.is_empty() {
            kept.push(buffer);
            continue;
        }
        if !rows && nvfp4.is_some() {
            return Err(error(buffer.span, "Q8 blocks take no NVFP4 transform; an operand's rows do (`operand rows .. nvfp4 (..)`)"));
        }
        let index = counts[usize::from(rows)];
        counts[usize::from(rows)] += 1;
        let Some(operand) = project.operands.get(index) else {
            return Err(error(
                project.span,
                format!("the projection holds more operands than the line names: `{}` is not named", buffer.name.name),
            ));
        };
        let writes = writes.unwrap_or_else(|| holds(true, buffer.span));
        forms[index].0[usize::from(rows)] = Some(buffer.name.clone());
        forms[index].1 = owner;
        if !rows {
            forms[index].2 = Some(writes.clone());
        }
        // Only a parameter's operand has a producer to pair with.
        if let OperandSourceDecl::Parameter { parameter, at, norm, .. } = &operand.source {
            buffer.operand = Some(ScratchOperandDecl::Formed {
                parameter: parameter.clone(),
                formation: formation.clone(),
                pairs: pairing(at, norm, when, buffer.span)?,
                rows,
                norm: norm.clone(),
                formed: writes,
                nvfp4: nvfp4.unwrap_or_else(|| holds(false, buffer.span)),
            });
        }
        kept.push(buffer);
    }
    let mut scratch = kept;
    if forms_any {
        let port_of = |weight: &Weight| {
            signature
                .params
                .iter()
                .find(|param| matches!(&param.ty.kind, TypeKind::Shaped { elem, .. } if elem.name == weight.element.name))
                .map(|param| param.name.clone())
                .ok_or_else(|| error(project.span, format!("`{}` is the element of no port of `{function}`", weight.element.name)))
        };
        let mut operands = Vec::new();
        for (operand, ([quantized, staged], owner, formed)) in project.operands.iter().zip(forms) {
            let (values, named) = match &operand.source {
                OperandSourceDecl::Parameter { parameter, at, within, .. } => (
                    FormedValuesDecl::Parameter { parameter: parameter.clone(), at: at.clone(), within: within.clone() },
                    format!("`{}`", parameter.name),
                ),
                OperandSourceDecl::Values(function) => {
                    (FormedValuesDecl::Function(function.clone()), format!("`values ({})`", function.name))
                }
                OperandSourceDecl::Emitted(span) => {
                    if operand.tiles.is_some() {
                        return Err(error(*span, "an `emitted` operand's Q8 blocks are its emitter's, never tiles"));
                    }
                    continue;
                }
            };
            let (Some(quantized), Some(staged), Some(formed)) = (quantized, staged, formed) else {
                return Err(error(
                    project.span,
                    format!("projection `{}` holds no operand for {named}: one Q8 buffer and one `operand rows` buffer each", projection.name.name),
                ));
            };
            let weight = owner.map(|owner| &instance.weights[owner]);
            let formal = |name: &str| -> Result<Expr, Diagnostic> {
                Ok(fold(instance.substitute(&Expr { kind: ExprKind::Name(Ident { name: name.to_owned(), span: project.span }), span: project.span }, weight)?))
            };
            // The entry's rows serve every weight without rows of its own.
            let weights = match weight {
                Some(weight) => vec![port_of(weight)?],
                None => instance.weights.iter().filter(|weight| weight.over.is_none()).map(|weight| port_of(weight)).collect::<Result<_, _>>()?,
            };
            operands.push(FormedOperandDecl {
                values,
                rows: formal("M")?,
                width: formal("K")?,
                groups: instance.g.clone(),
                staged,
                quantized,
                formed,
                weights,
                tiles: match (&operand.tiles, variant == "tiled") {
                    // The entry's own condition, over its dimensions.
                    (Some(tiles), true) => tiles.clone(),
                    (None, tiled) => holds(tiled, project.span),
                    (Some(tiles), false) => return Err(error(tiles.span, "only a `tiled` projection's Q8 blocks are tiles")),
                },
            });
        }
        *entry_formation = Some(NativeFormationDecl { launch: formation, operands });
    }
    scratch.extend(emitted);
    place_at_line(entry_scratch, project.scratch_before, scratch);
    Ok((forms_any, clause_reads))
}

/// The projection's launches the entry takes, in the projection's order,
/// with the entry's own (`entry_launches`) placed `before` them or after
/// them all; returns the parameters of the launches past the entry's rows.
/// Invariant: every `before` names a launch the variant has, a launch's
/// parameters are in the range its own `where` leaves ([`admit`]), and a
/// grid is read only from a buffer the entry has (`entry_scratch`).
fn launches(
    entry_launches: &mut Vec<NativeLaunchDecl>,
    entry_scratch: &[NativeScratchDecl],
    projection: &ProjectionDecl,
    instance: &Instance,
    forms_any: bool,
    rows: Option<u64>,
    gate: Option<&Gate>,
) -> Result<Vec<NativeParamDecl>, Diagnostic> {
    let variant = instance.variant;
    // Every buffer name the projection declares (per weight, its stems').
    let projection_buffers: BTreeSet<String> = projection
        .scratch
        .iter()
        .flat_map(|item| match item {
            ProjectionScratchDecl::One(one) => vec![one.item.name.name.clone()],
            ProjectionScratchDecl::Each(each) => instance
                .weights
                .iter()
                .flat_map(|weight| each.iter().map(move |one| format!("{}{}", weight.prefix, one.item.name.name)))
                .collect(),
        })
        .collect();
    let own = std::mem::take(entry_launches);
    // The parameters of the launches past the entry's rows.
    let mut dropped = Vec::new();
    let mut placed = vec![false; own.len()];
    let mut launches = Vec::new();
    for launch in &projection.launches {
        // The formation is an entry's that names an operand it forms.
        if !instance.takes(&launch.variants) || (launch.launch.kernel.name == FORM && !forms_any) {
            continue;
        }
        for (ordinal, entry) in own.iter().enumerate() {
            if entry.before.as_ref().is_some_and(|before| before.name == launch.launch.kernel.name) {
                placed[ordinal] = true;
                launches.push(NativeLaunchDecl {
                    before: None,
                    ..entry.clone()
                });
            }
        }
        let mut launch = instance.launch(launch)?;
        launch.conditions.extend(instance.guard.iter().cloned());
        // The launch's own admission (its `where`) gives its parameters' ranges.
        if let Some(constraint) = launch.constraint.take() {
            let span = constraint.span;
            let mut conjuncts = Vec::new();
            flatten(constraint, BinaryOp::And, &mut conjuncts);
            let conjuncts = admit(launch.params.iter_mut().collect(), conjuncts)?;
            launch.constraint = (!conjuncts.is_empty()).then(|| chain(BinaryOp::And, conjuncts, span));
        }
        // A grid read from a buffer of the projection's that the entry does
        // not have (its `when` decided false) is the launch's declared one.
        if launch.grid.as_ref().is_some_and(|grid| {
            !entry_scratch.iter().any(|buffer| buffer.name.name == grid.name) && projection_buffers.contains(&grid.name)
        }) {
            launch.grid = None;
        }
        if beyond(&launch.when, instance, rows) {
            dropped.extend(launch.params);
        } else {
            launch.when = gated(gate, launch.when);
            launches.push(launch);
        }
    }
    for (ordinal, entry) in own.into_iter().enumerate() {
        match &entry.before {
            Some(before) if !placed[ordinal] => {
                return Err(error(
                    before.span,
                    format!(
                        "projection `{}` has no `{variant}` launch `{}`",
                        projection.name.name, before.name
                    ),
                ));
            }
            Some(_) => {}
            None => launches.push(entry),
        }
    }
    *entry_launches = launches;
    Ok(dropped)
}

/// A per-row gathered entry's selected rows: the routed expansion of its
/// weights (`PER_ROW`), under the other value of `I` (`gate`, the dense
/// expansion's), named apart ([`gather`]) and added to `native`. Invariant:
/// the two expansions share no parameter, launch or buffer name, and each
/// one's conditions hold where the other's value of `I` is in force.
#[allow(clippy::too_many_arguments)]
fn expand_selected(
    native: &mut NativeDecl,
    project: &NativeProjectDecl,
    file: usize,
    projections: &BTreeMap<(String, String), (usize, ProjectionDecl)>,
    signatures: &BTreeMap<String, Vec<Signature>>,
    constants: &BTreeMap<String, u64>,
    choices: Expr,
    gate: Gate,
    gathered_restated: Vec<NativeRestateDecl>,
    gathers: &str,
    m: Expr,
    n: Option<Expr>,
) -> Result<(), Diagnostic> {
    let span = project.span;
    let ident = |name: &str| Ident { name: name.to_owned(), span };
    let line = NativeProjectDecl {
        projection: ident(PER_ROW.0),
        variant: ident(PER_ROW.1),
        weights: project.weights.clone(),
        weight_overs: vec![None; project.weights.len()],
        weight_planes: vec![Vec::new(); project.weights.len()],
        over: None,
        served: Vec::new(),
        shared: Vec::new(),
        down: None,
        groups: None,
        parts: None,
        mappings: Vec::new(),
        gathered: None,
        rows: None,
        publishes: None,
        ordered: Vec::new(),
        guard: project.guard.clone(),
        restated: gathered_restated
            .into_iter()
            .map(|mut restated| {
                restated.param.name.name.drain(..gathers.len());
                restated
            })
            .collect(),
        operands: Vec::new(),
        emits: Vec::new(),
        scratch_before: 0,
        span,
    };
    let selects = PerRow {
        m,
        choices,
        n: n.expect("a per-row gathered projection's weights share their rows"),
        gate: Gate { on: gate.off, off: gate.on },
    };
    let mut selections = NativeDecl {
        params: Vec::new(),
        constraint: None,
        scratch: Vec::new(),
        launches: Vec::new(),
        ..native.clone()
    };
    expand_native(&mut selections, line, file, projections, signatures, constants, Some(&selects))?;
    gather(&native.function.name, &mut selections, &gathers);
    native.params.extend(selections.params);
    let mut conjuncts = Vec::new();
    for constraint in native.constraint.take().into_iter().chain(selections.constraint) {
        flatten(constraint, BinaryOp::And, &mut conjuncts);
    }
    native.constraint = (!conjuncts.is_empty()).then(|| chain(BinaryOp::And, conjuncts, span));
    native.scratch.extend(selections.scratch);
    native.launches.extend(selections.launches);
    Ok(())
}

/// Name a per-row gathered entry's routed expansion (`expanded`, of entry
/// `entry`) apart from its own: each parameter `<parameters>NAME`, each
/// launch `<entry>_gather_NAME` and each buffer `gather_NAME`.
fn gather(entry: &str, expanded: &mut NativeDecl, parameters: &str) {
    let launch_params = expanded.launches.iter().flat_map(|launch| &launch.params);
    let names: BTreeMap<String, String> = expanded
        .params
        .iter()
        .chain(launch_params)
        .map(|param| (param.name.name.clone(), format!("{parameters}{}", param.name.name)))
        .collect();
    let buffers: BTreeMap<String, String> = expanded
        .scratch
        .iter()
        .map(|buffer| (buffer.name.name.clone(), format!("{GATHER}{}", buffer.name.name)))
        .collect();
    let rename_ident = |ident: &mut Ident, map: &BTreeMap<String, String>| {
        if let Some(renamed) = map.get(&ident.name) {
            ident.name = renamed.clone();
        }
    };
    let rename_default = |default: &mut NativeDefaultDecl| match default {
        NativeDefaultDecl::Value(expr) | NativeDefaultDecl::Least(expr) | NativeDefaultDecl::Greatest(expr) => {
            rename(expr, &names)
        }
    };
    let rename_param = |param: &mut NativeParamDecl| {
        rename_ident(&mut param.name, &names);
        rename_default(&mut param.default);
        if let Some(reference) = &mut param.reference {
            rename_default(reference);
        }
    };
    expanded.params.iter_mut().for_each(rename_param);
    expanded.constraint.iter_mut().for_each(|constraint| rename(constraint, &names));
    for buffer in &mut expanded.scratch {
        assert!(buffer.operand.is_none(), "a GPU projection's buffers hold no operand");
        rename_ident(&mut buffer.name, &buffers);
        rename(&mut buffer.bytes, &names);
        buffer.when.iter_mut().for_each(|when| rename(when, &names));
    }
    for launch in &mut expanded.launches {
        let own = launch.kernel.name.strip_prefix(entry).and_then(|name| name.strip_prefix('_'));
        launch.kernel.name = format!("{entry}_{GATHER}{}", own.expect("a projection launch is named for its entry"));
        launch.params.iter_mut().for_each(rename_param);
        launch.reads.iter_mut().for_each(|read| rename_ident(read, &names));
        launch.grid.iter_mut().for_each(|grid| rename_ident(grid, &buffers));
        let exprs = launch.when.iter_mut().chain(launch.constraint.iter_mut()).chain(launch.shared_bytes.iter_mut());
        let exprs = exprs.chain(launch.threadgroups.iter_mut()).chain(launch.threads_per_threadgroup.iter_mut());
        exprs.chain(launch.conditions.iter_mut().flat_map(|condition| condition.index.iter_mut())).for_each(|expr| rename(expr, &names));
    }
}

/// `expr` with each name `names` maps renamed.
fn rename(expr: &mut Expr, names: &BTreeMap<String, String>) {
    match &mut expr.kind {
        ExprKind::Name(name) => {
            if let Some(renamed) = names.get(&name.name) {
                name.name = renamed.clone();
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            rename(lhs, names);
            rename(rhs, names);
        }
        ExprKind::Unary { expr, .. } => rename(expr, names),
        ExprKind::Call { callee, args, bindings } => {
            let read = read_arguments(callee, args.len());
            args.iter_mut().take(read).for_each(|arg| rename(&mut arg.value, names));
            bindings.iter_mut().for_each(|(_, value)| rename(value, names));
        }
        ExprKind::Tuple(items) => items.iter_mut().for_each(|item| rename(item, names)),
        ExprKind::Attr { base, .. } => rename(base, names),
        _ => {}
    }
}

/// Whether any item of `projection` reads `name`.
fn reads_name(projection: &ProjectionDecl, name: &str) -> bool {
    let mut read = item_reads(projection);
    for term in &projection.terms {
        names(&term.item.value, &mut read);
    }
    for constraint in &projection.constraints {
        names(&constraint.item, &mut read);
    }
    read.contains(name)
}

/// Every name the launches and buffers of `projection` read, their operand
/// clauses' included.
fn item_reads(projection: &ProjectionDecl) -> BTreeSet<String> {
    let mut read = BTreeSet::new();
    for launch in &projection.launches {
        let launch = &launch.launch;
        let exprs = launch.when.iter().chain(launch.constraint.iter()).chain(launch.shared_bytes.iter());
        for expr in exprs.chain(launch.threadgroups.iter()).chain(launch.threads_per_threadgroup.iter()) {
            names(expr, &mut read);
        }
        read.extend(launch.reads.iter().map(|name| name.name.clone()));
    }
    for item in &projection.scratch {
        let buffers = match item {
            ProjectionScratchDecl::One(one) => std::slice::from_ref(one),
            ProjectionScratchDecl::Each(each) => each.as_slice(),
        };
        for buffer in buffers {
            // An operand buffer's clauses read parameters as its size does.
            let clauses = match &buffer.item.operand {
                Some(ScratchOperandDecl::Declared { when, formed, nvfp4, .. }) => [when, formed, nvfp4],
                _ => [&None, &None, &None],
            };
            let exprs = std::iter::once(&buffer.item.bytes).chain(buffer.item.when.iter());
            for expr in exprs.chain(clauses.into_iter().flatten()) {
                names(expr, &mut read);
            }
        }
    }
    read
}

/// `buffers` placed among a declaration's own `scratch` at its `project`
/// line, after the `before` buffers it declares ahead of the line. The
/// parser counts those buffers, so a count past the declaration's is an
/// invariant violation, never a position to clamp.
fn place_at_line(scratch: &mut Vec<NativeScratchDecl>, before: usize, buffers: Vec<NativeScratchDecl>) {
    assert!(before <= scratch.len(), "a `project` line after {before} of its declaration's {} buffers", scratch.len());
    let after = scratch.split_off(before);
    scratch.extend(buffers);
    scratch.extend(after);
}

/// An entry's range and default for a parameter the projection declares. A
/// restatement chooses a range and a default; it never changes what the
/// parameter is to the tuner and the checks (a reduction order stays
/// `arithmetic`) or admits a value the projection does not. A form is not
/// an entry's to restate: the projection's admission gives its range
/// ([`admit`]) and the projection its default.
fn restate(own: &NativeParamDecl, theirs: NativeParamDecl) -> Result<NativeParamDecl, Diagnostic> {
    if theirs.form {
        return Err(error(
            own.span,
            format!(
                "`{}` is a form of the projection: its `where` lines admit it over the entry's facts, and no entry restates it",
                own.name.name
            ),
        ));
    }
    if own.arithmetic != theirs.arithmetic
        || own.form != theirs.form
        || !own.values.iter().all(|value| theirs.values.contains(value))
    {
        return Err(error(
            own.span,
            format!(
                "`{}` restates the projection's parameter with other flags or values outside its own",
                own.name.name
            ),
        ));
    }
    Ok(own.clone())
}

/// The values the projection's admission leaves `params` for one entry.
/// `conjuncts` are those of its `where` lines (or of one launch's) with the
/// entry's facts substituted; one that they reduce to a condition on a
/// single parameter of `params`, decided at each of its values, is that
/// parameter's range rather than a condition of the entry: the values it
/// rejects leave the range, and it leaves the conjunction (`TALL == 0 or
/// (PARTED == 0 and ..)` is `TALL == 0` for a parted entry, whose `TALL` is
/// then `[0]`). The others are returned, in order, without those that hold.
/// A default the range no longer holds gives way to the first listed value
/// it does: a parameter lists its values in the order it prefers them. So
/// an entry's admitted configurations are unchanged, and no value its facts
/// exclude is one of its parameter's.
fn admit(mut params: Vec<&mut NativeParamDecl>, conjuncts: Vec<Expr>) -> Result<Vec<Expr>, Diagnostic> {
    let mut kept = Vec::new();
    for conjunct in conjuncts {
        if matches!(conjunct.kind, ExprKind::Bool(true)) {
            continue;
        }
        let mut read = BTreeSet::new();
        names(&conjunct, &mut read);
        let mut read = read.into_iter();
        let param = match (read.next(), read.next()) {
            (Some(name), None) => params.iter_mut().find(|param| param.name.name == name),
            _ => None,
        };
        let Some(param) = param else {
            kept.push(conjunct);
            continue;
        };
        let name = param.name.name.clone();
        let decided: Option<Vec<(u64, bool)>> = param
            .values
            .iter()
            .map(|value| {
                match fold(substitute_values(conjunct.clone(), &BTreeMap::from([(name.clone(), *value)]))).kind {
                    ExprKind::Bool(holds) => Some((*value, holds)),
                    _ => None,
                }
            })
            .collect();
        let Some(decided) = decided else {
            kept.push(conjunct);
            continue;
        };
        let values: Vec<u64> = decided.iter().filter(|(_, holds)| *holds).map(|(value, _)| *value).collect();
        let Some(first) = values.first().copied() else {
            return Err(error(conjunct.span, format!("the projection admits no value of `{name}` for this entry")));
        };
        if values.len() == param.values.len() {
            continue;
        }
        let narrow = |decl: &mut NativeDefaultDecl, kind: &str| -> Result<(), Diagnostic> {
            match decl {
                NativeDefaultDecl::Value(Expr { kind: ExprKind::Int(val), .. }) => {
                    if !values.contains(val) {
                        *decl = NativeDefaultDecl::Value(Expr { kind: ExprKind::Int(first), span: param.span });
                    }
                    Ok(())
                }
                NativeDefaultDecl::Least(_) | NativeDefaultDecl::Greatest(_) => Ok(()),
                NativeDefaultDecl::Value(_) => Err(error(
                    param.span,
                    format!("the projection's admission narrows `{name}` for this entry, so its {kind} is a literal value or `least`/`greatest`"),
                )),
            }
        };
        narrow(&mut param.default, "default")?;
        if let Some(reference) = &mut param.reference {
            narrow(reference, "reference")?;
        }
        param.values = values;
    }
    Ok(kept)
}

/// The most rows `predicates` (an entry contract's `where`) bind `m` at:
/// a conjunct `m <= c` or `m < c`.
fn int_or_const(expr: &Expr, constants: &BTreeMap<String, u64>) -> Option<u64> {
    match &expr.kind {
        ExprKind::Int(c) => Some(*c),
        ExprKind::Name(name) => constants.get(&name.name).copied(),
        _ => None,
    }
}

fn row_bound(predicates: &[Expr], m: &Expr, constants: &BTreeMap<String, u64>) -> Option<u64> {
    let ExprKind::Name(m) = &m.kind else { return None };
    let mut conjuncts = Vec::new();
    for predicate in predicates {
        flatten(predicate.clone(), BinaryOp::And, &mut conjuncts);
    }
    conjuncts
        .iter()
        .filter_map(|conjunct| match &conjunct.kind {
            ExprKind::Binary { op, lhs, rhs } => match (&lhs.kind, int_or_const(rhs, constants)) {
                (ExprKind::Name(name), Some(c)) if name.name == m.name => match op {
                    BinaryOp::Le => Some(c),
                    BinaryOp::Lt => c.checked_sub(1),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .min()
}

/// Whether `when` holds only past `rows` rows of `m`: a conjunct `m > c`
/// (`c >= rows`), `m >= c` or `m == c` (`c > rows`).
fn needs_more_rows(when: &Expr, m: &Expr, rows: Option<u64>, constants: &BTreeMap<String, u64>) -> bool {
    let (Some(rows), ExprKind::Name(m)) = (rows, &m.kind) else { return false };
    let mut conjuncts = Vec::new();
    flatten(when.clone(), BinaryOp::And, &mut conjuncts);
    conjuncts.iter().any(|conjunct| match &conjunct.kind {
        ExprKind::Binary { op, lhs, rhs } => match (&lhs.kind, int_or_const(rhs, constants)) {
            (ExprKind::Name(name), Some(c)) if name.name == m.name => match op {
                BinaryOp::Gt => c >= rows,
                BinaryOp::Ge | BinaryOp::Eq => c > rows,
                _ => false,
            },
            _ => false,
        },
        _ => false,
    })
}

/// The projection's parameters (`params`) that nothing `native` launches or
/// sizes reads, and those of the launches it does not emit (`dropped`), are
/// removed: they choose nothing, so a configuration is admitted where some
/// value of each admits it (the `where` conjuncts that read it, over each
/// value, joined by `or`). No configuration then tunes a choice that
/// changes nothing. `read` holds what the operand clauses of `native`'s
/// buffers read before the expansion took them (a Q8 buffer's `formed`
/// condition, read by an entry that writes its blocks itself).
fn fix_unread(native: &mut NativeDecl, params: &BTreeSet<String>, dropped: &[NativeParamDecl], mut read: BTreeSet<String>) {
    for launch in &native.launches {
        let exprs = launch.when.iter().chain(launch.constraint.iter()).chain(launch.shared_bytes.iter());
        for expr in exprs.chain(launch.threadgroups.iter()).chain(launch.threads_per_threadgroup.iter()) {
            names(expr, &mut read);
        }
        read.extend(launch.reads.iter().map(|name| name.name.clone()));
    }
    // The formation's shapes and conditions (its Q8 blocks' `formed`, read
    // even where its buffers hold no operand of a parameter's).
    for operand in native.formation.iter().flat_map(|formation| &formation.operands) {
        for expr in [&operand.rows, &operand.width, &operand.groups, &operand.formed, &operand.tiles] {
            names(expr, &mut read);
        }
    }
    for scratch in &native.scratch {
        let conditions = match &scratch.operand {
            Some(ScratchOperandDecl::Formed { pairs, formed, nvfp4, .. }) => vec![pairs, formed, nvfp4],
            Some(ScratchOperandDecl::Declared { when, formed, nvfp4, .. }) => [when, formed, nvfp4].into_iter().flatten().collect(),
            Some(ScratchOperandDecl::Emitted { .. }) | None => Vec::new(),
        };
        for expr in std::iter::once(&scratch.bytes).chain(scratch.when.iter()).chain(conditions) {
            names(expr, &mut read);
        }
    }
    let mut unread: Vec<(String, Vec<u64>)> = dropped
        .iter()
        .filter(|param| {
            !native.launches.iter().any(|launch| launch.params.iter().any(|own| own.name.name == param.name.name))
        })
        .map(|param| (param.name.name.clone(), param.values.clone()))
        .collect();
    native.params.retain(|param| {
        let keep = !params.contains(&param.name.name) || read.contains(&param.name.name);
        if !keep {
            unread.push((param.name.name.clone(), param.values.clone()));
        }
        keep
    });
    if let Some(constraint) = native.constraint.take() {
        let span = constraint.span;
        let mut conjuncts = Vec::new();
        flatten(constraint, BinaryOp::And, &mut conjuncts);
        for (name, values) in unread {
            // Only the conjuncts that read the parameter quantify over it.
            let (reading, others): (Vec<_>, Vec<_>) = conjuncts.into_iter().partition(|conjunct| {
                let mut read = BTreeSet::new();
                names(conjunct, &mut read);
                read.contains(&name)
            });
            conjuncts = others;
            if reading.is_empty() {
                continue;
            }
            let reading = chain(BinaryOp::And, reading, span);
            let each = values
                .iter()
                .map(|value| fold(substitute_values(reading.clone(), &BTreeMap::from([(name.clone(), *value)]))))
                .collect();
            flatten(fold(chain(BinaryOp::Or, each, span)), BinaryOp::And, &mut conjuncts);
        }
        conjuncts.retain(|conjunct| !matches!(conjunct.kind, ExprKind::Bool(true)));
        native.constraint = (!conjuncts.is_empty()).then(|| chain(BinaryOp::And, conjuncts, span));
    }
}

/// Every name `expr` reads.
fn names(expr: &Expr, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Name(name) => {
            out.insert(name.name.clone());
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            names(lhs, out);
            names(rhs, out);
        }
        ExprKind::Unary { expr, .. } => names(expr, out),
        ExprKind::Call { callee, args, bindings } => {
            args.iter().take(read_arguments(callee, args.len())).for_each(|arg| names(&arg.value, out));
            bindings.iter().for_each(|(_, value)| names(value, out));
        }
        ExprKind::Tuple(items) => items.iter().for_each(|item| names(item, out)),
        ExprKind::Attr { base, .. } => names(base, out),
        _ => {}
    }
}

/// `expr` with each name `values` holds replaced by its value.
fn substitute_values(expr: Expr, values: &BTreeMap<String, u64>) -> Expr {
    let span = expr.span;
    let kind = match expr.kind {
        ExprKind::Name(name) => match values.get(&name.name) {
            Some(value) => ExprKind::Int(*value),
            None => ExprKind::Name(name),
        },
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op,
            lhs: Box::new(substitute_values(*lhs, values)),
            rhs: Box::new(substitute_values(*rhs, values)),
        },
        ExprKind::Unary { op, expr } => ExprKind::Unary { op, expr: Box::new(substitute_values(*expr, values)) },
        ExprKind::Call { callee, bindings, args } => {
            let read = read_arguments(&callee, args.len());
            ExprKind::Call {
                callee,
                bindings: bindings.into_iter().map(|(name, value)| (name, substitute_values(value, values))).collect(),
                args: args
                    .into_iter()
                    .enumerate()
                    .map(|(at, arg)| match at < read {
                        true => Arg { name: arg.name, value: substitute_values(arg.value, values) },
                        false => arg,
                    })
                    .collect(),
            }
        }
        kind => kind,
    };
    Expr { kind, span }
}

/// The leading arguments of a call to `callee` (of `count`) that are
/// expressions: all but the names an element query reads by
/// (`form(E, NAME)`'s form of the backend, `axis(E, NAME)`'s decode axis,
/// `row_bytes(E, PART, VALUES)`'s planes and count), not parameters that
/// share their names.
fn read_arguments(callee: &Expr, count: usize) -> usize {
    match &callee.kind {
        ExprKind::Name(name) if matches!(name.name.as_str(), "form" | "axis" | "row_bytes") => count.min(1),
        _ => count,
    }
}

/// `lhs op rhs`, spanning both.
fn binary(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr {
        span: lhs.span.to(rhs.span),
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
    }
}

/// The operands of `expr`'s chain of `op` (`and`, `or`), left to right.
fn flatten(expr: Expr, op: BinaryOp, out: &mut Vec<Expr>) {
    match expr.kind {
        ExprKind::Binary { op: inner, lhs, rhs } if inner == op => {
            flatten(*lhs, op, out);
            flatten(*rhs, op, out);
        }
        kind => out.push(Expr { kind, span: expr.span }),
    }
}

/// `operands` joined by `op` (`and`, `or`, `+`) from the left; an empty `and`
/// is true and an empty `or` false.
fn chain(op: BinaryOp, operands: Vec<Expr>, span: Span) -> Expr {
    operands
        .into_iter()
        .reduce(|lhs, rhs| binary(op, lhs, rhs))
        .unwrap_or(Expr {
            kind: ExprKind::Bool(op == BinaryOp::And),
            span,
        })
}

impl Instance<'_> {
    /// Whether an item gated `for` `variants` is in force: one names the
    /// variant, or the kind of the rows (a GPU projection's: a CPU variant
    /// is no kind of rows, and the CPU's `plain` is not the rows' `plain`).
    fn takes(&self, variants: &[Ident]) -> bool {
        let cpu = CPU_VARIANTS.contains(&self.variant);
        let ordered = self.weights.iter().any(|w| w.ordered);
        variants.is_empty()
            || variants.iter().any(|name| {
                name.name == self.variant
                    || (!cpu && (name.name == self.rows.name() || (name.name == "ordered" && ordered)))
            })
    }

    fn formal(&self, name: &str) -> bool {
        match self.routed {
            Some(_) => {
                matches!(
                    name,
                    "M" | "K" | "N" | "W" | "WIDTH" | "STACKED" | "FIRST" | "SERVED" | "SHARED" | "RECORDS" | "WHOLE" | "PER_ROW"
                )
            }
            None => matches!(
                name,
                "G" | "P" | "M" | "K" | "N" | "W" | "OWN" | "EMITTED" | "WEIGHTS" | "SERVED" | "GATHERED" | "PARTED" | "PLANES"
                    | "PUBLISHED" | "ORDERED"
            ),
        }
    }

    /// A formal's expression, every position the reference's.
    fn formal_value(&self, value: &Expr, span: Span) -> Expr {
        fn at(expr: &mut Expr, span: Span) {
            expr.span = span;
            match &mut expr.kind {
                ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Inf | ExprKind::Bool(_) => {}
                ExprKind::Name(name) => name.span = span,
                ExprKind::Tuple(items) => items.iter_mut().for_each(|item| at(item, span)),
                ExprKind::Range { lo, hi } => {
                    at(lo, span);
                    at(hi, span);
                }
                ExprKind::Tensor { shape, elem } => {
                    shape.iter_mut().for_each(|dim| at(dim, span));
                    elem.span = span;
                }
                ExprKind::Call { callee, bindings, args } => {
                    at(callee, span);
                    for (name, value) in bindings {
                        name.span = span;
                        at(value, span);
                    }
                    for arg in args {
                        if let Some(name) = &mut arg.name {
                            name.span = span;
                        }
                        at(&mut arg.value, span);
                    }
                }
                ExprKind::Index { base, indices } => {
                    at(base, span);
                    for index in indices {
                        match index {
                            Index::Expr(expr) => at(expr, span),
                            Index::Slice { start, end } => {
                                start.iter_mut().chain(end.iter_mut()).for_each(|bound| at(bound, span))
                            }
                        }
                    }
                }
                ExprKind::Attr { base, name } => {
                    at(base, span);
                    name.span = span;
                }
                ExprKind::Unary { expr, .. } => at(expr, span),
                ExprKind::Binary { lhs, rhs, .. } => {
                    at(lhs, span);
                    at(rhs, span);
                }
            }
        }
        let mut value = value.clone();
        at(&mut value, span);
        value
    }

    fn span(&self, span: Span) -> Span {
        if self.same_file { span } else { self.at }
    }

    fn ident(&self, ident: &Ident) -> Ident {
        Ident {
            name: ident.name.clone(),
            span: self.span(ident.span),
        }
    }

    /// `expr` with the formals and terms substituted, folded.
    fn expr(&self, expr: &Expr) -> Result<Expr, Diagnostic> {
        Ok(fold(self.substitute(expr, None)?))
    }

    fn substitute(&self, expr: &Expr, weight: Option<&Weight>) -> Result<Expr, Diagnostic> {
        let span = self.span(expr.span);
        if let (Some(routed), ExprKind::Name(name)) = (&self.routed, &expr.kind) {
            // A routed projection's formals; its extents are its entry's names.
            let first = &self.weights[0];
            let weight_or_first = weight.unwrap_or(first);
            let kind = match name.name.as_str() {
                "M" => return Ok(self.formal_value(&self.m, span)),
                "K" => return Ok(self.formal_value(&routed.choices, span)),
                "XS" => {
                    if let Some(xs) = &routed.xs {
                        return Ok(self.formal_value(xs, span));
                    }
                    ExprKind::Name(Ident { name: "XS".to_owned(), span })
                }
                "N" => return Ok(self.formal_value(&weight_or_first.n, span)),
                "WIDTH" => return Ok(self.formal_value(&weight_or_first.width, span)),
                "STACKED" => ExprKind::Int(u64::from(weight_or_first.stacked)),
                "FIRST" => ExprKind::Int(u64::from(weight_or_first.segment == 0)),
                "SERVED" => ExprKind::Int(self.served as u64),
                "SHARED" => ExprKind::Int(u64::from(routed.shared)),
                "RECORDS" => ExprKind::Int(u64::from(!routed.records.is_empty())),
                "WHOLE" => ExprKind::Int(u64::from(routed.whole)),
                "PER_ROW" => ExprKind::Int(u64::from(routed.per_row)),
                "W" => match weight {
                    Some(weight) => ExprKind::Name(Ident { name: weight.element.name.clone(), span }),
                    None => {
                        return Err(error(
                            self.span(expr.span),
                            "`W` is a weight only in a weight quantifier or an `each W:` block",
                        ));
                    }
                },
                other => match self.terms.get(other) {
                    Some(value) => return Ok(value.clone()),
                    None if self.named.contains(other) => {
                        return Err(error(self.span(expr.span), format!("`{other}` has no `{}` value", self.variant)));
                    }
                    None => ExprKind::Name(Ident { name: other.to_owned(), span }),
                },
            };
            return Ok(Expr { kind, span });
        }
        let kind = match &expr.kind {
            ExprKind::Name(name) => match name.name.as_str() {
                "G" => return Ok(self.formal_value(&self.g, span)),
                "P" => return Ok(self.formal_value(&self.p, span)),
                "M" => {
                    let own = weight.and_then(|weight| weight.over.as_ref());
                    return Ok(self.formal_value(own.map_or(&self.m, |over| &over.rows), span));
                }
                "K" => {
                    let own = weight.and_then(|weight| weight.over.as_ref());
                    return Ok(self.formal_value(own.map_or(&self.k, |over| &over.width), span));
                }
                "N" => {
                    return match (&self.n, weight) {
                        (_, Some(weight)) => Ok(self.formal_value(&weight.n, span)),
                        (Some(n), None) => Ok(self.formal_value(n, span)),
                        (None, None) => Err(error(
                            self.span(expr.span),
                            "a `segmented` projection's `N` is a weight's: read it in `each`, `any`, `sum`, `tiles` or an `each W:` block",
                        )),
                    };
                }
                "WEIGHTS" => ExprKind::Int(self.weights.len() as u64),
                // The weights the segment of the quantified weight serves, else
                // the most weights one segment serves.
                "GATHERED" => ExprKind::Int(u64::from(self.gathered)),
                "PARTED" => ExprKind::Int(u64::from(self.parted)),
                "PLANES" => ExprKind::Int(u64::from(self.planes)),
                "PUBLISHED" => ExprKind::Int(u64::from(self.published)),
                // Whether the quantified weight's sums are in the reference order.
                "ORDERED" => match weight {
                    Some(weight) => ExprKind::Int(u64::from(weight.ordered)),
                    None => {
                        return Err(error(
                            self.span(expr.span),
                            "`ORDERED` is a weight's: read it in a weight quantifier or an `each W:` block",
                        ));
                    }
                },
                // Whether the quantified weight multiplies rows of its own.
                "OWN" => match weight {
                    Some(weight) => ExprKind::Int(u64::from(weight.over.is_some())),
                    None => {
                        return Err(error(
                            self.span(expr.span),
                            "`OWN` is a weight's: read it in a weight quantifier or an `each W:` block",
                        ));
                    }
                },
                // Whether the entry emits the quantified weight's own operand.
                "EMITTED" => match weight {
                    Some(weight) => ExprKind::Int(u64::from(weight.emitted)),
                    None => {
                        return Err(error(
                            self.span(expr.span),
                            "`EMITTED` is a weight's: read it in a weight quantifier or an `each W:` block",
                        ));
                    }
                },
                "SERVED" => ExprKind::Int(match weight {
                    Some(weight) => self.weights.iter().filter(|other| other.segment == weight.segment).count() as u64,
                    None => self.served as u64,
                }),
                "W" => match weight {
                    Some(weight) if weight.planes => {
                        return Err(error(
                            self.span(expr.span),
                            "a plane set has no element: its `W` is read only by `form`, `operands`, `axis`, `row_bytes`, `dense_bytes` or `bound`",
                        ));
                    }
                    Some(weight) => ExprKind::Name(Ident {
                        name: weight.element.name.clone(),
                        span,
                    }),
                    None => {
                        return Err(error(
                            self.span(expr.span),
                            "`W` is a weight only in a weight quantifier or an `each W:` block",
                        ));
                    }
                },
                other => match self.terms.get(other) {
                    Some(value) => return Ok(value.clone()),
                    None if self.named.contains(other) => {
                        return Err(error(
                            self.span(expr.span),
                            format!("`{other}` has no `{}` value", self.variant),
                        ));
                    }
                    None => ExprKind::Name(Ident { name: other.to_owned(), span }),
                },
            },
            ExprKind::Call { callee, bindings, args } => {
                let quantifier = match &callee.kind {
                    ExprKind::Name(name) if bindings.is_empty() => name.name.as_str(),
                    _ => "",
                };
                match quantifier {
                    "each" | "any" | "agree" | "sum" | "most" | "tiles" => return self.quantify(quantifier, args, weight, expr.span),
                    // A plane set takes no form a producer delivers, and has
                    // neither decode axes, packed rows nor dense storage.
                    "form" | "operands" | "axis" | "row_bytes" | "dense_bytes"
                        if weight.is_some_and(|weight| weight.planes)
                            && args.first().is_some_and(|arg| matches!(&arg.value.kind, ExprKind::Name(name) if name.name == "W")) =>
                    {
                        ExprKind::Int(0)
                    }
                    "bound" => {
                        let (Some(weight), [Arg { name: None, value }]) = (weight, args.as_slice()) else {
                            return Err(error(self.span(expr.span), "`bound(W)` reads the extent of a weight `W`"));
                        };
                        if !matches!(&value.kind, ExprKind::Name(name) if name.name == "W") {
                            return Err(error(self.span(expr.span), "`bound(W)` reads the extent of a weight `W`"));
                        }
                        // A weight without a bound port has none: extent 0.
                        match &weight.bound {
                            Some(bound) => ExprKind::Name(Ident { name: bound.name.clone(), span }),
                            None => ExprKind::Int(0),
                        }
                    }
                    _ => ExprKind::Call {
                        callee: Box::new(self.substitute(callee, weight)?),
                        bindings: bindings
                            .iter()
                            .map(|(name, value)| Ok((self.ident(name), self.substitute(value, weight)?)))
                            .collect::<Result<_, Diagnostic>>()?,
                        args: args
                            .iter()
                            .map(|arg| {
                                Ok(Arg {
                                    name: arg.name.as_ref().map(|name| self.ident(name)),
                                    value: self.substitute(&arg.value, weight)?,
                                })
                            })
                            .collect::<Result<_, Diagnostic>>()?,
                    },
                }
            }
            ExprKind::Attr { base, name } => ExprKind::Attr {
                base: Box::new(self.substitute(base, weight)?),
                name: self.ident(name),
            },
            ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
                op: *op,
                expr: Box::new(self.substitute(inner, weight)?),
            },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
                op: *op,
                lhs: Box::new(self.substitute(lhs, weight)?),
                rhs: Box::new(self.substitute(rhs, weight)?),
            },
            ExprKind::Tuple(items) => ExprKind::Tuple(
                items
                    .iter()
                    .map(|item| self.substitute(item, weight))
                    .collect::<Result<_, _>>()?,
            ),
            kind @ (ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Inf | ExprKind::Bool(_)) => kind.clone(),
            ExprKind::Range { .. } | ExprKind::Tensor { .. } | ExprKind::Index { .. } => {
                return Err(error(self.span(expr.span), "not an expression of a native declaration"));
            }
        };
        Ok(Expr { kind, span })
    }

    /// `each(W, C)`, `any(W, C)`, `agree(W, E)`, `sum(W, E)` or `most(W, E)`
    /// over the weights; `tiles(W, E)`, E over the projection's output rows: the
    /// weights' shared `N` once, or each segment's summed.
    fn quantify(
        &self,
        quantifier: &str,
        args: &[Arg],
        weight: Option<&Weight>,
        span: Span,
    ) -> Result<Expr, Diagnostic> {
        let [Arg { name: None, value: formal }, Arg { name: None, value: body }] = args else {
            return Err(error(self.span(span), format!("`{quantifier}(W, ..)` takes the weight `W` and one expression")));
        };
        if !matches!(&formal.kind, ExprKind::Name(name) if name.name == "W") {
            return Err(error(self.span(formal.span), format!("`{quantifier}` ranges over the weights, `W`")));
        }
        if weight.is_some() {
            return Err(error(self.span(span), "weight quantifiers do not nest"));
        }
        let each = self
            .weights
            .iter()
            .map(|weight| self.substitute(body, Some(weight)))
            .collect::<Result<Vec<_>, _>>()?;
        let span = self.span(span);
        Ok(match quantifier {
            "each" => chain(BinaryOp::And, each, span),
            "any" => chain(BinaryOp::Or, each, span),
            "sum" => chain(BinaryOp::Add, each, span),
            // The largest of the weights' values, each distinct one once.
            "most" => {
                let mut distinct: Vec<Expr> = Vec::new();
                for value in each {
                    if !distinct.iter().any(|seen| structure(seen) == structure(&value)) {
                        distinct.push(value);
                    }
                }
                distinct
                    .into_iter()
                    .reduce(|most, value| Expr {
                        span,
                        kind: ExprKind::Call {
                            callee: Box::new(Expr { kind: ExprKind::Name(Ident { name: "max".to_owned(), span }), span }),
                            bindings: Vec::new(),
                            args: vec![Arg { name: None, value: most }, Arg { name: None, value }],
                        },
                    })
                    .expect("a variant takes at least one weight")
            }
            // The weights' rows once where they share them, else each
            // segment's in turn.
            "tiles" if self.n.is_some() => each.into_iter().next().expect("a variant takes at least one weight"),
            // Each segment's rows once (the first weight of a served pair).
            "tiles" => {
                let firsts = (0..self.weights.len())
                    .filter(|index| *index == 0 || self.weights[*index].segment != self.weights[*index - 1].segment)
                    .map(|index| each[index].clone())
                    .collect();
                chain(BinaryOp::Add, firsts, span)
            }
            _ => {
                let first = each[0].clone();
                let pairs = each[1..]
                    .iter()
                    .map(|other| binary(BinaryOp::Eq, first.clone(), other.clone()))
                    .collect();
                chain(BinaryOp::And, pairs, span)
            }
        })
    }

    fn default_decl(&self, decl: &NativeDefaultDecl) -> Result<NativeDefaultDecl, Diagnostic> {
        Ok(match decl {
            NativeDefaultDecl::Value(value) => NativeDefaultDecl::Value(self.expr(value)?),
            NativeDefaultDecl::Least(condition) => NativeDefaultDecl::Least(self.expr(condition)?),
            NativeDefaultDecl::Greatest(condition) => NativeDefaultDecl::Greatest(self.expr(condition)?),
        })
    }

    fn param(&self, param: &NativeParamDecl) -> Result<NativeParamDecl, Diagnostic> {
        Ok(NativeParamDecl {
            name: self.ident(&param.name),
            arithmetic: param.arithmetic,
            form: param.form,
            values: param.values.clone(),
            default: self.default_decl(&param.default)?,
            reference: param.reference.as_ref().map(|r| self.default_decl(r)).transpose()?,
            span: self.span(param.span),
        })
    }

    fn optional(&self, expr: &Option<Expr>, weight: Option<&Weight>) -> Result<Option<Expr>, Diagnostic> {
        expr.as_ref()
            .map(|expr| Ok(fold(self.substitute(expr, weight)?)))
            .transpose()
    }

    fn scratch(&self, scratch: &NativeScratchDecl, weight: Option<&Weight>) -> Result<NativeScratchDecl, Diagnostic> {
        let name = match weight {
            Some(weight) => Ident {
                name: format!("{}{}", weight.prefix, scratch.name.name),
                span: self.span(scratch.name.span),
            },
            None => self.ident(&scratch.name),
        };
        let operand = match &scratch.operand {
            None => None,
            Some(ScratchOperandDecl::Declared { rows, when, formed, nvfp4 }) => Some(ScratchOperandDecl::Declared {
                rows: *rows,
                when: self.optional(when, weight)?,
                formed: self.optional(formed, weight)?,
                nvfp4: self.optional(nvfp4, weight)?,
            }),
            Some(_) => unreachable!("a projection's buffer is declared an operand, never expanded as one"),
        };
        Ok(NativeScratchDecl {
            name,
            bytes: fold(self.substitute(&scratch.bytes, weight)?),
            sync: scratch.sync,
            when: self.optional(&scratch.when, weight)?.filter(|when| !matches!(when.kind, ExprKind::Bool(true))),
            operand,
            emission: None,
            span: self.span(scratch.span),
        })
    }

    fn launch(&self, launch: &ProjectionLaunchDecl) -> Result<NativeLaunchDecl, Diagnostic> {
        let declared = &launch.launch;
        let mut params = Vec::new();
        let mut reads: Vec<Ident> = declared.reads.iter().map(|read| self.ident(read)).collect();
        for (param, variant) in declared.params.iter().zip(&launch.param_variants) {
            if self.takes(variant) && self.lifted.contains(&param.name.name) {
                reads.push(self.ident(&param.name));
            } else if self.takes(variant) {
                let theirs = self.param(param)?;
                let restated = self.restated.iter().find(|restated| {
                    restated.launch.as_ref().is_some_and(|launch| launch.name == declared.kernel.name)
                        && restated.param.name.name == param.name.name
                });
                params.push(match restated {
                    Some(restated) => restate(&restated.param, theirs)?,
                    None => theirs,
                });
            }
        }
        let three = |exprs: &[Expr; 3]| -> Result<[Expr; 3], Diagnostic> {
            Ok([self.expr(&exprs[0])?, self.expr(&exprs[1])?, self.expr(&exprs[2])?])
        };
        Ok(NativeLaunchDecl {
            kernel: Ident {
                name: format!("{}_{}", self.entry, declared.kernel.name),
                span: self.span(declared.kernel.span),
            },
            before: None,
            resident: declared.resident.as_ref().map(|barrier| self.ident(barrier)),
            when: self.optional(&declared.when, None)?.filter(|when| !matches!(when.kind, ExprKind::Bool(true))),
            params,
            constraint: self.optional(&declared.constraint, None)?,
            reads,
            alike: declared.alike.as_ref().map(|alike| Ident {
                name: format!("{}_{}", self.entry, alike.name),
                span: self.span(alike.span),
            }),
            staging: declared.staging,
            records: declared.records.as_ref().map(|records| {
                records
                    .iter()
                    .flat_map(|record| match (record, &self.routed) {
                        // A routed launch's `records (W)`: its entry's record ports.
                        (NativeRecordDecl::Parameter(name), Some(routed)) if name.name == "W" => routed.records.clone(),
                        _ => vec![record.clone()],
                    })
                    .map(|record| match &record {
                        NativeRecordDecl::Parameter(name) => NativeRecordDecl::Parameter(self.ident(name)),
                        NativeRecordDecl::Result { ordinal, span } => NativeRecordDecl::Result {
                            ordinal: *ordinal,
                            span: self.span(*span),
                        },
                    })
                    .collect()
            }),
            threadgroups: three(&declared.threadgroups)?,
            grid: declared.grid.as_ref().map(|grid| self.ident(grid)),
            conditions: declared
                .conditions
                .iter()
                .map(|condition| {
                    Ok(NativeConditionDecl {
                        encoded: condition.encoded,
                        word: self.ident(&condition.word),
                        index: self.optional(&condition.index, None)?,
                    })
                })
                .collect::<Result<_, Diagnostic>>()?,
            threads_per_threadgroup: three(&declared.threads_per_threadgroup)?,
            shared_bytes: self.optional(&declared.shared_bytes, None)?,
            span: self.span(declared.span),
        })
    }
}

/// `expr` with its constant parts evaluated: literal arithmetic, `min`,
/// `max`, `ceil_div` and comparisons, `x * 1`, `1 * x` and `x / 1`, `not`
/// of a literal, and conjunctions and disjunctions without their decided
/// operands (flat, from the left).
fn fold(expr: Expr) -> Expr {
    let span = expr.span;
    let kind = match expr.kind {
        ExprKind::Binary { op: op @ (BinaryOp::And | BinaryOp::Or), lhs, rhs } => {
            let mut operands = Vec::new();
            flatten(fold(*lhs), op, &mut operands);
            flatten(fold(*rhs), op, &mut operands);
            let absorbing = op == BinaryOp::Or;
            if operands.iter().any(|operand| matches!(operand.kind, ExprKind::Bool(value) if value == absorbing)) {
                return Expr { kind: ExprKind::Bool(absorbing), span };
            }
            operands.retain(|operand| !matches!(operand.kind, ExprKind::Bool(_)));
            let mut joined = chain(op, operands, span);
            joined.span = span;
            return joined;
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let (lhs, rhs) = (fold(*lhs), fold(*rhs));
            match (&lhs.kind, &rhs.kind) {
                (&ExprKind::Int(a), &ExprKind::Int(b)) => match op {
                    BinaryOp::Add => a.checked_add(b).map(ExprKind::Int),
                    BinaryOp::Sub => a.checked_sub(b).map(ExprKind::Int),
                    BinaryOp::Mul => a.checked_mul(b).map(ExprKind::Int),
                    BinaryOp::Div => a.checked_div(b).map(ExprKind::Int),
                    BinaryOp::Rem => a.checked_rem(b).map(ExprKind::Int),
                    BinaryOp::Eq => Some(ExprKind::Bool(a == b)),
                    BinaryOp::Ne => Some(ExprKind::Bool(a != b)),
                    BinaryOp::Lt => Some(ExprKind::Bool(a < b)),
                    BinaryOp::Le => Some(ExprKind::Bool(a <= b)),
                    BinaryOp::Gt => Some(ExprKind::Bool(a > b)),
                    BinaryOp::Ge => Some(ExprKind::Bool(a >= b)),
                    _ => None,
                },
                (_, ExprKind::Int(1)) if matches!(op, BinaryOp::Mul | BinaryOp::Div) => return lhs,
                (ExprKind::Int(1), _) if op == BinaryOp::Mul => return rhs,
                (_, ExprKind::Int(0)) | (ExprKind::Int(0), _) if op == BinaryOp::Mul => Some(ExprKind::Int(0)),
                (_, ExprKind::Int(0)) if matches!(op, BinaryOp::Add | BinaryOp::Sub) => return lhs,
                (ExprKind::Int(0), _) if op == BinaryOp::Add => return rhs,
                _ => None,
            }
            .unwrap_or(ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        ExprKind::Unary { op, expr: inner } => match (op, fold(*inner)) {
            (UnaryOp::Not, Expr { kind: ExprKind::Bool(value), .. }) => ExprKind::Bool(!value),
            (op, inner) => ExprKind::Unary {
                op,
                expr: Box::new(inner),
            },
        },
        ExprKind::Call { callee, bindings, args } => {
            let bindings: Vec<_> = bindings.into_iter().map(|(name, value)| (name, fold(value))).collect();
            let args: Vec<_> = args
                .into_iter()
                .map(|arg| Arg {
                    name: arg.name,
                    value: fold(arg.value),
                })
                .collect();
            if bindings.is_empty() {
                if let (
                    ExprKind::Name(name),
                    [
                        Arg { name: None, value: Expr { kind: ExprKind::Int(a), .. } },
                        Arg { name: None, value: Expr { kind: ExprKind::Int(b), .. } },
                    ],
                ) = (&callee.kind, args.as_slice())
                {
                    let folded = match name.name.as_str() {
                        "min" => Some((*a).min(*b)),
                        "max" => Some((*a).max(*b)),
                        "ceil_div" if *b > 0 => Some(a.div_ceil(*b)),
                        _ => None,
                    };
                    if let Some(value) = folded {
                        return Expr { kind: ExprKind::Int(value), span };
                    }
                }
            }
            ExprKind::Call { callee, bindings, args }
        }
        kind => kind,
    };
    Expr { kind, span }
}

#[cfg(test)]
mod tests {
    use crate::checked::{check_source, NativeImplementation, SourceFile, SourceSet};
    use crate::registry::BackendName;

    /// A paired and a single entry, each with a bound port per weight.
    const FUNCTIONS: &str = "fn pair[O, K, N, GB, UB](x: &tensor[O, K] f32, gate_weight: &tensor[N, K] GW, up_weight: &tensor[N, K] UW, gate_bound: &tensor[GB] f32, up_bound: &tensor[UB] f32) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\nfn one[O, K, N, DB](x: &tensor[O, K] f32, down_weight: &tensor[N, K] DW, down_bound: &tensor[DB] f32) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";

    const PROJECTION: &str = "projection lanes for metal:\n    let TOKENS = 128 / WEIGHTS\n    let SPLITS = 1 for paired\n    let SPLITS = SPLIT for single\n    params (arithmetic SPLIT in [1, 2] default 1 for single, form WIDE in [0, 1] default 0)\n    where (WIDE == 0 or (SPLITS == 1 and K % (64 / WEIGHTS) == 0 and each(W, form(W, INT8) == 1) and agree(W, form(W, INT8_INTERLEAVED))))\n    where (WIDE == 0 or any(W, bound(W) > 0 and form(W, PACK) == 1))\n    scratch partials bytes (SPLIT * M * N * 4) when M > 4 and SPLIT > 1 for single\n    scratch packed bytes (ceil_div(M, TOKENS) * TOKENS * K) when WIDE == 1\n    each W:\n        scratch factors bytes (N * (K / 32) * 4) when WIDE == 1\n        scratch scales bytes (N * 4) when WIDE == 1\n    launch gemv when M < 2:\n        params (ROWS in [1, 2] default 1 for paired, ROWS in [2, 1] default 2 for single)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (ROWS * (128 * WEIGHTS))\n    launch reduce for single when M > 4 and SPLIT > 1:\n        threadgroups (ceil_div(M * N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch tiles when M >= 2:\n        threadgroups (ceil_div(N, 32 / WEIGHTS), ceil_div(M, TOKENS) * (TOKENS / 4), SPLITS)\n        threads_per_threadgroup (128, 1, 1)\n\n";

    const PROJECTED: &str = "native pair for metal from \"pair.metal\":\n    static (K, N, GB, UB)\n    where N > 0\n    scratch staged bytes (O * K)\n    project lanes paired (gate_weight, up_weight)\n    launch pair_stage before tiles when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n\nnative one for metal from \"one.metal\":\n    static (K, N, DB)\n    project lanes single (down_weight)\n    scratch staged bytes (O * K)\n    launch one_stage before gemv when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n";

    /// What `PROJECTED` stands for, written by hand.
    const WRITTEN: &str = "native pair for metal from \"pair.metal\":\n    static (K, N, GB, UB)\n    params (form WIDE in [0, 1] default 0)\n    where N > 0 and (WIDE == 0 or (K % 32 == 0 and form(GW, INT8) == 1 and form(UW, INT8) == 1 and form(GW, INT8_INTERLEAVED) == form(UW, INT8_INTERLEAVED))) and (WIDE == 0 or (GB > 0 and form(GW, PACK) == 1) or (UB > 0 and form(UW, PACK) == 1))\n    scratch staged bytes (O * K)\n    scratch packed bytes (ceil_div(O, 64) * 64 * K) when WIDE == 1\n    scratch gate_factors bytes (N * (K / 32) * 4) when WIDE == 1\n    scratch gate_scales bytes (N * 4) when WIDE == 1\n    scratch up_factors bytes (N * (K / 32) * 4) when WIDE == 1\n    scratch up_scales bytes (N * 4) when WIDE == 1\n    launch pair_gemv when O < 2:\n        params (ROWS in [1, 2] default 1)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (ROWS * 256)\n    launch pair_stage when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch pair_tiles when O >= 2:\n        threadgroups (ceil_div(N, 16), ceil_div(O, 64) * 16, 1)\n        threads_per_threadgroup (128, 1, 1)\n\nnative one for metal from \"one.metal\":\n    static (K, N, DB)\n    params (arithmetic SPLIT in [1, 2] default 1, form WIDE in [0, 1] default 0)\n    where (WIDE == 0 or (SPLIT == 1 and K % 64 == 0 and form(DW, INT8) == 1)) and (WIDE == 0 or DB > 0 and form(DW, PACK) == 1)\n    scratch partials bytes (SPLIT * O * N * 4) when O > 4 and SPLIT > 1\n    scratch packed bytes (ceil_div(O, 128) * 128 * K) when WIDE == 1\n    scratch down_factors bytes (N * (K / 32) * 4) when WIDE == 1\n    scratch down_scales bytes (N * 4) when WIDE == 1\n    scratch staged bytes (O * K)\n    launch one_stage when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch one_gemv when O < 2:\n        params (ROWS in [2, 1] default 2)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (ROWS * 128)\n    launch one_reduce when O > 4 and SPLIT > 1:\n        threadgroups (ceil_div(O * N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch one_tiles when O >= 2:\n        threadgroups (ceil_div(N, 32), ceil_div(O, 128) * 32, SPLIT)\n        threads_per_threadgroup (128, 1, 1)\n";

    /// Two segmented entries of two weights each (the second's bounded),
    /// the second's contract bounding its rows at 16.
    const SEGMENTED_FUNCTIONS: &str = "fn seg[O, K, A, B, BB](x: &tensor[O, K] f32, a_weight: &tensor[A, K] AW, b_weight: &tensor[B, K] BW, b_bound: &tensor[BB] f32) -> tensor[O, A + B] f32:\n    let mut result = tensor[O, A + B] f32\n    parallel for row in 0..O:\n        parallel for col in 0..A + B:\n            result[row, col] = x[row, 0]\n    return result\n\nfn seg16[O, K, A, B, BB](x: &tensor[O, K] f32, a_weight: &tensor[A, K] AW, b_weight: &tensor[B, K] BW, b_bound: &tensor[BB] f32) -> tensor[O, A + B] f32\n    where O <= 16:\n    let mut result = tensor[O, A + B] f32\n    parallel for row in 0..O:\n        parallel for col in 0..A + B:\n            result[row, col] = x[row, 0]\n    return result\n\n";

    /// A projection whose `segmented` forms tile each weight's rows: a GEMV
    /// below two rows, and a wide form past 16 whose parameter (named as the
    /// form it admits) and table only it reads.
    const SEGMENTED_PROJECTION: &str = "projection rows for metal:\n    params (form PACK in [0, 1] default 0, form FAST in [0, 1] default 0)\n    where (PACK == 0 or any(W, N > 0 and N % 32 == 0 and form(W, PACK) == 1))\n    where (FAST == 0 or K % 64 == 0) for single | segmented\n    scratch table bytes (sum(W, N) * 4) when M > 16 and PACK == 1 for segmented\n    launch gemv when M < 2:\n        threadgroups (tiles(W, ceil_div(N, 32 / SERVED)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (FAST * 64)\n    launch wide when M > 16 and PACK == 1:\n        threadgroups (tiles(W, N / 32) * ceil_div(M, 64), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n\n";

    const SEGMENTED_PROJECTED: &str = "native seg for metal from \"seg.metal\":\n    static (K, A, B, BB)\n    project rows segmented (a_weight, b_weight)\n\nnative seg16 for metal from \"seg16.metal\":\n    static (K, A, B, BB)\n    project rows segmented (a_weight, b_weight)\n";

    /// What `SEGMENTED_PROJECTED` stands for, written by hand: the second
    /// entry has neither the wide form, its table nor its parameter, which
    /// its contract's rows never reach.
    const SEGMENTED_WRITTEN: &str = "native seg for metal from \"seg.metal\":\n    static (K, A, B, BB)\n    params (form PACK in [0, 1] default 0, form FAST in [0, 1] default 0)\n    where (PACK == 0 or (A > 0 and A % 32 == 0 and form(AW, PACK) == 1) or (B > 0 and B % 32 == 0 and form(BW, PACK) == 1)) and (FAST == 0 or K % 64 == 0)\n    scratch table bytes ((A + B) * 4) when O > 16 and PACK == 1\n    launch seg_gemv when O < 2:\n        threadgroups (ceil_div(A, 32) + ceil_div(B, 32), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (FAST * 64)\n    launch seg_wide when O > 16 and PACK == 1:\n        threadgroups ((A / 32 + B / 32) * ceil_div(O, 64), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n\nnative seg16 for metal from \"seg16.metal\":\n    static (K, A, B, BB)\n    params (form FAST in [0, 1] default 0)\n    where (FAST == 0 or K % 64 == 0)\n    launch seg16_gemv when O < 2:\n        threadgroups (ceil_div(A, 32) + ceil_div(B, 32), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (FAST * 64)\n";

    fn implementations(natives: &str) -> Vec<NativeImplementation> {
        checked(FUNCTIONS, natives, &["pair", "one"])
    }

    fn checked(functions: &str, natives: &str, entries: &[&str]) -> Vec<NativeImplementation> {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!("{functions}{natives}"),
        });
        let module = crate::checked::check_source_with(sources, &mut |_entry: &str, implementation: &NativeImplementation| {
            Ok::<_, std::convert::Infallible>(Some(crate::native_source::fixture_source(implementation, &[]).into()))
        })
        .unwrap_or_else(|failure| match failure {
            crate::checked::CheckFailure::Source(error) => panic!("{error}"),
            crate::checked::CheckFailure::Read(never) => match never {},
        });
        entries
            .iter()
            .copied()
            .map(|name| {
                let entry = module.entry_named(name).unwrap();
                module.native_implementation(entry, BackendName::Metal).unwrap().clone()
            })
            .collect()
    }

    /// An entry that projects checks to exactly the implementation it would
    /// write by hand: its parameters, conditions (each weight, some weight,
    /// agreement between weights; folded terms and formals), scratch per
    /// weight at the `project` line, and launches named for the entry in the
    /// projection's order, its own placed `before` them.
    #[test]
    fn a_projecting_entry_checks_to_the_declaration_it_stands_for() {
        let projected = implementations(&format!("{PROJECTION}{PROJECTED}"));
        let mut written = implementations(WRITTEN);
        for (written, projected) in written.iter_mut().zip(&projected) {
            written.entry = projected.entry;
        }
        assert_eq!(projected, written);
    }

    /// A segmented entry checks to its weights' own rows summed (`sum`,
    /// `tiles`) and quantified (`any`), and an entry whose contract bounds
    /// its rows keeps only the launches and scratch those rows reach, and
    /// only the parameters they read.
    #[test]
    fn a_segmented_entry_checks_to_its_weights_and_its_contracts_rows() {
        let entries = ["seg", "seg16"];
        let projected = checked(SEGMENTED_FUNCTIONS, &format!("{SEGMENTED_PROJECTION}{SEGMENTED_PROJECTED}"), &entries);
        let mut written = checked(SEGMENTED_FUNCTIONS, SEGMENTED_WRITTEN, &entries);
        // A hand-written declaration has no projection to group: the
        // projected one's grouping is its weights one a segment.
        for (written, projected) in written.iter_mut().zip(&projected) {
            written.entry = projected.entry;
            written.segments = Some(vec![1, 1]);
        }
        assert_eq!(projected, written);
    }

    /// A segmented entry may group its weights into served pairs: a pair is
    /// one segment (its rows once in `tiles`, at the segment's own `SERVED`),
    /// and `SERVED` outside a quantifier is the most weights a segment serves.
    #[test]
    fn a_segment_may_serve_a_pair() {
        const FUNCTIONS: &str = "fn conv[O, K, N](x: &tensor[O, K] f32, b_weight: &tensor[N, K] BW, x_weight: &tensor[N, K] XW, c_weight: &tensor[N, K] CW) -> tensor[O, N + N] f32:\n    let mut result = tensor[O, N + N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N + N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    params (ROWS in [1, 2] default 1)\n    launch gemv when M < 2:\n        threadgroups (tiles(W, ceil_div(N * SERVED, ROWS)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (SERVED * 64)\n\n";
        const PROJECTED: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    project grouped segmented ((b_weight, x_weight), c_weight)\n";
        const WRITTEN: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    params (ROWS in [1, 2] default 1)\n    launch conv_gemv when O < 2:\n        threadgroups (ceil_div(N * 2, ROWS) + ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (128)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTION}{PROJECTED}"), &["conv"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["conv"]);
        written[0].entry = projected[0].entry;
        written[0].segments = Some(vec![2, 1]);
        assert_eq!(projected, written);
    }

    /// A `gathered` weight's rows are the extent of the port that selects them,
    /// and `GATHERED` says so.
    #[test]
    fn a_gathered_projection_has_the_extent_of_its_selecting_port() {
        const FUNCTIONS: &str = "fn rows[O, V, K, S](x: &tensor[O, K] f32, weight: &tensor[V, K] W, selected: &tensor[S] i32) -> tensor[O, S] f32:\n    let mut result = tensor[O, S] f32\n    parallel for row in 0..O:\n        parallel for col in 0..S:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    params (ROWS in [1, 2] default 1)\n    launch gemv when M < 2:\n        threadgroups (tiles(W, ceil_div(N, ROWS)) + GATHERED, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        const PROJECTED: &str = "native rows for metal from \"rows.metal\":\n    static (K)\n    project grouped single (weight) gathered selected\n";
        const WRITTEN: &str = "native rows for metal from \"rows.metal\":\n    static (K)\n    params (ROWS in [1, 2] default 1)\n    launch rows_gemv when O < 2:\n        threadgroups (ceil_div(S, ROWS) + 1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTION}{PROJECTED}"), &["rows"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["rows"]);
        written[0].entry = projected[0].entry;
        assert_eq!(projected, written);
    }

    /// A port selecting rows per activation row (`tensor[M, I, N]`, `I`
    /// static) takes the line's projection over the weight's first `N` rows
    /// where `I` is 0 and the routed `choice` over each row's selections
    /// where it is 1, every item and condition gated by `I`, the routed
    /// expansion's parameters, launches and buffers named apart, and its
    /// forms for rows that share an expert (`PER_ROW == 0`) left out.
    #[test]
    fn a_per_row_gathered_entry_takes_the_routed_choice_where_it_selects() {
        const FUNCTIONS: &str = "fn pick[O, V, D, S, I](x: &tensor[O, D] f32, weight: &tensor[V, D] W, selected: &tensor[O, I, S] i32) -> tensor[O, S] f32\n    where S <= V and I <= 1:\n    let mut result = tensor[O, S] f32\n    parallel for row in 0..O:\n        parallel for col in 0..S:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTIONS: &str = "projection grouped for metal:\n    params (SIMDGROUPS in [1, 2] default 1)\n    where K % 32 == 0\n    scratch partials bytes (M * N * 4) when M > 1\n    launch gemv when M < 2:\n        threadgroups (ceil_div(N, SIMDGROUPS) + GATHERED, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\nprojection routed for metal:\n    let GROUPS = ceil_div(most(W, N), SIMDGROUPS * 32) for choice\n    params (SIMDGROUPS in [2, 4] default 2 for choice)\n    where K + SHARED <= 32 for choice\n    scratch staged bytes (M * K * WIDTH) for choice\n    launch choice for choice when RECORDS == 0:\n        params (LANES in [16, 32] default 16)\n        threadgroups (GROUPS, M * K, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n        shared_bytes (WIDTH * LANES)\n    launch grouped for choice when PER_ROW == 0:\n        threadgroups (GROUPS, M * K, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n\n";
        const PROJECTED: &str = "native pick for metal from \"pick.metal\":\n    static (V, D, S, I)\n    project grouped single (weight) gathered selected\n    restate GATHER_SIMDGROUPS in [4, 2] default 4\n";
        const WRITTEN: &str = "native pick for metal from \"pick.metal\":\n    static (V, D, S, I)\n    params (SIMDGROUPS in [1, 2] default 1, GATHER_SIMDGROUPS in [4, 2] default 4)\n    where (I > 0 or D % 32 == 0) and (I == 0 or I <= 32)\n    scratch partials bytes (O * S * 4) when I == 0 and O > 1\n    scratch gather_staged bytes (O * I * D) when I > 0\n    launch pick_gemv when I == 0 and O < 2:\n        threadgroups (ceil_div(S, SIMDGROUPS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch pick_gather_choice when I > 0:\n        params (GATHER_LANES in [16, 32] default 16)\n        threadgroups (ceil_div(S, GATHER_SIMDGROUPS * 32), O * I, 1)\n        threads_per_threadgroup (32 * GATHER_SIMDGROUPS, 1, 1)\n        shared_bytes (D * GATHER_LANES)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTIONS}{PROJECTED}"), &["pick"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["pick"]);
        written[0].entry = projected[0].entry;
        assert_eq!(projected, written);

        let refused = |native: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile { path: "ops.seismic".to_owned(), text: format!("{FUNCTIONS}{PROJECTIONS}{native}") });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        // Its choices are a static dimension, and its rows split no further.
        refused(&PROJECTED.replace("static (V, D, S, I)", "static (V, D, S)"), "one of a static `I`");
        refused(&PROJECTED.replace("single", "segmented"), "is `paired` or `single`");
    }

    /// An entry may restate a projection's parameter or a launch's, and only one that exists.
    #[test]
    fn an_entry_may_restate_a_projection_or_launch_parameter() {
        const FUNCTIONS: &str = "fn conv[O, K, N](x: &tensor[O, K] f32, weight: &tensor[N, K] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    params (ROWS in [1, 2, 4] default 1)\n    launch gemv when M < 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (tiles(W, ceil_div(N, ROWS * LANES)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch gemv_rows when M >= 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (tiles(W, ceil_div(N, ROWS * LANES)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        const PROJECTED: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    project grouped single (weight)\n    restate ROWS in [4, 2] default 4\n    restate gemv_rows LANES in [2, 1] default 2\n";
        const WRITTEN: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    params (ROWS in [4, 2] default 4)\n    launch conv_gemv when O < 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (ceil_div(N, ROWS * LANES), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch conv_gemv_rows when O >= 2:\n        params (LANES in [2, 1] default 2)\n        threadgroups (ceil_div(N, ROWS * LANES), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTION}{PROJECTED}"), &["conv"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["conv"]);
        written[0].entry = projected[0].entry;
        assert_eq!(projected, written);

        let refused = |restate: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{FUNCTIONS}{PROJECTION}native conv for metal from \"conv.metal\":\n    static (K, N)\n    project grouped single (weight)\n    {restate}\n"),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        // A restatement keeps the flags and admits no value the projection does not.
        refused("restate ROWS in [8, 2] default 8", "restates the projection's parameter");
        refused("restate arithmetic ROWS in [2, 1] default 2", "restates the projection's parameter");
        refused("restate gemv_rows LANES in [4, 1] default 4", "restates the projection's parameter");
        // And only a parameter the projection, or the launch, has.
        refused("restate WIDE in [0, 1] default 0", "has no parameter `WIDE`");
        refused("restate gemv_pair LANES in [2, 1] default 2", "has no `single` launch `gemv_pair`");
        refused("restate gemv_rows ROWS in [2, 1] default 2", "with a parameter `ROWS`");
    }

    /// A projection's `mapping` is a set of restates an entry takes by name: the entry
    /// that takes it declares what it would have restated line by line.
    #[test]
    fn an_entry_may_take_a_projections_mapping() {
        const FUNCTIONS: &str = "fn conv[O, K, N](x: &tensor[O, K] f32, weight: &tensor[N, K] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    params (ROWS in [1, 2, 4] default 1)\n    launch gemv when M < 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (tiles(W, ceil_div(N, ROWS * LANES)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch gemv_rows when M >= 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (tiles(W, ceil_div(N, ROWS * LANES)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    mapping fast:\n        restate ROWS in [4, 2] default 4\n        restate gemv_rows LANES in [2, 1] default 2\n\n";
        const PROJECTED: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    project grouped single (weight)\n    restate fast\n";
        const WRITTEN: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    params (ROWS in [4, 2] default 4)\n    launch conv_gemv when O < 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (ceil_div(N, ROWS * LANES), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch conv_gemv_rows when O >= 2:\n        params (LANES in [2, 1] default 2)\n        threadgroups (ceil_div(N, ROWS * LANES), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTION}{PROJECTED}"), &["conv"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["conv"]);
        written[0].entry = projected[0].entry;
        assert_eq!(projected, written);

        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!("{FUNCTIONS}{PROJECTION}{}", PROJECTED.replace("restate fast", "restate slow")),
        });
        let error = check_source(sources).expect_err("an unknown mapping").to_string();
        assert!(error.contains("has no mapping `slow`"), "{error}");
    }

    /// An entry-wide restate of a launch parameter is one parameter of the
    /// entry that every launch declaring it reads.
    #[test]
    fn an_entry_may_share_a_launch_parameter() {
        const FUNCTIONS: &str = "fn conv[O, K, N](x: &tensor[O, K] f32, weight: &tensor[N, K] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    launch gemv when M < 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (tiles(W, ceil_div(N, LANES)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch gemv_rows when M >= 2:\n        params (LANES in [1, 2] default 1)\n        threadgroups (tiles(W, ceil_div(N, LANES)), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        const PROJECTED: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    project grouped single (weight)\n    restate LANES in [2, 1] default 2\n";
        const WRITTEN: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    params (LANES in [2, 1] default 2)\n    launch conv_gemv when O < 2:\n        reads (LANES)\n        threadgroups (ceil_div(N, LANES), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch conv_gemv_rows when O >= 2:\n        reads (LANES)\n        threadgroups (ceil_div(N, LANES), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTION}{PROJECTED}"), &["conv"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["conv"]);
        written[0].entry = projected[0].entry;
        assert_eq!(projected, written);
    }

    /// An item gated `for rms` is the declaration's only where its rows are of that kind.
    #[test]
    fn a_kind_of_rows_gates_an_item() {
        const FUNCTIONS: &str = "fn conv[O, K, N](x: &tensor[O, K] f32, weight: &tensor[N, K] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    scratch normalized bytes (M * K * 2) when M > 1 for rms\n    launch gemv:\n        threadgroups (tiles(W, N), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        const WRITTEN: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    scratch normalized bytes (O * K * 2) when O > 1\n    launch conv_gemv:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        const PLAIN: &str = "native conv for metal from \"conv.metal\":\n    static (K, N)\n    launch conv_gemv:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        for (rows, written) in [("rms", WRITTEN), ("plain", PLAIN)] {
            let projected = format!("native conv for metal from \"conv.metal\":\n    static (K, N)\n    project grouped single (weight) rows {rows}\n");
            let projected = checked(FUNCTIONS, &format!("{PROJECTION}{projected}"), &["conv"]);
            let mut written = checked(FUNCTIONS, written, &["conv"]);
            written[0].entry = projected[0].entry;
            assert_eq!(projected, written, "rows {rows}");
        }
    }

    /// A form's values for an entry are those its projection's admission
    /// leaves over the entry's facts: a `where` line (the projection's, or a
    /// launch's over its own parameter) that the facts reduce to a condition
    /// on one parameter is its range, a default it excludes giving way to the
    /// first listed value it admits. No entry restates a form.
    #[test]
    fn a_form_has_the_values_its_admission_leaves_an_entry() {
        const FUNCTIONS: &str = "fn mix[O, S, H, N](x: &tensor[O, S * H] f32, weight: &tensor[N, S * H] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection parted for metal:\n    params (form TALL in [1, 0] default 1)\n    where (TALL == 0 or (PARTED == 0 and K % 64 == 0))\n    launch gemv:\n        params (TILED in [0, 1] default 0)\n        where (TILED == 0 or PARTED == 0)\n        threadgroups (P * tiles(W, N), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes ((TALL + TILED) * 64)\n\n";
        const PARTED: &str = "native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    params (form TALL in [0] default 0)\n    launch mix_gemv:\n        params (TILED in [0] default 0)\n        threadgroups (S * N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes ((TALL + TILED) * 64)\n";
        const WHOLE: &str = "native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    params (form TALL in [1, 0] default 1)\n    where (TALL == 0 or S * H % 64 == 0)\n    launch mix_gemv:\n        params (TILED in [0, 1] default 0)\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes ((TALL + TILED) * 64)\n";
        for (parts, written) in [(" parts S", PARTED), ("", WHOLE)] {
            let projected = format!("native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    project parted single (weight){parts}\n");
            let projected = checked(FUNCTIONS, &format!("{PROJECTION}{projected}"), &["mix"]);
            let mut written = checked(FUNCTIONS, written, &["mix"]);
            written[0].entry = projected[0].entry;
            assert_eq!(projected, written, "parts `{parts}`");
        }

        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!("{FUNCTIONS}{PROJECTION}native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    project parted single (weight)\n    restate form TALL in [0] default 0\n"),
        });
        let error = check_source(sources).expect_err("a restated form").to_string();
        assert!(error.contains("`TALL` is a form of the projection"), "{error}");
    }

    /// No entry of the kernels restates a form of its projection, in its own
    /// lines or through a mapping: the kernels' declarations, read as they
    /// are written.
    #[test]
    fn no_kernel_entry_names_a_form_of_its_projection() {
        use crate::syntax::ast::Decl;
        use std::collections::{BTreeMap, BTreeSet};

        let directory =
            std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
                .join("../../engine/kernels/kernels");
        let mut decls = Vec::new();
        for file in std::fs::read_dir(&directory).unwrap_or_else(|error| panic!("{}: {error}", directory.display())) {
            let path = file.expect("a directory entry").path();
            if path.extension().is_some_and(|extension| extension == "seismic") {
                let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                let file = crate::syntax::parse(&text, &BTreeMap::new(), &BTreeMap::new())
                    .unwrap_or_else(|error| panic!("{}: {error:?}", path.display()));
                decls.extend(file.decls);
            }
        }
        let projections: BTreeMap<(&str, &str), &crate::syntax::ast::ProjectionDecl> = decls
            .iter()
            .filter_map(|decl| match decl {
                Decl::Projection(projection) => {
                    Some(((projection.name.name.as_str(), projection.target.name.as_str()), projection))
                }
                _ => None,
            })
            .collect();
        let mut projecting = 0;
        for decl in &decls {
            let Decl::Native(native) = decl else { continue };
            let Some(project) = &native.project else { continue };
            projecting += 1;
            let target = native.target.name.as_str();
            // The line's projection, and the one a per-row gathered entry's
            // selected rows are projected by (its `GATHER_` restates).
            let taken = projections
                .get(&(project.projection.name.as_str(), target))
                .copied()
                .unwrap_or_else(|| panic!("`{}` projects by no projection for `{target}`", native.function.name));
            let per_row = projections.get(&(super::PER_ROW.0, target)).copied();
            let forms: BTreeSet<String> = std::iter::once(taken)
                .chain(per_row)
                .flat_map(|projection| {
                    let own = projection.params.iter().map(|param| &param.item);
                    own.chain(projection.launches.iter().flat_map(|launch| &launch.launch.params))
                })
                .filter(|param| param.form)
                .flat_map(|param| [param.name.name.clone(), format!("{}{}", super::GATHER.to_uppercase(), param.name.name)])
                .collect();
            let mappings = project.mappings.iter().flat_map(|name| {
                let mapping = taken.mappings.iter().find(|mapping| mapping.name.name == name.name);
                mapping.into_iter().flat_map(|mapping| &mapping.restated)
            });
            for restated in project.restated.iter().chain(mappings) {
                assert!(
                    !restated.param.form && !forms.contains(&restated.param.name.name),
                    "`{}` for `{target}` restates `{}`, a form of its projection",
                    native.function.name,
                    restated.param.name.name
                );
            }
        }
        assert!(projecting > 0, "no kernel declaration projects: {}", directory.display());
    }

    /// `parts P` names the parts of the columns (`P`, 1 where the line names
    /// none) and `PARTED` whether it names them, so an item over parts folds
    /// away for every other entry.
    #[test]
    fn an_entry_may_split_its_columns_into_parts() {
        const FUNCTIONS: &str = "fn mix[O, S, H, N](x: &tensor[O, S * H] f32, weight: &tensor[N, S * H] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection parted for metal:\n    scratch partials bytes (M * P * N * 4) when M < 4 and PARTED == 1\n    launch gemv:\n        threadgroups (P * tiles(W, N), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        const PARTED: &str = "native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    scratch partials bytes (O * S * N * 4) when O < 4\n    launch mix_gemv:\n        threadgroups (S * N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        const WHOLE: &str = "native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    launch mix_gemv:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        for (parts, written) in [(" parts S", PARTED), ("", WHOLE)] {
            let projected = format!("native mix for metal from \"mix.metal\":\n    static (S, H, N)\n    project parted single (weight){parts}\n");
            let projected = checked(FUNCTIONS, &format!("{PROJECTION}{projected}"), &["mix"]);
            let mut written = checked(FUNCTIONS, written, &["mix"]);
            written[0].entry = projected[0].entry;
            assert_eq!(projected, written, "parts `{parts}`");
        }
    }

    /// A plane set is a `single` projection's weight: its rows and width are
    /// its `over`'s, `PLANES` is 1, and its `form(W, ..)`, `operands(W)` and
    /// `bound(W)` are 0, so every item gated off a plane set folds away with
    /// the parameters only it read, and the forms gated on one stay.
    const PLANE_FUNCTIONS: &str = "fn read[O, V, D](x: &tensor[O, D] f32, top: &tensor[V, D] u32, scales: &tensor[V, D] f16) -> tensor[O, V] f32:\n    let mut result = tensor[O, V] f32\n    parallel for row in 0..O:\n        parallel for col in 0..V:\n            result[row, col] = x[row, 0]\n    return result\n\n";
    const PLANE_PROJECTION: &str = "projection planar for metal:\n    params (form INT8 in [0, 1] default 0)\n    where (INT8 == 0 or each(W, form(W, INT8) == 1 and operands(W) == 0 and bound(W) == 0))\n    where (PLANES == 1 or K % 64 == 0)\n    scratch quantized bytes (M * K) when INT8 == 1 and PLANES == 0\n    launch gemv when PLANES == 0:\n        params (ROWS in [1, 2] default 1)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch planes when PLANES == 1:\n        params (WARPS in [8, 4] default 8)\n        threadgroups (ceil_div(N, WARPS), ceil_div(M, 8), 1)\n        threads_per_threadgroup (WARPS * 32, 1, 1)\n\n";

    #[test]
    fn a_plane_set_projects_by_the_forms_gated_on_it() {
        let projected = format!(
            "{PLANE_PROJECTION}native read for metal from \"read.metal\":\n    static (V, D)\n    project planar single (planes (top, scales) over (V, D))\n    restate planes WARPS in [4] default 4\n"
        );
        let projected = checked(PLANE_FUNCTIONS, &projected, &["read"]);
        const WRITTEN: &str = "native read for metal from \"read.metal\":\n    static (V, D)\n    launch read_planes:\n        params (WARPS in [4] default 4)\n        threadgroups (ceil_div(V, WARPS), ceil_div(O, 8), 1)\n        threads_per_threadgroup (WARPS * 32, 1, 1)\n";
        let mut written = checked(PLANE_FUNCTIONS, WRITTEN, &["read"]);
        written[0].entry = projected[0].entry;
        assert_eq!(projected, written);
    }

    /// A plane set is one `single` projection's weight, of the entry's ports,
    /// giving its rows and width, through a projection that reads `PLANES`.
    #[test]
    fn a_plane_set_is_refused_where_it_cannot_project() {
        let refused = |projection: &str, project: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!(
                    "{PLANE_FUNCTIONS}{projection}native read for metal from \"read.metal\":\n    static (V, D)\n    {project}\n"
                ),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        const BLIND: &str = "projection planar for metal:\n    launch gemv:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        refused(BLIND, "project planar single (planes (top, scales) over (V, D))", "never reads `PLANES`");
        refused(PLANE_PROJECTION, "project planar single (planes (top, nope) over (V, D))", "`nope` is not a port");
        refused(PLANE_PROJECTION, "project planar single (planes (top, scales))", "gives its rows and width");
        refused(PLANE_PROJECTION, "project planar single (top over (V, D))", "only a CPU projection's weights");
        refused(
            PLANE_PROJECTION,
            "project planar paired (planes (top) over (V, D), planes (scales) over (V, D))",
            "the one weight of a GPU `single` projection",
        );
    }

    /// `ordered` is the fact `ORDERED`: a projection offers the forms that
    /// accumulate in the reference order to the entry that says so, and the
    /// others to every other entry.
    #[test]
    fn an_ordered_entry_takes_the_ordered_launches() {
        const FUNCTIONS: &str = "fn rows[O, K, N](x: &tensor[O, K] f32, weight: &tensor[N, K] W) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    launch gemv when each(W, ORDERED == 0):\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch ordered when any(W, ORDERED == 1):\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (8, 1, 1)\n\n";
        for (project, launch, threads) in [("", "gemv", 32), (" ordered (weight)", "ordered", 8)] {
            let projected = checked(FUNCTIONS, &format!("{PROJECTION}native rows for metal from \"rows.metal\":\n    static (K, N)\n    project grouped single (weight){project}\n"), &["rows"]);
            let mut written = checked(
                FUNCTIONS,
                &format!("native rows for metal from \"rows.metal\":\n    static (K, N)\n    launch rows_{launch}:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup ({threads}, 1, 1)\n"),
                &["rows"],
            );
            written[0].entry = projected[0].entry;
            assert_eq!(projected, written, "{launch}");
        }
    }

    /// A segment of a segmented projection may be one row, a `tensor[K]`.
    #[test]
    fn a_segment_may_be_one_row() {
        const FUNCTIONS: &str = "fn logits[O, K, E](x: &tensor[O, K] f32, router: &tensor[E, K] RW, gate: &tensor[K] GW) -> tensor[O, E + 1] f32:\n    let mut result = tensor[O, E + 1] f32\n    parallel for row in 0..O:\n        parallel for col in 0..E + 1:\n            result[row, col] = x[row, 0]\n    return result\n\n";
        const PROJECTION: &str = "projection grouped for metal:\n    launch gemv when M < 2:\n        threadgroups (tiles(W, N), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";
        const PROJECTED: &str = "native logits for metal from \"logits.metal\":\n    static (K, E)\n    project grouped segmented (router, gate)\n";
        const WRITTEN: &str = "native logits for metal from \"logits.metal\":\n    static (K, E)\n    launch logits_gemv when O < 2:\n        threadgroups (E + 1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let projected = checked(FUNCTIONS, &format!("{PROJECTION}{PROJECTED}"), &["logits"]);
        let mut written = checked(FUNCTIONS, WRITTEN, &["logits"]);
        written[0].entry = projected[0].entry;
        written[0].segments = Some(vec![1, 1]);
        assert_eq!(projected, written);
        // Only a segmented projection's weights may be one row.
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!(
                "{FUNCTIONS}{PROJECTION}native logits for metal from \"logits.metal\":\n    static (K, E)\n    project grouped paired (router, gate)\n"
            ),
        });
        let error = check_source(sources).expect_err("a paired one-row weight").to_string();
        assert!(error.contains("is not a `tensor[N, K]`"), "{error}");
    }

    /// `publishes (result N)` makes the operator's formed rows the entry's
    /// result: the projection's `staged` scratch and its stage launch, which
    /// read `PUBLISHED`, are the entry's own then, and a line's `unless` is
    /// held by every launch it expands to.
    const PUBLISHED_FUNCTIONS: &str = "fn route[O, K, N](x: &tensor[O, K] f32, weight: &tensor[N, K] W, next: &tensor[1] i32) -> (tensor[O, N] f32, tensor[O, K] f32):\n    let mut result = tensor[O, N] f32\n    let mut rows = tensor[O, K] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    parallel for row in 0..O:\n        parallel for col in 0..K:\n            rows[row, col] = x[row, col]\n    return (result, rows)\n\n";
    const PUBLISHED_PROJECTION: &str = "projection grouped for metal:\n    scratch staged bytes (M * K * 2) when PUBLISHED == 0\n    launch stage when PUBLISHED == 0:\n        threadgroups (M, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch gemv:\n        threadgroups (tiles(W, N), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\n";

    #[test]
    fn a_published_result_replaces_the_staged_rows_and_a_guard_holds_every_launch() {
        const PUBLISHED: &str = "native route for metal from \"route.metal\":\n    static (K, N)\n    project grouped single (weight) rows rms publishes (result 1)\n    unless (next)\n";
        const PUBLISHED_WRITTEN: &str = "native route for metal from \"route.metal\":\n    static (K, N)\n    launch route_gemv:\n        threadgroups (N, 1, 1)\n        unless (next)\n        threads_per_threadgroup (32, 1, 1)\n";
        const STAGED: &str = "native route for metal from \"route.metal\":\n    static (K, N)\n    project grouped single (weight) rows rms\n";
        const STAGED_WRITTEN: &str = "native route for metal from \"route.metal\":\n    static (K, N)\n    scratch staged bytes (O * K * 2)\n    launch route_stage:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch route_gemv:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        for (projected, written) in [(PUBLISHED, PUBLISHED_WRITTEN), (STAGED, STAGED_WRITTEN)] {
            let projected = checked(PUBLISHED_FUNCTIONS, &format!("{PUBLISHED_PROJECTION}{projected}"), &["route"]);
            let mut written = checked(PUBLISHED_FUNCTIONS, written, &["route"]);
            written[0].entry = projected[0].entry;
            assert_eq!(projected, written);
        }
    }

    /// A published result is a `tensor[M, K]` of the line's rows and width,
    /// and only a GPU projection of `rms` or `staged` rows publishes.
    #[test]
    fn a_published_result_is_refused_where_it_is_not_the_rows() {
        let refused = |project: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!(
                    "{PUBLISHED_FUNCTIONS}{PUBLISHED_PROJECTION}native route for metal from \"route.metal\":\n    static (K, N)\n    {project}\n"
                ),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        refused("project grouped single (weight) rows rms publishes (result 0)", "is not a `tensor[M, K]`");
        refused("project grouped single (weight) rows rms publishes (result 2)", "is not a `tensor[M, K]`");
        refused("project grouped single (weight) publishes (result 1)", "publishes them");
    }

    /// A projection is found by its name and the target of the entry's declaration.
    #[test]
    fn a_projection_is_per_target() {
        let refused = |natives: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{FUNCTIONS}{PROJECTION}{natives}"),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        refused(
            "native pair for cuda from \"pair.cu\":\n    static (K, N, GB, UB)\n    project lanes paired (gate_weight, up_weight)\n",
            "no projection `lanes` for `cuda`",
        );
        refused(PROJECTION, "is declared twice for `metal`");
    }

    #[test]
    fn a_projection_refuses_what_it_cannot_expand() {
        let refused = |natives: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{FUNCTIONS}{PROJECTION}{natives}"),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        let native = |project: &str, launch: &str| {
            format!("native pair for metal from \"pair.metal\":\n    {project}\n    launch pair_stage{launch}:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n")
        };
        refused(&native("project lanes paired (gate_weight)", ""), "takes 2 weight(s)");
        refused(&native("project lanes paired (gate_weight, up_weight)", " before reduce"), "no `paired` launch `reduce`");
        refused(&native("project lanes twice (gate_weight, up_weight)", ""), "not a projection variant");
        refused(&native("project lanes paired (gate_weight, x)", ""), "share `N` and `K`");
        refused(&native("project lanes paired (gate_weight, up_weight) groups K", ""), "a weight over `K` groups");
        refused(&native("scratch staged bytes (4)", " before gemv"), "does not project");
        refused(&native("project lanes segmented (gate_weight, up_weight)", ""), "`SPLITS` has no `segmented` value");
        refused(&native("project lanes paired (gate_weight, up_weight) rows wide", ""), "not a kind of rows");
    }

    /// The routed entries: an expert-stacked `choice` reading record ports, one
    /// with the shared expert beside it, and a `grouped` one over stacked and
    /// over dense weights. Their extents are their own names; a formal the
    /// entry's facts decide (`SHARED`, `RECORDS`, `STACKED`, `FIRST`) folds
    /// away, with every launch and buffer it decides false, and a grid read
    /// from a buffer the entry does not have.
    const ROUTED_FUNCTIONS: &str = "fn pick[M, H, E, K, F, XS](x: &tensor[M, H] f32, routes: &tensor[M, K] i32, sources: &tensor[XS, M, K] i32, expert_up: &tensor[E, F, H] UW, tier_up: &tensor[XS, F, H] UW) -> tensor[M, K, F] f32:\n    let mut result = tensor[M, K, F] f32\n    parallel for row in 0..M:\n        parallel for choice in 0..K:\n            parallel for col in 0..F:\n                result[row, choice, col] = x[row, 0]\n    return result\n\nfn both[M, H, E, K, F, S](x: &tensor[M, H] f32, routes: &tensor[M, K] i32, expert_up: &tensor[E, F, H] UW, shared_up: &tensor[S, H] SW) -> tensor[M, K, F] f32:\n    let mut result = tensor[M, K, F] f32\n    parallel for row in 0..M:\n        parallel for choice in 0..K:\n            parallel for col in 0..F:\n                result[row, choice, col] = x[row, 0]\n    return result\n\nfn group[M, H, E, F, B](x: &tensor[M, H] f32, bias: &tensor[H] f32, expert_up: &tensor[E, F, H] UW, expert_down: &tensor[E, H, F] DW) -> tensor[B, H] f32:\n    let mut result = tensor[B, H] f32\n    parallel for row in 0..B:\n        parallel for col in 0..H:\n            result[row, col] = x[0, col]\n    return result\n\nfn dense[M, H, S](x: &tensor[M, H] f32, shared_up: &tensor[S, H] UW, shared_down: &tensor[H, S] DW) -> tensor[M, H] f32:\n    let mut result = tensor[M, H] f32\n    parallel for row in 0..M:\n        parallel for col in 0..H:\n            result[row, col] = x[row, col]\n    return result\n\n";
    const ROUTED_PROJECTION: &str = "projection routed for metal:\n    let GROUPS = ceil_div(most(W, N), SIMDGROUPS * 32) for choice\n    params (SIMDGROUPS in [2, 4] default 2 for choice, TILE in [32, 64] default 32 for grouped)\n    where K + SHARED <= 32 for choice\n    scratch staged bytes (M * WIDTH * 2) when STACKED == 1 and any(W, FIRST == 1 and operands(W) == 1) for grouped\n    each W:\n        scratch factors bytes (N * 4) when FIRST == 0 for grouped\n    scratch expand_grid bytes (12) when STACKED == 1 for grouped\n    launch choice for choice when RECORDS == 0 or XS > 0:\n        threadgroups (GROUPS, M * K + SHARED, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n        shared_bytes (WIDTH * (1 + SHARED * (M - 1)))\n    launch resident for choice when RECORDS == 1 and XS == 0:\n        records (W)\n        threadgroups (GROUPS, M * K + SHARED, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n    launch expand for grouped:\n        threadgroups (ceil_div(N * SERVED, TILE), STACKED * B + (1 - STACKED) * ceil_div(M, TILE), 1)\n        grid (expand_grid)\n        threads_per_threadgroup (128, 1, 1)\n\n";
    const ROUTED_PROJECTED: &str = "native pick for metal from \"pick.metal\":\n    static (H, E, K, F, XS)\n    records (expert_up, tier_up)\n    project routed choice (expert_up)\n\nnative both for metal from \"both.metal\":\n    static (H, E, K, F, S)\n    project routed choice (expert_up) shared (shared_up)\n\nnative group for metal from \"group.metal\":\n    static (H, E, F)\n    project routed grouped (expert_up) down (expert_down)\n\nnative dense for metal from \"dense.metal\":\n    static (H, S)\n    project routed grouped (shared_up) down (shared_down)\n";
    const ROUTED_WRITTEN: &str = "native pick for metal from \"pick.metal\":\n    static (H, E, K, F, XS)\n    params (SIMDGROUPS in [2, 4] default 2)\n    records (expert_up, tier_up)\n    where K <= 32\n    launch pick_choice when XS > 0:\n        threadgroups (ceil_div(F, SIMDGROUPS * 32), M * K, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n        shared_bytes (H)\n    launch pick_resident when XS == 0:\n        records (expert_up, tier_up)\n        threadgroups (ceil_div(F, SIMDGROUPS * 32), M * K, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n\nnative both for metal from \"both.metal\":\n    static (H, E, K, F, S)\n    params (SIMDGROUPS in [2, 4] default 2)\n    where K + 1 <= 32\n    launch both_choice:\n        threadgroups (ceil_div(max(F, S), SIMDGROUPS * 32), M * K + 1, 1)\n        threads_per_threadgroup (32 * SIMDGROUPS, 1, 1)\n        shared_bytes (H * (1 + (M - 1)))\n\nnative group for metal from \"group.metal\":\n    static (H, E, F)\n    params (TILE in [32, 64] default 32)\n    scratch staged bytes (M * H * 2) when operands(UW) == 1\n    scratch expert_down_factors bytes (H * 4)\n    scratch expand_grid bytes (12)\n    launch group_expand:\n        threadgroups (ceil_div(F, TILE), B, 1)\n        grid (expand_grid)\n        threads_per_threadgroup (128, 1, 1)\n\nnative dense for metal from \"dense.metal\":\n    static (H, S)\n    params (TILE in [32, 64] default 32)\n    scratch shared_down_factors bytes (H * 4)\n    launch dense_expand:\n        threadgroups (ceil_div(S, TILE), ceil_div(M, TILE), 1)\n        threads_per_threadgroup (128, 1, 1)\n";

    #[test]
    fn a_routed_entry_checks_to_the_declaration_its_facts_decide() {
        let entries = ["pick", "both", "group", "dense"];
        let projected = checked(ROUTED_FUNCTIONS, &format!("{ROUTED_PROJECTION}{ROUTED_PROJECTED}"), &entries);
        let mut written = checked(ROUTED_FUNCTIONS, ROUTED_WRITTEN, &entries);
        for (written, projected) in written.iter_mut().zip(&projected) {
            written.entry = projected.entry;
        }
        assert_eq!(projected, written);
    }

    #[test]
    fn a_routed_projection_refuses_what_its_variant_does_not_take() {
        let refused = |project: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{ROUTED_FUNCTIONS}{ROUTED_PROJECTION}native group for metal from \"group.metal\":\n    static (H, E, F)\n    {project}\n"),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        refused("project routed grouped (expert_up)", "names its `down` weight");
        refused("project routed choice (expert_up) down (expert_down)", "names its `down` weight");
        refused("project routed grouped (expert_up) shared (expert_down) down (expert_down)", "projects a shared expert");
        refused("project routed grouped (expert_up, expert_up, expert_up) down (expert_down)", "takes 1 to 2 weight(s)");
        refused("project routed grouped (bias) down (expert_down)", "not a `tensor[N, K]` or `tensor[E, N, K]`");
    }

    /// A CPU entry names its weights' element parameters and gives the
    /// formals; it checks to the declaration it would write by hand.
    #[test]
    fn a_cpu_entry_projects_over_the_formals_it_gives() {
        const ROWS: &str = "fn rows[M, K, N, G](x: &tensor[M, K] f32, w: &tensor[G * N, K] RW) -> tensor[M, G * N] f32:\n    let mut result = tensor[M, G * N] f32\n    parallel for row in 0..M:\n        parallel for col in 0..G * N:\n            result[row, col] = x[row, 0] + f32(w[col, 0])\n    return result\n\n";
        const OPERATOR: &str = "projection quantized for cpu:\n    params (ROWS in [8, 4, 2, 1] default 8, arithmetic form INT8 in [0, 1] default 0)\n    where (INT8 == 0 or any(W, form(W, INT8) == 1))\n    scratch quantized bytes (G * ceil_div(M, Q8_ROWS) * Q8_ROWS * ceil_div(K, Q8_VALUES) * Q8_BYTES) for tiled\n    scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) for rows\n\n";
        const LAUNCH: &str = "    launch rows_stage:\n        threadgroups (M, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";
        let implementation = |natives: String| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{ROWS}{natives}"),
            });
            let module = crate::native_source::check_fixture(sources, &[]).unwrap_or_else(|error| panic!("{error}"));
            let entry = module.entry_named("rows").unwrap();
            module.native_implementation(entry, BackendName::Cpu).unwrap().clone()
        };
        let projected = implementation(format!(
            "{OPERATOR}native rows for cpu from \"rows.rs\":\n    static (G)\n    where G >= 1\n    project quantized tiled (RW) over (M, K) groups G\n{LAUNCH}"
        ));
        let mut written = implementation(format!(
            "native rows for cpu from \"rows.rs\":\n    static (G)\n    params (ROWS in [8, 4, 2, 1] default 8, arithmetic form INT8 in [0, 1] default 0)\n    where G >= 1 and (INT8 == 0 or form(RW, INT8) == 1)\n    scratch quantized bytes (G * ceil_div(M, Q8_ROWS) * Q8_ROWS * ceil_div(K, Q8_VALUES) * Q8_BYTES)\n{LAUNCH}"
        ));
        written.entry = projected.entry;
        assert_eq!(projected, written);
    }

    /// A weight with rows of its own gets its own buffer, named for it; the
    /// others share the buffer of the rows the line gives, which only a line
    /// giving them has.
    #[test]
    fn weights_with_rows_of_their_own_get_their_own_scratch() {
        const PAIR: &str = "fn pair[M, K, F, S](x: &tensor[M, K] f32, expert: &tensor[F, K] EW, shared: &tensor[S, K] SW) -> tensor[M, F] f32:\n    let mut result = tensor[M, F] f32\n    parallel for row in 0..M:\n        parallel for col in 0..F:\n            result[row, col] = x[row, 0] + f32(expert[col, 0]) + f32(shared[0, 0])\n    return result\n\n";
        const OPERATOR: &str = "projection quantized for cpu:\n    scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) for rows\n    each W:\n        scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) for rows\n\n";
        let scratch = |line: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{PAIR}{OPERATOR}native pair for cpu from \"pair.rs\":\n    {line}\n    launch pair_stage:\n        threadgroups (M, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n"),
            });
            let module = crate::native_source::check_fixture(sources, &[]).unwrap_or_else(|error| panic!("{error}"));
            let entry = module.entry_named("pair").unwrap();
            let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
            implementation.scratch.iter().map(|buffer| (buffer.name.clone(), format!("{:?}", buffer.bytes))).collect::<Vec<_>>()
        };
        let named = |buffers: Vec<(String, String)>| buffers.into_iter().map(|(name, _)| name).collect::<Vec<_>>();
        assert_eq!(named(scratch("project quantized rows (EW over (M * K, F), SW over (M, S))")), ["ew_quantized", "sw_quantized"]);
        let mixed = scratch("project quantized rows (EW, SW over (M * F, S)) over (M, K)");
        assert_eq!(named(mixed.clone()), ["quantized", "sw_quantized"]);
        assert_ne!(mixed[0].1, mixed[1].1, "each buffer is of its own rows");
    }

    /// A projection that quantizes refuses a weight that multiplies no rows.
    #[test]
    fn a_weight_without_rows_is_refused() {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: "fn pair[M, K](x: &tensor[M, K] f32, expert: &tensor[M, K] EW, shared: &tensor[M, K] SW) -> tensor[M, K] f32:\n    let mut result = tensor[M, K] f32\n    parallel for row in 0..M:\n        parallel for col in 0..K:\n            result[row, col] = x[row, 0]\n    return result\n\nprojection quantized for cpu:\n    scratch quantized bytes (M) for rows\n\nnative pair for cpu from \"pair.rs\":\n    project quantized rows (EW over (M, K), SW)\n    launch pair_stage:\n        threadgroups (M, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n"
                .to_owned(),
        });
        let error = check_source(sources).expect_err("a weight without rows").to_string();
        assert!(error.contains("`SW` multiplies no rows"), "{error}");
    }

    /// The CPU operator's operand buffers, over the formals of `rows`
    /// (`a_cpu_entry_projects_over_the_formals_it_gives`), and its `for
    /// emits` buffers.
    const OPERAND_ROWS: &str = "fn rows[M, K, N](x: &tensor[M, K] f32, picked: &tensor[M] i32, w: &tensor[N, K] RW) -> tensor[M, N] f32:
    let mut result = tensor[M, N] f32
    parallel for row in 0..M:
        parallel for col in 0..N:
            result[row, col] = x[row, 0] + f32(w[col, 0])
    return result

";
    const OPERAND_PARAMS: &str = "projection quantized for cpu:\n    params (ROWS in [8, 4, 2, 1] default 8, arithmetic form INT8 in [0, 1] default 0 for rows)\n    where (INT8 == 0 or any(W, form(W, INT8) == 1)) for rows\n";
    const OPERAND_EMITS: &str = "    scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) operand for emits\n    scratch rows bytes (M * K * 4) operand rows for emits\n    scratch arrivals bytes (M * 4) sync for emits\n";
    const OPERAND_FORM: &str = "    launch form for rows:\n        threadgroups (M, ceil_div(K, Q8_VALUES * M), 1)\n        threads_per_threadgroup (1, 1, 1)\n        shared_bytes (K * 4)\n\n";
    const OPERAND_LAUNCHES: &str = "    launch rows_project:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";

    const NORMED_ROWS: &str = "fn normed[M, K, N](x: &tensor[M, K] f32, norm: &tensor[K] f32, w: &tensor[N, K] RW, eps: f32) -> tensor[M, N] f32:\n    let mut result = tensor[M, N] f32\n    parallel for row in 0..M:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0] + f32(w[col, 0])\n    return result\n\n";

    /// The operator's consumer forms: the Q8 blocks, paired at one row, and
    /// the formed rows, NVFP4 operands where every weight takes them.
    fn operand_buffers() -> String {
        format!(
            "    scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) operand when (M == 1) formed (INT8 == 1) for rows\n    scratch rows bytes (M * K * 4) operand rows nvfp4 (each(W, operands(W) == 1)) for rows\n{OPERAND_EMITS}{OPERAND_FORM}"
        )
    }

    /// `rows` declared for the CPU with `native`'s lines, over the operator
    /// whose buffers and launches are `operator`.
    fn operand_module_over(operator: &str, native: &str) -> Result<crate::checked::CheckedModule, String> {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!(
                "{OPERAND_ROWS}{NORMED_ROWS}{OPERAND_PARAMS}{operator}native rows for cpu from \"rows.rs\":\n{native}{OPERAND_LAUNCHES}"
            ),
        });
        crate::native_source::check_fixture(sources, &[]).map_err(|error| error.to_string())
    }

    fn operand_module(native: &str) -> Result<crate::checked::CheckedModule, String> {
        operand_module_over(&operand_buffers(), native)
    }

    /// A parameter that only an operand clause reads (`formed (INT8 == 1)`)
    /// is kept where the entry names no operand and writes the Q8 blocks
    /// itself: its source reads the parameter.
    #[test]
    fn a_parameter_only_an_operand_clause_reads_is_kept() {
        let module = operand_module("    project quantized rows (RW) over (M, K)\n").unwrap();
        let entry = module.entry_named("rows").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        let names: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["quantized"]);
        assert!(implementation.params.iter().any(|param| param.name == "INT8"));
    }

    /// The native condition that `left == right` checks to.
    fn equal(left: crate::checked::NativeNatExpr, right: u64) -> crate::checked::NativeCondition {
        crate::checked::NativeCondition::Compare {
            comparison: crate::expr::CmpOp::Eq,
            left,
            right: crate::checked::NativeNatExpr::Constant(right),
        }
    }

    /// The native condition a constant `true` or `false` checks to.
    fn holds(value: bool) -> crate::checked::NativeCondition {
        crate::checked::NativeCondition::Compare {
            comparison: if value { crate::expr::CmpOp::Eq } else { crate::expr::CmpOp::Ne },
            left: crate::checked::NativeNatExpr::Constant(0),
            right: crate::checked::NativeNatExpr::Constant(0),
        }
    }

    /// A consumer's operand is the operator's: its Q8 blocks, written where
    /// `formed` holds and paired where `when` does, and its formed rows,
    /// always written, NVFP4 operands where every weight takes them, both
    /// formed by the operator's `form` launch, which the declaration's
    /// formation describes; an emitted result gets every `for emits` buffer
    /// over its rows and width, named for it and serving it (`emission`),
    /// its Q8 blocks and its rows its forms.
    #[test]
    fn operands_are_formed_by_the_operator_or_emitted_by_their_producer() {
        use crate::checked::{ElementQuery, NativeFormation, NativeFormedOperand, NativeFormedValues, NativeNatExpr, NativeOperand};
        let module = operand_module("    project quantized rows (RW) over (M, K) operand (x) emits (result 0 over (M, N))\n").unwrap();
        let entry = module.entry_named("rows").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        let names: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["quantized", "rows", "result_0_quantized", "result_0_rows", "result_0_arrivals"]);
        let kernels: Vec<_> = implementation.launches.iter().map(|launch| launch.kernel.as_str()).collect();
        assert_eq!(kernels, ["rows_form", "rows_project"]);
        assert_eq!(
            implementation.scratch[0].operand,
            Some(NativeOperand::Formed {
                parameter: 0,
                formation: 0,
                pairs: equal(NativeNatExpr::Dimension("M".to_owned()), 1),
                rows: false,
                norm: None,
                formed: equal(NativeNatExpr::Parameter("INT8".to_owned()), 1),
                nvfp4: holds(false),
            })
        );
        assert_eq!(
            implementation.scratch[1].operand,
            Some(NativeOperand::Formed {
                parameter: 0,
                formation: 0,
                pairs: holds(true),
                rows: true,
                norm: None,
                formed: holds(true),
                nvfp4: equal(NativeNatExpr::Element("RW".to_owned(), ElementQuery::Operands), 1),
            })
        );
        assert_eq!(
            implementation.formation,
            Some(NativeFormation {
                launch: 0,
                operands: vec![NativeFormedOperand {
                    values: NativeFormedValues::Parameter { at: None, within: None },
                    rows: NativeNatExpr::Dimension("M".to_owned()),
                    width: NativeNatExpr::Dimension("K".to_owned()),
                    groups: NativeNatExpr::Constant(1),
                    staged: 1,
                    quantized: 0,
                    formed: equal(NativeNatExpr::Parameter("INT8".to_owned()), 1),
                    weights: vec![2],
                    tiles: holds(false),
                }],
            })
        );
        assert_eq!(implementation.scratch[2].operand, Some(NativeOperand::Emitted { result: 0, rows: false, norm: None }));
        assert_eq!(implementation.scratch[3].operand, Some(NativeOperand::Emitted { result: 0, rows: true, norm: None }));
        assert!(implementation.scratch[4].sync && implementation.scratch[4].operand.is_none());
        let emission: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.emission).collect();
        assert_eq!(emission, [None, None, Some(0), Some(0), Some(0)]);
        assert!(implementation.params.iter().any(|param| param.name == "INT8"));

        // A gathered operand's rows are another tensor's: no producer's
        // form stands for them.
        let module = operand_module("    project quantized rows (RW) over (M, K) operand (x at (picked))\n").unwrap();
        let entry = module.entry_named("rows").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        assert!(matches!(
            implementation.scratch[1].operand,
            Some(NativeOperand::Formed { ref pairs, .. }) if *pairs == holds(false)
        ));
        assert_eq!(
            implementation.formation.as_ref().map(|formation| formation.operands[0].values.clone()),
            Some(NativeFormedValues::Parameter { at: Some(1), within: None })
        );

        // A producer alone takes the emitted buffers and nothing else.
        let module = operand_module("    project quantized emits (result 0 over (M, N))\n").unwrap();
        let entry = module.entry_named("rows").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        let names: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["result_0_quantized", "result_0_rows", "result_0_arrivals"]);
        assert!(implementation.scratch.iter().all(|buffer| buffer.emission == Some(0)));
        assert!(implementation.params.is_empty());
        assert!(implementation.formation.is_none());

        // A consumer that names no operand takes the Q8 blocks alone and no
        // formation.
        let module = operand_module("    project quantized rows (RW) over (M, K)\n").unwrap();
        let entry = module.entry_named("rows").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        let names: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["quantized"]);
        assert!(implementation.scratch.iter().all(|buffer| buffer.operand.is_none() && buffer.emission.is_none()));
        assert!(implementation.formation.is_none());
        let kernels: Vec<_> = implementation.launches.iter().map(|launch| launch.kernel.as_str()).collect();
        assert_eq!(kernels, ["rows_project"]);
    }

    #[test]
    fn operands_name_the_tensors_and_results_they_hold() {
        let refused = |native: &str, message: &str| {
            let error = operand_module(native).map(|_| ()).unwrap_err();
            assert!(error.contains(message), "{native}: {error}");
        };
        refused("    project quantized rows (RW) over (M, K) operand (y)\n", "`y` is not a tensor port of `rows`");
        refused("    project quantized rows (RW) over (M, K) operand (x, x)\n", "holds no operand for `x`");
        refused("    project quantized rows (RW) over (M, K) operand (x at (w))\n", "`w` is not a `tensor[ROWS]` port");
        refused("    project quantized rows (RW) over (M, K) operand (x normed (w, x))\n", "`w` is not a one-row norm");
        refused("    project quantized emits (result 1 over (M, N))\n", "no tensor result 1");
        // `plain` is a kind of rows only, never a CPU variant.
        refused("    project quantized plain (RW)\n", "is not a CPU projection variant");
        refused("    project quantized emits (0 over (M, N))\n", "names its result");
        // An entry's own buffer holds no form: the operator owns them all.
        assert!(operand_module("    scratch staged bytes (M * K * 4) operand rows\n    project quantized rows (RW) over (M, K) operand (x)\n").is_err());
        // The operator forms what the line names in its `form` launch.
        let formless = operand_buffers().replace(OPERAND_FORM, "\n");
        let error = operand_module_over(&formless, "    project quantized rows (RW) over (M, K) operand (x)\n").unwrap_err();
        assert!(error.contains("has no `form` launch"), "{error}");
        // A clause is a native condition.
        for (clause, message) in [("nvfp4 (RW)", "a native `nvfp4` condition"), ("formed (M + 1)", "a native `formed` condition")] {
            let operator = operand_buffers().replace("nvfp4 (each(W, operands(W) == 1))", clause);
            let error = operand_module_over(&operator, "    project quantized rows (RW) over (M, K) operand (x)\n").unwrap_err();
            assert!(error.contains(message), "{clause}: {error}");
        }
    }

    /// The operator's own clauses: a Q8 form has no NVFP4 transform, and an
    /// emitted form is written as its consumer's forms say, so it states no
    /// condition.
    #[test]
    fn operator_operand_clauses_are_refused_where_they_mean_nothing() {
        let native = "    project quantized rows (RW) over (M, K) operand (x) emits (result 0 over (M, N))\n";
        let refused = |operator: &str, message: &str| {
            let error = operand_module_over(operator, native).map(|_| ()).unwrap_err();
            assert!(error.contains(message), "{operator}: {error}");
        };
        let rows = "    scratch rows bytes (M * K * 4) operand rows for rows\n";
        refused(
            &format!("    scratch quantized bytes (M * 4) operand nvfp4 (INT8 == 1) for rows\n{rows}{OPERAND_EMITS}{OPERAND_FORM}"),
            "Q8 blocks take no NVFP4 transform",
        );
        let consumer = format!("    scratch quantized bytes (M * 4) operand formed (INT8 == 1) for rows\n{rows}");
        for emitted in [
            "    scratch quantized bytes (M * 4) operand formed (INT8 == 1) for emits\n    scratch rows bytes (M * K * 4) operand rows for emits\n",
            "    scratch quantized bytes (M * 4) operand for emits\n    scratch rows bytes (M * K * 4) operand rows nvfp4 (INT8 == 1) for emits\n",
            "    scratch quantized bytes (M * 4) operand for emits\n    scratch rows bytes (M * K * 4) operand rows when (M == 1) for emits\n",
        ] {
            refused(&format!("{consumer}{emitted}{OPERAND_FORM}"), "with no condition");
        }
        refused(
            &format!("{consumer}    scratch quantized bytes (M * 4) operand for emits\n{OPERAND_FORM}"),
            "no `for emits` buffer of an operand's formed rows",
        );
        refused(
            &format!("{consumer}    scratch rows bytes (M * K * 4) operand rows for emits\n{OPERAND_FORM}"),
            "no `for emits` buffer of an operand's Q8 blocks",
        );
    }

    /// A normed operand's forms: the operator's rows, formed under the norm
    /// the line names, which a producer's stand for only while a norm is
    /// bound (its extent past 0); a producer that emits a normed operand
    /// forms its rows under its own norm.
    #[test]
    fn a_normed_operand_holds_its_formed_rows_and_its_norm() {
        use crate::checked::{ElementQuery, NativeNatExpr, NativeOperand, NativeOperandNorm};
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!(
                "{NORMED_ROWS}{OPERAND_PARAMS}{}native normed for cpu from \"normed.rs\":\n    project quantized rows (RW) over (M, K) operand (x normed (norm, eps)) emits (result 0 over (M, N) normed (norm, eps))\n    launch normed_project:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
                operand_buffers()
            ),
        });
        let module = crate::native_source::check_fixture(sources, &[]).unwrap();
        let entry = module.entry_named("normed").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        let names: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["quantized", "rows", "result_0_quantized", "result_0_rows", "result_0_arrivals"]);
        let norm = Some(NativeOperandNorm { weight: 1, epsilon: 3 });
        let bound = crate::checked::NativeCondition::Compare {
            comparison: crate::expr::CmpOp::Gt,
            left: NativeNatExpr::Dimension("K".to_owned()),
            right: NativeNatExpr::Constant(0),
        };
        assert_eq!(
            implementation.scratch[1].operand,
            Some(NativeOperand::Formed {
                parameter: 0,
                formation: 0,
                pairs: bound,
                rows: true,
                norm,
                formed: holds(true),
                nvfp4: equal(NativeNatExpr::Element("RW".to_owned(), ElementQuery::Operands), 1),
            })
        );
        assert!(matches!(
            implementation.scratch[0].operand,
            Some(NativeOperand::Formed { parameter: 0, rows: false, norm: n, .. }) if n == norm
        ));
        assert_eq!(implementation.scratch[2].operand, Some(NativeOperand::Emitted { result: 0, rows: false, norm }));
        assert_eq!(implementation.scratch[3].operand, Some(NativeOperand::Emitted { result: 0, rows: true, norm }));
        assert_eq!(implementation.scratch[4].emission, Some(0));
    }

    /// One line mixes the entry's rows, formed by the operator from a window
    /// of a parameter's rows, with a weight's own rows that the entry's
    /// launch emits: the latter's buffers are the entry's plain buffers, and
    /// the `form` launch's rows count only what it forms (`EMITTED`). An
    /// entry function's rows are formed whole and never pair; the entry's
    /// rows are never `emitted`.
    #[test]
    fn operands_take_windows_functions_and_emitting_launches() {
        use crate::checked::{NativeFormedValues, NativeNatExpr};
        let mixed = |native: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!(
                    "fn mixed[M, K, N, F](x: &tensor[M, K] f32, first: &tensor[1] i32, count: &tensor[1] i32, w: &tensor[N, K] RW, v: &tensor[N, F] VW) -> tensor[M, N] f32:\n    let mut result = tensor[M, N] f32\n    parallel for row in 0..M:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0] + f32(w[col, 0]) + f32(v[col, 0])\n    return result\n\n\
                     projection quantized for cpu:\n    let FORM_ROWS = M + sum(W, (OWN - EMITTED) * M) for rows\n    params (ROWS in [8, 4, 2, 1] default 8, arithmetic form INT8 in [0, 1] default 0 for rows)\n    where (INT8 == 0 or any(W, form(W, INT8) == 1)) for rows\n    scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) operand formed (INT8 == 1) for rows\n    scratch rows bytes (M * K * 4) operand rows for rows\n    each W:\n        scratch quantized bytes (M * ceil_div(K, Q8_VALUES) * Q8_BYTES) operand formed (INT8 == 1) for rows\n        scratch rows bytes (M * K * 4) operand rows for rows\n    launch form for rows:\n        threadgroups (FORM_ROWS, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n\n\
                     native mixed for cpu from \"mixed.rs\":\n{native}    launch mixed_project:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n"
                ),
            });
            crate::native_source::check_fixture(sources, &[]).map_err(|error| error.to_string())
        };
        let module = mixed("    project quantized rows (RW, VW over (N, F)) over (M, K) operand (x within (first, count), emitted)\n").unwrap();
        let entry = module.entry_named("mixed").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        let names: Vec<_> = implementation.scratch.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["quantized", "rows", "vw_quantized", "vw_rows"]);
        assert!(implementation.scratch[..2].iter().all(|buffer| buffer.operand.is_some()));
        assert!(implementation.scratch[2..].iter().all(|buffer| buffer.operand.is_none()));
        assert_eq!(implementation.launches[0].kernel, "mixed_form");
        assert_eq!(implementation.launches[0].groups[0], NativeNatExpr::Dimension("M".to_owned()));
        let formation = implementation.formation.as_ref().expect("the entry's rows are formed");
        assert_eq!(formation.operands.len(), 1);
        assert_eq!(formation.operands[0].values, NativeFormedValues::Parameter { at: None, within: Some([1, 2]) });
        assert_eq!(formation.operands[0].weights, [3]);

        // An entry function's rows: the operator forms them, nothing pairs.
        let module = mixed("    project quantized rows (RW) over (M, K) operand (values (joined))\n").unwrap();
        let entry = module.entry_named("mixed").unwrap();
        let implementation = module.native_implementation(entry, BackendName::Cpu).unwrap();
        assert!(implementation.scratch.iter().all(|buffer| buffer.operand.is_none()));
        assert_eq!(
            implementation.formation.as_ref().map(|formation| formation.operands[0].values.clone()),
            Some(NativeFormedValues::Function("joined".to_owned()))
        );

        let error = mixed("    project quantized rows (RW, VW over (N, F)) over (M, K) operand (emitted, emitted)\n").unwrap_err();
        assert!(error.contains("only a weight's own rows"), "{error}");
    }

    #[test]
    fn void_entry_constant_row_bound_and_routed_constant_sources_expand() {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: "\
const DECODE_ROWS = 8
const SOURCE_ROWS = 2

fn hint[M, H, N](residual: &tensor[M, H] f32, weight: &tensor[N, H] W, out: &mut tensor[M, N] f32) where M <= DECODE_ROWS:
    parallel for row in 0..M:
        parallel for col in 0..N:
            out[row, col] = residual[row, 0]

fn down_project[M, K, E, H, F](
    sources: &tensor[SOURCE_ROWS, M, K] i32,
    activated: &tensor[M, K, F] f32,
    expert_down: &tensor[E, H, F] DW,
    experts: &tensor[M, K] i32,
    weights: &tensor[M, K] f32,
) -> tensor[M, H] f32:
    let mut out = tensor[M, H] f32
    parallel for row in 0..M:
        parallel for col in 0..H:
            out[row, col] = weights[row, 0]
    return out

projection dense for vulkan:
    scratch staged bytes (M * K * 2) when M > 8
    launch stage when M > 8:
        threadgroups (M, 1, 1)
        threads_per_threadgroup (32, 1, 1)
    launch gemv when M <= 8:
        threadgroups (ceil_div(N, 32), M, 1)
        threads_per_threadgroup (32, 1, 1)

projection routed for metal:
    launch resident for summed when XS == 0:
        threadgroups (ceil_div(N, 32), M, 1)
        threads_per_threadgroup (32, 1, 1)
    launch summed for summed when XS > 0:
        threadgroups (ceil_div(N, 32), M, 1)
        threads_per_threadgroup (32, 1, 1)

native hint for vulkan from \"hint.comp\":
    project dense single (weight)

native down_project for metal from \"down.metal\":
    project routed summed (expert_down)
"
            .to_owned(),
        });
        let module = crate::native_source::check_fixture(sources, &[]).unwrap();
        let hint = module.entry_named("hint").unwrap();
        let hint_impl = module.native_implementation(hint, BackendName::Vulkan).unwrap();
        assert!(hint_impl.scratch.is_empty());
        assert_eq!(hint_impl.launches.len(), 1);
        assert_eq!(hint_impl.launches[0].kernel, "hint_gemv");

        let down = module.entry_named("down_project").unwrap();
        let down_impl = module.native_implementation(down, BackendName::Metal).unwrap();
        assert_eq!(down_impl.launches.len(), 1);
        assert_eq!(down_impl.launches[0].kernel, "down_project_summed");
    }

    #[test]
    fn projection_templates_allow_device_properties() {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: "\
fn proj[M, H, N](x: &tensor[M, H] f32, weight: &tensor[N, H] W) -> tensor[M, N] f32:
    let mut out = tensor[M, N] f32
    parallel for row in 0..M:
        parallel for col in 0..N:
            out[row, col] = x[row, 0]
    return out

projection dense for metal:
    let HALF_SHARED = device.shared_bytes_per_group / 2
    params (SPLIT in [1, 2, 4] default least where SPLIT >= device.units or SPLIT == 4, SIMDS in [2, 4, 8] default 4)
    where SIMDS * device.subgroup_width <= device.max_group_threads
    where SIMDS * 1024 <= HALF_SHARED
    launch gemv:
        threadgroups (ceil_div(N, 32), SPLIT, 1)
        threads_per_threadgroup (SIMDS * device.subgroup_width, 1, 1)

native proj for metal from \"proj.metal\":
    static (H, N)
    project dense single (weight)
"
            .to_owned(),
        });
        let module = crate::native_source::check_fixture(sources, &[]).expect("projection with device.* checks");
        let entry = module.entry_named("proj").unwrap();
        let native = module.native_implementation(entry, BackendName::Metal).unwrap();
        let statics = crate::checked::NativeSpecialization::new()
            .with_static("H", 64)
            .with_static("N", 64);
        assert!(
            native.default_specialization_without_device(&statics).is_none(),
            "projection with device.* requires a device to resolve defaults and constraints"
        );
    }
}
