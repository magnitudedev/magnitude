//! What an entry's stored inputs are, derived from its contract.
//!
//! Everything here is read from what the language already states (the
//! portable reference program, representation decoding definitions and type
//! declarations); nothing is declared for it. Native kernels are not
//! described: these facts are what any implementation of the entry is held
//! to.
//!
//! - [`EntryAnalysis::parameters`]: for every tensor parameter, the one kind
//!   of value its stored elements are ([`Stored`]), which fixes the lattice
//!   each stored value lies on.
//! - [`EntryAnalysis::grids`]: for every coded value, its decoded step: how
//!   far the decoded value moves when the code moves to an adjacent level.
//! - [`EntryAnalysis::reductions`]: accumulations across loop iterations, with
//!   their extent and accumulator type (for inspecting rounding level only).

mod domain;
#[cfg(test)]
mod probe;
mod term;
#[cfg(test)]
mod tests;
mod trace;

pub use term::{CodeLevels, Neighbours, Term, TermError, Variable};

use crate::entry::{LogicalEntry, ParameterKind};
use crate::ids::RepresentationId;
use crate::registry::{self, DecodeStep, PlaneEncoding, PlaneField, RepresentationKind};
use crate::types::DType;
use term::Node;
use trace::Trace;

/// What the stored elements of one tensor parameter are.
#[derive(Clone, Debug, PartialEq)]
pub enum Stored {
    /// Floating values stored in `dtype`. Their lattice is the representable
    /// values of `dtype`.
    Float { dtype: DType },
    /// A registry-packed representation: the planes its registry decode
    /// reads. Every plane is stored and lies on its own lattice, the code
    /// planes and the scale and bias planes alike.
    Packed { planes: Vec<Plane> },
    /// Integer elements the portable body decodes by value: it converts an
    /// integer computed from one element to floating and writes
    /// `slope * code + intercept`, with slope and intercept computed from
    /// other stored values. `levels` are the integers the converted code
    /// takes; `coefficients` are the tensor parameters whose stored values
    /// the slope and intercept are computed from. Each decode has a
    /// [`Grid`].
    Codes {
        levels: CodeLevels,
        coefficients: Vec<String>,
    },
    /// Integer or boolean elements whose values the body never converts to
    /// floating: indices, selections, counts, masks. They have no resolution.
    Discrete,
    /// Integer elements whose values the body converts to floating, but not
    /// in a decode of the form [`Stored::Codes`] requires that the analysis
    /// resolves. A position scaled into an angle is of this kind, and so is
    /// a code assembled from several stored elements; the analysis does not
    /// tell them apart and states no lattice for either.
    Converted,
    /// Canonical bytes of an external artifact, which the language converts
    /// exactly and never reads by element. It states no lattice for them.
    External,
}

/// One plane of a packed representation, as its registry decode reads it:
/// the logical values at positions `v` of the packing axis with equal
/// `v / group` share `fields` consecutive entries.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub field: PlaneField,
    pub group: u32,
    pub fields: u32,
    pub lattice: Lattice,
}

/// The values one stored entry of a plane can take.
#[derive(Clone, Debug, PartialEq)]
pub enum Lattice {
    /// The representable values of a floating dtype.
    Float(DType),
    /// The interpreted values of a code.
    Codes(CodeLevels),
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredParameter {
    pub name: String,
    pub stored: Stored,
}

/// A loop coordinate free in a term, ranging over `start..end`.
#[derive(Clone, Debug, PartialEq)]
pub struct Coordinate {
    /// The [`Variable::Binder`] id.
    pub binder: u32,
    pub start: Term,
    pub end: Term,
}

/// The decoded step of one coded value: for every assignment of
/// `coordinates`, the decoded value is `slope * code` plus terms that do not
/// read the code, so moving the code from one of its `levels` to an adjacent
/// one moves the decoded value by `slope` times the distance between them.
///
/// The terms name parameter contents as the decode reads them. Where the
/// call has already written a parameter and the analysis did not resolve
/// whether that write replaced the element read, they are the contents the
/// call was given.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub parameter: String,
    /// For a packed representation, the plane and the entry of its group
    /// that hold the code. `None` for a parameter of [`Stored::Codes`].
    pub plane: Option<(PlaneField, u32)>,
    /// Binders `0..coordinates.len()`, innermost loop first.
    pub coordinates: Vec<Coordinate>,
    /// For a packed representation, the logical element decoded. For
    /// [`Stored::Codes`], the stored element the code is computed from.
    pub index: Vec<Term>,
    /// The code's value: the interpreted code of a packed plane, the integer
    /// the body converts to floating otherwise.
    pub code: Term,
    pub levels: CodeLevels,
    /// `|∂decoded/∂code|`.
    pub slope: Term,
}

/// How far a decoded value moves when its code moves to the adjacent level
/// on each side. The two differ on a non-uniform codebook; a code at an end
/// of its range has no step outward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steps {
    pub below: Option<f64>,
    pub above: Option<f64>,
}

impl Grid {
    /// The decoded steps at the point `value` supplies.
    pub fn steps(&self, value: &impl Fn(&Variable) -> Option<f64>) -> Result<Steps, TermError> {
        let slope = self.slope.evaluate(value)?;
        let code = self.code.evaluate(value)?;
        let neighbours = self.levels.neighbours(code);
        Ok(Steps {
            below: neighbours.below.map(|level| slope * (code - level)),
            above: neighbours.above.map(|level| slope * (level - code)),
        })
    }
}

/// An accumulation of a floating value across loop iterations.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    /// Accumulation steps into one accumulator.
    pub extent: Term,
    pub accumulator: DType,
    /// Loops whose coordinates remain free in `extent`.
    pub coordinates: Vec<Coordinate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EntryAnalysis {
    /// Every tensor parameter of the entry, in schema order.
    pub parameters: Vec<StoredParameter>,
    pub grids: Vec<Grid>,
    pub reductions: Vec<Reduction>,
}

/// Derive the analysis of `entry`.
pub fn analyze(entry: &LogicalEntry) -> EntryAnalysis {
    crate::cancellation::check();
    let view = entry.as_view();
    let mut trace = Trace::new(view);
    trace.run_entry();

    let mut grids: Vec<Grid> = Vec::new();
    // Per decode, the parameter decoded and those its coefficients read.
    let mut bound: Vec<(String, Vec<String>)> = Vec::new();
    for decode in &trace.decodes {
        let reads = [&decode.slope, &decode.intercept]
            .into_iter()
            .flat_map(|term| term.value_reads().expect("a recorded decode is resolved"))
            .filter_map(|variable| variable.storage().map(str::to_owned))
            .collect();
        bound.push((decode.parameter.clone(), reads));
        let grid = canonical(Grid {
            parameter: decode.parameter.clone(),
            plane: None,
            coordinates: decode.coordinates.iter().map(coordinate).collect(),
            index: decode.index.clone(),
            code: decode.code.clone(),
            levels: CodeLevels::Integers {
                min: decode.range.0 as i64,
                max: decode.range.1 as i64,
            },
            slope: decode.slope.clone().abs(),
        });
        if !grids.contains(&grid) {
            grids.push(grid);
        }
    }

    let mut parameters = Vec::new();
    for parameter in view.schema().parameters() {
        let ParameterKind::Tensor {
            representation,
            axes,
            ..
        } = &parameter.kind
        else {
            continue;
        };
        let stored = match &registry::representation_info(*representation).kind {
            RepresentationKind::Dense(dtype) if dtype.is_float() => Stored::Float { dtype: *dtype },
            RepresentationKind::Dense(_) => {
                let decoded = grids
                    .iter()
                    .filter(|grid| grid.parameter == parameter.name)
                    .filter_map(|grid| match grid.levels {
                        CodeLevels::Integers { min, max } => Some((min, max)),
                        CodeLevels::Table(_) => None,
                    })
                    .reduce(|(min, max), (low, high)| (min.min(low), max.max(high)));
                match decoded {
                    Some((min, max)) => {
                        let mut coefficients: Vec<String> = Vec::new();
                        for (_, reads) in bound.iter().filter(|(name, _)| *name == parameter.name) {
                            for read in reads {
                                if !coefficients.contains(read) {
                                    coefficients.push(read.clone());
                                }
                            }
                        }
                        Stored::Codes {
                            levels: CodeLevels::Integers { min, max },
                            coefficients,
                        }
                    }
                    None if trace.converted.contains(&parameter.name) => Stored::Converted,
                    None => Stored::Discrete,
                }
            }
            RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_) => {
                let coordinates = axes
                    .iter()
                    .enumerate()
                    .map(|(axis, extent)| Coordinate {
                        binder: axis as u32,
                        start: Term::constant(0.0),
                        end: trace.nat_term(*extent),
                    })
                    .collect::<Vec<_>>();
                let (planes, decoded) = packed(*representation, &parameter.name, coordinates);
                grids.extend(decoded);
                Stored::Packed { planes }
            }
            RepresentationKind::External(_) => Stored::External,
        };
        parameters.push(StoredParameter {
            name: parameter.name.clone(),
            stored,
        });
    }

    EntryAnalysis {
        parameters,
        grids,
        reductions: merge_reductions(&trace.reductions),
    }
}

fn coordinate(open: &trace::Coordinate) -> Coordinate {
    Coordinate {
        binder: open.binder,
        start: open.start.clone(),
        end: open.end.clone(),
    }
}

/// `grid` over only the coordinates its terms read, with every binder
/// numbered in order: the coordinates first, then binders bound inside its
/// terms.
fn canonical(mut grid: Grid) -> Grid {
    let reads = |term: &Term, binder: u32| term.mentions(&Term::variable(Variable::Binder(binder)));
    let mut kept: Vec<usize> = Vec::new();
    loop {
        let before = kept.len();
        for (position, candidate) in grid.coordinates.iter().enumerate() {
            let read = grid
                .index
                .iter()
                .chain([&grid.code, &grid.slope])
                .chain(kept.iter().flat_map(|kept| {
                    let kept = &grid.coordinates[*kept];
                    [&kept.start, &kept.end]
                }))
                .any(|term| reads(term, candidate.binder));
            if read && !kept.contains(&position) {
                kept.push(position);
            }
        }
        if kept.len() == before {
            break;
        }
    }
    kept.sort_unstable();
    grid.coordinates = kept
        .into_iter()
        .map(|position| grid.coordinates[position].clone())
        .collect();

    let mut binders: Vec<u32> = grid
        .coordinates
        .iter()
        .map(|coordinate| coordinate.binder)
        .collect();
    let mut note = |term: &Term| {
        term.visit(&mut |term| {
            let binder = match term.node() {
                Node::Variable(Variable::Binder(binder)) | Node::Fold { binder, .. } => *binder,
                _ => return,
            };
            if !binders.contains(&binder) {
                binders.push(binder);
            }
        })
    };
    for coordinate in &grid.coordinates {
        note(&coordinate.start);
        note(&coordinate.end);
    }
    grid.index
        .iter()
        .chain([&grid.code, &grid.slope])
        .for_each(&mut note);
    let number = |binder: u32| {
        binders
            .iter()
            .position(|known| *known == binder)
            .expect("every binder of the grid was numbered") as u32
    };
    let rename = |term: &Term| term.rename_binders(&number);
    Grid {
        coordinates: grid
            .coordinates
            .iter()
            .map(|coordinate| Coordinate {
                binder: number(coordinate.binder),
                start: rename(&coordinate.start),
                end: rename(&coordinate.end),
            })
            .collect(),
        index: grid.index.iter().map(rename).collect(),
        code: rename(&grid.code),
        slope: rename(&grid.slope),
        ..grid
    }
}

/// Sequential accumulations into the same storage under the same loops add.
fn merge_reductions(records: &[trace::ReductionRecord]) -> Vec<Reduction> {
    let mut merged: Vec<(Option<usize>, Reduction)> = Vec::new();
    for record in records {
        let reduction = Reduction {
            extent: record.extent.clone(),
            accumulator: record.accumulator,
            coordinates: record.coordinates.iter().map(coordinate).collect(),
        };
        match merged.iter_mut().find(|(storage, existing)| {
            storage.is_some()
                && *storage == record.storage
                && existing.coordinates == reduction.coordinates
        }) {
            Some((_, existing)) => existing.extent = existing.extent.clone().add(reduction.extent),
            None => merged.push((record.storage, reduction)),
        }
    }
    merged.into_iter().map(|(_, reduction)| reduction).collect()
}

/// The values an entry of a plane takes: for a coded plane, every value the
/// registry's interpretation gives a raw code.
fn lattice(encoding: &PlaneEncoding) -> Lattice {
    match encoding {
        PlaneEncoding::Dense(dtype) => Lattice::Float(*dtype),
        PlaneEncoding::Packed {
            bits,
            interpretation,
        } => Lattice::Codes(CodeLevels::of(
            interpretation.codes(*bits).into_iter().map(f64::from),
        )),
        PlaneEncoding::FloatCode { format } => Lattice::Codes(CodeLevels::of(
            (0..1u32 << format.bits())
                .map(|raw| f64::from(format.decode(raw)))
                .filter(|value| value.is_finite()),
        )),
    }
}

/// The planes of packed `representation` and, for the logical element at
/// `coordinates` of `parameter`, the grid of every code its registry decode
/// reads.
fn packed(
    representation: RepresentationId,
    parameter: &str,
    coordinates: Vec<Coordinate>,
) -> (Vec<Plane>, Vec<Grid>) {
    let info = registry::representation_info(representation);
    let recipe = registry::decode_recipe(representation, info.decoded)
        .expect("a packed representation has a decode recipe");
    let index = coordinates
        .iter()
        .map(|coordinate| Term::variable(Variable::Binder(coordinate.binder)))
        .collect::<Vec<_>>();
    let planes = recipe
        .planes()
        .iter()
        .map(|plane| Plane {
            field: plane.field,
            group: plane.group,
            fields: plane.fields,
            lattice: lattice(&plane.encoding),
        })
        .collect();

    // The decoded value over what each plane read yields: a code read is its
    // interpreted value, so interpretation steps pass it through.
    let mut codes: Vec<(Term, PlaneField, u32, CodeLevels)> = Vec::new();
    let mut temporaries: Vec<Option<Term>> = vec![None; recipe.temporary_count()];
    for step in recipe.steps() {
        let get = |temp| {
            temporaries[recipe.ordinal(temp)]
                .clone()
                .expect("a decode step reads temporaries defined before it")
        };
        let value = match step {
            DecodeStep::ReadPlaneField { plane, field, .. } => {
                let schema = recipe
                    .planes()
                    .iter()
                    .find(|schema| schema.ordinal == *plane)
                    .expect("a decode step reads a plane of its recipe");
                let read = Term::variable(Variable::Plane {
                    parameter: parameter.to_owned(),
                    plane: schema.field,
                    field: *field,
                    index: index.clone(),
                });
                if let Lattice::Codes(levels) = lattice(&schema.encoding) {
                    codes.push((read.clone(), schema.field, *field, levels));
                }
                read
            }
            // A function of the element's position alone.
            DecodeStep::ReadPlaneLane { .. } => {
                Term::opaque(index.last().map(Term::variables).unwrap_or_default())
            }
            DecodeStep::InterpretCode { raw: from, .. }
            | DecodeStep::InterpretGridCode { raw: from, .. }
            | DecodeStep::DecodeFloatCode { raw: from, .. }
            | DecodeStep::ConvertToF32 { from, .. } => get(*from),
            DecodeStep::Cast { from, to, .. } => Term::round(*to, get(*from)),
            DecodeStep::Multiply { left, right, .. } => get(*left).mul(get(*right)),
            DecodeStep::Negate { from, .. } => Term::unary(term::Unary::Neg, get(*from)),
            DecodeStep::MultiplyAdd {
                factor,
                multiplicand,
                addend,
                ..
            } => get(*factor).mul(get(*multiplicand)).add(get(*addend)),
        };
        temporaries[recipe.ordinal(step.defines())] = Some(value);
    }
    let decoded = temporaries[recipe.ordinal(recipe.output())]
        .clone()
        .expect("a decode recipe defines its output");
    let grids = codes
        .into_iter()
        .map(|(code, plane, field, levels)| {
            let (slope, _) = decoded
                .affine_in(&code)
                .expect("a registry decode is affine in every code it reads");
            Grid {
                parameter: parameter.to_owned(),
                plane: Some((plane, field)),
                coordinates: coordinates.clone(),
                index: index.clone(),
                code,
                levels,
                slope: slope.abs(),
            }
        })
        .collect();
    (planes, grids)
}
