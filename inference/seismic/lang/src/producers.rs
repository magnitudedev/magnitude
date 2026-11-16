//! The producer declaration: which decoders a backend's producers read, as
//! data its producer table exports, and the questions forms ask of it:
//! whether a form's producers read a weight ([`covers`]) and how its lane
//! producer enters the weight's codes ([`lanes`]).
//!
//! A format reaches a backend's fast path when, and only when, a producer of
//! the form decodes its axis values. Each backend's `producers.decl`
//! (`native-library/<backend>/`) is the one statement of that reach: form
//! admission asks [`covers`] and then states only the form's own arithmetic
//! over `Profile`, and the build renders the backend's predicates from the
//! same text (`build.rs`), which its decode functions are dispatched by
//! (C++ functions on Metal and CUDA; on Vulkan the preprocessor macro a
//! bound slot is held to, whose `DECODED` form admits the same weights,
//! requirements included).

use crate::forms::Form;
pub use crate::producer_decl::Axis;
use crate::producer_decl::{parse, Domain, Expr, Pattern};
use crate::registry::{
    self, BackendName, BiasKind, CodeInterpretation, CodeKind, CodingAxes, CoefficientKind, DecodeAxes, LocalCodes,
    RepresentationInfo, RepresentationKind, ValueCodes,
};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Where a hierarchical weight's local coefficient codes are read from, with
/// their interpretation; a direct weight has no local codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Locals {
    Direct,
    /// A scales plane, in every layout.
    Plane(CodeKind),
    /// The group's grid entries carry them.
    Carried(CodeKind),
}

/// The axis values a producer reads a weight's decoder by: its value codes
/// (kind, and the bits of the low and high code planes) and its coefficients'
/// local codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisKey {
    pub kind: CodeKind,
    pub low: u32,
    pub high: u32,
    pub locals: Locals,
}

impl DecodeAxes {
    /// The axis values a producer must read for this weight.
    pub fn axis_key(&self) -> AxisKey {
        let kind = match self.coding.codes() {
            ValueCodes::Fields(interpretation) => interpretation.kind(),
            ValueCodes::Grid(_) => CodeKind::Grid,
            ValueCodes::E4m3 => CodeKind::E4m3,
        };
        let locals = match &self.coding {
            CodingAxes::Direct { .. } => Locals::Direct,
            CodingAxes::Hierarchical { interpretation, locals, .. } => match locals {
                LocalCodes::Plane { .. } => Locals::Plane(interpretation.kind()),
                LocalCodes::Carried { .. } => Locals::Carried(interpretation.kind()),
            },
        };
        AxisKey { kind, low: self.low, high: self.high, locals }
    }
}

/// A packed-rows weight as its producers read it: its decode axes and its
/// storage group (`PackedRowLayout::group`).
#[derive(Clone, Debug)]
pub struct Weight {
    pub axes: DecodeAxes,
    pub storage: u32,
}

/// The ABI's number of `value` of an enumerated axis whose values are `all`:
/// its position + 1 (`SEISMIC_<AXIS>_<NAME>`), 0 for none.
pub fn number<T: PartialEq>(all: &[T], value: Option<T>) -> i64 {
    value.map_or(0, |value| {
        1 + all.iter().position(|candidate| *candidate == value).expect("a value of its axis") as i64
    })
}

/// A weight's value of an axis of its native decode ABI: a number, or a
/// value of an enumerated axis (none where the weight has none), which the
/// ABI names (`SEISMIC_<AXIS>_<NAME>`) and numbers ([`number`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisValue {
    Number(i64),
    Codes(Option<CodeKind>),
    Coeff(CoefficientKind),
    Bias(Option<BiasKind>),
}

impl AxisValue {
    /// Its number, as a requirement reads it.
    pub fn number(self) -> i64 {
        match self {
            Self::Number(value) => value,
            Self::Codes(kind) => number(&CodeKind::ALL, kind),
            Self::Coeff(kind) => number(&CoefficientKind::ALL, Some(kind)),
            Self::Bias(kind) => number(&BiasKind::ALL, kind),
        }
    }
}

impl Weight {
    /// The weight of representation `info`; `None` for storage other than
    /// packed rows.
    pub fn of(info: &RepresentationInfo) -> Option<Self> {
        let RepresentationKind::PackedRows(rows) = &info.kind else {
            return None;
        };
        Some(Self { axes: registry::decode_axes(info.id)?, storage: rows.group() })
    }

    /// The value of `axis` for this weight: the one computation of it, which
    /// the native decode ABI renders (`seismic_runtime`'s `CODING_AXES`) and
    /// requirements read.
    pub fn value(&self, axis: Axis) -> AxisValue {
        let number = |value: u32| AxisValue::Number(i64::from(value));
        let codes = self.axes.coding.codes();
        let (group, factor_group, bits, local, carried, bias) = match &self.axes.coding {
            CodingAxes::Direct { group, bias, .. } => (*group, 0, 0, None, 0, bias.then_some(BiasKind::Minimum)),
            CodingAxes::Hierarchical { factor_group, group, bits, interpretation, locals, .. } => (
                *group,
                *factor_group,
                *bits,
                Some(interpretation.kind()),
                match locals {
                    LocalCodes::Plane { .. } => 0,
                    LocalCodes::Carried { grid, .. } => grid.carried_bit(),
                },
                locals.bias().map(|bias| bias.kind()),
            ),
        };
        match axis {
            Axis::Codes => AxisValue::Codes(Some(self.axes.axis_key().kind)),
            Axis::CodeLowBits => number(self.axes.low),
            Axis::CodeHighBits => number(self.axes.high),
            Axis::CodesOffset => AxisValue::Number(match codes {
                ValueCodes::Fields(CodeInterpretation::Offset(offset) | CodeInterpretation::Odd(offset)) => {
                    i64::from(offset)
                }
                ValueCodes::Fields(_) | ValueCodes::Grid(_) | ValueCodes::E4m3 => 0,
            }),
            Axis::Coeff => AxisValue::Coeff(self.axes.coding.kind()),
            Axis::CoeffBias => AxisValue::Bias(bias),
            Axis::CoeffGroup => number(group),
            Axis::CoeffFactorGroup => number(factor_group),
            Axis::CoeffLocal => AxisValue::Codes(local),
            Axis::CoeffLocalBits => number(bits),
            Axis::CoeffCarried => number(carried),
            Axis::LogicalGroup => number(self.storage),
        }
    }
}

/// The ABI's number of the enumerated value `name` of `domain`.
fn constant(domain: Domain, name: &str) -> i64 {
    fn find<T: Copy + PartialEq>(all: &[T], label: fn(T) -> &'static str, name: &str) -> Option<i64> {
        all.iter().find(|value| label(**value) == name).map(|value| number(all, Some(*value)))
    }
    match domain {
        Domain::Codes => find(&CodeKind::ALL, CodeKind::name, name),
        Domain::Coeff => find(&CoefficientKind::ALL, CoefficientKind::name, name),
        Domain::BiasKind => find(&BiasKind::ALL, BiasKind::name, name),
    }
    .unwrap_or_else(|| panic!("no value `{}:{name}`", domain.name()))
}

/// Every enumerated value `expression` names is one its axis has.
fn check(expression: &Expr) {
    match expression {
        Expr::Number(_) | Expr::Axis(_) => {}
        Expr::Constant(domain, name) => {
            constant(*domain, name);
        }
        Expr::Not(operand) => check(operand),
        Expr::Binary(_, left, right) => {
            check(left);
            check(right);
        }
    }
}

fn parse_kind(name: &str) -> CodeKind {
    CodeKind::ALL
        .into_iter()
        .find(|kind| kind.name().eq_ignore_ascii_case(name))
        .unwrap_or_else(|| panic!("no code kind `{name}`"))
}

fn any_or<T: PartialEq>(allowed: &[T], value: &T) -> bool {
    allowed.is_empty() || allowed.contains(value)
}

/// Whether `pattern` holds `key`.
fn contains(pattern: &Pattern, key: &AxisKey) -> bool {
    let (placement, kind) = match key.locals {
        Locals::Direct => ("direct", None),
        Locals::Plane(kind) => ("plane", Some(kind)),
        Locals::Carried(kind) => ("carried", Some(kind)),
    };
    any_or(&pattern.planes, &(key.low, key.high))
        && (pattern.kinds.is_empty() || pattern.kinds.iter().any(|name| parse_kind(name) == key.kind))
        && (pattern.locals.is_empty()
            || pattern.locals.iter().any(|(allowed, local)| {
                allowed == placement && local.as_deref().is_none_or(|local| Some(parse_kind(local)) == kind)
            }))
}

/// How a form's lane producer enters a weight's codes: as pairs at any zero
/// from `zero` on, or as bytes at `zero` (`producers.decl`'s `enter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lanes {
    pub bytes: bool,
    pub zero: u32,
}

impl Lanes {
    /// Whether the codes enter as their values + `offset`, in a layout of
    /// bytes (`bytes`) or of pairs: a producer of bytes serves bytes at its
    /// zero and pairs at any zero (one add moves a pair's two codes), a
    /// producer of pairs only pairs, at any zero from its own. The form
    /// takes an offset at which its values are non-negative.
    pub fn enter_at(self, offset: u32, bytes: bool) -> bool {
        match (self.bytes, bytes) {
            (true, true) => offset == self.zero,
            (true, false) => true,
            (false, true) => false,
            (false, false) => offset >= self.zero,
        }
    }
}

/// The producers of one backend's form: the patterns of the weights they
/// read, and the requirements every such weight meets.
#[derive(Debug)]
pub struct Declaration {
    patterns: Vec<Pattern>,
    requires: Vec<Expr>,
}

impl Declaration {
    /// The pattern a producer reads every axis value of `weight` by; `None`
    /// where none does.
    fn pattern(&self, weight: &Weight) -> Option<&Pattern> {
        let key = weight.axes.axis_key();
        let mut held = self.patterns.iter().filter(|pattern| {
            contains(pattern, &key) && (!pattern.signed_bytes || weight.axes.profile().codes.fits_i8())
        });
        let pattern = held.next()?;
        assert!(
            held.all(|other| other.entry == pattern.entry),
            "a weight's codes enter a producer's lanes one way ({key:?})"
        );
        let meets = |requirement: &Expr| requirement.value(&|axis| weight.value(axis).number(), &constant) != 0;
        self.requires.iter().all(meets).then_some(pattern)
    }

    /// Whether a producer reads every axis value of `weight`.
    pub fn covers(&self, weight: &Weight) -> bool {
        self.pattern(weight).is_some()
    }

    /// How the form's lane producer enters `weight`'s codes; `None` where no
    /// producer reads them or their values + its zero are no bytes where it
    /// enters bytes. Codes enter at the zero the declaration states, or at
    /// their own: an unsigned code's 0, an offset code's offset, a two's
    /// complement code's sign bit (which its plane flips) and a grid value's
    /// 0 (codes of other kinds state theirs).
    pub fn lanes(&self, weight: &Weight) -> Option<Lanes> {
        let pattern = self.pattern(weight)?;
        let entry = pattern.entry.unwrap_or_else(|| panic!("a lane producer states how its codes enter ({pattern:?})"));
        let zero = match entry.zero {
            Some(zero) => zero,
            None => match weight.axes.coding.codes() {
                ValueCodes::Fields(CodeInterpretation::Unsigned) | ValueCodes::Grid(_) => 0,
                ValueCodes::Fields(CodeInterpretation::Offset(offset)) => u32::try_from(offset).ok()?,
                ValueCodes::Fields(CodeInterpretation::TwosComplement) => 1 << (weight.axes.low + weight.axes.high - 1),
                codes => panic!("a lane producer of {codes:?} states the zero they enter at"),
            },
        };
        let values = weight.axes.profile().codes;
        let byte = |value: i32| (0..=255).contains(&(i64::from(value) + i64::from(zero)));
        (!entry.bytes || (byte(values.least) && byte(values.greatest))).then_some(Lanes { bytes: entry.bytes, zero })
    }
}

type Declarations = HashMap<Form, Declaration>;

/// A backend's declaration, over the forms it names.
fn declarations(backend: BackendName, text: &str) -> Declarations {
    parse(text)
        .forms
        .into_iter()
        .map(|form| {
            let named = Form::named(backend, &form.name)
                .unwrap_or_else(|| panic!("`{}` declares producers of no form `{}`", backend.as_str(), form.name));
            form.requires.iter().for_each(check);
            (named, Declaration { patterns: form.patterns, requires: form.requires })
        })
        .collect()
}

static METAL: LazyLock<Declarations> =
    LazyLock::new(|| declarations(BackendName::Metal, include_str!("../native-library/metal/producers.decl")));
static CUDA: LazyLock<Declarations> =
    LazyLock::new(|| declarations(BackendName::Cuda, include_str!("../native-library/cuda/producers.decl")));
static VULKAN: LazyLock<Declarations> =
    LazyLock::new(|| declarations(BackendName::Vulkan, include_str!("../native-library/vulkan/producers.decl")));

/// The declaration of `form`'s producers; `form` must be one the backend's
/// table declares (every form whose reach its producers bound).
pub fn declaration(form: Form) -> &'static Declaration {
    let table = match form.backend() {
        BackendName::Metal => &METAL,
        BackendName::Cuda => &CUDA,
        BackendName::Vulkan => &VULKAN,
        BackendName::Cpu => panic!("`{}` declares no producers", form.backend().as_str()),
    };
    table
        .get(&form)
        .unwrap_or_else(|| panic!("`{}` declares no producers of `{}`", form.backend().as_str(), form.name()))
}

/// Whether `form`'s producers read every axis value of `weight`.
pub fn covers(form: Form, weight: &Weight) -> bool {
    declaration(form).covers(weight)
}

/// How `form`'s lane producer enters `weight`'s codes
/// ([`Declaration::lanes`]).
pub fn lanes(form: Form, weight: &Weight) -> Option<Lanes> {
    declaration(form).lanes(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every backend's declaration parses, names only forms the backend has
    /// (`declarations`) and declares the producers of some form.
    #[test]
    fn declarations_parse() {
        for (backend, table) in [(BackendName::Metal, &METAL), (BackendName::Cuda, &CUDA), (BackendName::Vulkan, &VULKAN)] {
            assert!(!table.is_empty(), "`{}` declares the producers of some form", backend.as_str());
        }
    }

    /// The lane producers read neither floating nor odd codes; PACK reads
    /// local codes of the scales plane and those grid entries carry
    /// (`packing_factors`).
    #[test]
    fn metal_lane_producers_read_integer_codes() {
        let (mut covered, mut carried) = (0, 0);
        for info in registry::representations() {
            let Some(weight) = Weight::of(info) else { continue };
            let key = weight.axes.axis_key();
            let covers = covers(crate::forms::METAL_PACK, &weight);
            if matches!(key.kind, CodeKind::E4m3 | CodeKind::Odd) {
                assert!(!covers, "{} is not read by a lane producer", info.name);
            }
            if matches!(key.locals, Locals::Carried(_)) {
                carried += usize::from(covers);
            }
            covered += usize::from(covers);
        }
        assert!(covered > 0 && carried > 0, "some representation has a lane producer, some with carried codes");
    }
}
