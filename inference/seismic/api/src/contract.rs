//! Compile-time generated entry contracts (`EntryContract`): admission,
//! default specialization parameters, and result/scratch/launch sizing
//! evaluated as compiled Rust without decoding `.seismicbundle` or opening a
//! device.

use crate::{BackendName, Element};
use std::fmt;

pub use seismic_lang::checked::NativeOperandNorm;

/// Failure evaluating an entry's compiled contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// The entry declares no native implementation for this backend.
    NoImplementation(BackendName),
    /// The element bindings are missing, unexpected, inadmissible for the
    /// entry's element domain, or lack a required element query value (such as
    /// `unit_columns` for a non-packed destination).
    UnsupportedElements,
    /// A required static or call dimension was not supplied in `dims`.
    MissingDimension(&'static str),
    /// A declared tuning parameter was missing, or an unknown parameter was
    /// supplied in `params`.
    InvalidParameter(&'static str),
    /// A tuning parameter value lies outside its declared finite domain.
    ParameterOutsideDomain {
        parameter: &'static str,
        value: u64,
    },
    /// The native `where` condition does not hold, or no configuration is
    /// admissible at the supplied static dimensions.
    DomainViolation,
    /// A tuning parameter's default has no listed value at the supplied static
    /// dimensions.
    NoDefault(&'static str),
    /// An arithmetic expression overflowed, underflowed, divided by zero, or
    /// produced an unrepresentable tensor byte length.
    ArithmeticOverflow,
    /// An expression reads a `device.NAME` property that requires an open
    /// device.
    DeviceDependent(&'static str),
    /// A launch condition reads `word[index]`, which was not supplied in `words`.
    MissingConditionWord {
        word: &'static str,
        index: u64,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(backend) => {
                write!(f, "no native implementation for `{}`", backend.as_str())
            }
            Self::UnsupportedElements => f.write_str("unsupported element bindings"),
            Self::MissingDimension(name) => write!(f, "missing dimension `{name}`"),
            Self::InvalidParameter(name) => write!(f, "invalid tuning parameter `{name}`"),
            Self::ParameterOutsideDomain { parameter, value } => {
                write!(f, "tuning parameter `{parameter}` does not admit {value}")
            }
            Self::DomainViolation => f.write_str("configuration violates native `where` condition"),
            Self::NoDefault(name) => write!(f, "default of tuning parameter `{name}` has no value"),
            Self::ArithmeticOverflow => {
                f.write_str("contract expression overflowed, underflowed, or divided by zero")
            }
            Self::DeviceDependent(name) => {
                write!(f, "contract reads `device.{name}`, which requires an open device")
            }
            Self::MissingConditionWord { word, index } => {
                write!(f, "launch condition reads `{word}[{index}]`, which was not supplied")
            }
        }
    }
}

impl std::error::Error for ContractError {}

/// One call-private scratch buffer sized by the entry's compiled contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchSpec {
    pub bytes: u64,
    pub sync: bool,
    /// The result ordinal whose emitted operand this buffer serves, if any.
    pub emission: Option<u32>,
}

/// Role of an operand scratch buffer in graph operand pairing (`operand_pairs`
/// and `same_norm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandRole {
    /// Formed from tensor parameter `parameter` by launch `formation`.
    Formed {
        parameter: u32,
        formation: u32,
        pairs: bool,
        formed: bool,
        nvfp4: bool,
    },
    /// Emitted for tensor result `result`.
    Emitted { result: u32 },
}

/// Operand form metadata attached to a scratch buffer at `scratch_index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandFormSpec {
    pub scratch_index: usize,
    pub role: OperandRole,
    pub rows: bool,
    pub norm: Option<NativeOperandNorm>,
}

/// One tensor result leaf sized by the entry's compiled contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorResultSpec {
    pub element: Element,
    pub bytes: u64,
}

/// One tensor parameter leaf sized by the entry's compiled contract
/// (`local_for`, `input_for`, `port_for`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorParamSpec {
    pub element: Element,
    pub extents: Vec<u64>,
    pub bytes: u64,
}

/// Compile-time static node shape for an entry on one backend: the exact
/// vector lengths and ABI counts that determine `plan_heap`, `recorded`
/// (`GraphCensus`), and disjoint-port checks without decoding `.seismicbundle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractNodeShape {
    /// Buffers in ABI order: tensor parameters, tensor results, scratch.
    pub buffers: usize,
    /// Buffers bound in the kernel argument table (all but host ports).
    pub arguments: usize,
    /// 64-bit argument words.
    pub words: usize,
    /// Declared launches, active or not.
    pub launches: usize,
    /// Total launch conditions declared across all launches.
    pub conditions: usize,
    /// Parameter ordinal pairs required disjoint by `AliasRule::Disjoint`.
    pub disjoint: &'static [[usize; 2]],
    /// Whether the entry parks (`CallSchema::parks`).
    pub parks: bool,
    /// Per schema result: `Some(rank)` for a tensor result, `None` for scalar/index/range.
    pub result_ranks: &'static [Option<usize>],
    /// Names of the native implementation's static dimensions.
    pub statics: &'static [&'static str],
    /// Sorted, deduplicated names of the entry's dynamic (non-static) dimensions
    /// referenced by `contract_sizing_at_point` (tensor result extents, scratch
    /// byte expressions, and scratch `when` predicates). Any dynamic dimension
    /// absent from this slice is never read by `contract_sizing_at_point`.
    pub sizing_dims: &'static [&'static str],
    /// Sorted, deduplicated subset of `sizing_dims` in which
    /// `contract_sizing_at_point` is proven at build time to be monotonically
    /// non-decreasing (holding all other dimensions, elements, and scratch
    /// configs fixed). For each dimension in this slice, the element-wise
    /// maximum over any regime slice box is achieved at that dimension's
    /// maximum value in the slice.
    pub monotonic_sizing_dims: &'static [&'static str],
    /// Sorted, deduplicated names of the entry's dynamic (non-static) dimensions
    /// referenced by any `launch.when` predicate on CUDA (empty on other backends
    /// or when no launch has a dynamic `when` predicate).
    pub launch_when_dims: &'static [&'static str],
}

/// Static metadata of one call-private scratch buffer, independent of
/// elements, dimensions, and tuning parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchMeta {
    pub sync: bool,
    /// The result ordinal whose emitted operand this buffer serves, if any.
    pub emission: Option<u32>,
    /// Static operand pairing role of this buffer, if any.
    pub operand: Option<crate::sizing::ScratchOperand>,
}

/// Compile-time static scratch metadata for an entry on one backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractScratchMeta {
    pub scratch: &'static [ScratchMeta],
}

/// Complete per-call sizing of an entry at one backend, element binding,
/// dimension assignment, and tuning parameter assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntrySizing {
    pub results: Vec<TensorResultSpec>,
    pub scratch: Vec<ScratchSpec>,
    pub active_launches: u64,
    pub operand_forms: Vec<OperandFormSpec>,
}

/// Admitted assignments of an entry's scratch-reading parameters at fixed statics.
/// Stored flat as `tuples` of stride `arity` (0 arity = 1 trivial admitted config).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchConfigs {
    pub arity: usize,
    pub tuples: Vec<u64>,
}

/// Compile-time generated contract for a checked Seismic entry.
///
/// Implemented automatically by `seismic-build` for every exported entry in
/// `OUT_DIR/<module>.rs`. Evaluates admission, default tuning parameters, and
/// per-call result/scratch/launch sizing as pure compiled Rust with no
/// `.seismicbundle` decoding and no `HashMap` allocation.
pub trait EntryContract: crate::Entry {
    fn contract_admit(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
        params: &[(&str, u64)],
    ) -> Result<(), ContractError>;

    fn contract_default_params(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
    ) -> Result<Vec<(&'static str, u64)>, ContractError>;

    fn contract_sizing(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
        params: &[(&str, u64)],
    ) -> Result<EntrySizing, ContractError>;

    /// Static sealed-node shape for `backend` (`plan_heap`, `GraphCensus`,
    /// disjoint parameter pairs, result ranks, and static dimension names).
    fn contract_node_shape(backend: BackendName) -> Result<ContractNodeShape, ContractError>;

    /// Static scratch buffer metadata (`sync`, `emission`, and static operand
    /// pairing role) for `backend`, with zero allocations or parameter loops.
    fn contract_scratch_meta(backend: BackendName) -> Result<ContractScratchMeta, ContractError>;

    /// Evaluate the `(element, extents, bytes)` of tensor parameter `parameter`
    /// at `(elements, dims)` for `local_for`/`input_for`/`port_for`.
    fn contract_param_spec(
        backend: BackendName,
        elements: &[(&str, Element)],
        parameter: &str,
        dims: &[(&str, u64)],
    ) -> Result<TensorParamSpec, ContractError>;

    /// Evaluate results, operand forms, active launches, and the **scratch
    /// bound** (maximum per-buffer charge over all admitted tuning parameter
    /// configurations, matching `NativeImplementation::scratch_bound`).
    fn contract_sizing_bound(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
    ) -> Result<EntrySizing, ContractError>;

    /// Compute the admitted assignments of this entry's scratch-reading parameters
    /// from its statics and elements alone (exact match with `NativeImplementation::scratch_configurations`).
    fn contract_scratch_configs(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
    ) -> Result<ScratchConfigs, ContractError>;

    /// Evaluate result and scratch byte sizes at `dims` over precomputed `configs`,
    /// updating `max_results[i] = max_results[i].max(bytes)` and
    /// `max_scratch[j] = max_scratch[j].max(bytes)` in place with ZERO heap allocations.
    fn contract_sizing_at_point(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
        configs: &ScratchConfigs,
        max_results: &mut [u64],
        max_scratch: &mut [u64],
    ) -> Result<(), ContractError>;

    /// Evaluate the active launch count at `(elements, dims, params)` under
    /// host-written condition word values `words` (`(parameter, index, nonzero)`),
    /// matching `NativeImplementation` launch filtering (`when`, `conditions`,
    /// and `repeat`).
    fn contract_active_launches(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
        params: &[(&str, u64)],
        words: &[(&str, u64, bool)],
    ) -> Result<u64, ContractError>;

    /// Enumerate admitted assignments of parameters read by any `launch.when`
    /// predicate on CUDA (mirroring `NativeImplementation::launch_configurations`).
    /// On non-CUDA backends, returns `Ok(vec![Vec::new()])`.
    fn contract_launch_configs(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
    ) -> Result<Vec<Vec<u64>>, ContractError>;

    /// Evaluate the `u64` bitmask of active launches (`launch.when`) on CUDA at
    /// `(elements, dims)` and one parameter assignment from
    /// [`Self::contract_launch_configs`]. Bit `i` is set iff launch `i` is
    /// active. On non-CUDA backends, returns `Ok(0)`.
    fn contract_launch_mask(
        backend: BackendName,
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
        params: &[u64],
    ) -> Result<u64, ContractError>;
}

/// Runtime-free helpers called by `seismic-build`-generated `EntryContract`
/// implementations.
#[doc(hidden)]
pub mod rt {
    use super::ContractError;
    use crate::{BackendName, Element};
    pub use seismic_lang::registry::RowPart;

    #[inline]
    pub fn element_named(name: &'static str) -> Element {
        Element::named(name).expect("checked entry names a registered representation")
    }

    #[inline]
    pub fn canonical_bytes(element: Element, extents: &[u64]) -> Result<u64, ContractError> {
        element
            .canonical_byte_len(extents)
            .map_err(|_| ContractError::ArithmeticOverflow)
    }

    #[inline]
    pub fn query_operands(element: Element) -> u64 {
        seismic_lang::checked::ElementQuery::Operands
            .of(element.id())
            .expect("operands query always has a value")
    }

    #[inline]
    pub fn query_form(backend: BackendName, name: &'static str, element: Element) -> u64 {
        let form = seismic_lang::forms::Form::named(backend, name)
            .expect("checked entry names a valid backend form");
        seismic_lang::checked::ElementQuery::Form(form)
            .of(element.id())
            .expect("form query always has a value")
    }

    #[inline]
    pub fn query_unit_columns(element: Element) -> Result<u64, ContractError> {
        seismic_lang::checked::ElementQuery::UnitColumns
            .of(element.id())
            .ok_or(ContractError::UnsupportedElements)
    }

    #[inline]
    pub fn query_packed(element: Element) -> u64 {
        seismic_lang::checked::ElementQuery::Packed
            .of(element.id())
            .expect("packed query always has a value")
    }

    #[inline]
    pub fn query_dense_bytes(element: Element) -> u64 {
        seismic_lang::checked::ElementQuery::DenseBytes
            .of(element.id())
            .expect("dense_bytes query always has a value")
    }

    #[inline]
    pub fn query_axis(name: &'static str, element: Element) -> u64 {
        let axis = seismic_lang::producers::Axis::named(name)
            .expect("checked entry names a valid decode axis");
        seismic_lang::checked::ElementQuery::Axis(axis)
            .of(element.id())
            .expect("axis query always has a value")
    }

    #[inline]
    pub fn query_row_bytes(part: RowPart, values: u64, element: Element) -> u64 {
        seismic_lang::checked::ElementQuery::RowBytes(part, values)
            .of(element.id())
            .expect("row_bytes query always has a value")
    }

    #[inline]
    pub fn ceil_div(lhs: u64, rhs: u64) -> Result<u64, ContractError> {
        if rhs == 0 {
            Err(ContractError::ArithmeticOverflow)
        } else {
            Ok(lhs.div_ceil(rhs))
        }
    }

    #[inline]
    pub fn align_up(lhs: u64, rhs: u64) -> Result<u64, ContractError> {
        if rhs == 0 {
            Err(ContractError::ArithmeticOverflow)
        } else {
            lhs.checked_next_multiple_of(rhs)
                .ok_or(ContractError::ArithmeticOverflow)
        }
    }
}
