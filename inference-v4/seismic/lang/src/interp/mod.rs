//! Semantic oracle over a monomorphized [`LogicalEntry`]. Deterministic
//! programs return one value; unordered floating associations return an
//! explicit allowed-outcome relation plus one non-authoritative witness.
//!
//! This is not a second compilation or production path. It executes the
//! sealed reference candidate of each semantic family in source order and is
//! used only by differential validation. It consumes no SIR, checker symbols,
//! ABI layouts, schedules, or backend code.

#[cfg(test)]
mod failure_tests;
mod oracle;
mod scalar;
mod tensor;
pub mod value;

pub use tensor::{round_to, TensorData};
pub use value::Value;

use crate::entry::{AssociationOutcome, LogicalEntry, LogicalEntryView};
use crate::failure::{SourceFailure, SourceFailureCause, SourceTermination};
use crate::ids::NodeId;
use crate::reference_math::ReferenceScalar;
use crate::types::DType;

/// One flattened invocation argument, in [`crate::entry::CallSchema`] order.
#[derive(Clone, Debug)]
pub enum Arg {
    Tensor(usize),
    Scalar(ReferenceScalar),
    Index(num_bigint::BigUint),
    Range(num_bigint::BigUint, num_bigint::BigUint),
}

/// A reference execution of exactly one monomorphized entry.
pub struct Interpreter<'a> {
    entry: LogicalEntryView<'a>,
    runtime_symbols: oracle::RuntimeSymbols,
    tensors: Vec<TensorData>,
    work: Option<WorkBudget>,
    memory: Option<std::rc::Rc<MemoryBudget>>,
    associations: Vec<AllowedAssociation>,
    parallel_regions: Vec<NodeId>,
    metadata_reservations: Vec<MemoryReservation>,
}

struct WorkBudget {
    initial: u64,
    remaining: std::cell::Cell<u64>,
}

/// Tracks owned reference payloads, including transferred inputs. Native
/// observations are accounted by the caller before readback.
#[derive(Debug)]
struct MemoryBudget {
    limit: u64,
    live: std::cell::Cell<u64>,
    peak: std::cell::Cell<u64>,
}
#[derive(Debug)]
pub(super) struct MemoryReservation {
    budget: Option<std::rc::Rc<MemoryBudget>>,
    bytes: u64,
}
impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if let Some(budget) = &self.budget {
            budget.live.set(budget.live.get() - self.bytes);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleError {
    InvalidInvocation(String),
    InterpreterDefect(String),
    WorkLimit { limit: u64 },
    MemoryLimit { limit: u64, required: u64 },
}
impl std::fmt::Display for OracleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInvocation(reason) => f.write_str(reason),
            Self::InterpreterDefect(reason) => write!(f, "reference interpreter defect: {reason}"),
            Self::MemoryLimit { limit, required } => {
                write!(
                    f,
                    "reference execution requires {required} live payload bytes, limit {limit}"
                )
            }
            Self::WorkLimit { limit } => {
                write!(f, "reference execution exceeded {limit} work units")
            }
        }
    }
}
impl std::error::Error for OracleError {}

/// Typed propagation inside source evaluation. Recipe failures acquire their
/// checked event at the operation dispatcher, before crossing a call boundary.
/// A contradiction of the checked entry graph is a panic, never a value here.
#[derive(Debug)]
pub(super) enum EvalError {
    Source(SourceFailure),
    Scalar(crate::reference_math::ScalarFailure),
    Service(OracleError),
}
impl From<OracleError> for EvalError {
    fn from(error: OracleError) -> Self {
        Self::Service(error)
    }
}
impl From<crate::reference_math::ScalarFailure> for EvalError {
    fn from(cause: crate::reference_math::ScalarFailure) -> Self {
        Self::Scalar(cause)
    }
}
impl EvalError {
    fn at(self, function: &crate::entry::SemanticFunction, node: NodeId) -> Self {
        match self {
            Self::Scalar(cause) => Self::Source(SourceFailure::at(
                function,
                node,
                SourceFailureCause::Scalar(cause),
            )),
            error => error,
        }
    }
}

/// Completed reference execution. All backing is owned here; no interpreter or
/// semantic arena is required to inspect it, and no writer is exposed.
#[derive(Debug)]
pub struct OracleOutcome {
    termination: SourceTermination<OracleReturns>,
    inputs: Vec<(usize, usize, crate::entry::TensorAccess)>,
    tensors: Vec<TensorData>,
    relation: Option<AllowedOutcomeRelation>,
    memory: std::rc::Rc<MemoryBudget>,
    _reservations: Vec<MemoryReservation>,
    work_units: u64,
}
#[derive(Debug)]
pub struct OracleReturns {
    values: Vec<Value>,
    results: Vec<(Vec<u32>, OutcomeKind)>,
}

#[derive(Clone, Copy, Debug)]
enum OutcomeKind {
    Tensor,
    Scalar(DType),
    Index,
    Range,
}
pub struct OutcomeResult<'a> {
    outcome: &'a OracleOutcome,
    index: usize,
}
pub enum OutcomeValue<'a> {
    Tensor(TensorReader<'a>),
    Scalar(ReferenceScalar),
    Index(&'a num_bigint::BigUint),
    Range(&'a num_bigint::BigUint, &'a num_bigint::BigUint),
}
pub struct FinalTensorInput<'a> {
    outcome: &'a OracleOutcome,
    index: usize,
}
enum TensorBackingRead<'a> {
    Argument(&'a TensorData),
    Owned(std::cell::Ref<'a, TensorData>),
}
/// Read-only logical view into completed reference storage.
pub struct TensorReader<'a> {
    backing: TensorBackingRead<'a>,
    view: Option<&'a value::TensorValue>,
}
impl TensorReader<'_> {
    fn storage(&self) -> &TensorData {
        match &self.backing {
            TensorBackingRead::Argument(t) => t,
            TensorBackingRead::Owned(t) => t,
        }
    }
    pub fn shape(&self) -> &[usize] {
        self.view
            .map(|v| v.shape())
            .unwrap_or_else(|| self.storage().shape())
    }
    pub fn representation(&self) -> crate::ids::RepresentationId {
        self.storage().representation()
    }
    pub fn element_count(&self) -> usize {
        self.shape().iter().product()
    }
    pub fn canonical_byte_len(&self) -> Result<usize, String> {
        if let crate::registry::RepresentationKind::Dense(dtype) =
            crate::registry::representation_info(self.representation()).kind
        {
            self.shape()
                .iter()
                .try_fold(dtype.bytes() as usize, |n, extent| n.checked_mul(*extent))
                .ok_or_else(|| "tensor byte geometry overflow".into())
        } else {
            tensor::encoded_bytes(self.representation(), self.shape())
        }
    }
    pub fn decoder_workspace_bytes(&self) -> u64 {
        let info = crate::registry::representation_info(self.representation());
        crate::registry::decode_recipe(self.representation(), info.decoded)
            .map(|r| r.temporary_count() as u64 * 8)
            .unwrap_or(0)
    }
    pub fn read(&self, index: usize) -> Result<f64, String> {
        if index >= self.element_count() {
            return Err("tensor index outside logical shape".into());
        }
        let flat = self.view.map(|v| v.positions[index]).unwrap_or(index);
        Ok(self.storage().read(flat))
    }
    /// Borrow canonical storage when this reader covers the complete backing.
    /// Views with a different logical order are inspected elementwise instead.
    pub fn canonical_bytes(&self) -> Result<Option<&[u8]>, String> {
        if let Some(v) = self.view {
            if v.shape() != self.storage().shape()
                || !v.positions.iter().enumerate().all(|(a, b)| a == *b)
            {
                return Ok(None);
            }
        }
        self.storage().canonical_bytes().map(Some)
    }
}
impl OracleOutcome {
    pub fn termination(&self) -> &SourceTermination<OracleReturns> {
        &self.termination
    }
    pub fn results(&self) -> impl ExactSizeIterator<Item = OutcomeResult<'_>> {
        let count = match &self.termination {
            SourceTermination::Returned(values) => values.results.len(),
            SourceTermination::Failed(_) => 0,
        };
        (0..count).map(|index| OutcomeResult {
            outcome: self,
            index,
        })
    }
    pub fn inputs(&self) -> impl ExactSizeIterator<Item = FinalTensorInput<'_>> {
        (0..self.inputs.len()).map(|index| FinalTensorInput {
            outcome: self,
            index,
        })
    }
    pub fn relation(&self) -> Option<&AllowedOutcomeRelation> {
        self.relation.as_ref()
    }
    pub fn retained_payload_bytes(&self) -> u64 {
        self.memory.live.get()
    }
    pub fn peak_payload_bytes(&self) -> u64 {
        self.memory.peak.get()
    }
    pub fn work_units(&self) -> u64 {
        self.work_units
    }
    fn tensor<'b>(&'b self, value: &'b value::TensorValue) -> TensorReader<'b> {
        // Called only with this outcome's private values.
        match &value.backing {
            value::Backing::Argument(i) => TensorReader {
                backing: TensorBackingRead::Argument(&self.tensors[*i]),
                view: Some(value),
            },
            value::Backing::Owned(t) => TensorReader {
                backing: TensorBackingRead::Owned(t.borrow()),
                view: Some(value),
            },
        }
    }
}
impl OutcomeResult<'_> {
    fn returned(&self) -> &OracleReturns {
        let SourceTermination::Returned(values) = &self.outcome.termination else {
            unreachable!("failed outcome has no successful result")
        };
        values
    }
    pub fn path(&self) -> &[u32] {
        &self.returned().results[self.index].0
    }
    pub fn value(&self) -> OutcomeValue<'_> {
        match (
            self.returned().results[self.index].1,
            &self.returned().values[self.index],
        ) {
            (OutcomeKind::Tensor, Value::Tensor(t)) => OutcomeValue::Tensor(self.outcome.tensor(t)),
            (OutcomeKind::Scalar(dtype), Value::Scalar(v)) => {
                assert_eq!(dtype, v.dtype());
                OutcomeValue::Scalar(*v)
            }
            (OutcomeKind::Index, Value::Index(v)) => OutcomeValue::Index(v),
            (OutcomeKind::Range, Value::Range(a, b)) => OutcomeValue::Range(a, b),
            _ => unreachable!("completed reference result disagrees with checked kind"),
        }
    }
}
impl FinalTensorInput<'_> {
    pub fn ordinal(&self) -> usize {
        self.outcome.inputs[self.index].0
    }
    pub fn access(&self) -> crate::entry::TensorAccess {
        self.outcome.inputs[self.index].2
    }
    pub fn tensor(&self) -> TensorReader<'_> {
        TensorReader {
            backing: TensorBackingRead::Argument(
                &self.outcome.tensors[self.outcome.inputs[self.index].1],
            ),
            view: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowedOutcomeRelation {
    associations: Vec<AllowedAssociation>,
    parallel_regions: Vec<NodeId>,
}

impl AllowedOutcomeRelation {
    /// Actually entered parallel regions can choose different permitted failure
    /// prefixes. This describes allowed behavior; it does not prove membership.
    pub fn parallel_regions(&self) -> &[NodeId] {
        &self.parallel_regions
    }
    pub fn associations(&self) -> &[AllowedAssociation] {
        &self.associations
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowedAssociation {
    pub node: NodeId,
    pub outcome: AssociationOutcome,
}

impl<'a> Interpreter<'a> {
    pub fn new(entry: &'a LogicalEntry) -> Self {
        Self::from_view(entry.as_view())
    }

    pub fn from_view(entry: LogicalEntryView<'a>) -> Self {
        Self {
            entry,
            runtime_symbols: oracle::runtime_symbols(entry.arena()),
            tensors: Vec::new(),
            work: None,
            memory: None,
            associations: Vec::new(),
            parallel_regions: Vec::new(),
            metadata_reservations: Vec::new(),
        }
    }

    pub fn run_bounded(self, arguments: &[Arg], limit: u64) -> Result<OracleOutcome, OracleError> {
        self.run_bounded_with_memory(arguments, limit, u64::MAX)
    }
    pub fn run(self, arguments: &[Arg]) -> Result<OracleOutcome, OracleError> {
        self.run_bounded(arguments, u64::MAX)
    }
    pub fn run_bounded_with_memory(
        mut self,
        arguments: &[Arg],
        work_limit: u64,
        memory_bytes: u64,
    ) -> Result<OracleOutcome, OracleError> {
        self.work = Some(WorkBudget {
            initial: work_limit,
            remaining: std::cell::Cell::new(work_limit),
        });
        self.memory = Some(std::rc::Rc::new(MemoryBudget {
            limit: memory_bytes,
            live: std::cell::Cell::new(0),
            peak: std::cell::Cell::new(0),
        }));
        for tensor in &self.tensors {
            self.metadata_reservations
                .push(self.reserve_memory(tensor.storage_bytes())?);
        }
        let termination = match self.run_reference(arguments) {
            Ok(values) => SourceTermination::Returned(values),
            Err(EvalError::Source(failure)) => SourceTermination::Failed(failure),
            Err(EvalError::Service(error)) => return Err(error),
            Err(EvalError::Scalar(_)) => {
                unreachable!("source scalar failure lost its operation identity")
            }
        };
        self.finish(arguments, termination)
    }
    fn finish(
        mut self,
        arguments: &[Arg],
        termination: SourceTermination<Vec<Value>>,
    ) -> Result<OracleOutcome, OracleError> {
        let values_bytes = match &termination {
            SourceTermination::Returned(values) => {
                values.capacity().checked_mul(std::mem::size_of::<Value>())
            }
            SourceTermination::Failed(failure) => Some(
                std::mem::size_of::<SourceFailure>()
                    + match &failure.cause {
                        SourceFailureCause::Check(crate::entry::CheckReason::Custom(text)) => {
                            text.capacity()
                        }
                        _ => 0,
                    },
            ),
        };
        let successful = matches!(termination, SourceTermination::Returned(_));
        use crate::entry::{ParameterKind, ResultKind};
        let bytes = self
            .entry
            .schema()
            .results()
            .iter()
            .filter(|_| successful)
            .try_fold(0usize, |sum, r| {
                sum.checked_add(std::mem::size_of::<(Vec<u32>, OutcomeKind)>())?
                    .checked_add(r.path.len().checked_mul(4)?)
            })
            .and_then(|n| {
                n.checked_add(arguments.len().checked_mul(std::mem::size_of::<(
                    usize,
                    usize,
                    crate::entry::TensorAccess,
                )>())?)
            })
            .and_then(|n| n.checked_add(values_bytes?))
            .and_then(|n| {
                n.checked_add(
                    self.tensors
                        .capacity()
                        .checked_mul(std::mem::size_of::<TensorData>())?,
                )
            })
            .ok_or_else(|| self.memory_size_overflow())?;
        self.metadata_reservations
            .push(self.reserve_memory(bytes as u64)?);
        let results = self
            .entry
            .schema()
            .results()
            .iter()
            .filter(|_| successful)
            .map(|r| {
                (
                    r.path.clone(),
                    match r.kind {
                        ResultKind::Tensor { .. } => OutcomeKind::Tensor,
                        ResultKind::Scalar(d) => OutcomeKind::Scalar(d),
                        ResultKind::Index { .. } => OutcomeKind::Index,
                        ResultKind::Range { .. } => OutcomeKind::Range,
                    },
                )
            })
            .collect();
        let inputs = self
            .entry
            .schema()
            .parameters()
            .iter()
            .zip(arguments)
            .enumerate()
            .filter_map(|(ordinal, (p, a))| match (&p.kind, a) {
                (ParameterKind::Tensor { access, .. }, Arg::Tensor(index)) => {
                    Some((ordinal, *index, *access))
                }
                _ => None,
            })
            .collect();
        let work = self.work.take().unwrap();
        let termination = match termination {
            SourceTermination::Returned(values) => {
                SourceTermination::Returned(OracleReturns { values, results })
            }
            SourceTermination::Failed(failure) => SourceTermination::Failed(failure),
        };
        Ok(OracleOutcome {
            termination,
            inputs,
            tensors: self.tensors,
            relation: (!self.associations.is_empty() || !self.parallel_regions.is_empty())
                .then_some(AllowedOutcomeRelation {
                    associations: self.associations,
                    parallel_regions: self.parallel_regions,
                }),
            memory: self.memory.take().unwrap(),
            _reservations: self.metadata_reservations,
            work_units: work.initial - work.remaining.get(),
        })
    }

    fn reserve_memory(&self, bytes: u64) -> Result<MemoryReservation, OracleError> {
        if let Some(budget) = &self.memory {
            let total = budget.live.get().checked_add(bytes);
            if total.is_none_or(|total| total > budget.limit) {
                return Err(OracleError::MemoryLimit {
                    limit: budget.limit,
                    required: total.unwrap_or(u64::MAX),
                });
            }
            budget.live.set(total.unwrap());
            budget.peak.set(budget.peak.get().max(total.unwrap()));
        }
        Ok(MemoryReservation {
            budget: self.memory.clone(),
            bytes,
        })
    }
    fn memory_size_overflow(&self) -> OracleError {
        OracleError::MemoryLimit {
            limit: self.memory.as_ref().map_or(u64::MAX, |budget| budget.limit),
            required: u64::MAX,
        }
    }
    fn reserve_elements<T>(&self, count: usize) -> Result<MemoryReservation, OracleError> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| self.memory_size_overflow())?;
        self.reserve_memory(bytes)
    }
    fn reserve_tensor_value(
        &self,
        representation: crate::ids::RepresentationId,
        shape: &[usize],
        owned: bool,
    ) -> Result<value::TensorMemory, OracleError> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, x| n.checked_mul(*x))
            .ok_or_else(|| self.memory_size_overflow())?;
        Ok(value::TensorMemory {
            _backing: if owned {
                Some(std::rc::Rc::new(
                    self.reserve_memory(
                        TensorData::allocation_bytes(representation, shape)
                            .map_err(|_| self.memory_size_overflow())?,
                    )?,
                ))
            } else {
                None
            },
            shape: std::rc::Rc::new(self.reserve_elements::<usize>(shape.len())?),
            positions: std::rc::Rc::new(self.reserve_elements::<usize>(count)?),
        })
    }

    fn charge_work(&self, units: u64) -> Result<(), OracleError> {
        if let Some(budget) = &self.work {
            let Some(remaining) = budget.remaining.get().checked_sub(units) else {
                return Err(OracleError::WorkLimit {
                    limit: budget.initial,
                });
            };
            budget.remaining.set(remaining);
        }
        Ok(())
    }

    fn charge_tensor(&self, shape: &[usize]) -> Result<(), OracleError> {
        let Some(count) = shape
            .iter()
            .try_fold(1u64, |count, extent| count.checked_mul(*extent as u64))
        else {
            // No work budget admits more than u64::MAX units.
            return Err(OracleError::WorkLimit {
                limit: self.work.as_ref().map_or(u64::MAX, |budget| budget.initial),
            });
        };
        self.charge_work(count)
    }

    pub fn add_tensor(&mut self, tensor: TensorData) -> usize {
        self.tensors.push(tensor);
        self.tensors.len() - 1
    }

    fn record_association(
        &mut self,
        node: NodeId,
        outcome: AssociationOutcome,
    ) -> Result<(), OracleError> {
        let association = AllowedAssociation { node, outcome };
        if !self.associations.contains(&association) {
            self.metadata_reservations
                .push(self.reserve_elements::<AllowedAssociation>(1)?);
            self.associations.reserve_exact(1);
            self.associations.push(association);
        }
        Ok(())
    }
    fn record_parallel(&mut self, node: NodeId) -> Result<(), OracleError> {
        if !self.parallel_regions.contains(&node) {
            self.metadata_reservations
                .push(self.reserve_elements::<NodeId>(1)?);
            self.parallel_regions.reserve_exact(1);
            self.parallel_regions.push(node);
        }
        Ok(())
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::checked::{check_source, SourceFile, SourceSet};
    use crate::entry::ElementBindings;
    fn entry(source: &str) -> LogicalEntry {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "bounded-reference.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap()
    }
    #[test]
    fn large_allocation_stops_before_materializing_elements() {
        let entry = entry("fn probe() -> tensor[1000000000] f32:\n    let mut output = tensor[1000000000] f32\n    parallel for i in 0..1000000000:\n        output[i] = 1.0\n    return output\n");
        assert!(matches!(
            Interpreter::new(&entry).run_bounded(&[], 128),
            Err(OracleError::WorkLimit { limit: 128 })
        ));
        assert!(matches!(
            Interpreter::new(&entry).run_bounded_with_memory(&[], u64::MAX, 1024),
            Err(OracleError::MemoryLimit { limit: 1024, .. })
        ));
    }
    #[test]
    fn transferred_inputs_are_charged_before_execution() {
        let entry =
            entry("fn probe[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n");
        let mut interpreter = Interpreter::new(&entry);
        let input =
            interpreter.add_tensor(TensorData::dense(DType::F32, vec![1000], vec![1.0; 1000]));
        assert!(matches!(
            interpreter.run_bounded_with_memory(&[Arg::Tensor(input)], u64::MAX, 1024),
            Err(OracleError::MemoryLimit { limit: 1024, .. })
        ));
    }
    #[test]
    fn outcome_owns_backing_after_entry_is_gone_without_copying_input() {
        let (outcome, address) = {
            let entry=entry("fn probe[N,W](x: &tensor[N,W] f32) -> tensor[W] f32:\n    let mut row = to_owned(x[0])\n    for j in 0..W:\n        row[j] = row[j] + 1.0\n    return row\n");
            let tensor = TensorData::dense(DType::F32, vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
            let address = match &tensor {
                TensorData::Dense { bytes, .. } => bytes.as_ptr(),
                _ => unreachable!(),
            };
            let mut interpreter = Interpreter::new(&entry);
            let index = interpreter.add_tensor(tensor);
            (
                interpreter
                    .run_bounded_with_memory(&[Arg::Tensor(index)], 1000, 4096)
                    .unwrap(),
                address,
            )
        };
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(row) = result.value() else {
            panic!()
        };
        assert_eq!(row.shape(), [3]);
        assert_eq!(
            (0..3).map(|i| row.read(i).unwrap()).collect::<Vec<_>>(),
            [2., 3., 4.]
        );
        let input = outcome.inputs().next().unwrap();
        assert_eq!(input.tensor().read(0).unwrap(), 1.);
        assert_eq!(
            match &outcome.tensors[0] {
                TensorData::Dense { bytes, .. } => bytes.as_ptr(),
                _ => unreachable!(),
            },
            address
        );
        assert!(outcome.retained_payload_bytes() > 0);
        assert!(outcome.peak_payload_bytes() >= outcome.retained_payload_bytes());
    }
    #[test]
    fn dense_view_snapshot_preserves_original_element_bits() {
        let entry = entry(
            "fn probe[W](x: &tensor[2,W] f32) -> tensor[W] f32:\n    return to_owned(x[1])\n",
        );
        let words = [1f32.to_bits(), 2f32.to_bits(), 0x8000_0000, 0x7f80_0001];
        let bytes = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter.add_tensor(
            TensorData::dense_from_bytes(DType::F32, vec![2, 2], bytes.clone()).unwrap(),
        );
        let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(snapshot) = result.value() else {
            panic!()
        };
        assert_eq!(snapshot.canonical_bytes().unwrap().unwrap(), &bytes[8..]);
        assert_eq!(
            outcome
                .inputs()
                .next()
                .unwrap()
                .tensor()
                .canonical_bytes()
                .unwrap()
                .unwrap(),
            bytes
        );
    }

    #[test]
    fn associations_follow_executed_helpers_and_not_inactive_branches() {
        let entry=entry("fn helper[N](x: &tensor[N] f32) -> f32:\n    return reduce(x, 0, sum, unordered=true)\n\nfn probe[N](x: &tensor[N] f32, enabled: bool) -> f32:\n    let mut result = 0.0\n    if enabled:\n        result = helper(x)\n    return result\n");
        for enabled in [false, true] {
            let mut interpreter = Interpreter::new(&entry);
            let index =
                interpreter.add_tensor(TensorData::dense(DType::F32, vec![3], vec![1., 2., 3.]));
            let outcome = interpreter
                .run(&[
                    Arg::Tensor(index),
                    Arg::Scalar(ReferenceScalar::Bool(enabled)),
                ])
                .unwrap();
            assert_eq!(outcome.relation().is_some(), enabled);
        }
    }
    #[test]
    fn final_mutated_input_and_snapshot_are_owned_together() {
        let entry=entry("fn probe[N](x: &mut tensor[N] f32) -> tensor[N] f32:\n    let snapshot = to_owned(x)\n    for i in 0..N:\n        x[i] = x[i] + 1.0\n    return snapshot\n");
        let mut interpreter = Interpreter::new(&entry);
        let index = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![2., 3.]));
        let outcome = interpreter.run(&[Arg::Tensor(index)]).unwrap();
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(snapshot) = result.value() else {
            panic!()
        };
        assert_eq!(snapshot.read(0).unwrap(), 2.);
        assert_eq!(
            outcome.inputs().next().unwrap().tensor().read(0).unwrap(),
            3.
        );
    }
    #[test]
    fn shared_arguments_keep_one_backing_and_repeated_associations_stay_bounded() {
        let entry=entry("fn helper[N](x: &tensor[N] f32) -> f32:\n    return reduce(x, 0, sum, unordered=true)\n\nfn probe[N](x: &tensor[N] f32, y: &tensor[N] f32, times: range[N]) -> f32:\n    let mut result = 0.0\n    for i in times:\n        result = helper(x) + helper(y)\n    return result\n");
        let mut counts = Vec::new();
        for times in [2, 2, 2] {
            let mut interpreter = Interpreter::new(&entry);
            let index =
                interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![1., 2.]));
            let outcome = interpreter
                .run(&[Arg::Tensor(index), Arg::Tensor(index), Arg::Range(0u8.into(), u32::try_from(times).unwrap().into())])
                .unwrap();
            assert_eq!(outcome.tensors.len(), 1);
            assert_eq!(outcome.inputs().count(), 2);
            counts.push(
                outcome
                    .relation()
                    .map(|r| r.associations().len())
                    .unwrap_or(0),
            );
        }
        assert_eq!(counts[0], counts[1]);
        assert!(counts[1] > 0);
        assert_eq!(counts[1], counts[2]);
    }
    #[test]
    fn range_loops_preserve_actual_endpoints_and_executed_associations() {
        let entry = entry("fn helper(x: &tensor[4] f32) -> f32:\n    return reduce(x, 0, sum, unordered=true)\n\nfn probe(x: &tensor[4] f32, times: range[4]) -> f32:\n    let mut result = 0.0\n    for i in times:\n        result = result + x[i] + helper(x)\n    return result\n");
        for (start, end, expected) in [
            (0, 0, 0.0),
            (2, 2, 0.0),
            (0, 1, 11.0),
            (1, 3, 25.0),
            (0, 4, 50.0),
        ] {
            let mut interpreter = Interpreter::new(&entry);
            let input = interpreter.add_tensor(TensorData::dense(
                DType::F32,
                vec![4],
                vec![1., 2., 3., 4.],
            ));
            let outcome = interpreter
                .run(&[Arg::Tensor(input), Arg::Range(u32::try_from(start).unwrap().into(), u32::try_from(end).unwrap().into())])
                .unwrap();
            assert!(
                matches!(outcome.results().next().unwrap().value(), OutcomeValue::Scalar(ReferenceScalar::F32(actual)) if f32::from_bits(actual) as f64 == expected)
            );
            assert_eq!(outcome.relation().is_some(), start < end);
        }
    }

    #[test]
    fn statically_empty_loop_does_not_collect_body_associations() {
        let entry=entry("fn helper[N](x: &tensor[N] f32) -> f32:\n    return reduce(x, 0, sum, unordered=true)\n\nfn probe[N](x: &tensor[N] f32) -> f32:\n    let mut result = 0.0\n    for i in 0..0:\n        result = helper(x)\n    return result\n");
        let mut interpreter = Interpreter::new(&entry);
        let index = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![1., 2.]));
        let outcome = interpreter.run(&[Arg::Tensor(index)]).unwrap();
        assert!(outcome.relation().is_none());
    }
    #[test]
    fn retry_constructs_a_new_execution() {
        let entry = entry("fn probe(x: f32) -> f32:\n    return x + x\n");
        let args = [Arg::Scalar(ReferenceScalar::F32(2f32.to_bits()))];
        assert!(matches!(
            Interpreter::new(&entry).run_bounded(&args, 0),
            Err(OracleError::WorkLimit { limit: 0 })
        ));
        let outcome = Interpreter::new(&entry).run_bounded(&args, 100).unwrap();
        assert!(matches!(
            outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::F32(0x4080_0000))
        ));
    }
    #[test]
    fn narrow_payloads_survive_helpers_views_and_scalar_publication() {
        let entry = entry("fn keep(x: f16) -> f16:\n    return x\n\nfn probe(x: &tensor[2] f16) -> tensor[2] f16:\n    let mut result = to_owned(x)\n    result[0] = keep(x[0])\n    result[1] = abs(x[1])\n    return result\n");
        let bytes = [0xfc01u16, 0xfc01]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter
            .add_tensor(TensorData::dense_from_bytes(DType::F16, vec![2], bytes.clone()).unwrap());
        let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(result) = result.value() else {
            panic!()
        };
        assert_eq!(
            result.canonical_bytes().unwrap().unwrap(),
            &[1, 0xfc, 1, 0x7c]
        );
        assert_eq!(
            outcome
                .inputs()
                .next()
                .unwrap()
                .tensor()
                .canonical_bytes()
                .unwrap()
                .unwrap(),
            bytes
        );
    }
    #[test]
    fn natural_argument_and_result_transport_retains_the_upper_sign_bit() {
        let entry = entry("fn keep[N](shape: &tensor[0, N] f32, x: index[N]) -> index[N]:\n    return x\n\nfn probe[N](shape: &tensor[0, N] f32, x: index[N]) -> index[N]:\n    return keep(shape, x)\n");
        let value = (1u64 << 63) + 17;
        let mut interpreter = Interpreter::new(&entry);
        // A zero-element tensor supplies a full-width natural shape binding without
        // allocating storage or relying on the current signed shape-literal parser.
        let shape =
            interpreter.add_tensor(TensorData::dense(DType::F32, vec![0, usize::MAX], vec![]));
        let outcome = interpreter
            .run(&[Arg::Tensor(shape), Arg::Index(value.into())])
            .unwrap();
        assert!(
            matches!(outcome.results().next().unwrap().value(), OutcomeValue::Index(actual) if actual == &value.into())
        );
    }
}
