//! Consuming execution ownership for an already-admitted workflow run.
//!
//! This module sequences admitted schedules and completed producer publication.
//! The workflow continuation retains argument binding, variant selection, and
//! resource admission authority; execution invokes it only after the prior
//! native prefix is complete. Terminal ownership retains every issued resource.

use crate::api::kernel::{DecodedValue, WorkflowCompletionAny};
use crate::api::CallError;
use crate::driver::{AdmittedCommand, Allocation, Opened, Service};
use crate::resources::AdmittedResources;
use seismic_compiler::errors::{CheckFailure, ExecutionError};
use seismic_compiler::executable::{NativeExecution, NativeExecutor, NativeSubmission};
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::expr::SymbolValue;
use seismic_lang::failure::SourceTermination;
use seismic_native_target::TargetFamily;
use std::sync::Arc;

type Submission<T, E> = <E as NativeExecutor<T>>::Submission;
type SubmittedExecution<T, E> = <Submission<T, E> as NativeSubmission<T>>::Execution;

/// A result descriptor whose layout and allocation have already been closed
/// by admission. Decoding cannot consult an executable expression.
pub(crate) enum AdmittedOutput {
    PendingTensor {
        path: Vec<u32>,
    },
    Tensor {
        allocation: Arc<Allocation>,
        byte_offset: u64,
        byte_len: u64,
        representation: seismic_lang::ids::RepresentationId,
        extents: Vec<u64>,
        strides: Vec<u64>,
    },
    Scalar(AdmittedScalarOutput),
}

pub(crate) enum AdmittedScalarOutput {
    Value {
        dtype: seismic_lang::types::DType,
        symbol: seismic_lang::expr::SymbolId,
    },
    Index {
        symbol: seismic_lang::expr::SymbolId,
    },
    Range {
        start: seismic_lang::expr::SymbolId,
        end: seismic_lang::expr::SymbolId,
    },
}

pub(crate) struct AdmittedNode<T: TargetFamily, E: NativeExecutor<T>> {
    pub(crate) command: AdmittedCommand<T, E>,
    pub(crate) outputs: Vec<AdmittedOutput>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> AdmittedNode<T, E> {
    pub(crate) fn new(command: AdmittedCommand<T, E>, outputs: Vec<AdmittedOutput>) -> Self {
        Self { command, outputs }
    }

    fn allocated_bytes(&self) -> u64 {
        self.command.allocated_bytes()
    }

    fn decode_values(&self) -> Result<Vec<DecodedValue>, CallError> {
        let Self { command, outputs } = self;
        outputs
            .iter()
            .map(|output| output.decode(command.values(), command.output_device()))
            .collect()
    }
}

/// The only input accepted by execution: a fully admitted run with a distinct
/// backend submission and opaque retained admission ownership.
pub(crate) struct AdmittedRun<T: TargetFamily, E: NativeExecutor<T>> {
    identity: u64,
    device: Arc<Service<T, E>>,
    opened: Arc<Opened<T, E>>,
    nodes: Vec<AdmittedNode<T, E>>,
    retained: AdmittedResources,
    submission: Submission<T, E>,
    continuation: Option<crate::driver::workflow::native::WorkflowContinuation<T, E>>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> AdmittedRun<T, E> {
    pub(crate) fn new(
        identity: u64,
        device: Arc<Service<T, E>>,
        opened: Arc<Opened<T, E>>,
        nodes: Vec<AdmittedNode<T, E>>,
        retained: AdmittedResources,
        submission: Submission<T, E>,
    ) -> Self {
        Self {
            identity,
            device,
            opened,
            nodes,
            retained,
            submission,
            continuation: None,
        }
    }

    pub(crate) fn set_continuation(
        &mut self,
        continuation: crate::driver::workflow::native::WorkflowContinuation<T, E>,
    ) {
        self.continuation = Some(continuation);
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.nodes
            .iter()
            .map(AdmittedNode::allocated_bytes)
            .fold(self.retained.reached_allocated(), u64::saturating_add)
    }

    /// The values produced by ordinary binding for each admitted node.
    pub(crate) fn invocation_values(&self) -> impl ExactSizeIterator<Item = &InvocationValues> {
        self.nodes.iter().map(|node| node.command.values())
    }

    pub(crate) fn submit(mut self) -> Result<SubmittedRun<T, E>, CallError> {
        let mut issued = Ok(());
        let mut termination = SourceTermination::Returned(());
        let mut deferred_error = None;
        let mut ordinal = 0usize;
        while ordinal < self.nodes.len() {
            let node = &mut self.nodes[ordinal];
            if let Err(error) = node.command.issue(
                &mut self.submission,
                &self.device,
                &mut self.retained,
                &self.opened,
            ) {
                match error {
                    ExecutionError::DataCheckFailed(failure) => {
                        termination = SourceTermination::Failed(failure)
                    }
                    error => issued = Err(error),
                }
                break;
            }
            let published = match node.command.published_tensors(&self.retained) {
                Ok(published) => published,
                Err(error) => {
                    issued = Err(error);
                    break;
                }
            };
            let mut published = published
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            for output in &mut node.outputs {
                if let AdmittedOutput::PendingTensor { path } = output {
                    let Some(value) = published.remove(path) else {
                        issued = Err(ExecutionError::ConstructionContradiction(
                            "successful command omitted tensor publication".into(),
                        ));
                        break;
                    };
                    *output = value;
                }
            }
            if issued.is_err() {
                break;
            }
            if let Some(continuation) = &mut self.continuation {
                if let Err(error) = self.submission.complete_prefix() {
                    issued = Err(error);
                    break;
                }
                if let Err(error) = continuation.completed(
                    ordinal as u32,
                    node,
                    self.opened.identity(),
                    &mut self.retained,
                ) {
                    deferred_error = Some(error);
                    break;
                }
                match continuation.next(&self.opened, &mut self.retained) {
                    Ok(Some(next)) => self.nodes.push(next),
                    Ok(None) => {}
                    Err(error) => {
                        deferred_error = Some(error);
                        break;
                    }
                }
            }
            ordinal += 1;
        }

        // Even a partially issued schedule is submitted and synchronized. The
        // completion owner retains the complete admission transaction until
        // that synchronization finishes.
        let native = self.submission.submit();
        let completion = CompletionOwner::new(
            BackendCompletion(native),
            SubmittedState {
                nodes: self.nodes,
                termination,
                retained: self.retained,
            },
        );
        let completion = finish_issue(completion, issued).map_err(CallError::Execution)?;
        if let Some(error) = deferred_error {
            completion
                .complete_and_take()
                .map_err(CallError::Execution)?;
            return Err(error);
        }
        Ok(SubmittedRun {
            identity: self.identity,
            completion,
        })
    }
}

pub(crate) struct SubmittedRun<T: TargetFamily, E: NativeExecutor<T>> {
    identity: u64,
    completion: CompletionOwner<BackendCompletion<T, E>, SubmittedState<T, E>>,
}

struct SubmittedState<T: TargetFamily, E: NativeExecutor<T>> {
    nodes: Vec<AdmittedNode<T, E>>,
    termination: SourceTermination<(), CheckFailure>,
    retained: AdmittedResources,
}

impl<T: TargetFamily, E: NativeExecutor<T>> SubmittedRun<T, E> {
    pub(crate) fn complete(self) -> Result<CompletedRun<T, E>, CallError> {
        self.completion
            .complete_and_take()
            .map(|state| CompletedRun { state })
            .map_err(CallError::Execution)
    }

    pub(crate) fn complete_outcome(
        self,
    ) -> Result<SourceTermination<Vec<Vec<DecodedValue>>, CheckFailure>, CallError> {
        self.complete()?.outcome()
    }

    pub(crate) fn into_completion(self) -> WorkflowCompletionAny
    where
        T: 'static,
    {
        let identity = self.identity;
        WorkflowCompletionAny::pending(identity, move || self.complete_outcome())
    }
}

/// Native completion has finished. Output decoding/checking is outside the
/// submission-through-completion measurement endpoint.
pub(crate) struct CompletedRun<T: TargetFamily, E: NativeExecutor<T>> {
    state: SubmittedState<T, E>,
}
impl<T: TargetFamily, E: NativeExecutor<T>> CompletedRun<T, E> {
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.state
            .nodes
            .iter()
            .map(AdmittedNode::allocated_bytes)
            .fold(self.state.retained.reached_allocated(), u64::saturating_add)
    }
    pub(crate) fn read_tensor(
        &self,
        tensor: &crate::api::tensor::TensorInner,
    ) -> Result<Vec<u8>, ExecutionError> {
        tensor.read_with(self.state.retained.access(tensor.allocation()))
    }

    /// Observe the completed outcome while admission permits and backing remain
    /// owned by this run. Failed source execution has no return product.
    pub(crate) fn outcome(
        &self,
    ) -> Result<SourceTermination<Vec<Vec<DecodedValue>>, CheckFailure>, CallError> {
        match &self.state.termination {
            SourceTermination::Returned(()) => self
                .state
                .nodes
                .iter()
                .map(AdmittedNode::decode_values)
                .collect::<Result<_, _>>()
                .map(SourceTermination::Returned),
            SourceTermination::Failed(failure) => Ok(SourceTermination::Failed(failure.clone())),
        }
    }

    pub(crate) fn values(self) -> Result<Vec<Vec<DecodedValue>>, CallError> {
        match self.outcome()? {
            SourceTermination::Returned(values) => Ok(values),
            SourceTermination::Failed(failure) => Err(CallError::Execution(
                ExecutionError::DataCheckFailed(failure),
            )),
        }
    }
}

trait TerminalCompletion {
    type Error;
    /// Returns only after the owned work is terminal, including on `Err`.
    /// The mutable borrow lets the owner retain the completion handle if an
    /// implementation bug panics before establishing that fact.
    fn complete(&mut self) -> Result<(), Self::Error>;
}

struct BackendCompletion<T: TargetFamily, E: NativeExecutor<T>>(SubmittedExecution<T, E>);

impl<T: TargetFamily, E: NativeExecutor<T>> TerminalCompletion for BackendCompletion<T, E> {
    type Error = ExecutionError;

    fn complete(&mut self) -> Result<(), Self::Error> {
        self.0.complete()
    }
}

/// Couples terminal device completion to opaque retained ownership. Retained
/// state can be released only after completion returns a terminal result. A
/// completion panic deliberately leaks both owned values because terminality
/// was not established.
struct CompletionOwner<C: TerminalCompletion, R> {
    completion: Option<C>,
    retained: Option<R>,
}

impl<C: TerminalCompletion, R> CompletionOwner<C, R> {
    fn new(completion: C, retained: R) -> Self {
        Self {
            completion: Some(completion),
            retained: Some(retained),
        }
    }

    fn complete_and_take(mut self) -> Result<R, C::Error> {
        let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.completion
                .as_mut()
                .expect("completion owner consumed more than once")
                .complete()
        }));
        match completed {
            Ok(Ok(())) => {
                drop(self.completion.take());
                Ok(self
                    .retained
                    .take()
                    .expect("completion owner retained state consumed more than once"))
            }
            Ok(Err(error)) => {
                // An error result is still a terminal completion result by
                // contract, so ordinary destruction is now safe.
                drop(self.completion.take());
                Err(error)
            }
            Err(payload) => {
                // A panic proves no terminal state. Releasing either the
                // completion handle or its retained resources could create a
                // use-after-free in the backend, so ownership is intentionally
                // leaked before the bug continues unwinding.
                self.leak_in_flight();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn leak_in_flight(&mut self) {
        if let Some(completion) = self.completion.take() {
            std::mem::forget(completion);
        }
        if let Some(retained) = self.retained.take() {
            std::mem::forget(retained);
        }
    }
}

impl<C: TerminalCompletion, R> Drop for CompletionOwner<C, R> {
    fn drop(&mut self) {
        if self.completion.is_some() {
            let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.completion
                    .as_mut()
                    .expect("completion owner changed during destruction")
                    .complete()
            }));
            match completed {
                Ok(_) => {
                    drop(self.completion.take());
                }
                Err(_) => {
                    // Drop cannot report the backend bug. Preserve ownership
                    // safety exactly as the explicit path does.
                    self.leak_in_flight();
                }
            }
        }
    }
}

fn finish_issue<C: TerminalCompletion<Error = ExecutionError>, R>(
    completion: CompletionOwner<C, R>,
    issued: Result<(), ExecutionError>,
) -> Result<CompletionOwner<C, R>, ExecutionError> {
    match issued {
        Ok(()) => Ok(completion),
        Err(issue) => match completion.complete_and_take() {
            Ok(retained) => {
                drop(retained);
                Err(issue)
            }
            Err(terminal) => Err(ExecutionError::IssueAndCompletionFailed {
                issue: Box::new(issue),
                completion: Box::new(terminal),
            }),
        },
    }
}

impl AdmittedOutput {
    fn decode(
        &self,
        values: &InvocationValues,
        device: &Arc<crate::api::device::DeviceInner>,
    ) -> Result<DecodedValue, CallError> {
        match self {
            Self::PendingTensor { .. } => Err(CallError::Execution(
                ExecutionError::ConstructionContradiction("tensor output was not completed".into()),
            )),
            Self::Tensor {
                allocation,
                byte_offset,
                byte_len,
                representation,
                extents,
                strides,
            } => Ok(DecodedValue::Tensor(Arc::new(
                crate::api::tensor::TensorInner::new_view(
                    device.clone(),
                    allocation.clone(),
                    *byte_offset,
                    *byte_len,
                    *representation,
                    extents.clone(),
                    strides.clone(),
                ),
            ))),
            Self::Scalar(AdmittedScalarOutput::Value { dtype, symbol }) => {
                Ok(DecodedValue::Scalar(scalar_result(*dtype, *symbol, values)))
            }
            Self::Scalar(AdmittedScalarOutput::Index { symbol }) => {
                let Some(SymbolValue::Nat(value)) = values.get(*symbol) else {
                    return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                        "admitted index result slot has the wrong scalar sort".to_owned(),
                    )));
                };
                Ok(DecodedValue::Scalar(ArgumentValue::Index(value)))
            }
            Self::Scalar(AdmittedScalarOutput::Range { start, end }) => {
                let (Some(SymbolValue::Nat(start)), Some(SymbolValue::Nat(end))) =
                    (values.get(*start), values.get(*end))
                else {
                    return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                        "admitted range result slots have the wrong scalar sort".to_owned(),
                    )));
                };
                Ok(DecodedValue::Scalar(ArgumentValue::Range { start, end }))
            }
        }
    }
}

fn scalar_result(
    dtype: seismic_lang::types::DType,
    symbol: seismic_lang::expr::SymbolId,
    values: &InvocationValues,
) -> ArgumentValue {
    match (dtype, values.get(symbol)) {
        (seismic_lang::types::DType::F32, Some(SymbolValue::F32(v))) => ArgumentValue::F32(v),
        (seismic_lang::types::DType::F16, Some(SymbolValue::F16(v))) => ArgumentValue::F16(v),
        (seismic_lang::types::DType::BF16, Some(SymbolValue::BF16(v))) => ArgumentValue::BF16(v),
        (seismic_lang::types::DType::I32, Some(SymbolValue::I32(v))) => ArgumentValue::I32(v),
        (seismic_lang::types::DType::U32, Some(SymbolValue::U32(v))) => ArgumentValue::U32(v),
        (seismic_lang::types::DType::Bool, Some(SymbolValue::Bool(v))) => ArgumentValue::Bool(v),
        _ => panic!("closed scalar result slot has the wrong scalar sort"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct FakeCompletion {
        live: Arc<AtomicBool>,
        completions: Arc<AtomicUsize>,
        result: FakeCompletionResult,
    }

    #[derive(Clone, Copy)]
    enum FakeCompletionResult {
        Ok,
        Error,
        Panic,
    }

    impl TerminalCompletion for FakeCompletion {
        type Error = ExecutionError;

        fn complete(&mut self) -> Result<(), Self::Error> {
            assert!(self.live.load(Ordering::SeqCst));
            self.completions.fetch_add(1, Ordering::SeqCst);
            match self.result {
                FakeCompletionResult::Ok => Ok(()),
                FakeCompletionResult::Error => {
                    Err(ExecutionError::SynchronizationFailed("completion".into()))
                }
                FakeCompletionResult::Panic => panic!("completion fixture panic"),
            }
        }
    }

    struct RetainedProbe(Arc<AtomicBool>);

    impl Drop for RetainedProbe {
        fn drop(&mut self) {
            assert!(self.0.swap(false, Ordering::SeqCst));
        }
    }

    fn fixture_with(
        result: FakeCompletionResult,
    ) -> (
        CompletionOwner<FakeCompletion, RetainedProbe>,
        Arc<AtomicBool>,
        Arc<AtomicUsize>,
    ) {
        let live = Arc::new(AtomicBool::new(true));
        let completions = Arc::new(AtomicUsize::new(0));
        (
            CompletionOwner::new(
                FakeCompletion {
                    live: live.clone(),
                    completions: completions.clone(),
                    result,
                },
                RetainedProbe(live.clone()),
            ),
            live,
            completions,
        )
    }

    fn fixture() -> (
        CompletionOwner<FakeCompletion, RetainedProbe>,
        Arc<AtomicBool>,
        Arc<AtomicUsize>,
    ) {
        fixture_with(FakeCompletionResult::Ok)
    }

    #[test]
    fn resources_live_through_explicit_completion() {
        let (owner, live, completions) = fixture();
        let retained = owner.complete_and_take().unwrap();
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        assert!(live.load(Ordering::SeqCst));
        drop(retained);
        assert!(!live.load(Ordering::SeqCst));
    }

    #[test]
    fn partial_issue_failure_synchronizes_before_release() {
        let (owner, live, completions) = fixture();
        assert!(matches!(
            finish_issue(
                owner,
                Err(ExecutionError::SubmissionFailed("issue".into()))
            ),
            Err(ExecutionError::SubmissionFailed(ref message)) if message == "issue"
        ));
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        assert!(!live.load(Ordering::SeqCst));
    }

    #[test]
    fn dropping_unresolved_owner_synchronizes_before_release() {
        let (owner, live, completions) = fixture();
        drop(owner);
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        assert!(!live.load(Ordering::SeqCst));
    }

    #[test]
    fn issue_and_terminal_failures_are_both_preserved() {
        let (owner, live, completions) = fixture_with(FakeCompletionResult::Error);
        let Err(error) = finish_issue(owner, Err(ExecutionError::SubmissionFailed("issue".into())))
        else {
            panic!("dual failure unexpectedly retained a submitted completion")
        };
        assert_eq!(
            error,
            ExecutionError::IssueAndCompletionFailed {
                issue: Box::new(ExecutionError::SubmissionFailed("issue".into())),
                completion: Box::new(ExecutionError::SynchronizationFailed("completion".into())),
            }
        );
        assert_eq!(
            std::error::Error::source(&error)
                .expect("aggregate exposes its issue as the primary source")
                .to_string(),
            "submission failed: issue"
        );
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        assert!(!live.load(Ordering::SeqCst));
    }

    #[test]
    fn explicit_completion_panic_leaks_in_flight_ownership() {
        let (owner, live, completions) = fixture_with(FakeCompletionResult::Panic);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = owner.complete_and_take();
        }));
        assert!(panicked.is_err());
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        assert!(live.load(Ordering::SeqCst));
    }

    #[test]
    fn drop_swallows_completion_panic_and_leaks_in_flight_ownership() {
        let (owner, live, completions) = fixture_with(FakeCompletionResult::Panic);
        let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
        assert!(dropped.is_ok());
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        assert!(live.load(Ordering::SeqCst));
    }
}
