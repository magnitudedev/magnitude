//! Typed callable numerical programs and their submission lifecycle.

pub(crate) mod graph;
pub(crate) mod invocation_demand;
pub(crate) mod native_constants;
pub(crate) mod native_draft;
pub(crate) mod native_drafter;
pub(crate) mod native_head;
pub(crate) mod native_import;
pub(crate) mod native_shared_head;
pub(crate) mod native_state;
pub(crate) mod native_target;
pub(crate) mod native_target_graph;
pub(crate) mod native_vision;
mod submission;

pub use graph::GraphError;
pub use graph::readout::PreparedTargetReadoutGraphs;
pub use native_draft::PreparedDraftGraphs;
pub use native_drafter::PreparedDrafterGraphs;
pub use native_head::PreparedHeadGraphs;
pub use native_state::PreparedStateCopyGraphs;
pub use native_target::{CommitSpan, StoredRead, TargetOutput, TargetReadoutGraphResult};
pub use native_target_graph::{
    PlannedSealedGraphs, PreparedTargetGraphs, SealReport, SealedGraphsHeap,
};
pub use native_vision::PreparedVisionGraphs;
pub use submission::{
    CompletedWork, DeviceSubmission, ProgramSubmission, ReadySubmission, SubmittedHead,
    SubmittedTarget,
};

use crate::{
    GraphOutputTensor, HeadLaunchCore, ImportLaunchCore, InvariantError,
    ResidentWeightSlot, StateLaunchCore, SubmitError, TargetLaunchCore, ValidatedHeadLaunch,
    ValidatedImportLaunch, ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch,
    VisionLaunchCore,
};

/// Submit one ready native graph run and wait for its outcome. Programs
/// that chain several runs submit them without waiting instead and check
/// every completion once all are queued.
pub(crate) fn run_graph(
    ready: seismic::ReadyNativeGraphRun<'_>,
) -> Result<seismic::NativeGraphOutputs, seismic::CallError> {
    let (outputs, completion) = ready.submit()?;
    completion.wait()?;
    Ok(outputs)
}

pub type CompletedTargetWork = CompletedWork<TargetLaunchCore, TargetOutput>;
pub type CompletedHeadWork = CompletedWork<HeadLaunchCore, HeadOutput>;

/// What a head step leaves the host.
pub struct HeadOutput {
    /// What its selections make.
    pub selected: HeadSelected,
    /// The step's landed reads of table rows its device work selected
    /// (`DeviceTable::read_selected`), completed once the step completed.
    pub landed_reads: Vec<crate::device_table::SubmittedRead>,
}

/// What a head step's selections make.
pub enum HeadSelected {
    /// Nothing: a causal-only step selects nothing.
    Causal,
    /// A drafting step's proposals, with the drafter's confidence in each.
    Proposals(HeadSelections),
    /// A front launch's chain (`guess_chain`), which alone reads its
    /// selections.
    Chain(FrontOutput),
}

/// What a front launch's chain (`guess_chain`) leaves: the pass's next
/// guesses, and the token rows and guesses of the step queued behind it.
pub struct FrontOutput {
    /// `[entry row class, 2]` (guess, kind) rows: the chain of the slot
    /// whose exact selection is entry row `s` holds its guess at depth `d`
    /// in row `s + d - 1`, kind 0 where it holds none.
    pub guesses: GraphOutputTensor,
    /// `[VERIFY_ROWS, 2]` (token, status) rows, the token rows of the step
    /// queued behind (`TargetTokens::Selected::selected`): row `r` the one
    /// its `continuing` row `r` names, (0, 0) where it names none.
    pub next_selected: GraphOutputTensor,
    /// `[VERIFY_ROWS, 2]` (guess, kind) rows of the step queued behind
    /// (`TargetTokens::Selected::guesses`): its front row `r` reads row
    /// `r - 1` (`routed_step`).
    pub next_guesses: GraphOutputTensor,
}

impl FrontOutput {
    /// The chain's guesses, read once the launch completed: per entry row
    /// of its class, the guess its row holds, none of kind 0.
    pub fn read_guesses(&self) -> Result<Vec<Option<crate::Guess>>, SubmitError> {
        let invalid = |detail: String| SubmitError::Invariant(InvariantError { context: "front guesses", detail });
        let bytes = self
            .guesses
            .tensor()
            .read_to_host()
            .map_err(|error| SubmitError::Device(crate::DeviceError::Transfer(error.to_string())))?;
        if !bytes.len().is_multiple_of(8) {
            return Err(invalid("guess byte count is not a row multiple".into()));
        }
        bytes
            .chunks_exact(8)
            .map(|row| {
                let word = |at: usize| i32::from_le_bytes(row[at..at + 4].try_into().expect("four bytes"));
                crate::GuessKind::of_word(word(4))
                    .map(|kind| {
                        let token = u32::try_from(word(0))
                            .map_err(|_| invalid(format!("guessed token {} is negative", word(0))))?;
                        Ok(crate::Guess { token: crate::TokenId(token), kind })
                    })
                    .transpose()
            })
            .collect()
    }
}

/// A drafting head step's selections.
pub struct HeadSelections {
    /// `[steps, slot class, 2]` (token, status) rows in step-major order.
    pub tokens: GraphOutputTensor,
    /// Each selection's probability under the logits it was sampled from
    /// (`readout::select`), `[steps · slot class]` f32 in the tokens' order;
    /// absent for a drafter that chooses by path scores, which give no
    /// probability (`DraftVariant::samples_proposals`).
    pub confidence: Option<GraphOutputTensor>,
}

impl HeadOutput {
    /// A front launch's chain; none of any other step.
    pub fn front(&self) -> Option<&FrontOutput> {
        match &self.selected {
            HeadSelected::Chain(chain) => Some(chain),
            HeadSelected::Causal | HeadSelected::Proposals(_) => None,
        }
    }

    /// The step completed: its landed reads bring the rows they landed into
    /// slots of their own (`SubmittedRead::complete`). Fails when one could
    /// not land its rows: the step read other rows.
    pub fn complete_landed_reads(&mut self) -> Result<(), SubmitError> {
        std::mem::take(&mut self.landed_reads)
            .into_iter()
            .try_for_each(crate::device_table::SubmittedRead::complete)
            .map_err(|detail| {
                SubmitError::Invariant(InvariantError {
                    context: "landed table rows",
                    detail,
                })
            })
    }
}
pub type CompletedVisionWork = CompletedWork<VisionLaunchCore, GraphOutputTensor>;
pub type CompletedStateWork = CompletedWork<StateLaunchCore, ()>;
pub type CompletedImportWork = CompletedWork<ImportLaunchCore, ResidentWeightSlot>;

/// Target launches, or their submissions, in chain order: never empty. A
/// step is a chain of one.
pub struct Chain<T> {
    first: T,
    rest: Vec<T>,
}

impl<T> Chain<T> {
    pub fn one(first: T) -> Self {
        Self {
            first,
            rest: Vec::new(),
        }
    }

    /// Append `item` behind the chain's last.
    pub fn push(&mut self, item: T) {
        self.rest.push(item);
    }

    pub fn len(&self) -> usize {
        1 + self.rest.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        std::iter::once(&mut self.first).chain(&mut self.rest)
    }

    /// The chain's last item.
    pub fn last_mut(&mut self) -> &mut T {
        self.rest.last_mut().unwrap_or(&mut self.first)
    }

    /// The chain's first item and the ones behind it.
    pub fn split_first(self) -> (T, Vec<T>) {
        (self.first, self.rest)
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Chain<U> {
        Chain {
            first: f(self.first),
            rest: self.rest.into_iter().map(f).collect(),
        }
    }

    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<Chain<U>, E> {
        Ok(Chain {
            first: f(self.first)?,
            rest: self.rest.into_iter().map(f).collect::<Result<_, _>>()?,
        })
    }

    pub fn as_ref(&self) -> Chain<&T> {
        Chain {
            first: &self.first,
            rest: self.rest.iter().collect(),
        }
    }

    pub fn as_mut(&mut self) -> Chain<&mut T> {
        Chain {
            first: &mut self.first,
            rest: self.rest.iter_mut().collect(),
        }
    }

}

impl<T> Chain<T> {
    /// Lend every item to `f` at once, mutably, and pair each item with the
    /// result `f` makes of it. What `f` is lent and returns are chains of
    /// one scope (`Branded`), made from the lent chain only by mapping it
    /// item for item, so each result is its item's by construction. Why `f`
    /// failed, with the items. The companion owns work using those items;
    /// it is returned on success/refusal and dropped before them on unwind.
    pub fn scoped<R, E, O>(
        mut self,
        owner: O,
        f: impl for<'b> FnOnce(Branded<'b, &'b mut T>, &mut O) -> Result<Branded<'b, R>, E>,
    ) -> (O, Result<Chain<(T, R)>, (E, Self)>) {
        // The companion owns work using the lent items. On unwind it must
        // retire that work before the chain releases those items.
        let mut owner = owner;
        let results = match f(Branded::lent(self.as_mut()), &mut owner) {
            Ok(results) => results.chain,
            Err(error) => return (owner, Err((error, self))),
        };
        let paired = Chain {
            first: (self.first, results.first),
            rest: self.rest.into_iter().zip(results.rest).collect(),
        };
        (owner, Ok(paired))
    }

}

/// A chain of one scope (`Chain::scoped`, its lifetime `'b`, which no other
/// scope shares): the scope's lent chain or one mapped from it item for
/// item, so it has the lent chain's length and order.
pub struct Branded<'b, T> {
    chain: Chain<T>,
    brand: std::marker::PhantomData<fn(&'b ()) -> &'b ()>,
}

impl<'b, T> Branded<'b, T> {
    fn lent(chain: Chain<T>) -> Self {
        Self { chain, brand: std::marker::PhantomData }
    }

    pub fn len(&self) -> usize {
        self.chain.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chain.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.chain.iter_mut()
    }

    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Branded<'b, U> {
        Branded { chain: self.chain.map(f), brand: self.brand }
    }

    pub fn try_map<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<Branded<'b, U>, E> {
        Ok(Branded { chain: self.chain.try_map(f)?, brand: self.brand })
    }

    /// The items, out of the scope's chains.
    pub fn into_chain(self) -> Chain<T> {
        self.chain
    }
}

impl<'b, A, B> Branded<'b, (A, B)> {
    /// The scope's chain of each pair's first and its chain of each second.
    pub fn unzip(self) -> (Branded<'b, A>, Branded<'b, B>) {
        let (a, b) = self.chain.unzip();
        (Branded { chain: a, brand: self.brand }, Branded { chain: b, brand: self.brand })
    }
}

impl<T> IntoIterator for Chain<T> {
    type Item = T;
    type IntoIter = std::iter::Chain<std::iter::Once<T>, std::vec::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.first).chain(self.rest)
    }
}

impl<A, B> Chain<(A, B)> {
    /// The chain of each pair's first and the chain of its second.
    pub fn unzip(self) -> (Chain<A>, Chain<B>) {
        let (rest_a, rest_b) = self.rest.into_iter().unzip();
        (
            Chain {
                first: self.first.0,
                rest: rest_a,
            },
            Chain {
                first: self.first.1,
                rest: rest_b,
            },
        )
    }
}

/// One typed numerical lane. A submission may still be executing on the
/// device when it is returned; its completion reports when it is not.
pub trait TargetProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedTargetWork>;

    /// Submit a chain of launches, each chained on the one before: every
    /// launch's entry, then each block over every launch in chain order,
    /// then every launch's readout, one submission per launch. Each graph
    /// of a launch reads only the latest output of that launch's graph
    /// before it, so between graphs every launch holds one output: the
    /// chain's graphs rotate its launches' output slots and `spare`, each
    /// writing the one free slot and freeing the one it read. A rejected
    /// chain returns its still-owned launches so the domain can terminate
    /// their groups without losing state ownership. A streamed load's chain
    /// reads memory by `pressure` between its submissions, and cuts the
    /// expert bank's one store beside its running steps where memory runs
    /// short (`ExpertBank::release_running`).
    fn submit(
        &mut self,
        chain: Chain<ValidatedTargetLaunch>,
        spare: crate::TargetGraphOutputLease,
        pressure: &dyn Fn() -> Option<crate::domain::RunningPressure>,
    ) -> Result<Chain<Self::Submission>, ChainRefusal>;
}

/// A refused chain: why, and its still-owned launches.
pub struct ChainRefusal {
    pub error: SubmitError,
    pub chain: Chain<ValidatedTargetLaunch>,
}

pub trait HeadProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedHeadWork>;

    fn submit(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)>;
}

pub trait VisionProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedVisionWork>;

    fn submit(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedVisionLaunch)>;
}

pub trait StateProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedStateWork>;

    fn submit(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedStateLaunch)>;
}

pub trait ImportProgram {
    type Submission: ProgramSubmission<CompletedWork = CompletedImportWork>;

    /// Submit `launch`, its calls allocating through `charges`, the
    /// import's, handed at the call so the program holds none of its own.
    fn submit(
        &mut self,
        charges: &seismic::Charges,
        launch: ValidatedImportLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedImportLaunch)>;
}

#[cfg(test)]
mod scoped_lifetime_tests {
    use super::Chain;
    use std::cell::RefCell;

    #[test]
    fn scoped_work_retires_before_chain_items_on_unwind() {
        struct Owner<'a>(&'a RefCell<Vec<&'static str>>, &'static str);
        impl Drop for Owner<'_> {
            fn drop(&mut self) { self.0.borrow_mut().push(self.1); }
        }
        let order = RefCell::new(Vec::new());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let items = Chain::one(Owner(&order, "step"));
            let _ = items.scoped::<(), (), _>(Owner(&order, "submitted work"), |_, _| {
                panic!("queueing failed after submission")
            });
        }));
        assert!(result.is_err());
        assert_eq!(*order.borrow(), ["submitted work", "step"]);
    }
}
