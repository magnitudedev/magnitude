//! Initialization belongs to actual storage contents. Views and child argument
//! proxies retain the caller's existing allocation identity; they never acquire
//! independent initialization flags. The language owns all region reasoning.
use seismic_ir::storage::{AnyBufferView, ViewBase};
use seismic_lang::expr::{BoolExpr, IntExpr, NatExpr, SymbolId};
use seismic_lang::initialization::{
    InitializationArgument, InitializationContext, InitializationContract, InitializationFailure,
    InitializationState, InitializationView,
};
use std::collections::BTreeMap;

pub(crate) type StoredView = seismic_ir::tensor_view::TensorView<AnyBufferView, NatExpr>;

#[derive(Clone, Debug)]
pub(crate) struct StoredTensor {
    pub(crate) view: StoredView,
    pub(crate) root: ViewBase,
    pub(crate) initialized_view: InitializationView,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StorageContents {
    roots: BTreeMap<ViewBase, InitializationState>,
}

/// Mirrors actual call leaves. Tensor arguments obtain their state exclusively
/// from their shared storage root; scalar arguments carry the language value.
pub(crate) enum CallArgument<'a> {
    Tensor(&'a StoredTensor),
    Scalar(InitializationArgument),
    Computed(InitializationView),
}

impl StorageContents {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Describe the root view of an actual allocation. Repeated physical uses
    /// preserve existing contents and add only initialization produced by the
    /// constructing operation. Aliasing views use `alias`.
    pub(crate) fn root(
        &mut self,
        context: &mut InitializationContext<'_>,
        root: ViewBase,
        view: StoredView,
        axes: &[IntExpr],
        initial: InitializationState,
    ) -> StoredTensor {
        self.roots
            .entry(root)
            .and_modify(|state| *state = state.union(&initial))
            .or_insert(initial);
        StoredTensor {
            view,
            root,
            initialized_view: context.root(axes),
        }
    }

    pub(crate) fn alias(
        &self,
        original: &StoredTensor,
        view: StoredView,
        initialized_view: InitializationView,
    ) -> StoredTensor {
        self.state(original);
        StoredTensor {
            view,
            root: original.root,
            initialized_view,
        }
    }

    /// Transfer the final contents retained by an escaping child product.
    /// Historical physical descriptors are remapped independently; dead local
    /// allocations need not exist in the child's final contents environment.
    pub(crate) fn import_allocation(
        &mut self,
        child: &Self,
        value: &StoredTensor,
        imported_root: ViewBase,
    ) {
        let incoming = child.state(value).clone();
        self.roots.entry(imported_root).or_insert(incoming);
    }

    pub(crate) fn state(&self, value: &StoredTensor) -> &InitializationState {
        self.roots
            .get(&value.root)
            .expect("stored tensor belongs to this construction's contents")
    }

    pub(crate) fn write(&mut self, context: &mut InitializationContext<'_>, value: &StoredTensor) {
        let next = context.write(self.state(value), &value.initialized_view);
        self.roots.insert(value.root, next);
    }

    pub(crate) fn copy_allocation(
        &mut self,
        context: &mut InitializationContext<'_>,
        source: &StoredTensor,
        destination: &StoredTensor,
    ) {
        let state = context.relocate(
            self.state(source),
            &source.initialized_view,
            &destination.initialized_view,
        );
        self.roots.insert(destination.root, state);
    }

    pub(crate) fn arguments(&self, arguments: &[CallArgument<'_>]) -> Vec<InitializationArgument> {
        arguments
            .iter()
            .map(|argument| match argument {
                CallArgument::Tensor(value) => InitializationArgument::Tensor {
                    state: self.state(value).clone(),
                    view: value.initialized_view.clone(),
                },
                CallArgument::Computed(view) => InitializationArgument::Tensor {
                    state: InitializationState::full(),
                    view: view.clone(),
                },
                CallArgument::Scalar(argument) => {
                    assert!(
                        !matches!(argument, InitializationArgument::Tensor { .. }),
                        "tensor call state must come from actual shared storage contents"
                    );
                    argument.clone()
                }
            })
            .collect()
    }

    /// Candidate selection checks the actual incoming state and the reference
    /// post-state together. This read-only query does not publish an authority.
    pub(crate) fn applicable(
        &self,
        context: &mut InitializationContext<'_>,
        candidate: &InitializationContract,
        reference: &InitializationContract,
        arguments: &[CallArgument<'_>],
    ) -> Result<(), InitializationFailure> {
        context
            .applicable(candidate, reference, &self.arguments(arguments))
            .map(|_| ())
    }

    /// Advance shared contents at the call's successful source continuation.
    /// If two parameter leaves alias one root, their produced regions union in
    /// that same root rather than overwriting each other's effects.
    pub(crate) fn call(
        &mut self,
        context: &mut InitializationContext<'_>,
        contract: &InitializationContract,
        arguments: &[CallArgument<'_>],
    ) -> Result<(), InitializationFailure> {
        let values = self.arguments(arguments);
        let outputs = context.apply(contract, &values)?;
        for (argument, output) in arguments.iter().zip(outputs) {
            if let (CallArgument::Tensor(value), Some(output)) = (argument, output) {
                let next = self.state(value).union(&output);
                self.roots.insert(value.root, next);
            }
        }
        Ok(())
    }

    fn completed_outputs(
        &mut self,
        arguments: &[CallArgument<'_>],
        outputs: Vec<Option<InitializationState>>,
    ) {
        for (argument, output) in arguments.iter().zip(outputs) {
            if let (CallArgument::Tensor(value), Some(output)) = (argument, output) {
                self.roots
                    .insert(value.root, self.state(value).union(&output));
            }
        }
    }

    pub(crate) fn loop_header(
        &mut self,
        context: &mut InitializationContext<'_>,
        transfer: &seismic_lang::initialization::LoopInitialization,
        arguments: &[CallArgument<'_>],
        start: IntExpr,
        current: IntExpr,
    ) {
        let actual = self.arguments(arguments);
        let completed = context
            .loop_completed(transfer, &actual, start, current)
            .expect("checked loop guarantees have no incoming read obligation");
        self.completed_outputs(arguments, completed);
        for (ordinal, state) in context.loop_carried(transfer, &actual) {
            if let CallArgument::Tensor(value) = &arguments[ordinal] {
                self.roots.insert(value.root, state);
            }
        }
    }

    pub(crate) fn loop_exit(
        &mut self,
        context: &mut InitializationContext<'_>,
        transfer: &seismic_lang::initialization::LoopInitialization,
        arguments: &[CallArgument<'_>],
        start: IntExpr,
        end: IntExpr,
    ) {
        let actual = self.arguments(arguments);
        let completed = context
            .loop_completed(transfer, &actual, start, end)
            .expect("checked loop guarantees have no incoming read obligation");
        self.completed_outputs(arguments, completed);
    }

    pub(crate) fn branch(
        &mut self,
        context: &mut InitializationContext<'_>,
        condition: BoolExpr,
        binders: &[SymbolId],
        then_contents: &Self,
        else_contents: &Self,
    ) {
        let mut roots = then_contents
            .roots
            .keys()
            .chain(else_contents.roots.keys())
            .copied()
            .collect::<Vec<_>>();
        roots.sort();
        roots.dedup();
        for root in roots {
            let state = match (
                then_contents.roots.get(&root),
                else_contents.roots.get(&root),
            ) {
                (Some(left), Some(right)) => context.branch(condition, binders, left, right),
                // An allocation created in only one arm has no contents in the
                // other arm. Its handle can escape only through that arm's
                // selected product, which retains the condition of its use.
                (Some(local), None) | (None, Some(local)) => local.clone(),
                (None, None) => unreachable!("root was collected from an arm"),
            };
            self.roots.insert(root, state);
        }
    }

    /// Captured roots accumulate completed iteration writes. Body-private roots
    /// are carried only through the ordinary complete value result mechanism.
    pub(crate) fn completed_loop(
        &mut self,
        context: &mut InitializationContext<'_>,
        before: &Self,
        iteration: &Self,
        binder: SymbolId,
        start: IntExpr,
        end: IntExpr,
    ) {
        for (&root, initial) in &before.roots {
            let body = iteration.roots.get(&root).unwrap_or(initial);
            self.roots.insert(
                root,
                context.completed_loop(initial, body, binder, start, end),
            );
        }
    }
}
