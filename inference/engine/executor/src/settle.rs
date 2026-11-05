//! What a load owes before it ends: the durable effects its owners make
//! once serving stops (the rows a streamed table keeps for the next load,
//! the expert bank's account of its landings). They are settled in one step
//! ([`settle`]), which the execution owner runs as it stops, so a worker
//! process can then end without tearing its device down: what remains is
//! release, which the process's end does.
//!
//! An owner owes its effects on the thread it is made on, the execution
//! thread, for as long as it lives.

use std::cell::RefCell;

thread_local! {
    /// Each owed effect, run by a settle: false once its owner is gone.
    static OWED: RefCell<Vec<Box<dyn Fn() -> bool>>> = RefCell::new(Vec::new());
}

/// An owner that settles its durable effects ([`settle`]).
pub(crate) trait Owed {
    fn settle(&self);
}

/// Owe `owner`'s effects on this thread for as long as it lives.
pub(crate) fn owe<O: Owed + ?Sized + 'static>(owner: std::rc::Weak<O>) {
    OWED.with_borrow_mut(|owed| owed.push(Box::new(move || owner.upgrade().map(|owner| owner.settle()).is_some())));
}

/// [`owe`] for an owner shared across threads.
pub(crate) fn owe_shared<O: Owed + Send + Sync + ?Sized + 'static>(owner: std::sync::Weak<O>) {
    OWED.with_borrow_mut(|owed| owed.push(Box::new(move || owner.upgrade().map(|owner| owner.settle()).is_some())));
}

/// Settle every durable effect this thread's live owners owe, in the order
/// they were made; those of owners gone are forgotten.
pub fn settle() {
    let owed = OWED.take();
    let live = owed.into_iter().filter(|settle| settle()).collect::<Vec<_>>();
    OWED.with_borrow_mut(|owed| {
        let made_since = std::mem::take(owed);
        *owed = live;
        owed.extend(made_since);
    });
}

/// Whether an owner's effects are made: by the settle, or else as it drops,
/// once either way.
#[derive(Default)]
pub(crate) struct Once(std::sync::atomic::AtomicBool);

impl Once {
    /// True the first time only.
    pub(crate) fn first(&self) -> bool {
        !self.0.swap(true, std::sync::atomic::Ordering::AcqRel)
    }
}
