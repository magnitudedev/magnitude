//! Host-written device resources reused across work in flight: a ring with
//! one entry for each use the submitter can have in flight at once.
//!
//! An entry's turn comes again only after the ring's capacity of later uses,
//! and the capacity is what the submitter can have in flight, so the work
//! that last used it has completed by then. Reuse is guarded by that work's
//! completion alone: an entry whose work has not completed when its turn
//! comes is an invariant violation, and the ring panics rather than wait.

use std::collections::VecDeque;

/// Device work whose completion releases a ring entry it used.
pub trait FlightCompletion {
    type Error;
    /// Whether the work has completed, without waiting.
    fn is_complete(&self) -> bool;
    /// The completed work's outcome. Called only once it is complete, so it
    /// does not wait.
    fn finish(self) -> Result<(), Self::Error>;
}

impl FlightCompletion for crate::NativeTensorBatchCompletion {
    type Error = crate::CallError;

    fn is_complete(&self) -> bool {
        crate::NativeTensorBatchCompletion::is_complete(self)
    }

    fn finish(self) -> Result<(), Self::Error> {
        self.wait()
    }
}

struct Entry<T, C> {
    value: T,
    /// The work that last used it, until its next use observes it.
    last: Option<C>,
}

impl<T, C: FlightCompletion> Entry<T, C> {
    fn complete(&self) -> bool {
        self.last.as_ref().is_none_or(FlightCompletion::is_complete)
    }
}

/// A ring of `T` reused in turn, each entry only once the work that last
/// used it has completed.
pub struct FlightRing<T, C: FlightCompletion> {
    /// In turn: the front is the least recently used.
    entries: VecDeque<Entry<T, C>>,
    /// Entries a shrink released while their work was in flight, kept until
    /// it completes so nothing the device reads is freed.
    retiring: Vec<Entry<T, C>>,
}

impl<T, C: FlightCompletion> Default for FlightRing<T, C> {
    fn default() -> Self {
        Self { entries: VecDeque::new(), retiring: Vec::new() }
    }
}

impl<T, C: FlightCompletion> FlightRing<T, C> {
    /// Entries in turn.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hold `entries` entries, each new one made by `make`. A shrink releases
    /// the least recently used; one whose work is in flight retires once it
    /// completes. On a failure the entries made before it are held.
    pub fn resize<E>(&mut self, entries: usize, mut make: impl FnMut() -> Result<T, E>) -> Result<(), E> {
        self.retire();
        while self.entries.len() > entries {
            let released = self.entries.pop_front().expect("a ring longer than its target has an entry");
            if !released.complete() {
                self.retiring.push(released);
            }
        }
        while self.entries.len() < entries {
            self.entries.push_back(Entry { value: make()?, last: None });
        }
        Ok(())
    }

    /// The next entry in turn, the work that last used it observed: its
    /// outcome is returned before the entry is.
    ///
    /// # Panics
    /// The ring is empty, or the entry's work has not completed: more uses
    /// are in flight than the capacity the ring was sized for.
    pub fn next(&mut self) -> Result<FlightUse<'_, T, C>, C::Error> {
        self.retire();
        let mut entry = self.entries.pop_front().expect("FlightRing::next precondition: the ring holds an entry");
        assert!(
            entry.complete(),
            "a flight ring's entry is reused while the work that last used it is in flight: \
             more uses are in flight than the {} its capacity holds",
            self.entries.len() + 1
        );
        let observed = entry.last.take().map_or(Ok(()), FlightCompletion::finish);
        // The entry stays in turn whatever its last work's outcome.
        let entry = FlightUse { ring: self, entry: Some(entry) };
        observed.map(|()| entry)
    }

    /// Every entry held, in turn and retiring.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().chain(&self.retiring).map(|entry| &entry.value)
    }

    /// Drop the retiring entries whose work has completed.
    fn retire(&mut self) {
        self.retiring.retain(|entry| !entry.complete());
    }
}

/// One use of a ring entry. Its work's completion, once submitted, guards
/// the entry's next use ([`Self::submitted`]); a use dropped unsubmitted
/// returns the entry with nothing in flight.
pub struct FlightUse<'a, T, C: FlightCompletion> {
    ring: &'a mut FlightRing<T, C>,
    entry: Option<Entry<T, C>>,
}

impl<T, C: FlightCompletion> FlightUse<'_, T, C> {
    pub fn value(&mut self) -> &mut T {
        &mut self.entry.as_mut().expect("a live use holds its entry").value
    }

    /// The work that uses the entry was submitted: its completion guards
    /// the entry's next use.
    pub fn submitted(mut self, completion: C) {
        self.entry.as_mut().expect("a live use holds its entry").last = Some(completion);
    }
}

impl<T, C: FlightCompletion> Drop for FlightUse<'_, T, C> {
    fn drop(&mut self) {
        if let Some(entry) = self.entry.take() {
            self.ring.entries.push_back(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Work that completes when its flag is set.
    struct Work(Rc<Cell<bool>>);

    impl FlightCompletion for Work {
        type Error = ();
        fn is_complete(&self) -> bool {
            self.0.get()
        }
        fn finish(self) -> Result<(), ()> {
            Ok(())
        }
    }

    fn ring(entries: usize) -> FlightRing<usize, Work> {
        let mut ring = FlightRing::default();
        let mut made = 0;
        ring.resize::<()>(entries, || {
            made += 1;
            Ok(made - 1)
        })
        .unwrap();
        ring
    }

    /// At depths 1, 2 and 3, with that many uses in flight, every entry is
    /// used once before any is used again, and each is reused only after
    /// its work completed.
    #[test]
    fn an_entry_is_reused_only_after_its_work_completed() {
        for depth in 1..=3 {
            let mut ring = ring(depth);
            let mut flying: VecDeque<(usize, Rc<Cell<bool>>)> = VecDeque::new();
            for step in 0..4 * depth {
                if flying.len() == depth {
                    let (_, done) = flying.pop_front().unwrap();
                    done.set(true);
                }
                let mut entry = ring.next().unwrap();
                let value = *entry.value();
                assert_eq!(value, step % depth, "depth {depth}: entries are used in turn");
                assert!(flying.iter().all(|(used, _)| *used != value), "depth {depth}: an entry in flight is never reused");
                let done = Rc::new(Cell::new(false));
                entry.submitted(Work(done.clone()));
                flying.push_back((value, done));
            }
        }
    }

    /// One use beyond the capacity in flight is refused.
    #[test]
    #[should_panic(expected = "reused while the work that last used it is in flight")]
    fn a_use_beyond_the_capacity_panics() {
        let mut ring = ring(2);
        for _ in 0..3 {
            ring.next().unwrap().submitted(Work(Rc::new(Cell::new(false))));
        }
    }

    /// A shrink keeps an entry in flight until its work completes; a use
    /// dropped unsubmitted returns its entry with nothing in flight.
    #[test]
    fn a_shrink_retires_entries_in_flight() {
        let mut ring = ring(2);
        let done = Rc::new(Cell::new(false));
        ring.next().unwrap().submitted(Work(done.clone()));
        drop(ring.next().unwrap());
        ring.resize::<()>(1, || unreachable!()).unwrap();
        assert_eq!((ring.len(), ring.retiring.len()), (1, 1));
        done.set(true);
        ring.resize::<()>(1, || unreachable!()).unwrap();
        assert!(ring.retiring.is_empty());
        assert_eq!(*ring.next().unwrap().value(), 1);
    }
}
