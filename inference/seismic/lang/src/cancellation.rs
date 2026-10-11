//! Cooperative cancellation of long compiler work by a wall-clock deadline.
//!
//! A deadline is installed for the extent of one [`within`] call on the
//! current thread. Work that can run long calls [`check`] at its natural step
//! boundaries; once the deadline has passed, `check` abandons the whole
//! computation by unwinding to the enclosing `within`, which reports
//! [`Expired`]. Nothing the abandoned computation built escapes: the caller
//! receives only the typed expiry, never a partial result.

use std::cell::Cell;
use std::time::Instant;

/// The deadline of the enclosing [`within`] passed before its work finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expired;

impl std::fmt::Display for Expired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the work's deadline passed before it finished")
    }
}

impl std::error::Error for Expired {}

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Run `work` under `deadline` (or under the enclosing deadline when it is
/// earlier). Returns `Err(Expired)` when a [`check`] inside `work` observed
/// the deadline passed, or when it had passed before `work` began. Any other
/// panic continues to unwind unchanged.
pub fn within<T>(deadline: Option<Instant>, work: impl FnOnce() -> T) -> Result<T, Expired> {
    let enclosing = DEADLINE.with(Cell::get);
    let effective = match (enclosing, deadline) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    if effective.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(Expired);
    }
    struct Restore(Option<Instant>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DEADLINE.with(|cell| cell.set(self.0));
        }
    }
    DEADLINE.with(|cell| cell.set(effective));
    let _restore = Restore(enclosing);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<Expired>() {
            // An enclosing `within` with this same deadline also observes it
            // passed; one with a later deadline continues its own work.
            Ok(_) => Err(Expired),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

/// Abandon the enclosing [`within`] when its deadline has passed. Outside any
/// `within`, or before the deadline, this returns immediately.
pub fn check() {
    if DEADLINE
        .with(Cell::get)
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        // `resume_unwind` does not run the panic hook: expiry is not a defect
        // and prints nothing.
        std::panic::resume_unwind(Box::new(Expired));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn work_finishing_before_its_deadline_returns_its_value() {
        let deadline = Instant::now() + Duration::from_secs(60);
        assert_eq!(
            within(Some(deadline), || {
                check();
                7
            }),
            Ok(7)
        );
        assert_eq!(
            within(None, || {
                check();
                8
            }),
            Ok(8)
        );
    }

    #[test]
    fn a_passed_deadline_abandons_the_work_at_its_next_check() {
        let deadline = Instant::now();
        let mut reached = false;
        let result = within(Some(deadline + Duration::from_millis(1)), || {
            std::thread::sleep(Duration::from_millis(5));
            check();
            reached = true;
        });
        assert_eq!(result, Err(Expired));
        assert!(!reached);
        // The deadline is scoped: later work on this thread is unaffected.
        check();
    }

    #[test]
    fn a_nested_deadline_never_extends_its_enclosing_one() {
        let outer = Instant::now() + Duration::from_millis(1);
        let result = within(Some(outer), || {
            within(Some(outer + Duration::from_secs(60)), || {
                std::thread::sleep(Duration::from_millis(5));
                check();
            })
        });
        assert_eq!(result, Ok(Err(Expired)));
    }

    #[test]
    fn other_panics_are_not_expiry() {
        let result = std::panic::catch_unwind(|| {
            within(Some(Instant::now() + Duration::from_secs(60)), || {
                panic!("defect")
            })
        });
        assert!(result.is_err());
    }
}
