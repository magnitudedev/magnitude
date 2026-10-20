//! The consumer's cutoff for optional measuring work. Expiry is incomplete
//! work, never an execution or numerical rejection.

use std::time::{Duration, Instant};

/// When optional work must end; none for work without a cutoff.
#[derive(Clone, Copy, Debug, Default)]
pub struct Deadline(Option<Instant>);

/// Work that ended at its deadline, or failed.
#[derive(Debug)]
pub enum WorkError<E> {
    /// The deadline passed before the work was done.
    Expired,
    /// The work failed with this error.
    Failed(E),
}

impl<E> From<E> for WorkError<E> {
    fn from(error: E) -> Self {
        Self::Failed(error)
    }
}

impl<E> WorkError<E> {
    /// The same outcome with the failure mapped by `map`.
    pub fn map<F>(self, map: impl FnOnce(E) -> F) -> WorkError<F> {
        match self {
            Self::Expired => WorkError::Expired,
            Self::Failed(error) => WorkError::Failed(map(error)),
        }
    }
}

impl Deadline {
    /// A cutoff at `before`; none when `before` is none.
    pub fn new(before: Option<Instant>) -> Self {
        Self(before)
    }

    /// Whether the cutoff has passed.
    pub fn expired(self) -> bool {
        self.0.is_some_and(|before| Instant::now() >= before)
    }

    /// A known submission must fit as well as start before the cutoff.
    /// Its price remains a forecast; work already submitted is drained.
    pub fn check<E>(self, forecast: Duration) -> Result<(), WorkError<E>> {
        if self.0.is_some_and(|before| {
            let now = Instant::now();
            now >= before || forecast > before.saturating_duration_since(now)
        }) {
            Err(WorkError::Expired)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_batch_must_fit_the_existing_cutoff() {
        let deadline = Deadline::new(Some(Instant::now() + Duration::from_secs(60)));
        assert!(deadline.check::<()>(Duration::from_secs(120)).is_err());
        assert!(deadline.check::<()>(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn expiry_does_not_authorize_even_an_unpriced_submission() {
        let deadline = Deadline::new(Some(Instant::now()));
        assert!(matches!(
            deadline.check::<()>(Duration::ZERO),
            Err(WorkError::Expired)
        ));
    }
}
