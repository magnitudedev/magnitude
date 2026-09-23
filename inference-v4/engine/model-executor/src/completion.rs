use crate::error::DeviceError;

/// Exactly one wake for an outstanding executor completion, independent of the
/// bounded control queue. Completion owners may move it across threads.
pub struct CompletionWake(Option<Box<dyn FnOnce() + Send>>);

impl CompletionWake {
    pub fn new(wake: impl FnOnce() + Send + 'static) -> Self {
        Self(Some(Box::new(wake)))
    }

    pub fn complete(mut self) {
        self.0.take().expect("one completion wake")();
    }
}

/// Completion covers numerical execution, deferred transfers, and selection.
/// A result may be read only after completion, including failed submissions.
pub trait Completion {
    fn is_complete(&self) -> bool;
    fn result(&mut self) -> Result<(), DeviceError>;

    /// Register the reserved host wake after all completion obligations.
    /// Invoke it immediately if already complete. This must not poll or block.
    fn notify(&mut self, wake: CompletionWake);
}

/// Synchronous phase-one completion returned after all native calls finish.
pub struct Completed {
    result: Result<(), DeviceError>,
}

impl Completed {
    pub fn success() -> Self {
        Self { result: Ok(()) }
    }

    pub fn failure(error: DeviceError) -> Self {
        Self { result: Err(error) }
    }
}

impl Completion for Completed {
    fn is_complete(&self) -> bool {
        true
    }

    fn result(&mut self) -> Result<(), DeviceError> {
        self.result.clone()
    }

    fn notify(&mut self, wake: CompletionWake) {
        wake.complete();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[test]
    fn completed_wakes_immediately_and_preserves_result() {
        let fired = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&fired);
        let mut completion =
            Completed::failure(DeviceError::Execution("native stage failed".into()));
        completion.notify(CompletionWake::new(move || {
            observed.store(true, Ordering::SeqCst);
        }));
        assert!(fired.load(Ordering::SeqCst));
        assert!(completion.is_complete());
        assert_eq!(
            completion.result(),
            Err(DeviceError::Execution("native stage failed".into()))
        );
    }
}
