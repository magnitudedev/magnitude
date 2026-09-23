use magnitude_model_executor::programs::{ProgramSubmission, ReadySubmission};
use std::{cell::Cell, rc::Rc};

struct Workspace(Rc<Cell<bool>>);

impl Drop for Workspace {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

#[test]
fn ready_submission_retains_launch_and_output_until_workspace_finishes() {
    let released = Rc::new(Cell::new(false));
    let mut submission = ReadySubmission::new(
        String::from("validated launch"),
        Workspace(released.clone()),
        String::from("leased output"),
    );
    assert!(submission.completion().is_complete());
    assert!(!released.get());

    let completed = submission.finish().expect("ready native completion");
    assert!(released.get());
    let (launch, output) = completed.into_parts();
    assert_eq!(launch, "validated launch");
    assert_eq!(output, "leased output");
}
