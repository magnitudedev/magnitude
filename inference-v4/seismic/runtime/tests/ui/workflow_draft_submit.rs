use seismic_runtime::api::kernel::WorkflowDraftAny;

fn submit_without_admission(draft: WorkflowDraftAny) {
    let _ = draft.submit();
}

fn main() {}
