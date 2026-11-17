//! An engine process's start: every engine binary's `main` is [`main`], the
//! one owner of what such a process sets up before its work, which runs as
//! seismic-threads' program ([`seismic::threads::main`]).

/// Run `program`, an engine process's whole work, and return what it
/// returned. Before any thread exists the GPU watchdog is relaxed for the
/// process (its children inherit it); then, on the program's thread, a
/// probe process serves its probe and exits, and every other process
/// reports its fault or its stacks on demand ([`crate::diagnostics`])
/// before `program` runs. A panic anywhere in it ends the process at the
/// panic.
pub fn main<R: Send>(name: impl Into<String>, program: impl FnOnce() -> R + Send) -> R {
    relax_gpu_watchdog();
    seismic::threads::main(name, || {
        // The measurement job's device probes run in processes of the
        // engine's executables.
        seismic::serve_probe_if_asked();
        crate::diagnostics::install();
        program()
    })
}

/// Relaxes the macOS GPU watchdog for this process. M1 and M2 GPUs cannot
/// preempt a long prefill attention launch, so while the window server waits
/// for the GPU, macOS kills the command buffer ("Impacting Interactivity")
/// and the model is lost. `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` is read by Apple's
/// GPU driver; llama.cpp sets it for the same failure. It is set before any
/// thread or Metal device exists; child processes inherit it, and an
/// explicit setting is kept.
fn relax_gpu_watchdog() {
    const RELAX: &str = "AGX_RELAX_CDM_CTXSTORE_TIMEOUT";
    if cfg!(target_os = "macos") && std::env::var_os(RELAX).is_none() {
        std::env::set_var(RELAX, "1");
    }
}
