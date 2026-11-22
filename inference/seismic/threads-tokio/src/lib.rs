//! Tokio runtime construction under [`seismic_threads`]'s share, stack, and
//! panic policy.
//!
//! Kept in a separate crate from `seismic-threads` so the process's thread
//! policy (depended on by `seismic-lang`, `seismic-runtime`, `magnitude-kernels`,
//! etc.) never carries a `tokio` dependency whose downstream feature unification
//! would invalidate `magnitude-kernels` across binaries.

#![allow(
    clippy::disallowed_methods,
    reason = "the one place the engine builds a Tokio runtime"
)]

/// A multi-threaded Tokio runtime whose worker count is [`seismic_threads::share`]
/// and whose workers and blocking threads have [`seismic_threads::STACK_BYTES`] of
/// stack: every runtime of the process is built here, so work that blocks on one
/// runs on the owner's stack too and respects the process's thread share. Its
/// threads work for the thread that builds it, which blocks on it: a panic on
/// them ends as one there would, so within [`seismic_threads::main`] a task's
/// panic ends the process at the panic.
pub fn runtime() -> std::io::Result<tokio::runtime::Runtime> {
    let init = seismic_threads::worker_init();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(seismic_threads::share().get())
        .enable_all()
        .thread_stack_size(seismic_threads::STACK_BYTES)
        .on_thread_start(init)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deep(bytes: usize) -> u8 {
        #[inline(never)]
        fn frame(left: usize) -> u8 {
            let mut page = [0u8; 16 << 10];
            page[left % page.len()] = left as u8;
            let below = if left > page.len() { frame(left - page.len()) } else { 0 };
            std::hint::black_box(&page)[0].wrapping_add(below)
        }
        frame(bytes)
    }

    const CHILD: &str = "SEISMIC_THREADS_TOKIO_PANIC_CHILD";

    fn child_as(test: &str, case: &str) -> std::process::Output {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads", "1"])
            .env(CHILD, case)
            .output()
            .unwrap()
    }

    #[test]
    fn a_runtimes_task_or_detached_panic_ends_the_process_before_it_unwinds() {
        const NAME: &str = "tests::a_runtimes_task_or_detached_panic_ends_the_process_before_it_unwinds";
        if let Some(case) = std::env::var_os(CHILD) {
            struct Unwound;
            impl Drop for Unwound {
                fn drop(&mut self) {
                    eprintln!("unwound past the panic");
                }
            }
            match case.to_str() {
                Some("task") => seismic_threads::main("program", || {
                    let _unwound = Unwound;
                    let _ = runtime().unwrap().block_on(async {
                        tokio::spawn(async {
                            let _unwound = Unwound;
                            panic!("an invariant violated")
                        })
                        .await
                    });
                }),
                Some("detached") => {
                    let _ = runtime().unwrap().block_on(async {
                        tokio::spawn(seismic_threads::detached("an owner", async {
                            let _unwound = Unwound;
                            panic!("an invariant violated")
                        }))
                        .await
                    });
                }
                _ => return,
            }
            eprintln!("the process outlived the panic");
            return;
        }
        for case in ["task", "detached"] {
            let ended = child_as(NAME, case);
            let stderr = String::from_utf8_lossy(&ended.stderr);
            assert!(!ended.status.success(), "{case}: {stderr}");
            #[cfg(unix)]
            assert_eq!(std::os::unix::process::ExitStatusExt::signal(&ended.status), Some(6), "{case} aborted: {stderr}");
            assert!(stderr.contains("an invariant violated"), "{case}: the panic is reported: {stderr}");
            assert!(stderr.contains("the process ends rather than run on the state it left"), "{case}: {stderr}");
            if case == "detached" {
                assert!(stderr.contains("(task an owner) panicked at"), "the report names the task: {stderr}");
            }
            assert!(!stderr.contains("unwound past the panic"), "{case}: nothing unwinds: {stderr}");
            assert!(!stderr.contains("outlived"), "{case}: {stderr}");
        }
    }

    #[test]
    fn a_runtimes_threads_are_the_owners() {
        let wanted = seismic_threads::STACK_BYTES / 2;
        let runtime = runtime().unwrap();
        runtime.block_on(async move {
            tokio::spawn(async move { deep(wanted) }).await.unwrap();
            tokio::task::spawn_blocking(move || deep(wanted)).await.unwrap();
        });
    }
}
