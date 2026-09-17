use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use clap::Args;

use crate::installation::Installation;

#[derive(Clone, Debug)]
pub(crate) enum NativeRuntimeAuthority {
    Installation(Installation),
    Development,
}

impl NativeRuntimeAuthority {
    pub(crate) fn installed(installation: Installation) -> Self {
        Self::Installation(installation)
    }

    pub(crate) fn development() -> Self {
        Self::Development
    }

    pub(crate) fn installation(&self) -> Option<&Installation> {
        match self {
            Self::Installation(installation) => Some(installation),
            Self::Development => None,
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct NativeWorkerArgs {
    #[arg(long)]
    installation: Option<PathBuf>,
    #[arg(long)]
    development_runtime: bool,
}

impl NativeWorkerArgs {
    pub(crate) fn authority(self) -> anyhow::Result<NativeRuntimeAuthority> {
        anyhow::ensure!(
            self.installation.is_some() != self.development_runtime,
            "native worker requires exactly one installation or development runtime authority"
        );
        match self.installation {
            Some(path) => Installation::load(&path).map(NativeRuntimeAuthority::installed),
            None => Ok(NativeRuntimeAuthority::development()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeWorkerRole {
    Planning,
    Inference,
}

impl NativeWorkerRole {
    fn subcommand(self) -> &'static str {
        match self {
            Self::Planning => "planning-worker",
            Self::Inference => "inference-worker",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeWorkerLauncher {
    authority: NativeRuntimeAuthority,
}

#[cfg(unix)]
pub(crate) type PlanningWorkerChild = tokio::process::Child;

#[cfg(windows)]
pub(crate) struct PlanningWorkerChild {
    // Drop containment before the asynchronous pipe adapters: killing the job closes child
    // pipe ends and releases any pending Tokio blocking I/O, including canceled pool tasks.
    owner: icn_utils::windows_process::OwnedWindowsChild,
    pub(crate) stdin: Option<tokio::process::ChildStdin>,
    pub(crate) stdout: Option<tokio::process::ChildStdout>,
    pub(crate) stderr: Option<tokio::process::ChildStderr>,
}

#[cfg(windows)]
impl PlanningWorkerChild {
    fn from_owner(
        mut owner: icn_utils::windows_process::OwnedWindowsChild,
    ) -> anyhow::Result<Self> {
        use std::os::windows::io::OwnedHandle;
        let stdin = tokio::process::ChildStdin::from_std(std::process::ChildStdin::from(
            OwnedHandle::from(
                owner
                    .stdin
                    .take()
                    .context("planning worker stdin missing")?,
            ),
        ))?;
        let stdout = tokio::process::ChildStdout::from_std(std::process::ChildStdout::from(
            OwnedHandle::from(
                owner
                    .stdout
                    .take()
                    .context("planning worker stdout missing")?,
            ),
        ))?;
        let stderr = tokio::process::ChildStderr::from_std(std::process::ChildStderr::from(
            OwnedHandle::from(
                owner
                    .stderr
                    .take()
                    .context("planning worker stderr missing")?,
            ),
        ))?;
        Ok(Self {
            owner,
            stdin: Some(stdin),
            stdout: Some(stdout),
            stderr: Some(stderr),
        })
    }

    pub(crate) fn start_kill(&mut self) -> std::io::Result<()> {
        self.owner.start_kill()
    }

    pub(crate) async fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        loop {
            if let Some(status) = self.owner.try_retirement()? {
                return Ok(status);
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

impl NativeWorkerLauncher {
    pub(crate) fn new(authority: NativeRuntimeAuthority) -> Self {
        Self { authority }
    }

    #[cfg(test)]
    pub(crate) fn development() -> Self {
        Self::new(NativeRuntimeAuthority::development())
    }

    pub(crate) fn command(&self, role: NativeWorkerRole) -> anyhow::Result<Command> {
        let executable =
            std::env::current_exe().context("failed to locate ICN worker executable")?;
        let mut command = Command::new(executable);
        configure_parent_lifetime(&mut command)?;
        command
            .arg(role.subcommand())
            .env("MAGNITUDE_OTEL", "0")
            .env("RUST_LOG", "error")
            .env_remove("MAGNITUDE_OTEL_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT");
        match &self.authority {
            NativeRuntimeAuthority::Installation(installation) => {
                command
                    .arg("--installation")
                    .arg(installation.declaration_path());
            }
            NativeRuntimeAuthority::Development => {
                command.arg("--development-runtime");
            }
        }
        Ok(command)
    }

    #[cfg(windows)]
    pub(crate) fn spawn_inference(
        &self,
    ) -> anyhow::Result<icn_utils::windows_process::OwnedWindowsChild> {
        self.spawn_contained(NativeWorkerRole::Inference)
    }

    pub(crate) fn spawn_planning(&self) -> anyhow::Result<PlanningWorkerChild> {
        #[cfg(unix)]
        {
            use std::process::Stdio;
            tokio::process::Command::from(self.command(NativeWorkerRole::Planning)?)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .context("failed to acquire the planning worker")
        }
        #[cfg(windows)]
        {
            PlanningWorkerChild::from_owner(self.spawn_contained(NativeWorkerRole::Planning)?)
        }
    }

    #[cfg(windows)]
    fn spawn_contained(
        &self,
        role: NativeWorkerRole,
    ) -> anyhow::Result<icn_utils::windows_process::OwnedWindowsChild> {
        use icn_utils::windows_process::{OwnedWindowsChild, merge_environment};
        // This composition constructs only executable, arguments and inherited-environment edits;
        // it never sets opaque std::Command flags or a shell command line.
        let command = self.command(role)?;
        let environment = merge_environment(
            std::env::vars_os().collect(),
            command
                .get_envs()
                .map(|(key, value)| (key.to_owned(), value.map(ToOwned::to_owned))),
        )?;
        let arguments = command
            .get_args()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        OwnedWindowsChild::spawn(
            std::path::Path::new(command.get_program()),
            &arguments,
            &environment,
        )
        .with_context(|| format!("failed to acquire the contained {}", role.subcommand()))
    }
}

#[cfg(unix)]
fn configure_parent_lifetime(command: &mut Command) -> anyhow::Result<()> {
    use std::os::unix::process::CommandExt as _;
    let parent = std::process::id() as libc::pid_t;
    command.env("MAGNITUDE_ICN_PARENT_PID", parent.to_string());
    // SAFETY: only async-signal-safe libc calls execute between fork and exec.
    unsafe {
        command.pre_exec(move || {
            let limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // Linux PDEATHSIG follows the spawning thread, which may be a temporary
            // blocking-pool thread. The worker's process-parent watchdog owns lifetime.
            if libc::getppid() != parent {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(windows)]
fn configure_parent_lifetime(command: &mut Command) -> anyhow::Result<()> {
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: GetCurrentProcess returns a borrowed pseudo-handle; it is never closed.
    let creation = unsafe { windows_process_creation(GetCurrentProcess()) }?;
    command.env("MAGNITUDE_ICN_PARENT_PID", std::process::id().to_string());
    command.env("MAGNITUDE_ICN_PARENT_CREATION", creation.to_string());
    Ok(())
}

#[cfg(windows)]
unsafe fn windows_process_creation(
    handle: windows_sys::Win32::Foundation::HANDLE,
) -> std::io::Result<u64> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetProcessTimes;
    let mut created = FILETIME::default();
    let mut exited = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: the caller retains a query-capable process handle for this invocation.
    if unsafe { GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime))
}

/// Both worker roles install protection before telemetry or native backend initialization.
#[cfg(unix)]
pub(crate) fn install_parent_watchdog() -> anyhow::Result<()> {
    let parent: libc::pid_t = std::env::var("MAGNITUDE_ICN_PARENT_PID")
        .context("native worker has no owning ICN identity")?
        .parse()
        .context("invalid owning ICN identity")?;
    anyhow::ensure!(parent > 1, "invalid owning ICN identity");
    // Preserve the spawning parent's identity across exec; never accept an already-reparented PID.
    anyhow::ensure!(
        unsafe { libc::getppid() } == parent,
        "owning ICN exited during worker startup"
    );
    std::thread::Builder::new()
        .name("icn-worker-parent-watchdog".to_owned())
        .spawn(move || {
            loop {
                if unsafe { libc::getppid() } != parent {
                    unsafe { libc::_exit(91) };
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        })
        .context("cannot start native worker parent watchdog")?;
    Ok(())
}

#[cfg(windows)]
pub(crate) fn install_parent_watchdog() -> anyhow::Result<()> {
    use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{
        INFINITE, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
        WaitForSingleObject,
    };
    let parent: u32 = std::env::var("MAGNITUDE_ICN_PARENT_PID")
        .context("native worker has no owning ICN identity")?
        .parse()
        .context("invalid owning ICN identity")?;
    let creation: u64 = std::env::var("MAGNITUDE_ICN_PARENT_CREATION")
        .context("native worker has no owning ICN creation identity")?
        .parse()
        .context("invalid owning ICN creation identity")?;
    anyhow::ensure!(parent > 0 && creation > 0, "invalid owning ICN identity");
    // SAFETY: acquire only query/synchronization rights, with inheritance disabled.
    let raw = unsafe {
        OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            parent,
        )
    };
    if raw.is_null() {
        return Err(std::io::Error::last_os_error()).context("cannot observe owning ICN");
    }
    // SAFETY: this successful OpenProcess handle is uniquely owned and CloseHandle-compatible.
    let owner = unsafe { OwnedHandle::from_raw_handle(raw) };
    anyhow::ensure!(
        unsafe { windows_process_creation(owner.as_raw_handle()) }? == creation,
        "owning ICN identity changed during worker startup"
    );
    // SAFETY: owner retains the synchronization handle; zero timeout does not block startup.
    let initial = unsafe { WaitForSingleObject(owner.as_raw_handle(), 0) };
    anyhow::ensure!(
        initial == WAIT_TIMEOUT,
        "owning ICN is not alive during worker startup"
    );
    std::thread::Builder::new()
        .name("icn-worker-parent-watchdog".to_owned())
        .spawn(move || {
            // SAFETY: the native thread owns the handle for its entire blocking wait.
            let result = unsafe { WaitForSingleObject(owner.as_raw_handle(), INFINITE) };
            if result != WAIT_OBJECT_0 {
                eprintln!("ICN parent observation failed");
            }
            // Both parent loss and observation failure terminate before further worker work.
            std::process::exit(91);
        })
        .context("cannot start native worker parent watchdog")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use icn_contracts::bootstrap_protocol::{IcnInstallationBackend, IcnInstallationDeclaration};

    use super::{NativeWorkerArgs, NativeWorkerLauncher, NativeWorkerRole};

    #[cfg(target_os = "linux")]
    #[test]
    fn worker_survives_the_spawning_thread() {
        let mut child = std::thread::spawn(|| {
            let mut command = std::process::Command::new("/bin/sh");
            super::configure_parent_lifetime(&mut command).unwrap();
            command.args(["-c", "sleep 0.2"]).spawn().unwrap()
        })
        .join()
        .unwrap();
        assert!(child.wait().unwrap().success());
    }

    // Re-enter the test executable to exercise the real watchdog across process death.
    #[cfg(target_os = "linux")]
    #[test]
    fn parent_lifetime_fixture() {
        let Ok(directory) = std::env::var("ICN_PARENT_LIFETIME_FIXTURE") else {
            return;
        };
        let directory = std::path::PathBuf::from(directory);
        if std::env::var_os("ICN_PARENT_LIFETIME_WORKER").is_some() {
            super::install_parent_watchdog().unwrap();
            fs::write(directory.join("ready"), b"ready").unwrap();
        } else {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            super::configure_parent_lifetime(&mut command).unwrap();
            let child = command
                .args(["--exact", "worker_process::tests::parent_lifetime_fixture"])
                .env("ICN_PARENT_LIFETIME_WORKER", "1")
                .spawn()
                .unwrap();
            fs::write(directory.join("pid.tmp"), child.id().to_string()).unwrap();
            fs::rename(directory.join("pid.tmp"), directory.join("pid")).unwrap();
        }
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn worker_exits_when_its_parent_process_dies() {
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};

        struct Fixture(std::process::Child, Option<libc::pid_t>);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
                if let Some(pid) = self.1 {
                    // This private fixture worker cannot outlive a failed assertion.
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                }
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let parent = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "worker_process::tests::parent_lifetime_fixture"])
            .env("ICN_PARENT_LIFETIME_FIXTURE", directory.path())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let mut fixture = Fixture(parent, None);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !directory.path().join("ready").exists() || fixture.1.is_none() {
            if let Ok(pid) = fs::read_to_string(directory.path().join("pid")) {
                fixture.1 = Some(pid.parse().unwrap());
            }
            assert!(
                fixture.0.try_wait().unwrap().is_none(),
                "fixture parent exited"
            );
            assert!(Instant::now() < deadline, "worker readiness timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
        fixture.0.kill().unwrap();
        fixture.0.wait().unwrap();
        let pid = fixture.1.unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match fs::read_to_string(format!("/proc/{pid}/stat")) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Ok(stat) if stat.rsplit_once(") ").unwrap().1.starts_with('Z') => break,
                Err(error) => panic!("cannot observe fixture worker: {error}"),
                Ok(_) => {}
            }
            assert!(
                Instant::now() < deadline,
                "worker survived parent process loss"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        fixture.1 = None;
    }

    #[test]
    fn worker_runtime_authority_is_explicit() {
        assert!(
            NativeWorkerArgs {
                installation: None,
                development_runtime: false,
            }
            .authority()
            .is_err()
        );
        assert!(
            NativeWorkerArgs {
                installation: None,
                development_runtime: true,
            }
            .authority()
            .is_ok()
        );
        assert!(
            NativeWorkerArgs {
                installation: Some("installation.json".into()),
                development_runtime: true,
            }
            .authority()
            .is_err()
        );
    }

    #[test]
    fn every_worker_role_uses_the_same_runtime_command_boundary() {
        let launcher = NativeWorkerLauncher::development();
        for (role, expected_subcommand) in [
            (NativeWorkerRole::Planning, "planning-worker"),
            (NativeWorkerRole::Inference, "inference-worker"),
        ] {
            let command = launcher.command(role).expect("worker command");
            assert_eq!(
                command
                    .get_args()
                    .map(|argument| argument.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
                vec![expected_subcommand, "--development-runtime"]
            );
            let parent = command
                .get_envs()
                .find(|(key, _)| *key == "MAGNITUDE_ICN_PARENT_PID")
                .and_then(|(_, value)| value)
                .expect("worker retains spawning parent identity");
            assert_eq!(parent, std::process::id().to_string().as_str());
            #[cfg(windows)]
            {
                let creation = command
                    .get_envs()
                    .find(|(key, _)| *key == "MAGNITUDE_ICN_PARENT_CREATION")
                    .and_then(|(_, value)| value)
                    .expect("worker retains parent creation identity");
                assert!(creation.to_str().unwrap().parse::<u64>().unwrap() > 0);
            }
        }
    }

    #[test]
    fn installed_worker_command_uses_the_verified_declaration() {
        let root = tempfile::tempdir().expect("installation root");
        for directory in ["bin", "catalog", "runtime", "backends"] {
            fs::create_dir(root.path().join(directory)).expect("installation directory");
        }
        for (path, contents) in [
            (
                root.path()
                    .join("bin")
                    .join(crate::installation::executable_name()),
                b"executable".as_slice(),
            ),
            (
                root.path().join("catalog/model-planner-inputs.bundle"),
                b"planner".as_slice(),
            ),
            (
                root.path().join("backends/backend-cpu"),
                b"backend".as_slice(),
            ),
        ] {
            fs::write(path, contents).expect("installation file");
        }
        let declaration_path = root.path().join("installation.json");
        fs::write(
            &declaration_path,
            serde_json::to_vec(&IcnInstallationDeclaration {
                schema_version: 1,
                backend: IcnInstallationBackend::Cpu,
                native_build: "native-build".to_owned(),
                backend_module_abi: "backend-abi".to_owned(),
            })
            .expect("serialize declaration"),
        )
        .expect("installation declaration");

        let authority = NativeWorkerArgs {
            installation: Some(declaration_path),
            development_runtime: false,
        }
        .authority()
        .expect("installed authority");
        let installation_path = authority
            .installation()
            .expect("installed runtime")
            .declaration_path();
        let command = NativeWorkerLauncher::new(authority)
            .command(NativeWorkerRole::Inference)
            .expect("worker command");

        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            vec![
                std::ffi::OsStr::new("inference-worker"),
                std::ffi::OsStr::new("--installation"),
                installation_path.as_os_str(),
            ]
        );
    }
}

#[cfg(all(test, windows))]
mod windows_pipe_tests {
    use super::PlanningWorkerChild;
    use icn_utils::windows_process::{OwnedWindowsChild, merge_environment};
    use std::io::{BufRead, Write};
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::time::Duration;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };

    #[test]
    fn pipe_child() {
        if std::env::var("MAGNITUDE_TEST_PLANNING_PIPE_CHILD").as_deref() != Ok("1") {
            return;
        }
        println!("MAGNITUDE_PIPE_READY");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).unwrap();
        println!("MAGNITUDE_PIPE_ECHO:{}", line.trim_end());
        std::io::stdout().flush().unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }

    #[test]
    fn planning_pipes_exchange_and_owner_drop_releases_pending_read() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let environment = merge_environment(
                std::env::vars_os().collect(),
                [(
                    "MAGNITUDE_TEST_PLANNING_PIPE_CHILD".into(),
                    Some("1".into()),
                )],
            )
            .unwrap();
            let owner = OwnedWindowsChild::spawn(
                &std::env::current_exe().unwrap(),
                &[
                    "--exact".into(),
                    "worker_process::windows_pipe_tests::pipe_child".into(),
                    "--nocapture".into(),
                ],
                &environment,
            )
            .unwrap();
            // SAFETY: this PID names the child still retained by owner. The returned independent
            // synchronization handle remains open across owner disposal to observe that same process.
            let raw = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, owner.id()) };
            assert!(!raw.is_null());
            let observed = unsafe { OwnedHandle::from_raw_handle(raw) };
            let mut child = PlanningWorkerChild::from_owner(owner).unwrap();
            let mut lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    let line = lines
                        .next_line()
                        .await
                        .unwrap()
                        .expect("worker exited before pipe readiness");
                    if line == "MAGNITUDE_PIPE_READY" {
                        break;
                    }
                }
                let stdin = child.stdin.as_mut().unwrap();
                stdin
                    .write_all("模型 pipe test\n".as_bytes())
                    .await
                    .unwrap();
                stdin.flush().await.unwrap();
                assert_eq!(
                    lines.next_line().await.unwrap().as_deref(),
                    Some("MAGNITUDE_PIPE_ECHO:模型 pipe test")
                );
            })
            .await
            .unwrap();
            let pending = tokio::spawn(async move { lines.next_line().await });
            tokio::task::yield_now().await;
            drop(child);
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), pending)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap(),
                None
            );
            // SAFETY: retained synchronization handle, bounded wait; never a fresh PID lookup.
            assert_eq!(
                unsafe { WaitForSingleObject(observed.as_raw_handle(), 5_000) },
                WAIT_OBJECT_0
            );
        });
    }
}
