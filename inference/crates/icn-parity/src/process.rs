use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, bail};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::Instant;

use crate::model::CommandSpec;
use crate::provenance::resolve_program;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
pub const DEFAULT_MAX_STDOUT: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_STDERR: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessLimits {
    pub timeout: Duration,
    pub max_stdout_bytes: usize,
    pub max_stderr_bytes: usize,
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            max_stdout_bytes: DEFAULT_MAX_STDOUT,
            max_stderr_bytes: DEFAULT_MAX_STDERR,
        }
    }
}

#[derive(Debug)]
pub struct ProcessOutput {
    pub program: PathBuf,
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub timed_out: bool,
    pub elapsed: Duration,
}

impl ProcessOutput {
    pub fn success(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

pub async fn run_bounded(
    spec: &CommandSpec,
    stdin: Option<&[u8]>,
    defaults: ProcessLimits,
) -> anyhow::Result<ProcessOutput> {
    let program = resolve_program(&spec.program)?;
    // Producer declarations may tighten a profile's limits, but they may not
    // relax them.  The profile/default limits are the outer sandbox contract.
    let timeout = spec
        .timeout_seconds
        .map(Duration::from_secs)
        .map_or(defaults.timeout, |declared| declared.min(defaults.timeout));
    if timeout.is_zero() {
        bail!("subprocess timeout must be greater than zero");
    }
    let max_stdout = usize::try_from(
        spec.max_stdout_bytes
            .unwrap_or(defaults.max_stdout_bytes as u64)
            .min(defaults.max_stdout_bytes as u64),
    )
    .context("max_stdout_bytes does not fit this platform")?;
    let max_stderr = usize::try_from(
        spec.max_stderr_bytes
            .unwrap_or(defaults.max_stderr_bytes as u64)
            .min(defaults.max_stderr_bytes as u64),
    )
    .context("max_stderr_bytes does not fit this platform")?;
    if max_stdout == 0 || max_stderr == 0 {
        bail!("subprocess output limits must be greater than zero");
    }

    let mut command = Command::new(&program);
    command
        .args(&spec.args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(cwd) = &spec.cwd {
        command.current_dir(cwd);
    }
    if spec.clear_env {
        command.env_clear();
    }
    command.envs(&spec.env);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.as_std_mut().process_group(0);
    }

    let started = Instant::now();
    let deadline = started + timeout;
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start {}", program.display()))?;
    let process_id = child.id();
    let stdout = child
        .stdout
        .take()
        .context("subprocess stdout was unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("subprocess stderr was unavailable")?;
    let stdout_task = tokio::spawn(read_limited(stdout, max_stdout));
    let stderr_task = tokio::spawn(read_limited(stderr, max_stderr));
    let stdout_abort = stdout_task.abort_handle();
    let stderr_abort = stderr_task.abort_handle();
    let stdin_task = child.stdin.take().map(|mut pipe| {
        let bytes = stdin.unwrap_or_default().to_vec();
        tokio::spawn(async move {
            pipe.write_all(&bytes).await?;
            pipe.shutdown().await
        })
    });

    let stdin_abort = stdin_task
        .as_ref()
        .map(tokio::task::JoinHandle::abort_handle);

    let status = match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(status) => status.context("failed to wait for subprocess")?,
        Err(_) => {
            kill_process_tree(&mut child, process_id).await;
            let _ = child.wait().await;
            stdout_abort.abort();
            stderr_abort.abort();
            if let Some(abort) = stdin_abort {
                abort.abort();
            }
            return Ok(ProcessOutput {
                program,
                exit_code: None,
                stdout: Vec::new(),
                stderr: Vec::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                timed_out: true,
                elapsed: started.elapsed(),
            });
        }
    };
    let drained = tokio::time::timeout_at(deadline, async move {
        if let Some(task) = stdin_task {
            // Broken pipes are expected when a producer rejects input or exits early.
            let _ = task.await;
        }
        let stdout = stdout_task.await.context("stdout capture task failed")??;
        let stderr = stderr_task.await.context("stderr capture task failed")??;
        Ok::<_, anyhow::Error>((stdout, stderr))
    })
    .await;
    let ((stdout, stdout_truncated), (stderr, stderr_truncated)) = match drained {
        Ok(Ok(streams)) => streams,
        Ok(Err(error)) => {
            kill_process_tree(&mut child, process_id).await;
            return Err(error);
        }
        Err(_) => {
            // The direct child may have exited while a descendant retained one
            // of its pipes. The original process group still identifies those
            // descendants, so terminate it and classify the whole invocation
            // as timed out.
            kill_process_tree(&mut child, process_id).await;
            stdout_abort.abort();
            stderr_abort.abort();
            if let Some(abort) = stdin_abort {
                abort.abort();
            }
            return Ok(ProcessOutput {
                program,
                exit_code: status.code(),
                stdout: Vec::new(),
                stderr: Vec::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                timed_out: true,
                elapsed: started.elapsed(),
            });
        }
    };

    Ok(ProcessOutput {
        program,
        exit_code: status.code(),
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        timed_out: false,
        elapsed: started.elapsed(),
    })
}

async fn read_limited<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut output = Vec::with_capacity(limit.min(64 * 1024));
    let mut truncated = false;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(output.len());
        let retained = remaining.min(read);
        output.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    Ok((output, truncated))
}

async fn kill_process_tree(child: &mut tokio::process::Child, process_id: Option<u32>) {
    #[cfg(unix)]
    if let Some(process_id) = process_id {
        // The command is placed in a fresh process group above. A negative PID
        // therefore terminates descendants as well as the direct producer.
        unsafe {
            libc::kill(-(process_id as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::model::{DecoderKind, StdinKind};

    fn shell(script: &str) -> CommandSpec {
        CommandSpec {
            program: "sh".into(),
            args: vec!["-c".to_owned(), script.to_owned()],
            cwd: None,
            env: BTreeMap::new(),
            clear_env: false,
            stdin: StdinKind::None,
            decoder: DecoderKind::Json,
            timeout_seconds: None,
            max_stdout_bytes: None,
            max_stderr_bytes: None,
        }
    }

    #[tokio::test]
    async fn captures_both_streams_without_deadlock() {
        let output = run_bounded(
            &shell("printf ok; printf err >&2"),
            None,
            ProcessLimits::default(),
        )
        .await
        .unwrap();
        assert!(output.success());
        assert_eq!(output.stdout, b"ok");
        assert_eq!(output.stderr, b"err");
    }

    #[tokio::test]
    async fn times_out_processes() {
        let limits = ProcessLimits {
            timeout: Duration::from_millis(30),
            ..ProcessLimits::default()
        };
        let output = run_bounded(&shell("sleep 5"), None, limits).await.unwrap();
        assert!(output.timed_out);
        assert!(!output.success());
    }

    #[tokio::test]
    async fn caps_retained_output() {
        let limits = ProcessLimits {
            max_stdout_bytes: 8,
            ..ProcessLimits::default()
        };
        let output = run_bounded(&shell("printf 0123456789"), None, limits)
            .await
            .unwrap();
        assert_eq!(output.stdout, b"01234567");
        assert!(output.stdout_truncated);
    }

    #[tokio::test]
    async fn descendant_cannot_hold_capture_open_after_parent_exits() {
        let limits = ProcessLimits {
            timeout: Duration::from_millis(50),
            ..ProcessLimits::default()
        };
        let output = run_bounded(&shell("(sleep 60) & exit 0"), None, limits)
            .await
            .unwrap();
        assert!(output.timed_out);
        assert!(output.elapsed < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn producer_cannot_relax_outer_profile_limits() {
        let mut spec = shell("printf 0123456789; sleep 5");
        spec.timeout_seconds = Some(60);
        spec.max_stdout_bytes = Some(1024);
        let limits = ProcessLimits {
            timeout: Duration::from_millis(30),
            max_stdout_bytes: 4,
            max_stderr_bytes: 4,
        };
        let output = run_bounded(&spec, None, limits).await.unwrap();
        assert!(output.timed_out);

        let mut spec = shell("printf 0123456789");
        spec.max_stdout_bytes = Some(1024);
        let output = run_bounded(
            &spec,
            None,
            ProcessLimits {
                timeout: Duration::from_secs(1),
                max_stdout_bytes: 4,
                max_stderr_bytes: 4,
            },
        )
        .await
        .unwrap();
        assert_eq!(output.stdout, b"0123");
        assert!(output.stdout_truncated);
    }
}
