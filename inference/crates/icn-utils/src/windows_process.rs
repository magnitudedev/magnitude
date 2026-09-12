//! Native Windows child acquisition with containment established by CreateProcess itself.
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::ExitStatusExt;
use std::path::Path;
use std::process::ExitStatus;
use std::ptr::{null, null_mut};
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::Globalization::CompareStringOrdinal;
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::System::JobObjects::*;
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::*;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn wide(value: &OsStr) -> io::Result<Vec<u16>> {
    let value: Vec<_> = value.encode_wide().collect();
    if value.contains(&0) {
        return Err(invalid("Windows command contains NUL"));
    }
    Ok(value)
}

fn argument(output: &mut Vec<u16>, value: &[u16]) {
    output.push(b'"' as u16);
    let mut slashes = 0;
    for &unit in value {
        if unit == b'\\' as u16 {
            slashes += 1;
            continue;
        }
        output.extend(std::iter::repeat_n(
            b'\\' as u16,
            if unit == b'"' as u16 {
                slashes * 2 + 1
            } else {
                slashes
            },
        ));
        output.push(unit);
        slashes = 0;
    }
    output.extend(std::iter::repeat_n(b'\\' as u16, slashes * 2));
    output.push(b'"' as u16);
}

fn compare_names(a: &[u16], b: &[u16]) -> io::Result<std::cmp::Ordering> {
    let a_len = i32::try_from(a.len()).map_err(|_| invalid("Environment name is too long"))?;
    let b_len = i32::try_from(b.len()).map_err(|_| invalid("Environment name is too long"))?;
    // SAFETY: each pointer addresses the declared UTF-16 slice length.
    match unsafe { CompareStringOrdinal(a.as_ptr(), a_len, b.as_ptr(), b_len, 1) } {
        1 => Ok(std::cmp::Ordering::Less),
        2 => Ok(std::cmp::Ordering::Equal),
        3 => Ok(std::cmp::Ordering::Greater),
        _ => Err(io::Error::last_os_error()),
    }
}

fn environment_name(name: &OsStr) -> io::Result<Vec<u16>> {
    let encoded = wide(name)?;
    // CreateProcess environment blocks preserve per-drive current directories as =C:=path.
    let drive_directory = encoded.len() == 3
        && encoded[0] == b'=' as u16
        && ((b'A' as u16..=b'Z' as u16).contains(&encoded[1])
            || (b'a' as u16..=b'z' as u16).contains(&encoded[1]))
        && encoded[2] == b':' as u16;
    if !drive_directory && (encoded.is_empty() || encoded.contains(&(b'=' as u16))) {
        return Err(invalid("Invalid Windows environment name"));
    }
    Ok(encoded)
}

/// Apply explicit edits to a complete environment using the same native name identity as spawning.
pub fn merge_environment(
    mut environment: Vec<(OsString, OsString)>,
    edits: impl IntoIterator<Item = (OsString, Option<OsString>)>,
) -> io::Result<Vec<(OsString, OsString)>> {
    environment_block(&environment)?;
    for (key, value) in edits {
        let encoded = environment_name(&key)?;
        let mut retained = Vec::with_capacity(environment.len() + 1);
        for entry in environment {
            if compare_names(&wide(&entry.0)?, &encoded)? != std::cmp::Ordering::Equal {
                retained.push(entry);
            }
        }
        if let Some(value) = value {
            wide(&value)?;
            retained.push((key, value));
        }
        environment = retained;
    }
    Ok(environment)
}

fn environment_block(environment: &[(OsString, OsString)]) -> io::Result<Vec<u16>> {
    let mut entries = environment
        .iter()
        .map(|(key, value)| {
            let key = environment_name(key)?;
            Ok((key, wide(value)?))
        })
        .collect::<io::Result<Vec<_>>>()?;
    // Keep native comparison failures in the error channel instead of inventing an ordering.
    for index in 1..entries.len() {
        let mut position = index;
        while position > 0 {
            match compare_names(&entries[position - 1].0, &entries[position].0)? {
                std::cmp::Ordering::Equal => {
                    return Err(invalid("Duplicate Windows environment name"));
                }
                std::cmp::Ordering::Less => break,
                std::cmp::Ordering::Greater => entries.swap(position - 1, position),
            }
            position -= 1;
        }
    }
    let mut block = Vec::new();
    for (key, value) in entries {
        block.extend(key);
        block.push(b'=' as u16);
        block.extend(value);
        block.push(0);
    }
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    Ok(block)
}

struct Attributes {
    storage: Vec<usize>,
}
impl Attributes {
    fn new() -> io::Result<Self> {
        let mut bytes = 0;
        // SAFETY: the first call obtains the required size; storage is pointer-aligned and retained.
        unsafe {
            InitializeProcThreadAttributeList(null_mut(), 2, 0, &mut bytes);
        }
        if bytes == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut storage = vec![0usize; bytes.div_ceil(size_of::<usize>())];
        if unsafe {
            InitializeProcThreadAttributeList(storage.as_mut_ptr().cast(), 2, 0, &mut bytes)
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { storage })
    }
    fn pointer(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.storage.as_mut_ptr().cast()
    }
}
impl Drop for Attributes {
    fn drop(&mut self) {
        // SAFETY: construction initialized this retained attribute list exactly once.
        unsafe {
            DeleteProcThreadAttributeList(self.pointer());
        }
    }
}

fn pipe(parent_reads: bool) -> io::Result<(File, OwnedHandle)> {
    // SAFETY: returned handles are transferred immediately to RAII and are valid pipe endpoints.
    unsafe {
        let security = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: null_mut(),
            bInheritHandle: 1,
        };
        let (mut read, mut write) = (null_mut(), null_mut());
        if CreatePipe(&mut read, &mut write, &security, 0) == 0 {
            return Err(io::Error::last_os_error());
        }
        let (read, write) = (
            OwnedHandle::from_raw_handle(read),
            OwnedHandle::from_raw_handle(write),
        );
        let (parent, child) = if parent_reads {
            (read, write)
        } else {
            (write, read)
        };
        if SetHandleInformation(parent.as_raw_handle(), HANDLE_FLAG_INHERIT, 0) == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((File::from(parent), child))
    }
}

/// The process and job remain retained after a retirement timeout; the caller must not replace it.
pub struct OwnedWindowsChild {
    process: OwnedHandle,
    job: OwnedHandle,
    pid: u32,
    pub stdin: Option<File>,
    pub stdout: Option<File>,
    pub stderr: Option<File>,
}

impl OwnedWindowsChild {
    /// `environment` is the complete child environment, not a set of edits to ambient state.
    pub fn spawn(
        executable: &Path,
        arguments: &[OsString],
        environment: &[(OsString, OsString)],
    ) -> io::Result<Self> {
        if !executable.is_absolute() {
            return Err(invalid("Windows child executable must be absolute"));
        }
        let mut application = wide(executable.as_os_str())?;
        let mut command = Vec::new();
        argument(&mut command, &application);
        application.push(0);
        for value in arguments {
            command.push(b' ' as u16);
            argument(&mut command, &wide(value)?);
        }
        command.push(0);
        if command.len() > 32767 {
            return Err(invalid("Windows command line exceeds its native limit"));
        }
        let environment = environment_block(environment)?;
        // SAFETY: all input structures, attribute values and inherited handles remain alive until
        // CreateProcess returns. Only the three child pipe ends are inheritable by this child.
        unsafe {
            let raw_job = CreateJobObjectW(null(), null());
            if raw_job.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = OwnedHandle::from_raw_handle(raw_job);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                raw_job,
                JobObjectExtendedLimitInformation,
                (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let (stdin, child_in) = pipe(false)?;
            let (stdout, child_out) = pipe(true)?;
            let (stderr, child_err) = pipe(true)?;
            let inherited = [
                child_in.as_raw_handle(),
                child_out.as_raw_handle(),
                child_err.as_raw_handle(),
            ];
            let jobs = [raw_job];
            let mut attributes = Attributes::new()?;
            for (kind, value, bytes) in [
                (
                    PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                    inherited.as_ptr().cast(),
                    size_of_val(&inherited),
                ),
                (
                    PROC_THREAD_ATTRIBUTE_JOB_LIST,
                    jobs.as_ptr().cast(),
                    size_of_val(&jobs),
                ),
            ] {
                if UpdateProcThreadAttribute(
                    attributes.pointer(),
                    0,
                    kind as usize,
                    value,
                    bytes,
                    null_mut(),
                    null(),
                ) == 0
                {
                    return Err(io::Error::last_os_error());
                }
            }
            let mut startup: STARTUPINFOEXW = zeroed();
            startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = inherited[0];
            startup.StartupInfo.hStdOutput = inherited[1];
            startup.StartupInfo.hStdError = inherited[2];
            startup.lpAttributeList = attributes.pointer();
            let mut information: PROCESS_INFORMATION = zeroed();
            if CreateProcessW(
                application.as_ptr(),
                command.as_mut_ptr(),
                null(),
                null(),
                1,
                EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                environment.as_ptr().cast(),
                null(),
                &startup.StartupInfo,
                &mut information,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let process = OwnedHandle::from_raw_handle(information.hProcess);
            let _thread = OwnedHandle::from_raw_handle(information.hThread);
            Ok(Self {
                process,
                job,
                pid: information.dwProcessId,
                stdin: Some(stdin),
                stdout: Some(stdout),
                stderr: Some(stderr),
            })
        }
    }

    pub fn id(&self) -> u32 {
        self.pid
    }

    pub fn try_wait(&self) -> io::Result<Option<ExitStatus>> {
        // SAFETY: the retained process handle has synchronization and query access.
        unsafe {
            match WaitForSingleObject(self.process.as_raw_handle(), 0) {
                WAIT_TIMEOUT => Ok(None),
                WAIT_OBJECT_0 => {
                    let mut code = 0;
                    if GetExitCodeProcess(self.process.as_raw_handle(), &mut code) == 0 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(Some(ExitStatus::from_raw(code)))
                }
                _ => Err(io::Error::last_os_error()),
            }
        }
    }

    /// Root exit alone does not prove that a worker's descendants released their resources.
    pub fn try_retirement(&self) -> io::Result<Option<ExitStatus>> {
        let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { zeroed() };
        // SAFETY: this object retains the queried job and the output buffer has the exact API size.
        if unsafe {
            QueryInformationJobObject(
                self.job.as_raw_handle(),
                JobObjectBasicAccountingInformation,
                (&mut accounting as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast(),
                size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        if accounting.ActiveProcesses == 0 {
            self.try_wait()
        } else {
            Ok(None)
        }
    }

    pub fn start_kill(&self) -> io::Result<()> {
        // SAFETY: this object retains the private job throughout termination.
        if unsafe { TerminateJobObject(self.job.as_raw_handle(), 1) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn retire(&self, timeout: Duration) -> io::Result<ExitStatus> {
        self.start_kill()?;
        let started = Instant::now();
        loop {
            if let Some(status) = self.try_retirement()? {
                return Ok(status);
            }
            if started.elapsed() >= timeout {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Windows child job retirement remains unproven",
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};

    #[test]
    fn command_arguments_preserve_quotes_empty_values_and_trailing_slashes() {
        for (input, expected) in [
            ("", "\"\""),
            ("a b", "\"a b\""),
            ("a\"b", "\"a\\\"b\""),
            ("C:\\models\\", "\"C:\\models\\\\\""),
        ] {
            let mut output = Vec::new();
            argument(&mut output, &wide(OsStr::new(input)).unwrap());
            assert_eq!(String::from_utf16(&output).unwrap(), expected);
        }
        assert!(wide(OsStr::new("bad\0value")).is_err());
    }

    #[test]
    fn environment_uses_native_order_and_rejects_ambiguous_names() {
        assert_eq!(environment_block(&[]).unwrap(), vec![0, 0]);
        let environment = [("z".into(), "模型".into()), ("A".into(), "".into())];
        assert_eq!(
            String::from_utf16(&environment_block(&environment).unwrap()).unwrap(),
            "A=\0z=模型\0\0"
        );
        for environment in [
            vec![("Path".into(), "a".into()), ("PATH".into(), "b".into())],
            vec![("=C:".into(), "a".into()), ("=c:".into(), "b".into())],
            vec![("=".into(), "x".into())],
            vec![("=1:".into(), "x".into())],
            vec![("bad=name".into(), "x".into())],
            vec![("A".into(), "bad\0value".into())],
        ] {
            assert!(environment_block(&environment).is_err());
        }
    }

    #[test]
    fn environment_preserves_and_edits_drive_directories() {
        let environment = vec![
            ("PATH".into(), "tools".into()),
            ("=C:".into(), "C:\\work".into()),
        ];
        assert_eq!(
            String::from_utf16(&environment_block(&environment).unwrap()).unwrap(),
            "=C:=C:\\work\0PATH=tools\0\0"
        );
        let merged =
            merge_environment(environment, [("=c:".into(), Some("C:\\next".into()))]).unwrap();
        assert_eq!(
            String::from_utf16(&environment_block(&merged).unwrap()).unwrap(),
            "=c:=C:\\next\0PATH=tools\0\0"
        );
    }

    #[test]
    fn environment_edits_replace_and_remove_using_windows_name_identity() {
        let merged = merge_environment(
            vec![
                ("Path".into(), "old".into()),
                ("REMOVE".into(), "value".into()),
            ],
            [
                ("PATH".into(), Some("new".into())),
                ("remove".into(), None),
                ("EMPTY".into(), Some("".into())),
            ],
        )
        .unwrap();
        assert_eq!(
            String::from_utf16(&environment_block(&merged).unwrap()).unwrap(),
            "EMPTY=\0PATH=new\0\0"
        );
    }

    #[test]
    // This fixture intentionally exits before its descendant; the test's retained job owns cleanup.
    #[allow(clippy::zombie_processes)]
    fn fixture_child() {
        match std::env::var("ICN_WINDOWS_JOB_FIXTURE").as_deref() {
            Ok("leaf") => std::thread::sleep(Duration::from_secs(60)),
            Ok("root") => {
                assert_eq!(
                    std::env::args().next_back().as_deref(),
                    Some("模型 \"quoted\" \\")
                );
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                assert_eq!(input, "request\n");
                let leaf = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "windows_process::tests::fixture_child",
                        "--nocapture",
                    ])
                    .env("ICN_WINDOWS_JOB_FIXTURE", "leaf")
                    .spawn()
                    .unwrap();
                println!("LEAF {}", leaf.id());
                // Root exit intentionally leaves a descendant in the inherited job.
                std::io::stdout().flush().unwrap();
            }
            _ => {}
        }
    }

    #[test]
    fn root_exit_does_not_bypass_retirement_of_its_descendant() {
        verify_descendant_cleanup(false);
    }

    #[test]
    fn dropping_the_owner_closes_its_kill_on_close_job() {
        verify_descendant_cleanup(true);
    }

    fn verify_descendant_cleanup(drop_owner: bool) {
        let environment = std::env::vars_os()
            .filter(|(key, _)| key != "ICN_WINDOWS_JOB_FIXTURE")
            .chain([(
                OsString::from("ICN_WINDOWS_JOB_FIXTURE"),
                OsString::from("root"),
            )])
            .collect::<Vec<_>>();
        // libtest's --skip accepts an arbitrary filter; it also exercises native argument quoting.
        let arguments = [
            "--exact",
            "windows_process::tests::fixture_child",
            "--nocapture",
            "--skip",
            "模型 \"quoted\" \\",
        ]
        .map(OsString::from);
        let mut child =
            OwnedWindowsChild::spawn(&std::env::current_exe().unwrap(), &arguments, &environment)
                .unwrap();
        child.stdin.take().unwrap().write_all(b"request\n").unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.unwrap();
                if let Some((_, pid)) = line.split_once("LEAF ") {
                    sender.send(pid.trim().parse::<u32>().unwrap()).unwrap();
                    return;
                }
            }
        });
        let pid = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        // SAFETY: observation retains this live descendant before the root/job is retired.
        let observed = unsafe {
            let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
            assert!(!handle.is_null());
            OwnedHandle::from_raw_handle(handle)
        };
        let started = Instant::now();
        while child.try_wait().unwrap().is_none() {
            assert!(started.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            unsafe { WaitForSingleObject(observed.as_raw_handle(), 0) },
            WAIT_TIMEOUT
        );
        assert!(child.try_retirement().unwrap().is_none());
        if drop_owner {
            drop(child);
        } else {
            child.retire(Duration::from_secs(2)).unwrap();
        }
        assert_eq!(
            unsafe { WaitForSingleObject(observed.as_raw_handle(), 2000) },
            WAIT_OBJECT_0
        );
        reader.join().unwrap();
    }
}
