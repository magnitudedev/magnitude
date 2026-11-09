//! A test's setup, read by a setup run without a device
//! (`seismic_native_tests::setup`): the tests of `setup_run` are its cases,
//! ignored in an ordinary run, and run by the one test here in a setup run of
//! this executable.

use seismic::{BackendName, NativeSpecialization};
use seismic_native_tests::split_sum;

mod setup_run {
    use super::*;

    fn configuration(n: u64, parts: u64) -> NativeSpecialization {
        NativeSpecialization::new().with_static("N", n).with_param("PARTS", parts).with_param("WIDTH", 32)
    }

    /// Forms two requests its declarations take, then would work on its
    /// devices: a setup run ends it before that.
    #[test]
    #[ignore = "a case of a setup run (`a_setup_run_reads_each_setup_without_a_device`)"]
    fn adopted() {
        let devices = seismic_native_tests::setup(&[BackendName::Cpu, BackendName::Metal], |setup| {
            for parts in [1, 2] {
                setup.admit(
                    format_args!("split_sum PARTS {parts}"),
                    split_sum::native_admission(setup.backend(), &configuration(2, parts)),
                );
            }
        });
        panic!("its device work ran on {} devices", devices.len());
    }

    /// Forms a request the `where` condition refuses at N = 2.
    #[test]
    #[ignore = "a case of a setup run (`a_setup_run_reads_each_setup_without_a_device`)"]
    fn refused() {
        seismic_native_tests::setup(&[BackendName::Cpu], |setup| {
            setup.admit("split_sum PARTS 4", split_sum::native_admission(setup.backend(), &configuration(2, 4)));
        });
        panic!("its device work ran");
    }

    /// Opens its devices with no setup.
    #[test]
    #[ignore = "a case of a setup run (`a_setup_run_reads_each_setup_without_a_device`)"]
    fn unadopted() {
        let devices = seismic_native_tests::every_device();
        panic!("its device work ran on {} devices", devices.len());
    }
}

/// A setup run of this executable on the CPU backend ends each case where
/// its setup ends, no device discovered: the adopted one with what it
/// checked, the one whose request the declaration refuses with
/// preparation's message, and the one that reaches its devices with no setup
/// as unchecked.
#[test]
fn a_setup_run_reads_each_setup_without_a_device() {
    let run = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--include-ignored", "--exact", "--test-threads", "1"])
        .args(["setup_run::adopted", "setup_run::refused", "setup_run::unadopted"])
        .env("SEISMIC_TEST_SETUP", "1")
        .env("SEISMIC_TEST_BACKEND", "cpu")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    eprintln!("{stdout}{}", String::from_utf8_lossy(&run.stderr));
    // The message a case's panic left in its captured output.
    let ended = |case: &str| {
        let section = stdout
            .split(&format!("---- setup_run::{case} stdout ----"))
            .nth(1)
            .unwrap_or_else(|| panic!("{case} has no captured output"));
        section.lines().skip_while(|line| !line.contains("panicked at")).nth(1).unwrap_or("").to_owned()
    };
    assert_eq!(
        ended("adopted"),
        format!("{} 2 requests admitted on [Cpu]", seismic::testing::SETUP_CHECKED)
    );
    assert_eq!(
        ended("refused"),
        "split_sum PARTS 4 on Cpu: preparation refuses it: `split_sum`: configuration violates the native `where` condition"
    );
    assert_eq!(
        ended("unadopted"),
        format!(
            "{}: it discovers devices outside a setup, which a setup run never does",
            seismic::testing::SETUP_UNCHECKED
        )
    );
    assert!(!stdout.contains("device work ran"), "a setup run ran a case's device work");
}
