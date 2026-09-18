use std::process::Command;
#[test]
fn explicit_compilation_commands_are_unavailable() {
    for command in ["run", "lower", "native", "inspect", "explore", "choices", "account", "calibrate"] {
        let output = Command::new(env!("CARGO_BIN_EXE_seismic")).arg(command).output().unwrap();
        assert!(!output.status.success(), "{command} bypassed automatic selection");
        assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command"));
    }
}
#[test]
fn implementation_flags_are_not_accepted_by_source_tools() {
    for flag in ["--lowering-path", "--execution-path", "--loads", "--piece", "--per-item", "--split", "--sg-per-tg", "--threads-per-block"] {
        let output = Command::new(env!("CARGO_BIN_EXE_seismic")).args(["check", flag, "1"]).output().unwrap();
        assert!(!output.status.success(), "{flag} was accepted");
        assert!(String::from_utf8_lossy(&output.stderr).contains("implementation choices are compiler-owned"));
    }
}
