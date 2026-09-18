use std::process::Command;
#[test]
fn generic_entry_is_bound_consistently_for_run_account_and_lower() {
    let dir = std::env::temp_dir().join(format!("seismic-cli-elements-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("generic.seismic.portable");
    std::fs::write(
        &path,
        "fn copy[N](x: tensor[N] ACTIVATION, out: tensor[N] f32):\n  t = load(x)\n  store(t,out)\n",
    )
    .unwrap();
    for command in ["run", "account", "lower"] {
        let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
            .args([
                command,
                path.to_str().unwrap(),
                "--fn",
                "copy",
                "--shape",
                "N=4",
                "--element",
                "ACTIVATION=bf16",
                "--target",
                "cpu",
                "--iters",
                "1",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{command}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args([
            "lower",
            path.to_str().unwrap(),
            "--fn",
            "copy",
            "--shape",
            "N=4",
            "--element",
            "ACTIVATION=not_a_type",
            "--target",
            "cpu",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn exploration_reports_budget_coverage_and_accounts_actual_materialization_choices() {
    let dir = std::env::temp_dir().join(format!("seismic-cli-explore-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("choices.seismic.portable");
    std::fs::write(&path, "fn evaluate[N](x: tensor[N] f32, out: tensor[N] f32):\n  a = load(x)\n  producer = tile[N] f32\n  for i in owned(producer): producer[i] = a[i] * 2.0\n  consumer = tile[N] f32\n  for i in owned(consumer): consumer[i] = producer[i] + 1.0\n  store(consumer,out)\n").unwrap();
    for budget in [1, 2] {
        let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
            .args([
                "explore",
                path.to_str().unwrap(),
                "--fn",
                "evaluate",
                "--shape",
                "N=73",
                "--target",
                "cpu",
                "--candidate-budget",
                &budget.to_string(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rows: Vec<serde_json::Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let coverage = rows.last().unwrap();
        assert_eq!(coverage["exhausted"], budget == 2);
        assert_eq!(coverage["failed_attempts"], 0);
        assert_eq!(coverage["optimality_certified"], false);
        assert_eq!(coverage["attempted"], budget);
        if budget == 2 {
            let a = &rows[0]["realization"]["phases"][0];
            let b = &rows[1]["realization"]["phases"][0];
            assert_ne!(a["scalar_ir_sha256"], b["scalar_ir_sha256"]);
            assert!(
                a["private_scratch_bytes_per_invocation"].as_u64().unwrap()
                    > b["private_scratch_bytes_per_invocation"].as_u64().unwrap()
            );
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
