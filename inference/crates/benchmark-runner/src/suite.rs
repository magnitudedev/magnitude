use std::collections::BTreeSet;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::model::{BenchmarkError, FixtureCatalog, Profile, SUITE_VERSION};

pub async fn load_profile(root: &Path, id: &str) -> Result<Profile, BenchmarkError> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(BenchmarkError::Asset(format!(
            "profile id {id:?} is not a safe filename stem"
        )));
    }
    let path = root.join("profiles").join(format!("{id}.toml"));
    let bytes = tokio::fs::read(&path).await.map_err(|error| {
        BenchmarkError::Asset(format!("failed to read {}: {error}", path.display()))
    })?;
    let profile: Profile = toml::from_str(std::str::from_utf8(&bytes).map_err(|error| {
        BenchmarkError::Asset(format!("{} is not UTF-8: {error}", path.display()))
    })?)
    .map_err(|error| {
        BenchmarkError::Asset(format!("failed to decode {}: {error}", path.display()))
    })?;
    validate_profile(&profile)?;
    Ok(profile)
}

pub async fn load_fixtures(root: &Path) -> Result<(FixtureCatalog, String), BenchmarkError> {
    let path = root.join("fixtures").join("core.json");
    let bytes = tokio::fs::read(&path).await.map_err(|error| {
        BenchmarkError::Asset(format!("failed to read {}: {error}", path.display()))
    })?;
    let fixtures: FixtureCatalog = serde_json::from_slice(&bytes)?;
    validate_fixtures(&fixtures)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok((fixtures, digest))
}

pub async fn validate_assets(root: &Path) -> Result<(), BenchmarkError> {
    let (fixtures, _) = load_fixtures(root).await?;
    if fixtures.suite_version != SUITE_VERSION {
        return Err(BenchmarkError::Asset(format!(
            "fixture suite version {} does not match runner version {SUITE_VERSION}",
            fixtures.suite_version
        )));
    }
    if load_profiles(root).await?.is_empty() {
        return Err(BenchmarkError::Asset(
            "benchmark contains no profile definitions".into(),
        ));
    }
    Ok(())
}

async fn load_profiles(root: &Path) -> Result<Vec<Profile>, BenchmarkError> {
    let profiles_path = root.join("profiles");
    let mut entries = tokio::fs::read_dir(&profiles_path).await.map_err(|error| {
        BenchmarkError::Asset(format!(
            "failed to read {}: {error}",
            profiles_path.display()
        ))
    })?;
    let mut profiles = Vec::new();
    while let Some(entry) = entries.next_entry().await.map_err(|error| {
        BenchmarkError::Asset(format!(
            "failed to enumerate {}: {error}",
            profiles_path.display()
        ))
    })? {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("toml") {
            continue;
        }
        let id = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| BenchmarkError::Asset("profile filename is not UTF-8".into()))?;
        let profile = load_profile(root, id).await?;
        if profile.id != id {
            return Err(BenchmarkError::Asset(format!(
                "profile file {id}.toml declares id {}",
                profile.id
            )));
        }
        profiles.push(profile);
    }
    profiles.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(profiles)
}

fn validate_profile(profile: &Profile) -> Result<(), BenchmarkError> {
    if profile.schema_version != 2 {
        return Err(BenchmarkError::Asset(format!(
            "profile {} uses unsupported schema version {}",
            profile.id, profile.schema_version
        )));
    }
    if profile.repetitions == 0 {
        return Err(BenchmarkError::Asset(format!(
            "profile {} requires positive repetitions",
            profile.id
        )));
    }
    if profile.cases.is_empty() {
        return Err(BenchmarkError::Asset(format!(
            "profile {} must select at least one case",
            profile.id
        )));
    }
    let mut cases = BTreeSet::new();
    let mut wildcards = BTreeSet::new();
    for case in &profile.cases {
        let Some((experiment, arm)) = case.split_once('/') else {
            return Err(BenchmarkError::Asset(format!(
                "profile {} case {case} must use EXPERIMENT/ARM syntax",
                profile.id
            )));
        };
        if !valid_case(experiment, arm) || !cases.insert(case) {
            return Err(BenchmarkError::Asset(format!(
                "profile {} selects invalid case {case}",
                profile.id
            )));
        }
        if arm == "*" {
            wildcards.insert(experiment);
        }
    }
    if profile.cases.iter().any(|case| {
        case.split_once('/')
            .is_some_and(|(experiment, arm)| arm != "*" && wildcards.contains(experiment))
    }) {
        return Err(BenchmarkError::Asset(format!(
            "profile {} mixes a wildcard with redundant cases from the same experiment",
            profile.id
        )));
    }
    let selects_e2 = profile.cases.iter().any(|case| case.starts_with("E2/"));
    let expands_e2 = profile.cases.iter().any(|case| case == "E2/*");
    let concurrency_valid = !profile.concurrency.contains(&0)
        && profile.concurrency.windows(2).all(|pair| pair[0] < pair[1]);
    if selects_e2
        && (profile.closed_loop_multiplier == 0
            || (expands_e2 && (profile.concurrency.is_empty() || !concurrency_valid))
            || (!expands_e2 && !profile.concurrency.is_empty()))
    {
        return Err(BenchmarkError::Asset(format!(
            "profile {} must configure closed_loop_multiplier and use concurrency only to expand E2/*",
            profile.id
        )));
    }
    if !selects_e2 && (profile.closed_loop_multiplier != 0 || !profile.concurrency.is_empty()) {
        return Err(BenchmarkError::Asset(format!(
            "profile {} configures E2 parameters without selecting E2",
            profile.id
        )));
    }
    if profile.controlled
        && (!profile.paired
            || profile.min_paired_repetitions < 2
            || profile.max_paired_repetitions < profile.min_paired_repetitions
            || !(0.0..1.0).contains(&profile.confidence_half_width_ratio))
    {
        return Err(BenchmarkError::Asset(format!(
            "profile {} has invalid controlled-run stability settings",
            profile.id
        )));
    }
    if !profile.controlled
        && (profile.min_paired_repetitions != 0 || profile.max_paired_repetitions != 0)
    {
        return Err(BenchmarkError::Asset(format!(
            "profile {} configures controlled-run stability without enabling controlled mode",
            profile.id
        )));
    }
    if profile.paired && !(0.0..1.0).contains(&profile.confidence_half_width_ratio) {
        return Err(BenchmarkError::Asset(format!(
            "profile {} requires a paired precision threshold between zero and one",
            profile.id
        )));
    }
    if !profile.paired && profile.confidence_half_width_ratio != 0.0 {
        return Err(BenchmarkError::Asset(format!(
            "profile {} configures paired precision without enabling paired mode",
            profile.id
        )));
    }
    Ok(())
}

fn valid_case(experiment: &str, arm: &str) -> bool {
    if arm == "*" {
        return matches!(experiment, "E1" | "E2" | "E3" | "E4" | "E5" | "E6" | "E7");
    }
    match experiment {
        "E1" => matches!(arm, "ss" | "ls" | "sl" | "ll"),
        "E2" => arm
            .split_once(".c")
            .and_then(|(phase, concurrency)| {
                (matches!(phase, "prefill" | "decode"))
                    .then(|| concurrency.parse::<usize>().ok())
                    .flatten()
            })
            .is_some_and(|concurrency| concurrency > 0),
        "E3" => arm == "prefill-arrives-during-decode",
        "E4" => matches!(arm, "exact" | "partial" | "unrelated"),
        "E5" => matches!(arm, "shared" | "independent"),
        "E6" => arm == "forced-edit",
        "E7" => arm == "decode-cancel-recovery",
        _ => false,
    }
}

fn validate_fixtures(fixtures: &FixtureCatalog) -> Result<(), BenchmarkError> {
    if fixtures.schema_version != 1 || fixtures.suite_version != SUITE_VERSION {
        return Err(BenchmarkError::Asset(
            "core fixture schema or suite version is unsupported".into(),
        ));
    }
    if fixtures.prompt_short_tokens == 0
        || fixtures.prompt_long_tokens <= fixtures.prompt_short_tokens
        || fixtures.output_short_tokens == 0
        || fixtures.output_long_tokens <= fixtures.output_short_tokens
        || fixtures.carrier_block.is_empty()
        || fixtures.answer_short.is_empty()
        || fixtures.answer_long.len() <= fixtures.answer_short.len()
    {
        return Err(BenchmarkError::Asset(
            "core fixture sizes and carrier block are invalid".into(),
        ));
    }
    let tool = &fixtures.tool_fixture;
    if tool.old_text.is_empty()
        || tool.old_text == tool.new_text
        || tool.before.matches(&tool.old_text).count() != 1
        || tool.tool_name.is_empty()
        || tool.final_acknowledgement.is_empty()
    {
        return Err(BenchmarkError::Asset(
            "tool fixture must define one unique replacement and one exact acknowledgement".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn profile_discovery_tracks_directory_contents_without_a_registry() {
        let directory = tempfile::tempdir().unwrap();
        let profiles = directory.path().join("profiles");
        tokio::fs::create_dir(&profiles).await.unwrap();
        let profile = |id: &str| {
            format!(
                "schema_version = 2\nid = \"{id}\"\ndescription = \"test\"\ncases = [\"E1/ss\"]\nrepetitions = 2\nwarmups = 0\nrequest_timeout_seconds = 30\n"
            )
        };

        tokio::fs::write(profiles.join("first.toml"), profile("first"))
            .await
            .unwrap();
        assert_eq!(
            load_profiles(directory.path())
                .await
                .unwrap()
                .into_iter()
                .map(|profile| profile.id)
                .collect::<Vec<_>>(),
            ["first"]
        );

        tokio::fs::remove_file(profiles.join("first.toml"))
            .await
            .unwrap();
        tokio::fs::write(profiles.join("replacement.toml"), profile("replacement"))
            .await
            .unwrap();
        assert_eq!(
            load_profiles(directory.path())
                .await
                .unwrap()
                .into_iter()
                .map(|profile| profile.id)
                .collect::<Vec<_>>(),
            ["replacement"]
        );
    }
}
