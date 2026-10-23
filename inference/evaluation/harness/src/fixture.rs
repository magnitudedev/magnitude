//! Fixture sets: deterministic chat conversations, the only model input
//! (contract "Fixture").

use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

use crate::digest::sha256_hex;

/// The fields of a fixture the harness renders. Family, source and the
/// generation's expected answer belong to the builders and the bench.
#[derive(Clone, Debug, Deserialize)]
pub struct Fixture {
    pub id: String,
    #[serde(default)]
    pub tools: Vec<Value>,
    pub messages: Vec<Map<String, Value>>,
    #[serde(default)]
    pub generation: Option<Generation>,
    pub sizing: Sizing,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Generation {
    pub prefix: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "unit", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Sizing {
    Turn { keep_head: usize },
    Fixed,
}

/// A fixture set as read: its fixtures in file order, its path and digest
/// (SHA-256 of the file bytes).
pub struct FixtureSet {
    pub path: PathBuf,
    pub digest: String,
    pub fixtures: Vec<Fixture>,
}

impl FixtureSet {
    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|error| format!("fixtures {}: {error}", path.display()))?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|error| format!("fixtures {}: {error}", path.display()))?;
        let fixtures = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| {
                let fixture: Fixture = serde_json::from_str(line).map_err(|error| {
                    format!("fixtures {} line {}: {error}", path.display(), index + 1)
                })?;
                fixture.validate()?;
                Ok(fixture)
            })
            .collect::<Result<Vec<_>, String>>()?;
        if fixtures.is_empty() {
            return Err(format!("fixtures {} holds no fixture", path.display()));
        }
        Ok(Self {
            path: path.to_path_buf(),
            digest: sha256_hex(&bytes),
            fixtures,
        })
    }
}

impl Fixture {
    fn validate(&self) -> Result<(), String> {
        let invalid = |reason: &str| Err(format!("fixture {}: {reason}", self.id));
        if self.messages.is_empty() {
            return invalid("no messages");
        }
        for message in &self.messages {
            let role = message.get("role").and_then(Value::as_str);
            if let Some(scored) = message.get("scored") {
                if scored != &Value::Bool(true) || role != Some("assistant") {
                    return invalid("only assistant messages may carry scored: true");
                }
            }
            if message.contains_key("tools") && role != Some("user") {
                return invalid("only a unit's opening user message may carry tools");
            }
        }
        if let Some(generation) = &self.generation {
            if generation.prefix == 0 || generation.prefix >= self.messages.len() + 1 {
                return invalid("generation.prefix is outside the messages");
            }
        }
        if let Sizing::Turn { keep_head } = self.sizing {
            let units = self.units()?;
            let probe = units.last().expect("units are nonempty");
            let scored_outside = self
                .messages
                .iter()
                .enumerate()
                .any(|(index, message)| message.contains_key("scored") && index < probe.start);
            if scored_outside {
                return invalid("scored messages live only in the probe");
            }
            if keep_head > probe.start {
                return invalid("keep_head reaches into the probe");
            }
            if let Some(generation) = &self.generation {
                if generation.prefix <= probe.start {
                    return invalid("generation.prefix ends before the probe");
                }
            }
        }
        Ok(())
    }

    /// The sizing units after the kept head (`turn`: one user message through
    /// its final assistant or tool reply), as message ranges in order. The
    /// last is the probe.
    pub fn units(&self) -> Result<Vec<std::ops::Range<usize>>, String> {
        let Sizing::Turn { keep_head } = self.sizing else {
            return Err(format!("fixture {}: a fixed fixture has no units", self.id));
        };
        let mut units: Vec<std::ops::Range<usize>> = Vec::new();
        for index in keep_head..self.messages.len() {
            let opens = self.messages[index].get("role").and_then(Value::as_str) == Some("user");
            match units.last_mut() {
                Some(unit) if !opens => unit.end = index + 1,
                _ => units.push(index..index + 1),
            }
        }
        if units.is_empty() {
            return Err(format!("fixture {}: no unit after the kept head", self.id));
        }
        if self.messages[units[0].start]
            .get("role")
            .and_then(Value::as_str)
            != Some("user")
        {
            return Err(format!(
                "fixture {}: the first unit after the head does not open with a user message",
                self.id
            ));
        }
        Ok(units)
    }
}
