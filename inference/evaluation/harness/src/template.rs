//! Template files: one benchmark request's sessions, the only model input.
//! The rig builds them from a template name (`magnitude_fixtures`); the
//! harness reads the name as an opaque label and sizes every lane to the
//! file's context.

use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

use crate::digest::sha256_hex;

/// One lane's conversation: history turns after a kept head, then the probe.
/// The fixture, response kind and expected answer belong to the builders
/// and the bench.
#[derive(Clone, Debug, Deserialize)]
pub struct Session {
    #[serde(default)]
    pub tools: Vec<Value>,
    pub messages: Vec<Map<String, Value>>,
    pub generation: Generation,
    pub keep_head: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Generation {
    pub prefix: usize,
}

/// The template file's contents as written by the rig.
#[derive(Deserialize)]
struct Contents {
    template: String,
    context: usize,
    sessions: Vec<Session>,
}

/// A template file as read: its name, its context, its sessions in lane
/// order, its path and digest (SHA-256 of the file bytes).
pub struct TemplateFile {
    pub path: PathBuf,
    pub digest: String,
    pub name: String,
    pub context: usize,
    pub sessions: Vec<Session>,
}

impl TemplateFile {
    pub fn read(path: &Path) -> Result<Self, String> {
        let failed = |error: String| format!("template {}: {error}", path.display());
        let bytes = std::fs::read(path).map_err(|error| failed(error.to_string()))?;
        let contents: Contents =
            serde_json::from_slice(&bytes).map_err(|error| failed(error.to_string()))?;
        if contents.sessions.is_empty() {
            return Err(failed("it holds no session".into()));
        }
        if contents.context == 0 {
            return Err(failed("its context is zero".into()));
        }
        for (lane, session) in contents.sessions.iter().enumerate() {
            session
                .validate()
                .map_err(|reason| failed(format!("lane {lane}: {reason}")))?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            digest: sha256_hex(&bytes),
            name: contents.template,
            context: contents.context,
            sessions: contents.sessions,
        })
    }

    /// Every file of `paths`, in order.
    pub fn read_all(paths: &[PathBuf]) -> Result<Vec<Self>, String> {
        if paths.is_empty() {
            return Err("a run takes at least one --template".into());
        }
        paths.iter().map(|path| Self::read(path)).collect()
    }
}

impl Session {
    fn validate(&self) -> Result<(), String> {
        if self.messages.is_empty() {
            return Err("no messages".into());
        }
        for message in &self.messages {
            let role = message.get("role").and_then(Value::as_str);
            if let Some(scored) = message.get("scored") {
                if scored != &Value::Bool(true) || role != Some("assistant") {
                    return Err("only assistant messages may carry scored: true".into());
                }
            }
            if message.contains_key("tools") && role != Some("user") {
                return Err("only a unit's opening user message may carry tools".into());
            }
        }
        if self.generation.prefix == 0 || self.generation.prefix >= self.messages.len() {
            return Err("generation.prefix is outside the messages".into());
        }
        let units = self.units()?;
        let probe = units.last().expect("units are nonempty");
        let scored_outside = self
            .messages
            .iter()
            .enumerate()
            .any(|(index, message)| message.contains_key("scored") && index < probe.start);
        if scored_outside {
            return Err("scored messages live only in the probe".into());
        }
        if self.generation.prefix <= probe.start {
            return Err("generation.prefix ends before the probe".into());
        }
        Ok(())
    }

    /// The sizing units after the kept head (one user message through its
    /// final assistant or tool reply), as message ranges in order. The last
    /// is the probe.
    pub fn units(&self) -> Result<Vec<std::ops::Range<usize>>, String> {
        let mut units: Vec<std::ops::Range<usize>> = Vec::new();
        for index in self.keep_head..self.messages.len() {
            let opens = self.messages[index].get("role").and_then(Value::as_str) == Some("user");
            match units.last_mut() {
                Some(unit) if !opens => unit.end = index + 1,
                _ => units.push(index..index + 1),
            }
        }
        if units.is_empty() {
            return Err("no unit after the kept head".into());
        }
        if self.messages[units[0].start].get("role").and_then(Value::as_str) != Some("user") {
            return Err("the first unit after the head does not open with a user message".into());
        }
        Ok(units)
    }
}
