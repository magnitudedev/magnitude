use magnitude_templates::{PreparedRequest, Request as NativeRequest, SpecialTokens, Template};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemplateVariant {
    pub name: String,
    pub source: String,
    pub provenance: String,
}

/// Validated artifact template metadata. Selection never invents a default source.
pub struct TemplateBundle {
    variants: BTreeMap<String, TemplateVariant>,
    default: String,
    special_tokens: SpecialTokens,
    profiles: RefCell<super::reasoning::ProfileCache>,
}
#[derive(Default)]
pub struct TemplateSelection<'a> {
    pub variant: Option<&'a str>,
    pub source_override: Option<&'a TemplateVariant>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Named(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub messages: Vec<serde_json::Value>,
    pub now: i64,
    pub tools: Vec<serde_json::Value>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub template_arguments: serde_json::Map<String, serde_json::Value>,
    pub json_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}
impl ChatRequest {
    pub fn new(messages: Vec<serde_json::Value>, now: i64) -> Self {
        Self {
            messages,
            now,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            template_arguments: Default::default(),
            json_schema: None,
            reasoning_effort: None,
        }
    }
}

fn validate_variant(variant: &TemplateVariant) -> Result<(), String> {
    if variant.name.is_empty() || variant.source.is_empty() || variant.provenance.is_empty() {
        return Err("template variant requires a name, source, and provenance".into());
    }
    Ok(())
}
impl TemplateBundle {
    pub fn new(
        variants: Vec<TemplateVariant>,
        default: String,
        special_tokens: SpecialTokens,
    ) -> Result<Self, String> {
        let mut declared = BTreeMap::new();
        for variant in variants {
            validate_variant(&variant)?;
            if declared.insert(variant.name.clone(), variant).is_some() {
                return Err("duplicate template variant".into());
            }
        }
        if !declared.contains_key(&default) {
            return Err("template bundle requires a declared default".into());
        }
        if special_tokens.keys().any(String::is_empty) {
            return Err("special token name must be nonempty".into());
        }
        Ok(Self {
            variants: declared,
            default,
            special_tokens,
            profiles: RefCell::new(super::reasoning::ProfileCache::new(16, 1024 * 1024)),
        })
    }
    pub fn select<'a>(
        &'a self,
        tools_offered: bool,
        selection: &TemplateSelection<'a>,
    ) -> Result<&'a TemplateVariant, String> {
        if let Some(source) = selection.source_override {
            if selection.variant.is_some() {
                return Err("configure a source override or a variant, not both".into());
            }
            validate_variant(source)?;
            return Ok(source);
        }
        let selected = selection.variant.unwrap_or_else(|| {
            if tools_offered && self.variants.contains_key("tool_use") {
                "tool_use"
            } else {
                &self.default
            }
        });
        self.variants
            .get(selected)
            .ok_or_else(|| format!("unknown template variant: {selected}"))
    }
    /// Resolve effective tools before selecting a variant. Cloning normalization
    /// preserves caller-owned messages and historical tool-call arguments.
    /// This prepares text/grammar/parser semantics; it does not bind a tokenizer
    /// or admit numerical generation with an unbound grammar.
    pub fn prepare(
        &self,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<PreparedRequest, String> {
        let choice = &request.tool_choice;
        let effort = request.reasoning_effort.as_deref();
        if effort.is_some()
            && super::reasoning::CONTROLS
                .iter()
                .any(|key| request.template_arguments.contains_key(*key))
        {
            return Err(
                "normalized reasoning effort conflicts with raw template reasoning controls".into(),
            );
        }
        let mut request = NativeRequest {
            messages: request.messages.clone(),
            now: request.now,
            tools: request.tools.clone(),
            tool_choice: magnitude_templates::ToolChoice::Auto,
            parallel_tool_calls: request.parallel_tool_calls,
            template_arguments: request.template_arguments.clone(),
            json_schema: request.json_schema.clone(),
        };
        let mut names = HashSet::new();
        for tool in &request.tools {
            if tool.get("type").and_then(|x| x.as_str()) != Some("function") {
                return Err("only function tools are supported".into());
            }
            let name = tool
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty())
                .ok_or("tool requires a nonempty function name")?;
            if !names.insert(name.to_owned()) {
                return Err("tools must have unique names".into());
            }
        }
        match choice {
            ToolChoice::None => request.tools.clear(),
            ToolChoice::Named(name) => {
                if !names.contains(name) {
                    return Err("tool choice names an unavailable function".into());
                }
                request
                    .tools
                    .retain(|tool| tool["function"]["name"].as_str() == Some(name));
            }
            ToolChoice::Required if request.tools.is_empty() => {
                return Err("required tool choice needs a tool".into())
            }
            _ => {}
        }
        let required = matches!(choice, ToolChoice::Required | ToolChoice::Named(_));
        request.tool_choice = if required {
            magnitude_templates::ToolChoice::Required
        } else {
            magnitude_templates::ToolChoice::Auto
        };
        for message in &mut request.messages {
            let message = message
                .as_object_mut()
                .ok_or("chat message must be an object")?;
            if let Some(calls) = message.get_mut("tool_calls") {
                for call in calls.as_array_mut().ok_or("tool_calls must be an array")? {
                    let function = call
                        .get_mut("function")
                        .and_then(|x| x.as_object_mut())
                        .ok_or("historical tool call requires a function")?;
                    let arguments = function
                        .get_mut("arguments")
                        .ok_or("historical tool call requires arguments")?;
                    if let Some(text) = arguments.as_str() {
                        *arguments = if text.trim().is_empty() {
                            serde_json::json!({})
                        } else {
                            serde_json::from_str(text)
                                .map_err(|e| format!("historical tool arguments: {e}"))?
                        };
                    }
                    if !arguments.is_object() {
                        return Err("historical tool arguments must be a JSON object".into());
                    }
                }
            }
        }
        let variant = self.select(!request.tools.is_empty(), selection)?;
        let template =
            Template::new(&variant.source, &self.special_tokens).map_err(|e| e.to_string())?;
        if effort.is_some() {
            let profile = self
                .profiles
                .borrow_mut()
                .inspect(&template, &request.template_arguments)?;
            request.template_arguments.extend(profile.resolve(effort)?);
        }
        let prepared = template.prepare(&request).map_err(|e| e.to_string())?;
        if (required || request.json_schema.is_some()) && prepared.description().grammar.is_empty()
        {
            return Err("selected template did not produce required output constraints".into());
        }
        Ok(prepared)
    }
}
