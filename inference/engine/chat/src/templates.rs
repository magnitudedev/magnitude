use super::reasoning::{ReasoningIntent, ResolvedReasoning};
use super::ChatError;
use magnitude_grammar::{CompileCache, Compiled, GrammarError};
use magnitude_templates::{PreparedRequest, Request as NativeRequest, SpecialTokens, Template};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

/// Bounded concurrent single-flight cache: the mutex is held only to look up
/// or insert a per-key `OnceLock` cell (evicting the least-recently-used entry
/// when full), never while computing a value.
struct BoundedOnceCache<K, V> {
    max_entries: usize,
    entries: Mutex<VecDeque<(K, Arc<OnceLock<V>>)>>,
}

impl<K: PartialEq, V> BoundedOnceCache<K, V> {
    fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            entries: Mutex::new(VecDeque::new()),
        }
    }

    fn cell(&self, key: K) -> Arc<OnceLock<V>> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(index) = entries.iter().position(|(existing, _)| *existing == key) {
            let entry = entries.remove(index).unwrap();
            let cell = Arc::clone(&entry.1);
            entries.push_back(entry);
            return cell;
        }
        let cell = Arc::new(OnceLock::new());
        if self.max_entries > 0 {
            while entries.len() >= self.max_entries {
                entries.pop_front();
            }
            entries.push_back((key, Arc::clone(&cell)));
        }
        cell
    }

    fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            let mut canonical = serde_json::Map::new();
            for (k, v) in entries {
                canonical.insert(k.clone(), canonical_json(v));
            }
            serde_json::Value::Object(canonical)
        }
        other => other.clone(),
    }
}

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
    profiles: BoundedOnceCache<(String, Vec<u8>), Result<Arc<super::reasoning::ReasoningProfile>, String>>,
    grammars: CompileCache,
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
    /// Offer only the named subset; `required` demands one of them be called.
    Allowed {
        names: Vec<String>,
        required: bool,
    },
}

/// Template-level chat input: OpenAI-shaped messages and tools as the native
/// template consumes them.
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
    /// A caller-supplied GBNF output grammar; exclusive with offered tools
    /// and JSON output.
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub reasoning: ReasoningIntent,
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
            grammar: None,
            reasoning: ReasoningIntent::ModelDefault,
        }
    }
}

fn validate_variant(variant: &TemplateVariant) -> Result<(), String> {
    if variant.name.is_empty() || variant.source.is_empty() || variant.provenance.is_empty() {
        return Err("template variant requires a name, source, and provenance".into());
    }
    Ok(())
}

fn apply_tool_selection_instruction(
    request: &mut NativeRequest,
    choice: &ToolChoice,
) -> Result<(), String> {
    let instruction = match choice {
        ToolChoice::Named(name) => {
            format!("Call the supplied tool '{name}' to answer this request.")
        }
        ToolChoice::Required | ToolChoice::Allowed { required: true, .. }
            if request.parallel_tool_calls =>
        {
            "Call at least one of the supplied tools to answer this request.".into()
        }
        ToolChoice::Required | ToolChoice::Allowed { required: true, .. } => {
            "Call one of the supplied tools to answer this request.".into()
        }
        ToolChoice::Auto
        | ToolChoice::None
        | ToolChoice::Allowed {
            required: false, ..
        } => return Ok(()),
    };
    if let Some(system) = request
        .messages
        .first_mut()
        .and_then(serde_json::Value::as_object_mut)
        .filter(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("system"))
    {
        match system.get_mut("content") {
            Some(serde_json::Value::String(content)) if !content.is_empty() => {
                content.push_str("\n\n");
                content.push_str(&instruction);
            }
            Some(serde_json::Value::Null) | None => {
                system.insert("content".into(), serde_json::Value::String(instruction));
            }
            Some(serde_json::Value::Array(parts)) => parts.push(serde_json::json!({
                "type": "text",
                "text": instruction,
            })),
            Some(_) => return Err("system message content must be text, parts, or null".into()),
        }
    } else {
        request.messages.insert(
            0,
            serde_json::json!({"role":"system", "content":instruction}),
        );
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
            profiles: BoundedOnceCache::new(16),
            grammars: CompileCache::new(16, 16 * 1024 * 1024),
        })
    }
    /// The compiled output constraint for a GBNF source, cached per model:
    /// a conversation's turns repeat their template grammar.
    pub fn compile_grammar(&self, gbnf: &str) -> Result<Arc<Compiled>, GrammarError> {
        self.grammars.compile(gbnf)
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
    ) -> Result<(PreparedRequest, ResolvedReasoning), ChatError> {
        self.prepare_native(request, selection, None)
            .map_err(|error| match error {
                Prepare::Invalid(message) => ChatError::InvalidRequest(message),
                Prepare::Reasoning(error) => ChatError::Reasoning(error),
            })
    }

    fn prepare_native(
        &self,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
        preparsed: Option<(&str, &Template)>,
    ) -> Result<(PreparedRequest, ResolvedReasoning), Prepare> {
        let choice = &request.tool_choice;
        let intent = &request.reasoning;
        if *intent != ReasoningIntent::ModelDefault
            && super::reasoning::CONTROLS
                .iter()
                .any(|key| request.template_arguments.contains_key(*key))
        {
            return Err(Prepare::Invalid(
                "reasoning selection conflicts with raw template reasoning controls".into(),
            ));
        }
        if request.grammar.is_some()
            && (request.json_schema.is_some()
                || (!request.tools.is_empty() && *choice != ToolChoice::None))
        {
            return Err(Prepare::Invalid(
                "a custom grammar cannot be combined with enabled tools or JSON output".into(),
            ));
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
            ToolChoice::Allowed { names: allowed, .. } => {
                if allowed.is_empty() || allowed.iter().any(|name| !names.contains(name)) {
                    return Err("allowed tools must name available functions".into());
                }
                request.tools.retain(|tool| {
                    tool["function"]["name"]
                        .as_str()
                        .is_some_and(|name| allowed.iter().any(|allowed| allowed == name))
                });
            }
            ToolChoice::Required if request.tools.is_empty() => {
                return Err("required tool choice needs a tool".into())
            }
            _ => {}
        }
        apply_tool_selection_instruction(&mut request, choice)?;
        let required = matches!(
            choice,
            ToolChoice::Required
                | ToolChoice::Named(_)
                | ToolChoice::Allowed { required: true, .. }
        );
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
        let owned_template;
        let template = match preparsed {
            Some((name, template))
                if selection.source_override.is_none() && variant.name == name =>
            {
                template
            }
            _ => {
                owned_template = Template::new(&variant.source, &self.special_tokens)
                    .map_err(|e| e.to_string())?;
                &owned_template
            }
        };
        // The default renders with no reasoning arguments, so it
        // needs no inspection; every other intent resolves against the profile.
        let reasoning = if *intent == ReasoningIntent::ModelDefault {
            ResolvedReasoning {
                effort: None,
                controls: Default::default(),
            }
        } else {
            let arguments_key = serde_json::to_vec(&canonical_json(&serde_json::Value::Object(
                request.template_arguments.clone(),
            )))
            .map_err(|error| error.to_string())?;
            let cell = self
                .profiles
                .cell((template.identity().to_owned(), arguments_key));
            let profile = cell
                .get_or_init(|| {
                    super::reasoning::inspect_reasoning(
                        template,
                        &variant.source,
                        &request.template_arguments,
                    )
                    .map(Arc::new)
                })
                .clone()?;
            profile.resolve(intent).map_err(Prepare::Reasoning)?
        };
        for (key, value) in &reasoning.controls {
            if request
                .template_arguments
                .insert(key.clone(), value.clone())
                .is_some()
            {
                return Err(Prepare::Invalid(format!(
                    "chat_template_kwargs conflicts with resolved reasoning control: {key}"
                )));
            }
        }
        let prepared = template.prepare(&request).map_err(|e| e.to_string())?;
        if (required || request.json_schema.is_some()) && prepared.description().grammar.is_empty()
        {
            return Err("selected template did not produce required output constraints".into());
        }
        Ok((prepared, reasoning))
    }
}

/// Template preparation failure before it is classified for callers.
enum Prepare {
    Invalid(String),
    Reasoning(super::reasoning::ReasoningError),
}
impl From<String> for Prepare {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}
impl From<&str> for Prepare {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}

/// Version of the template fingerprint derivation. Changes whenever the
/// digest's inputs change.
pub const TEMPLATE_FINGERPRINT_VERSION: &str = "v4-template-fingerprint-1";

/// Header-only chat facts of a bundle, established through the same
/// preparation path requests use. Nothing is tokenized or generated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateInspection {
    /// A required-tool request prepares with tool-call constraints.
    pub tools: bool,
    /// A JSON-schema request prepares with output constraints.
    pub structured_output: bool,
    /// Reasoning controls of the default variant without template arguments.
    pub reasoning: super::reasoning::ReasoningProfile,
    /// SHA-256 over the fingerprint version, the default variant's name, each
    /// variant's name and native template identity (source, special tokens
    /// and native build) in name order, and the reasoning profile's
    /// fingerprint. Equal fingerprints render and parse every request alike.
    pub fingerprint: String,
}

const MAX_CACHED_INSPECTIONS: usize = 64;
static INSPECTIONS: OnceLock<BoundedOnceCache<Vec<u8>, Result<TemplateInspection, String>>> =
    OnceLock::new();

impl TemplateBundle {
    /// Clear the process-wide template inspection cache (for cold/held-out benchmarking).
    #[doc(hidden)]
    pub fn clear_inspection_cache() {
        if let Some(cache) = INSPECTIONS.get() {
            cache.clear();
        }
    }

    /// Reuse exact header inspection for bundles with identical literal
    /// variants and special tokens. Preparation still uses this bundle's own
    /// sources; the cache contains only the immutable inspection result.
    /// Each bundle has its own cell: the map is locked only to find it, so
    /// distinct bundles are inspected concurrently and equal ones once.
    pub fn inspect_cached(&self) -> Result<TemplateInspection, String> {
        let sources = self
            .variants
            .values()
            .map(|variant| (&variant.name, &variant.source))
            .collect::<Vec<_>>();
        let key = serde_json::to_vec(&(&sources, &self.default, &self.special_tokens))
            .map_err(|error| error.to_string())?;
        let inspection = INSPECTIONS
            .get_or_init(|| BoundedOnceCache::new(MAX_CACHED_INSPECTIONS))
            .cell(key);
        inspection.get_or_init(|| self.inspect()).clone()
    }

    /// The default variant's source text.
    pub fn default_source(&self) -> &str {
        &self.variants[&self.default].source
    }

    #[cfg(test)]
    pub(crate) fn special_token(&self, name: &str) -> Option<&str> {
        self.special_tokens.get(name).map(String::as_str)
    }

    /// The native template's capability flags for the default variant.
    pub fn capabilities(&self) -> Result<BTreeMap<String, bool>, String> {
        Template::new(self.default_source(), &self.special_tokens)
            .and_then(|template| template.capabilities())
            .map_err(|e| e.to_string())
    }

    pub fn inspect(&self) -> Result<TemplateInspection, String> {
        self.inspect_inner(true)
    }

    #[doc(hidden)]
    pub fn inspect_unpruned(&self) -> Result<TemplateInspection, String> {
        self.inspect_inner(false)
    }

    fn inspect_inner(&self, pruned: bool) -> Result<TemplateInspection, String> {
        let default = self
            .variants
            .get(&self.default)
            .ok_or("template bundle lost its default variant")?;
        let default_template =
            Template::new(&default.source, &self.special_tokens).map_err(|e| e.to_string())?;
        let question = serde_json::json!({"role":"user","content":"What is the weather in Paris?"});
        let mut tool_request = ChatRequest::new(vec![question.clone()], 946684800);
        tool_request.tools = vec![serde_json::json!({"type":"function","function":{
            "name":"weather","description":"Get the current weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
        }})];
        tool_request.tool_choice = ToolChoice::Required;
        let mut schema_request = ChatRequest::new(vec![question], 946684800);
        schema_request.json_schema = Some(serde_json::json!({
            "type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]
        }));
        let (reasoning, identities, tools, structured_output) =
            if pruned && self.variants.len() == 1 {
                let preparsed = Some((default.name.as_str(), &default_template));
                let (reasoning, (tools, structured_output)) = seismic_threads::pool::join(
                    || {
                        super::reasoning::inspect_reasoning(
                            &default_template,
                            &default.source,
                            &Default::default(),
                        )
                    },
                    || {
                        seismic_threads::pool::join(
                            || {
                                self.prepare_native(
                                    &tool_request,
                                    &TemplateSelection::default(),
                                    preparsed,
                                )
                                .is_ok()
                            },
                            || {
                                self.prepare_native(
                                    &schema_request,
                                    &TemplateSelection::default(),
                                    preparsed,
                                )
                                .is_ok()
                            },
                        )
                    },
                );
                let reasoning = reasoning?;
                if reasoning.default_effort.is_none() {
                    return Err(
                        "the chat template's default reasoning is not exactly one of its efforts"
                            .into(),
                    );
                }
                let identities = vec![serde_json::json!({
                    "name": default.name,
                    "identity": default_template.identity(),
                })];
                (reasoning, identities, tools, structured_output)
            } else {
                let reasoning = if pruned {
                    super::reasoning::inspect_reasoning(
                        &default_template,
                        &default.source,
                        &Default::default(),
                    )?
                } else {
                    super::reasoning::inspect_reasoning_unpruned(
                        &default_template,
                        &Default::default(),
                    )?
                };
                if reasoning.default_effort.is_none() {
                    return Err(
                        "the chat template's default reasoning is not exactly one of its efforts"
                            .into(),
                    );
                }
                let mut tool_use_template: Option<Template> = None;
                let identities = self
                    .variants
                    .values()
                    .map(|variant| {
                        if pruned && variant.name == self.default {
                            Ok(serde_json::json!({
                                "name": variant.name,
                                "identity": default_template.identity(),
                            }))
                        } else {
                            let template = Template::new(&variant.source, &self.special_tokens)
                                .map_err(|e| e.to_string())?;
                            let identity = serde_json::json!({
                                "name": variant.name,
                                "identity": template.identity(),
                            });
                            if pruned && variant.name == "tool_use" {
                                tool_use_template = Some(template);
                            }
                            Ok(identity)
                        }
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                // A rejected probe is the capability's absence: the engine rejects
                // every request of that kind for this bundle.
                let accepted = |request: &ChatRequest| {
                    let preparsed = if !pruned {
                        None
                    } else if !request.tools.is_empty() {
                        tool_use_template
                            .as_ref()
                            .map(|template| ("tool_use", template))
                            .or(Some((default.name.as_str(), &default_template)))
                    } else {
                        Some((default.name.as_str(), &default_template))
                    };
                    self.prepare_native(request, &TemplateSelection::default(), preparsed)
                        .is_ok()
                };
                (
                    reasoning,
                    identities,
                    accepted(&tool_request),
                    accepted(&schema_request),
                )
            };
        let digest = sha2::Sha256::digest(
            serde_json::to_vec(&serde_json::json!({
                "version": TEMPLATE_FINGERPRINT_VERSION,
                "default": self.default,
                "templates": identities,
                "reasoning": reasoning.fingerprint(),
            }))
            .map_err(|e| e.to_string())?,
        );
        Ok(TemplateInspection {
            tools,
            structured_output,
            reasoning,
            fingerprint: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn native(messages: Vec<serde_json::Value>, parallel: bool) -> NativeRequest {
        let mut request = NativeRequest::new(messages, 0);
        request.parallel_tool_calls = parallel;
        request
    }

    #[test]
    fn tool_selection_instructions_cover_named_and_required_modes() {
        let original = vec![json!({"role":"user", "content":"weather"})];
        let mut named = native(original.clone(), true);
        apply_tool_selection_instruction(&mut named, &ToolChoice::Named("lookup".into())).unwrap();
        assert_eq!(
            named.messages[0]["content"],
            "Call the supplied tool 'lookup' to answer this request."
        );
        assert_eq!(original, vec![json!({"role":"user", "content":"weather"})]);

        let mut parallel = native(
            vec![json!({"role":"system", "content":"Be concise."})],
            true,
        );
        apply_tool_selection_instruction(&mut parallel, &ToolChoice::Required).unwrap();
        assert_eq!(
            parallel.messages[0]["content"],
            "Be concise.\n\nCall at least one of the supplied tools to answer this request."
        );

        let mut single = native(vec![json!({"role":"user", "content":"weather"})], false);
        apply_tool_selection_instruction(&mut single, &ToolChoice::Required).unwrap();
        assert_eq!(
            single.messages[0]["content"],
            "Call one of the supplied tools to answer this request."
        );
    }
}
