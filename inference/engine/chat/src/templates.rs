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

struct VariantEntry {
    variant: TemplateVariant,
    compiled: OnceLock<Result<Template, String>>,
}

impl VariantEntry {
    fn new(variant: TemplateVariant) -> Self {
        Self {
            variant,
            compiled: OnceLock::new(),
        }
    }

    fn template(&self, special_tokens: &SpecialTokens) -> Result<&Template, String> {
        self.compiled
            .get_or_init(|| {
                Template::new(&self.variant.source, special_tokens).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

enum SelectedVariant<'a, 'b> {
    Declared(&'a VariantEntry),
    Override(&'b TemplateVariant),
}

enum SelectedTemplate<'a> {
    Cached(&'a Template),
    Owned(Template),
}

impl SelectedTemplate<'_> {
    fn as_template(&self) -> &Template {
        match self {
            Self::Cached(template) => template,
            Self::Owned(template) => template,
        }
    }
}

/// Validated artifact template metadata. Selection never invents a default source.
pub struct TemplateBundle {
    variants: BTreeMap<String, VariantEntry>,
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
            Some(serde_json::Value::String(content)) => {
                if !content.is_empty() {
                    content.push_str("\n\n");
                }
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
            if declared
                .insert(variant.name.clone(), VariantEntry::new(variant))
                .is_some()
            {
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

    fn select_variant<'a, 'b>(
        &'a self,
        tools_offered: bool,
        selection: &TemplateSelection<'b>,
    ) -> Result<SelectedVariant<'a, 'b>, String> {
        if let Some(source) = selection.source_override {
            if selection.variant.is_some() {
                return Err("configure a source override or a variant, not both".into());
            }
            validate_variant(source)?;
            return Ok(SelectedVariant::Override(source));
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
            .map(SelectedVariant::Declared)
            .ok_or_else(|| format!("unknown template variant: {selected}"))
    }

    pub fn select<'a>(
        &'a self,
        tools_offered: bool,
        selection: &TemplateSelection<'a>,
    ) -> Result<&'a TemplateVariant, String> {
        match self.select_variant(tools_offered, selection)? {
            SelectedVariant::Declared(entry) => Ok(&entry.variant),
            SelectedVariant::Override(source) => Ok(source),
        }
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
        self.prepare_native(request, selection)
            .map_err(Prepare::into_chat_error)
    }

    /// Render only the prompt text for `request`, reusing the bundle's cached
    /// native template handle without allocating a native request handle or
    /// materializing grammar/parser descriptions.
    pub fn render_prompt(
        &self,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<String, ChatError> {
        let (native, selected, _, _) = self
            .normalize_request(request, selection)
            .map_err(Prepare::into_chat_error)?;
        let probe = selected
            .as_template()
            .prepare_probe(&native)
            .map_err(|e| ChatError::InvalidRequest(e.to_string()))?;
        Ok(probe.prompt)
    }

    fn normalize_request<'a>(
        &'a self,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<(NativeRequest, SelectedTemplate<'a>, ResolvedReasoning, bool), Prepare> {
        let choice = &request.tool_choice;
        let intent = &request.reasoning;
        let has_raw_reasoning = super::reasoning::CONTROLS
            .iter()
            .any(|key| request.template_arguments.contains_key(*key));
        if *intent != ReasoningIntent::ModelDefault && has_raw_reasoning {
            return Err(Prepare::Invalid(
                "reasoning selection conflicts with raw template reasoning controls".into(),
            ));
        }
        let has_raw_grammar = request.grammar.is_some();
        if has_raw_grammar
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
            pure_content: has_raw_grammar,
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
        let (source, selected) = match self.select_variant(!request.tools.is_empty(), selection)? {
            SelectedVariant::Declared(entry) => (
                entry.variant.source.as_str(),
                SelectedTemplate::Cached(entry.template(&self.special_tokens)?),
            ),
            SelectedVariant::Override(source) => (
                source.source.as_str(),
                SelectedTemplate::Owned(
                    Template::new(&source.source, &self.special_tokens)
                        .map_err(|e| e.to_string())?,
                ),
            ),
        };
        // The default renders with no reasoning arguments, so it needs no
        // inspection — except for a caller-supplied raw GBNF grammar, which
        // constrains token 0 directly and disables thinking whenever the
        // template exposes a disabled mode.
        let inspect_profile = || -> Result<Arc<super::reasoning::ReasoningProfile>, Prepare> {
            let arguments_key = serde_json::to_vec(&canonical_json(&serde_json::Value::Object(
                request.template_arguments.clone(),
            )))
            .map_err(|error| error.to_string())?;
            let cell = self
                .profiles
                .cell((selected.as_template().identity().to_owned(), arguments_key));
            cell.get_or_init(|| {
                super::reasoning::inspect_reasoning(
                    selected.as_template(),
                    source,
                    &request.template_arguments,
                )
                .map(Arc::new)
            })
            .clone()
            .map_err(Prepare::Invalid)
        };
        let reasoning = if *intent == ReasoningIntent::ModelDefault {
            if has_raw_grammar && !has_raw_reasoning {
                match inspect_profile()
                    .ok()
                    .and_then(|profile| profile.resolve(&ReasoningIntent::Disabled).ok())
                {
                    Some(disabled) => disabled,
                    None => ResolvedReasoning {
                        effort: None,
                        controls: Default::default(),
                    },
                }
            } else {
                ResolvedReasoning {
                    effort: None,
                    controls: Default::default(),
                }
            }
        } else {
            inspect_profile()?
                .resolve(intent)
                .map_err(Prepare::Reasoning)?
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
        Ok((request, selected, reasoning, required))
    }

    fn prepare_native(
        &self,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<(PreparedRequest, ResolvedReasoning), Prepare> {
        let (request, selected, reasoning, required) = self.normalize_request(request, selection)?;
        let prepared = selected
            .as_template()
            .prepare(&request)
            .map_err(|e| e.to_string())?;
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
impl Prepare {
    fn into_chat_error(self) -> ChatError {
        match self {
            Self::Invalid(message) => ChatError::InvalidRequest(message),
            Self::Reasoning(error) => ChatError::Reasoning(error),
        }
    }
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
            .map(|entry| (&entry.variant.name, &entry.variant.source))
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
        &self.variants[&self.default].variant.source
    }

    fn default_entry(&self) -> Result<&VariantEntry, String> {
        self.variants
            .get(&self.default)
            .ok_or_else(|| "template bundle lost its default variant".into())
    }

    #[cfg(test)]
    pub(crate) fn special_token(&self, name: &str) -> Option<&str> {
        self.special_tokens.get(name).map(String::as_str)
    }

    /// The native template's capability flags for the default variant.
    pub fn capabilities(&self) -> Result<BTreeMap<String, bool>, String> {
        self.default_entry()?
            .template(&self.special_tokens)?
            .capabilities()
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
        let default = self.default_entry()?;
        let default_template = default.template(&self.special_tokens)?;
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
                let (reasoning, (tools, structured_output)) = seismic_threads::pool::join(
                    || {
                        super::reasoning::inspect_reasoning(
                            default_template,
                            &default.variant.source,
                            &Default::default(),
                        )
                    },
                    || {
                        seismic_threads::pool::join(
                            || {
                                self.prepare_native(&tool_request, &TemplateSelection::default())
                                    .is_ok()
                            },
                            || {
                                self.prepare_native(&schema_request, &TemplateSelection::default())
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
                    "name": default.variant.name,
                    "identity": default_template.identity(),
                })];
                (reasoning, identities, tools, structured_output)
            } else {
                let reasoning = if pruned {
                    super::reasoning::inspect_reasoning(
                        default_template,
                        &default.variant.source,
                        &Default::default(),
                    )?
                } else {
                    super::reasoning::inspect_reasoning_unpruned(
                        default_template,
                        &Default::default(),
                    )?
                };
                if reasoning.default_effort.is_none() {
                    return Err(
                        "the chat template's default reasoning is not exactly one of its efforts"
                            .into(),
                    );
                }
                let identities = self
                    .variants
                    .values()
                    .map(|entry| {
                        let template = entry.template(&self.special_tokens)?;
                        Ok(serde_json::json!({
                            "name": entry.variant.name,
                            "identity": template.identity(),
                        }))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                // A rejected probe is the capability's absence: the engine rejects
                // every request of that kind for this bundle.
                let accepted = |request: &ChatRequest| {
                    self.prepare_native(request, &TemplateSelection::default())
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

    fn chatml_bundle() -> TemplateBundle {
        TemplateBundle::new(
            vec![TemplateVariant {
                name: "default".into(),
                source: "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}".into(),
                provenance: "test".into(),
            }],
            "default".into(),
            Default::default(),
        )
        .unwrap()
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

        let mut empty_system = native(
            vec![
                json!({"role":"system", "content":""}),
                json!({"role":"user", "content":"weather"}),
            ],
            false,
        );
        apply_tool_selection_instruction(&mut empty_system, &ToolChoice::Named("lookup".into()))
            .unwrap();
        assert_eq!(
            empty_system.messages[0]["content"],
            "Call the supplied tool 'lookup' to answer this request."
        );
    }

    #[test]
    fn variant_templates_compile_lazily_and_reuse_cached_handle() {
        let bundle = chatml_bundle();
        let entry = bundle.default_entry().unwrap();
        assert!(entry.compiled.get().is_none());

        let request = ChatRequest::new(vec![json!({"role":"user", "content":"hello"})], 0);
        let (prepared, _) = bundle
            .prepare(&request, &TemplateSelection::default())
            .unwrap();
        let first_ptr: *const Template = entry.compiled.get().unwrap().as_ref().unwrap();

        let prompt = bundle
            .render_prompt(&request, &TemplateSelection::default())
            .unwrap();
        assert_eq!(prompt, prepared.description().prompt);

        let _ = bundle.capabilities().unwrap();
        let second_ptr: *const Template = entry.compiled.get().unwrap().as_ref().unwrap();
        assert_eq!(first_ptr, second_ptr);
    }

    #[test]
    fn render_prompt_matches_prepare_with_override() {
        let bundle = chatml_bundle();
        let override_variant = TemplateVariant {
            name: "override".into(),
            source: "OVERRIDE:{{ messages[0].content }}".into(),
            provenance: "test".into(),
        };
        let selection = TemplateSelection {
            variant: None,
            source_override: Some(&override_variant),
        };
        let request = ChatRequest::new(vec![json!({"role":"user", "content":"ping"})], 0);
        let (prepared, _) = bundle.prepare(&request, &selection).unwrap();
        let prompt = bundle.render_prompt(&request, &selection).unwrap();
        assert_eq!(prompt, prepared.description().prompt);
    }

    #[test]
    fn concurrent_inspect_cached_on_shared_template_never_deadlocks() {
        use std::sync::Barrier;

        for iteration in 0..8 {
            let source = format!(
                "{{% for m in messages %}}<|im_start|>{{{{ m.role }}}}\n{{{{ m.content }}}}<|im_end|>\n{{% endfor %}}<!-- {iteration} -->"
            );
            let barrier = Arc::new(Barrier::new(6));
            let mut handles = Vec::new();
            for _ in 0..6 {
                let source = source.clone();
                let barrier = Arc::clone(&barrier);
                handles.push(
                    seismic_threads::spawn("test-inspect-cached", move || {
                        let bundle = TemplateBundle::new(
                            vec![TemplateVariant {
                                name: "default".into(),
                                source,
                                provenance: "test".into(),
                            }],
                            "default".into(),
                            SpecialTokens::default(),
                        )
                        .unwrap();
                        barrier.wait();
                        let (inspected, _) = seismic_threads::pool::join(
                            || bundle.inspect_cached(),
                            || 1usize,
                        );
                        inspected.expect("concurrent cached template inspection succeeded")
                    })
                    .expect("spawn test thread"),
                );
            }
            for handle in handles {
                let _ = handle.join().expect("inspection thread did not deadlock");
            }
        }
    }
}
