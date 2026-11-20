//! Generated values checked against the schemas they were generated under.
//! Output is published as generated either way; a violation is reported, and
//! under a grammar that enforces its schema exactly it is a grammar defect.
use super::{
    request::{ChatInput, OutputFormat},
    schema::JsonSchema,
    PreparedChat, ToolChoice,
};
use magnitude_templates::{Relaxation, RelaxationSubject};
use serde_json::Value;
use std::collections::BTreeMap;

/// How far the grammar a value was generated under enforces its schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Enforcement {
    Exact,
    /// Every keyword except these.
    Loosened(Vec<Relaxation>),
    /// Generation was not constrained.
    Unconstrained,
}

/// How a generated value relates to its schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Conformance {
    Conforms,
    /// The value fails its schema where the grammar does not enforce it.
    Unenforced {
        violations: Vec<String>,
        enforcement: Enforcement,
    },
    /// The grammar enforces the schema exactly, yet admitted this value.
    GrammarDefect {
        violations: Vec<String>,
    },
}

struct Checked {
    schema: JsonSchema,
    enforcement: Enforcement,
}

impl Checked {
    fn check(&self, text: &str) -> Conformance {
        let violations = match serde_json::from_str::<Value>(text) {
            Ok(value) => self.schema.violations(&value),
            Err(error) => vec![format!("not JSON: {error}")],
        };
        if violations.is_empty() {
            Conformance::Conforms
        } else if self.enforcement == Enforcement::Exact {
            Conformance::GrammarDefect { violations }
        } else {
            Conformance::Unenforced {
                violations,
                enforcement: self.enforcement.clone(),
            }
        }
    }
}

/// The schemas of one prepared request's tools and JSON output.
pub struct OutputSchemas {
    tools: BTreeMap<String, Checked>,
    output: Option<Checked>,
    constrained: bool,
}

impl OutputSchemas {
    pub fn new(input: &ChatInput, prepared: &PreparedChat) -> Self {
        let constrained = prepared.constraint().is_some();
        let enforcement = |subject: &RelaxationSubject| {
            if !constrained {
                return Enforcement::Unconstrained;
            }
            let relaxations = prepared
                .native()
                .description()
                .relaxations
                .iter()
                .filter(|relaxation| relaxation.subject == *subject)
                .cloned()
                .collect::<Vec<_>>();
            if relaxations.is_empty() {
                Enforcement::Exact
            } else {
                Enforcement::Loosened(relaxations)
            }
        };
        let offered = |name: &str| match input.tools.choice() {
            ToolChoice::None => false,
            ToolChoice::Named(selected) => name == selected,
            ToolChoice::Allowed { names, .. } => names.iter().any(|allowed| allowed == name),
            ToolChoice::Auto | ToolChoice::Required => true,
        };
        let tools = input
            .tools
            .definitions()
            .iter()
            .filter(|tool| offered(&tool.name))
            .map(|tool| {
                let checked = Checked {
                    schema: tool.parameters.clone(),
                    enforcement: enforcement(&RelaxationSubject::Tool {
                        name: tool.name.clone(),
                    }),
                };
                (tool.name.clone(), checked)
            })
            .collect();
        let output = match &input.output {
            OutputFormat::JsonSchema { schema, .. } => Some(Checked {
                schema: schema.clone(),
                enforcement: enforcement(&RelaxationSubject::Output),
            }),
            OutputFormat::Text | OutputFormat::JsonObject | OutputFormat::Grammar(_) => None,
        };
        Self {
            tools,
            output,
            constrained,
        }
    }

    /// A completed tool call's arguments text against its tool's schema.
    pub fn tool_call(&self, name: &str, arguments: &str) -> Conformance {
        match self.tools.get(name) {
            Some(tool) => tool.check(arguments),
            None => {
                let violations = vec![format!("no offered tool is named {name}")];
                if self.constrained {
                    Conformance::GrammarDefect { violations }
                } else {
                    Conformance::Unenforced {
                        violations,
                        enforcement: Enforcement::Unconstrained,
                    }
                }
            }
        }
    }

    /// Whether the output is constrained by a JSON schema.
    pub fn constrains_output(&self) -> bool {
        self.output.is_some()
    }

    /// Completed output text against the output schema; `None` when the
    /// output has no JSON schema.
    pub fn output(&self, text: &str) -> Option<Conformance> {
        self.output.as_ref().map(|output| output.check(text.trim()))
    }
}
