use crate::ImageInput;

use super::{NonEmptyText, NonEmptyVec, ToolExchange};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceContext {
    system: Option<NonEmptyText>,
    entries: NonEmptyVec<ContextEntry>,
}

impl InferenceContext {
    #[must_use]
    pub fn new(system: Option<NonEmptyText>, entries: NonEmptyVec<ContextEntry>) -> Self {
        Self { system, entries }
    }

    #[must_use]
    pub fn system(&self) -> Option<&NonEmptyText> {
        self.system.as_ref()
    }

    #[must_use]
    pub fn entries(&self) -> &[ContextEntry] {
        self.entries.as_slice()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextEntry {
    User { entry: UserEntry },
    Assistant { entry: AssistantEntry },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UserEntry {
    content: Vec<UserContent>,
}

impl UserEntry {
    #[must_use]
    pub fn new(content: Vec<UserContent>) -> Self {
        Self { content }
    }

    #[must_use]
    pub fn content(&self) -> &[UserContent] {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    Text { text: NonEmptyText },
    Image { image: ImageInput },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssistantEntry {
    reasoning: Option<NonEmptyText>,
    text: Option<NonEmptyText>,
    tool_calls: Vec<ToolExchange>,
}

impl AssistantEntry {
    #[must_use]
    pub fn new(
        reasoning: Option<NonEmptyText>,
        text: Option<NonEmptyText>,
        tool_calls: Vec<ToolExchange>,
    ) -> Self {
        Self {
            reasoning,
            text,
            tool_calls,
        }
    }

    #[must_use]
    pub fn reasoning(&self) -> Option<&NonEmptyText> {
        self.reasoning.as_ref()
    }

    #[must_use]
    pub fn text(&self) -> Option<&NonEmptyText> {
        self.text.as_ref()
    }

    #[must_use]
    pub fn tool_calls(&self) -> &[ToolExchange] {
        &self.tool_calls
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn text(value: &str) -> NonEmptyText {
        NonEmptyText::try_new(value, "test text").expect("nonempty fixture")
    }

    #[test]
    fn accepts_empty_assistant_entry() {
        let entry = AssistantEntry::new(None, None, Vec::new());
        assert!(entry.reasoning().is_none());
        assert!(entry.text().is_none());
        assert!(entry.tool_calls().is_empty());
    }

    #[test]
    fn accepts_empty_user_entry() {
        let entry = UserEntry::new(Vec::new());
        assert!(entry.content().is_empty());
    }

    #[test]
    fn rejects_empty_text_inside_present_fields() {
        for value in [
            json!({ "reasoning": "", "text": null, "tool_calls": [] }),
            json!({ "reasoning": null, "text": "", "tool_calls": [] }),
        ] {
            serde_json::from_value::<AssistantEntry>(value)
                .expect_err("present canonical text must be nonempty");
        }
    }

    #[test]
    fn rejects_empty_context_entries_during_deserialization() {
        let error = serde_json::from_value::<InferenceContext>(json!({
            "system": "system",
            "entries": []
        }))
        .expect_err("empty context must be rejected");
        assert!(error.to_string().contains("collection must not be empty"));
    }

    #[test]
    fn accepts_empty_assistant_entry_during_deserialization() {
        let entry = serde_json::from_value::<AssistantEntry>(json!({
            "reasoning": null,
            "text": null,
            "tool_calls": []
        }))
        .expect("empty assistant entry is a meaningful conversation container");
        assert!(entry.tool_calls().is_empty());
    }

    #[test]
    fn system_is_structurally_separate_from_entries() {
        let entries = NonEmptyVec::try_new(
            vec![ContextEntry::User {
                entry: UserEntry::new(vec![UserContent::Text {
                    text: text("hello"),
                }]),
            }],
            "context entries",
        )
        .expect("nonempty fixture");
        let context = InferenceContext::new(Some(text("system")), entries);

        assert_eq!(context.system().map(NonEmptyText::as_str), Some("system"));
        assert!(matches!(context.entries(), [ContextEntry::User { .. }]));
    }
}
