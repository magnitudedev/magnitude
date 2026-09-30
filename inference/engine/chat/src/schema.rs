//! Admitted JSON Schemas: the one authority for whether a schema is valid and
//! whether a value satisfies it. Constraint grammars approximate these schemas
//! from inside; this decides.
use super::ChatError;
use serde_json::{Map, Value};
use std::sync::Arc;

/// A JSON Schema valid under its draft, with every reference resolvable within
/// the document. The draft is the one `$schema` names; a document naming none,
/// or a dialect this does not know, is read as 2020-12. Formats are
/// annotations, as JSON Schema 2019-09 and later define them.
#[derive(Clone)]
pub struct JsonSchema {
    source: Map<String, Value>,
    validator: Arc<jsonschema::Validator>,
}

impl JsonSchema {
    /// Admits a schema. The only failure is a schema that is not a valid JSON
    /// Schema or names a resource outside itself.
    pub fn new(source: Map<String, Value>) -> Result<Self, ChatError> {
        let mut document = source.clone();
        if let Some(Value::String(dialect)) = source.get("$schema") {
            if jsonschema::Draft::from_schema_uri(dialect) == jsonschema::Draft::Unknown {
                document.remove("$schema");
            }
        }
        let validator = jsonschema::options()
            .should_validate_formats(false)
            .build(&Value::Object(document))
            .map_err(|error| {
                ChatError::InvalidRequest(format!(
                    "invalid JSON schema at {}: {error}",
                    pointer(&error.schema_path().to_string())
                ))
            })?;
        Ok(Self {
            source,
            validator: Arc::new(validator),
        })
    }

    pub fn source(&self) -> &Map<String, Value> {
        &self.source
    }

    /// Every way `value` fails the schema, each as its location and reason;
    /// empty when it conforms.
    pub fn violations(&self, value: &Value) -> Vec<String> {
        self.validator
            .iter_errors(value)
            .map(|error| format!("{}: {error}", pointer(&error.instance_path().to_string())))
            .collect()
    }
}

fn pointer(location: &str) -> String {
    format!("#{location}")
}

impl PartialEq for JsonSchema {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl std::fmt::Debug for JsonSchema {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("JsonSchema")
            .field(&self.source)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn admit(schema: Value) -> Result<JsonSchema, ChatError> {
        let Value::Object(source) = schema else {
            panic!("schema fixtures are objects")
        };
        JsonSchema::new(source)
    }

    #[test]
    fn only_invalid_schemas_are_rejected() {
        for valid in [
            json!({}),
            json!({"type": "number", "exclusiveMinimum": 0}),
            json!({"$schema": "http://json-schema.org/draft-04/schema#", "type": "integer", "minimum": 0, "exclusiveMinimum": true}),
            json!({"not": {}, "if": true, "then": false, "uniqueItems": true, "x-vendor": [1]}),
            json!({"$defs": {"a": {"$anchor": "foo"}}, "properties": {"x": {"$ref": "#foo"}}}),
            json!({"format": "not-a-known-format"}),
            json!({"$schema": "http://json-schema.org/draft-06/schema#", "type": "string"}),
            json!({"$schema": "https://spec.openapis.org/oas/3.1/dialect/base", "type": "string"}),
            json!({"$schema": "urn:example:custom", "type": "string"}),
        ] {
            assert!(admit(valid.clone()).is_ok(), "{valid}");
        }
        for invalid in [
            json!({"type": "integr"}),
            json!({"minimum": "0"}),
            json!({"required": "a"}),
            json!({"$ref": "#/$defs/missing"}),
            json!({"$ref": "https://example.com/schema.json"}),
            json!({"$schema": "http://json-schema.org/draft-04/schema#", "exclusiveMinimum": 0}),
        ] {
            let error = admit(invalid.clone()).expect_err(&invalid.to_string());
            assert!(matches!(error, ChatError::InvalidRequest(_)), "{invalid}");
        }
    }

    #[test]
    fn violations_locate_each_failure() {
        let schema = admit(json!({
            "type": "object",
            "properties": {"timeout": {"type": "number", "exclusiveMinimum": 0}, "email": {"format": "email"}},
            "required": ["command"]
        }))
        .unwrap();
        assert!(schema
            .violations(&json!({"command": "ls", "email": "not an email"}))
            .is_empty());
        let violations = schema.violations(&json!({"timeout": 0}));
        assert_eq!(violations.len(), 2, "{violations:?}");
        assert!(
            violations.iter().any(|v| v.starts_with("#/timeout: ")),
            "{violations:?}"
        );
    }
}
