//! No-fail recovery of recomputable JSON: a value of a cache that decodes partly keeps what
//! decodes.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde_json::Value;

/// Decodes independently meaningful map entries and discards only entries that do not decode.
pub fn recover_map<T: DeserializeOwned>(
    value: Option<Value>,
    maximum_entries: usize,
) -> BTreeMap<String, T> {
    let Some(Value::Object(entries)) = value else {
        return BTreeMap::new();
    };
    entries
        .into_iter()
        .take(maximum_entries)
        .filter_map(|(key, value)| serde_json::from_value(value).ok().map(|value| (key, value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_map_siblings_survive() {
        let recovered = recover_map::<u64>(
            Some(serde_json::json!({
                "one": 1,
                "broken": "no",
                "two": 2
            })),
            3,
        );
        assert_eq!(
            recovered,
            BTreeMap::from([("one".to_owned(), 1), ("two".to_owned(), 2)])
        );
        assert!(recover_map::<u64>(Some(serde_json::json!([1, 2])), 3).is_empty());
    }
}
