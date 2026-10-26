//! The native spelling of a name in emitted macros: ASCII letters and digits
//! uppercased, every other character `_`.

pub fn native_macro(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}
