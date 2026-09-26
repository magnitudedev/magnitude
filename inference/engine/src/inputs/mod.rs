//! Logical input boundaries and tokenization, independent of physical pages.
pub use magnitude_artifacts::media;
pub use magnitude_artifacts::{BoundaryRule, InputLayout, InputSpan, TokenId};
pub use magnitude_chat::artifacts;
pub use magnitude_chat::{BpeConfig, ByteBpeTokenizer, PieceKind, SpecialTokens, TokenDecoder};
