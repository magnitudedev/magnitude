//! What every model family does the same way when it interprets a header:
//! strict metadata reading, strict tensor binding, rotary tables in the
//! engine's pairing, and text-only input. A family crate keeps only what
//! varies between families: which keys and tensors mean what.

mod error;
#[cfg(feature = "catalog-headers")]
pub mod headers;
mod inputs;
mod metadata;
pub mod rotary;
mod tensors;

pub use error::HeaderError;
pub use inputs::TextInput;
pub use metadata::Metadata;
pub use tensors::Tensors;
