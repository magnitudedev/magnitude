//! Family-neutral model artifact and immutable host-payload ownership.
//!
//! This crate validates containers, exposes literal metadata/tensor inventory,
//! and owns bounded prepared media values before device execution. Model
//! families assign numerical meaning to those facts elsewhere.

mod error;
pub mod gguf;
mod identity;
mod layout;
pub mod media;
mod package;
mod payload;
mod preprocessing;
mod source;

pub use error::Error;
pub use identity::{ArtifactIdentity, PackageIdentity};
pub use layout::{BoundaryRule, InputLayout, InputSpan};
pub use package::{ComponentManifest, Package, PackageHeaders, PackageManifest};
pub use payload::{TemplatePayload, TemplateSource, TokenizerPayload};
pub use preprocessing::{ImageProcessor, ImageProcessorConfig, MAX_IMAGES_PER_REQUEST};
#[cfg(unix)]
pub use source::MappedWindow;
pub use source::{FileSource, SourceReader};
pub use token::TokenId;

mod token;
