//! Family-neutral model artifact and immutable host-payload ownership.
//!
//! This crate validates containers, exposes literal metadata/tensor inventory,
//! and owns bounded prepared media values before device execution. Model
//! families assign numerical meaning to those facts elsewhere.

pub mod catalog_headers;
mod error;
pub mod gguf;
mod identity;
mod layout;
pub mod media;
mod package;
mod payload;
mod preprocessing;
mod source;
mod reader;
pub mod support;

pub use error::Error;
pub use identity::{ArtifactIdentity, PackageIdentity};
pub use layout::{BoundaryRule, InputLayout, InputSpan};
pub use package::{ComponentFile, ComponentManifest, Package, PackageHeaders, PackageManifest};
pub use payload::{TemplatePayload, TemplateSource, TokenizerPayload};
pub use preprocessing::{
    ImageProcessor, ImageProcessorConfig, ImageResize, Resampling, MAX_IMAGES_PER_REQUEST,
};
#[cfg(unix)]
pub use source::MappedWindow;
pub use source::{FileSource, SourceReader};
pub use support::{UnsupportedReason, WeightTypeCount};
pub use reader::{
    adopt_volume, cached_pages, engine_work, probe_volume, read_cores, ArtifactReader, Delivery, EngineWork, Flight, RangeReader, CachePolicy, Ended, Job, Panic,
    Outstanding, Priority, Range, Urgent, ALIGNMENT, VolumeChoice,
};
pub use token::TokenId;

mod token;
