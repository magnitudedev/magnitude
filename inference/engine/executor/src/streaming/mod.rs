//! Expert streaming: a mixture-of-experts load whose routed weights do not
//! all fit, served from banks of `S < E` slots per layer with the rest
//! fetched from the model file when routed (specs/26-10-11/expert-streaming.md).

pub(crate) mod bank;
pub(crate) mod layout;
pub(crate) mod runtime;
pub(crate) mod source;
pub(crate) mod words;

pub(crate) mod shared_bank;

pub(crate) mod arena;
pub(crate) mod shared_directory;
