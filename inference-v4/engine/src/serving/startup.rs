//! Explicit composition root. Local artifact interpretation precedes execution
//! owner creation; this API neither downloads artifacts nor binds a listener.
use super::{Config, Server};
use crate::{
    generation::constraints::{CacheLimits, Vocabulary},
    inputs::ByteBpeTokenizer,
    models::qwen35::{loading::Model, service::QwenExecutor},
    service::{
        policy::Limits,
        runtime::{Runtime, Service},
    },
};
use seismic::{Device, PrecisionPolicy};
use std::{path::Path, rc::Rc, sync::Arc};

pub struct ExecutionLimits {
    pub storage_bytes: usize,
    pub control_capacity: usize,
    pub scheduler: Limits,
    pub grammar_cache: CacheLimits,
}

/// Settings and device construction are supplied by the host. They are created
/// on the worker, and must use measured backend-compatible hardware inputs.
/// HTTP vocabulary must equal the artifact projection; context may be smaller
/// than the artifact maximum. Failure returns before any listener is served.
pub fn qwen(
    path: impl AsRef<Path>,
    config: Config,
    limits: ExecutionLimits,
    execution: impl FnOnce() -> Result<(Device, PrecisionPolicy), String> + Send + 'static,
) -> Result<Server<QwenExecutor>, String> {
    if limits.storage_bytes == 0 || limits.control_capacity == 0 {
        return Err("startup requires positive storage and control budgets".into());
    }
    let path = path.as_ref();
    let model = Model::open(path)?;
    let description = model.description();
    let projection = usize::try_from(description.geometry.vocabulary)
        .map_err(|_| "vocabulary exceeds host domain")?;
    if config.vocabulary != projection
        || config.context_tokens == 0
        || config.context_tokens as u64 > description.geometry.context_limit
    {
        return Err("HTTP limits disagree with artifact geometry".into());
    }
    let tokenizer = Arc::new(ByteBpeTokenizer::new(model.tokenizer_config()?)?);
    let templates = model.templates()?;
    config.validate(&tokenizer, &templates)?;
    crate::service::policy::Scheduler::new(limits.scheduler.clone())?;
    let context = config.context_tokens;
    let worker_tokenizer = tokenizer.clone();
    let service = Service::spawn(
        move || {
            let vocabulary = Vocabulary::new(worker_tokenizer, projection, limits.grammar_cache)?;
            let (device, precision) = execution()?;
            let storage_bytes = u64::try_from(limits.storage_bytes)
                .map_err(|_| "storage budget exceeds the device accounting domain")?;
            device
                .set_memory_limit(Some(storage_bytes))
                .map_err(|e| e.to_string())?;
            let decoder = model
                .load(
                    Rc::new(device),
                    precision,
                    context,
                    limits.scheduler.max_requests,
                )
                .map_err(|error| error.to_string())?;
            Runtime::new(QwenExecutor::new(decoder)?, vocabulary, limits.scheduler)
        },
        limits.control_capacity,
    )?;
    Server::new(service, tokenizer, templates, config)
}
