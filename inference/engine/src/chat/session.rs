//! One chat generation on the host: prepare the request against the model's
//! host artifacts, admit it to the numerical worker, parse accepted tokens
//! into the semantic output stream, and report exactly one terminal outcome.
//! Dropping the event receiver cancels the request.
use super::operations::{prepare, InputBound, PreparedInput};
use crate::error::RequestError;
use crate::host::HostArtifacts;
use crate::worker::{
    protocol::{RequestState, RetentionPolicy},
    EngineClient, RequestEvent, RequestOptions,
};
use magnitude_chat::{
    conformance::OutputSchemas,
    generation::{generation_options, ModelLimits},
    output::{
        Completion, GenerationTimings, OutputEvent, Progress, Termination, TimingSnapshot,
        TokenUsage,
    },
    request::PromptCache,
    ChatError, Event, FinishReason, GenerationRequest, SpecialTokens, TerminalCause,
    TokenChatStream,
};
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

/// Per-request host bounds chosen by the serving composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionLimits {
    /// Output batches buffered for one request on each side of the worker.
    pub output_capacity: usize,
    /// Bound on parsed output bytes per request.
    pub max_output_bytes: usize,
}

/// Why a generation did not complete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionError {
    /// The request could not be prepared for the model.
    Chat(ChatError),
    /// The engine refused or ended the request.
    Request(RequestError),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(error) => error.fmt(formatter),
            Self::Request(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SessionError {}

/// A generation's lifecycle. `Admitted` precedes all output; exactly one of
/// `Completed` and `Failed` is last.
#[derive(Debug)]
pub enum SessionEvent {
    Progress(Progress),
    Admitted {
        prompt_tokens: u64,
    },
    Output {
        event: OutputEvent,
        snapshot: Option<TimingSnapshot>,
    },
    Completed(Completion),
    Failed(SessionError),
}

/// How often scheduling progress is sampled before the first output token.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(200);

/// The consumer is gone; the request is abandoned (cancelled on drop).
struct Abandoned;

struct Events {
    sender: mpsc::Sender<SessionEvent>,
}

impl Events {
    async fn send(&self, event: SessionEvent) -> Result<(), Abandoned> {
        self.sender.send(event).await.map_err(|_| Abandoned)
    }
}

/// Run one generation, publishing its lifecycle to `events`.
pub async fn generate(
    host: Arc<HostArtifacts>,
    engine: EngineClient,
    request: GenerationRequest,
    limits: SessionLimits,
    events: mpsc::Sender<SessionEvent>,
) {
    let events = Events { sender: events };
    let _ = run(host, engine, request, limits, &events).await;
}

async fn run(
    host: Arc<HostArtifacts>,
    engine: EngineClient,
    request: GenerationRequest,
    limits: SessionLimits,
    events: &Events,
) -> Result<(), Abandoned> {
    let started = Instant::now();
    events.send(SessionEvent::Progress(Progress::Preparing)).await?;
    let prepared = {
        let host = host.clone();
        let input = request.input.clone();
        tokio::task::spawn_blocking(move || prepare(&host, &input, InputBound::Served))
            .await
            .map_err(|error| ChatError::Internal(format!("request preparation failed: {error}")))
            .and_then(|prepared| prepared)
    };
    let PreparedInput { chat, input } = match prepared {
        Ok(prepared) => prepared,
        Err(error) => return events.send(SessionEvent::Failed(SessionError::Chat(error))).await,
    };
    let ready = engine.ready_info();
    let model_limits = ModelLimits {
        context_tokens: host.definition().decoder.context_limit as usize,
        vocabulary: host.definition().decoder.vocabulary as usize,
        output_capacity: limits.output_capacity,
        forced_quantum: 0,
        method: ready.model.method.policy(),
    };
    let prompt_tokens = input.tokens().len();
    let prepared = generation_options(
        &chat,
        prompt_tokens,
        &request.controls,
        host.tokenizer(),
        &model_limits,
    )
    .and_then(|options| {
        let parser = TokenChatStream::new(
            &chat,
            host.tokenizer(),
            request.controls.stops.clone(),
            limits.max_output_bytes,
        )
        .map_err(ChatError::Internal)?;
        Ok((options, parser))
    });
    let (options, mut parser) = match prepared {
        Ok(prepared) => prepared,
        Err(error) => return events.send(SessionEvent::Failed(SessionError::Chat(error))).await,
    };
    let admission = engine
        .admit(RequestOptions {
            input,
            options,
            constraint: chat.input().constraint.clone(),
            retention: match request.controls.prompt_cache {
                PromptCache::Allowed => RetentionPolicy::Retain,
                PromptCache::Disabled => RetentionPolicy::Transient,
            },
            output_capacity: limits.output_capacity,
        })
        .await;
    let mut admitted = match admission {
        Ok(admitted) => admitted,
        Err(error) => {
            return events
                .send(SessionEvent::Failed(SessionError::Request(error)))
                .await
        }
    };
    let admitted_at = Instant::now();
    events
        .send(SessionEvent::Admitted {
            prompt_tokens: prompt_tokens as u64,
        })
        .await?;
    events.send(SessionEvent::Progress(Progress::Queued)).await?;

    let mut clock = Clock {
        started,
        admitted: admitted_at,
        first_output: None,
        prompt_tokens: prompt_tokens as u64,
        generated: 0,
        parser: Duration::ZERO,
    };
    let mut semantics = Semantics::new(OutputSchemas::new(&request.input, &chat));
    let mut prefill_reported = None;
    loop {
        let event = if clock.first_output.is_none() {
            let received = tokio::select! {
                received = tokio::time::timeout(PROGRESS_INTERVAL, admitted.receive()) => received,
                () = events.sender.closed() => return Err(Abandoned),
            };
            match received {
                Ok(event) => event,
                Err(_) => {
                    if let Some(progress) = admitted.status().await {
                        let completed = progress.resident_tokens.min(progress.prompt_tokens);
                        let running = matches!(
                            progress.state,
                            RequestState::Runnable | RequestState::AwaitingCompletion
                        );
                        if running && prefill_reported != Some(completed) {
                            prefill_reported = Some(completed);
                            events
                                .send(SessionEvent::Progress(Progress::Prefill {
                                    completed_tokens: completed as u64,
                                    total_tokens: progress.prompt_tokens as u64,
                                    cached_tokens: progress.cached_tokens as u64,
                                }))
                                .await?;
                        }
                    }
                    continue;
                }
            }
        } else {
            tokio::select! {
                event = admitted.receive() => event,
                () = events.sender.closed() => return Err(Abandoned),
            }
        };
        let Some(event) = event else {
            return events
                .send(SessionEvent::Failed(SessionError::Request(
                    RequestError::WorkerLost {
                        reason: "the request stream ended without a terminal outcome".into(),
                    },
                )))
                .await;
        };
        match event {
            RequestEvent::Output(tokens) => {
                let mut produced = Vec::new();
                if clock.first_output.is_none() {
                    clock.first_output = Some(Instant::now());
                    events.send(SessionEvent::Progress(Progress::Generating)).await?;
                    produced.push(OutputEvent::Started);
                }
                let mut stopped = false;
                for token in &tokens {
                    clock.generated += 1;
                    let parsing = Instant::now();
                    let parsed = parser.feed(token);
                    clock.parser += parsing.elapsed();
                    match parsed.and_then(|parsed| semantics.translate(parsed)) {
                        Ok(parsed) => produced.extend(parsed),
                        Err(error) => {
                            return fail_internal(events, error).await;
                        }
                    }
                    if parser.stopped() {
                        stopped = true;
                        break;
                    }
                }
                emit(events, produced, Some(clock.snapshot())).await?;
                if stopped {
                    // A caller or template stop string ended the output: drain
                    // the ordered stop for terminal usage.
                    let terminal = admitted.stop().await;
                    return finish(events, terminal, &mut parser, &mut semantics, &mut clock, &host)
                        .await;
                }
            }
            terminal => {
                return finish(
                    events,
                    Some(terminal),
                    &mut parser,
                    &mut semantics,
                    &mut clock,
                    &host,
                )
                .await;
            }
        }
    }
}

async fn fail_internal(events: &Events, reason: String) -> Result<(), Abandoned> {
    events
        .send(SessionEvent::Failed(SessionError::Request(
            RequestError::Internal { reason },
        )))
        .await
}

async fn emit(
    events: &Events,
    produced: Vec<OutputEvent>,
    snapshot: Option<TimingSnapshot>,
) -> Result<(), Abandoned> {
    let last = produced.len().saturating_sub(1);
    for (index, event) in produced.into_iter().enumerate() {
        events
            .send(SessionEvent::Output {
                event,
                // Cumulative timing rides on the last event of each batch.
                snapshot: (index == last).then_some(snapshot).flatten(),
            })
            .await?;
    }
    Ok(())
}

async fn finish(
    events: &Events,
    terminal: Option<RequestEvent>,
    parser: &mut TokenChatStream<'_>,
    semantics: &mut Semantics,
    clock: &mut Clock,
    host: &HostArtifacts,
) -> Result<(), Abandoned> {
    let Some(terminal) = terminal else {
        return events
            .send(SessionEvent::Failed(SessionError::Request(
                RequestError::WorkerLost {
                    reason: "the request stream ended without a terminal outcome".into(),
                },
            )))
            .await;
    };
    let (finish, usage, timings) = match terminal {
        RequestEvent::Completed {
            finish,
            usage,
            method: _,
            timings,
        } => (finish, usage, timings),
        RequestEvent::Failed(error) => {
            return events
                .send(SessionEvent::Failed(SessionError::Request(error)))
                .await
        }
        RequestEvent::Output(_) => {
            return fail_internal(events, "output followed the terminal outcome".into()).await
        }
    };
    let mut produced = Vec::new();
    if clock.first_output.is_none() {
        clock.first_output = Some(Instant::now());
        produced.push(OutputEvent::Started);
    }
    if !parser.stopped() {
        let parsing = Instant::now();
        let parsed = parser.finish(finish);
        clock.parser += parsing.elapsed();
        match parsed.and_then(|parsed| semantics.translate(parsed)) {
            Ok(parsed) => produced.extend(parsed),
            Err(error) => return fail_internal(events, error).await,
        }
    }
    emit(events, produced, None).await?;
    let termination = if semantics.tool_calls > 0 {
        Termination::ToolCalls
    } else if let Some(stop) = parser.matched_stop() {
        Termination::StopSequence(stop.to_owned())
    } else {
        match finish {
            FinishReason::Length | FinishReason::Context => Termination::OutputLimit,
            FinishReason::Stop | FinishReason::Cancelled | FinishReason::Failed => {
                Termination::Natural
            }
        }
    };
    let reasoning_output_tokens = if semantics.reasoning.is_empty() {
        0
    } else {
        match host
            .tokenizer()
            .encode(&semantics.reasoning, SpecialTokens::Recognize)
        {
            Ok(tokens) => tokens.len() as u64,
            Err(error) => return fail_internal(events, error).await,
        }
    };
    let first_output = clock.first_output.expect("first output was recorded");
    events
        .send(SessionEvent::Completed(Completion {
            usage: TokenUsage {
                input_tokens: usage.prompt_tokens as u64,
                cached_input_tokens: usage.cached_tokens as u64,
                output_tokens: usage.completion_tokens as u64,
                reasoning_output_tokens,
            },
            termination,
            timings: GenerationTimings {
                prompt_ms: timings.prompt_ns as f64 / 1e6,
                decode_ms: timings.predicted_ns as f64 / 1e6,
                time_to_first_token_ms: millis(first_output - clock.started),
                sampler_ms: 0.0,
                parser_ms: millis(clock.parser),
                draft_tokens: usage.draft_n as u64,
                accepted_draft_tokens: usage.draft_n_accepted as u64,
            },
        }))
        .await
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Host-observed request timing for progressive snapshots.
struct Clock {
    started: Instant,
    admitted: Instant,
    first_output: Option<Instant>,
    prompt_tokens: u64,
    generated: u64,
    parser: Duration,
}

impl Clock {
    fn snapshot(&self) -> TimingSnapshot {
        let now = Instant::now();
        let first = self.first_output.unwrap_or(now);
        TimingSnapshot {
            cached_prompt_tokens: 0,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated,
            timings: GenerationTimings {
                prompt_ms: millis(first - self.admitted),
                decode_ms: millis(now - first),
                time_to_first_token_ms: millis(first - self.started),
                sampler_ms: 0.0,
                parser_ms: millis(self.parser),
                draft_tokens: 0,
                accepted_draft_tokens: 0,
            },
        }
    }
}

/// Maps native parser events onto the semantic output stream, and checks each
/// completed value against its schema. Values are published as generated;
/// violations are reported.
struct Semantics {
    tool_calls: usize,
    reasoning: String,
    schemas: OutputSchemas,
    /// Open tool calls: name and arguments so far.
    calls: BTreeMap<u32, (String, String)>,
    /// Content, when it is JSON output with a schema.
    content: String,
}

impl Semantics {
    fn new(schemas: OutputSchemas) -> Self {
        Self {
            tool_calls: 0,
            reasoning: String::new(),
            schemas,
            calls: BTreeMap::new(),
            content: String::new(),
        }
    }

    fn translate(&mut self, events: Vec<Event>) -> Result<Vec<OutputEvent>, String> {
        let mut output = Vec::new();
        for event in events {
            match event {
                Event::Content { text } if !text.is_empty() => {
                    if self.schemas.constrains_output() {
                        self.content.push_str(&text);
                    }
                    output.push(OutputEvent::TextDelta(text));
                }
                Event::Reasoning { text } if !text.is_empty() => {
                    self.reasoning.push_str(&text);
                    output.push(OutputEvent::ReasoningDelta(text));
                }
                Event::Content { .. } | Event::Reasoning { .. } => {}
                Event::ToolStart { index, name, id } => {
                    self.tool_calls += 1;
                    self.calls.insert(index, (name.clone(), String::new()));
                    output.push(OutputEvent::ToolCallStarted {
                        index: index as usize,
                        id,
                        name,
                    });
                }
                Event::ToolArguments { index, text } if !text.is_empty() => {
                    let (_, arguments) = self
                        .calls
                        .get_mut(&index)
                        .ok_or("tool arguments precede their call")?;
                    arguments.push_str(&text);
                    output.push(OutputEvent::ToolInputDelta {
                        index: index as usize,
                        fragment: text,
                    });
                }
                Event::ToolArguments { .. } => {}
                Event::ToolComplete { index } => {
                    let (name, arguments) = self
                        .calls
                        .remove(&index)
                        .ok_or("a tool call completed before it started")?;
                    crate::telemetry::span_nonconforming_output(
                        &format!("tool {name}"),
                        &self.schemas.tool_call(&name, &arguments),
                    );
                    output.push(OutputEvent::ToolCallFinished {
                        index: index as usize,
                    });
                }
                Event::Finish { cause } => {
                    // Output that stopped early is incomplete, not nonconforming.
                    if cause == TerminalCause::Natural {
                        if let Some(conformance) = self.schemas.output(&self.content) {
                            crate::telemetry::span_nonconforming_output("output", &conformance);
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}
